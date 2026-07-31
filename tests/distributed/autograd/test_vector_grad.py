"""Paired-real statevector kernels retain native real autograd graphs."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from aicir import PauliString
from aicir.core.circuit import Circuit
from aicir.core.circuit import crx, cry, crz, rx, rxx, ry, rz, rzz, u2, u3
from aicir.distributed import parameter_shift_gradient
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.backend import DistNPUBackend
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.qml.deriv import psr4


def test_pair_vector_kernel_applies_trainable_local_matrix_with_real_leaves(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    theta = torch.tensor(0.37, dtype=torch.float32, requires_grad=True)
    matrix = torch.stack(
        (
            torch.stack((torch.cos(theta / 2.0), torch.zeros_like(theta))),
            torch.stack((torch.zeros_like(theta), torch.cos(theta / 2.0))),
        )
    ).to(torch.complex64)
    plan = _GatePlanner(backend, layout, 1).plan_matrix(
        matrix,
        (0,),
        instruction_index=0,
    )
    state = _Pair(
        torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
    )

    result = _PairVectorKernel(backend).apply(state, plan, operation_index=0)
    result.real.sum().backward()

    assert theta.grad is not None
    torch.testing.assert_close(theta.grad, -0.5 * torch.sin(theta / 2.0))


def test_pair_reducer_probabilities_are_normalized_and_differentiable(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )
    real = torch.tensor([[3.0], [4.0]], dtype=torch.float32, requires_grad=True)
    pair = _Pair(real, torch.zeros_like(real))

    probabilities = _PairReducer(backend).probabilities(pair, spec=None)
    probabilities[0].backward()

    torch.testing.assert_close(probabilities.reshape(-1), torch.tensor([0.36, 0.64]))
    torch.testing.assert_close(real.grad.reshape(-1), torch.tensor([0.1536, -0.1152]))


@pytest.mark.parametrize(
    ("gate_factory", "parameters", "four_term"),
    (
        (lambda p: rx(p[0], 0), (0.31,), False),
        (lambda p: ry(p[0], 0), (-0.47,), False),
        (lambda p: rz(p[0], 0), (0.29,), False),
        (lambda p: crx(p[0], 1, (0,)), (0.23,), True),
        (lambda p: cry(p[0], 1, (0,)), (-0.41,), True),
        (lambda p: crz(p[0], 1, (0,)), (0.37,), True),
        (lambda p: rzz(p[0], 0, 1), (-0.19,), False),
        (lambda p: rxx(p[0], 0, 1), (0.53,), False),
        (lambda p: u2(p[0], p[1], 0), (0.17, -0.29), False),
        (lambda p: u3(p[0], p[1], p[2], 0), (0.21, -0.33, 0.45), False),
    ),
    ids=("rx", "ry", "rz", "crx", "cry", "crz", "rzz", "rxx", "u2", "u3"),
)
def test_parameterized_gate_gradients_match_parameter_shift(monkeypatch, gate_factory, parameters, four_term):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    n_qubits = 2
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=0)
    spec = _ShardSpec.build(n_qubits, 1, 0, "vector", layout)
    initial_real = torch.tensor([[0.5], [0.5], [0.5], [0.5]], dtype=torch.float32)
    initial_imag = torch.tensor([[0.1], [-0.2], [0.3], [-0.4]], dtype=torch.float32)
    initial = _Pair(initial_real, initial_imag)
    observable = PauliString("XI", n_qubits=2)

    def objective(values, *, gradient=False):
        values_t = tuple(
            torch.tensor(float(value), dtype=torch.float32, requires_grad=gradient)
            for value in values
        )
        plan = _GatePlanner(backend, layout, n_qubits).plan(
            gate_factory(values_t), instruction_index=0
        )
        evolved = _PairVectorKernel(backend).apply(initial, plan, operation_index=0)
        value = _PairReducer(backend).expectation(evolved, spec, observable)
        return value, values_t

    value, leaves = objective(parameters, gradient=True)
    value.backward()
    native = np.array([float(leaf.grad) for leaf in leaves])
    oracle = lambda values: float(objective(values)[0].detach())
    shifted = (psr4 if four_term else parameter_shift_gradient)(
        oracle, np.asarray(parameters, dtype=np.float64)
    )

    np.testing.assert_allclose(native, shifted, atol=1e-4, rtol=1e-4)


def test_thirty_two_parameter_circuit_gradient_matches_parameter_shift(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    parameters = torch.linspace(-0.4, 0.5, 32, dtype=torch.float32, requires_grad=True)
    state = _Pair(
        torch.tensor([[2**-0.5], [2**-0.5]], dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
    )
    kernel = _PairVectorKernel(backend)
    planner = _GatePlanner(backend, layout, 1)
    for index, parameter in enumerate(parameters):
        state = kernel.apply(state, planner.plan(ry(parameter, 0), index), operation_index=index)
    value = _PairReducer(backend).expectation(state, spec, PauliString("Z", n_qubits=1))
    value.backward()

    total = float(parameters.detach().sum())
    expected = -np.cos(total) * np.ones(32, dtype=np.float64)
    np.testing.assert_allclose(parameters.grad.detach().numpy(), expected, atol=2e-4, rtol=2e-4)


def test_custom_unitary_uses_the_same_paired_real_kernel(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    unitary = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex64)
    instruction = {"type": "unitary", "parameter": unitary, "n_qubits": 1}
    plan = _GatePlanner(backend, layout, 1).plan(instruction, instruction_index=0)
    state = _Pair(
        torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True),
        torch.zeros((2, 1), dtype=torch.float32, requires_grad=True),
    )

    result = _PairVectorKernel(backend).apply(state, plan, operation_index=0)
    loss = result.abs_sq()[1].sum()
    loss.backward()

    torch.testing.assert_close(result.real, torch.zeros_like(result.real))
    torch.testing.assert_close(result.imag, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(state.real.grad, torch.tensor([[2.0], [0.0]]))


def test_shared_trainable_leaf_reduces_its_accumulated_adjoint_once(monkeypatch):
    class RecordingCommunicator:
        world_size = 2

        def __init__(self):
            self.calls = 0

        def all_reduce_sum_real(self, gradient):
            self.calls += 1
            return gradient * self.world_size

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    backend._communicator = RecordingCommunicator()
    planner = _GatePlanner(
        backend,
        _Layout.explicit((0, 1), n_qubits=2, distributed_axes=1),
        2,
    )
    theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)

    first = planner.plan(ry(theta, 1), instruction_index=0)
    second = planner.plan(ry(theta, 1), instruction_index=1)
    (first.local_matrix.real.sum() + second.local_matrix.real.sum()).backward()

    assert backend.communicator.calls == 1


def test_trainable_gate_planning_never_enters_complex_matrix_route(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    monkeypatch.setattr(
        "aicir.distributed.gates._gate_local_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("complex matrix route invoked")),
    )
    theta = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

    plan = _GatePlanner(backend, _Layout.explicit((0,), n_qubits=1, distributed_axes=0), 1).plan(
        ry(theta, 0), instruction_index=0
    )

    assert isinstance(plan.local_matrix, _Pair)

"""Paired-real statevector kernels retain native real autograd graphs."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from aicir import Hamiltonian, PauliString
from aicir.ir import Observable
from aicir.core.circuit import Circuit
from aicir.core.circuit import crx, cry, crz, rx, rxx, ry, rz, rzz, u2, u3
from aicir.distributed import parameter_shift_gradient
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.backend import DistNPUBackend
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.qml.deriv import psr4


def _float64_gate_objective(name, values):
    """Independent complex128 full-state oracle for the native gate suite."""

    values = np.asarray(values, dtype=np.float64)
    c, s = np.cos(values[0] / 2.0), np.sin(values[0] / 2.0)
    if name in {"rx", "crx"}: base = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    elif name in {"ry", "cry"}: base = np.array([[c, -s], [s, c]], dtype=np.complex128)
    elif name in {"rz", "crz"}: base = np.diag([c-1j*s, c+1j*s]).astype(np.complex128)
    elif name == "rzz": base = np.diag([c-1j*s, c+1j*s, c+1j*s, c-1j*s]).astype(np.complex128)
    elif name == "rxx": base = c*np.eye(4, dtype=np.complex128)-1j*s*np.kron(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
    elif name == "u2":
        phi, lam = values; base = np.array([[1, -np.exp(1j*lam)], [np.exp(1j*phi), np.exp(1j*(phi+lam))]], dtype=np.complex128)/np.sqrt(2)
    elif name == "u3":
        theta, phi, lam = values; base = np.array([[np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]], dtype=np.complex128)
    elif name == "custom": base = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    else: raise AssertionError(name)
    if name.startswith("cr"):
        base = np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), base]])
    elif base.shape == (2,2): base = np.kron(base, np.eye(2))
    psi = np.array([.5+.1j, .5-.2j, .5+.3j, .5-.4j], dtype=np.complex128)
    observable = np.kron(np.array([[0,1],[1,0]], dtype=np.complex128), np.eye(2))
    out = base @ psi
    return float(np.real(np.vdot(out, observable @ out)))


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
    matrix = _Pair(
        torch.stack(
            (
                torch.stack((torch.cos(theta / 2.0), torch.zeros_like(theta))),
                torch.stack((torch.zeros_like(theta), torch.cos(theta / 2.0))),
            )
        ),
        torch.zeros((2, 2), dtype=torch.float32),
    )
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


@pytest.mark.parametrize(
    ("name", "factory", "values", "four_term"),
    (
        ("rx", lambda p: rx(p[0], 0), (0.31,), False), ("ry", lambda p: ry(p[0], 0), (-.47,), False),
        ("rz", lambda p: rz(p[0], 0), (.29,), False), ("crx", lambda p: crx(p[0], 1, (0,)), (.23,), True),
        ("cry", lambda p: cry(p[0], 1, (0,)), (-.41,), True), ("crz", lambda p: crz(p[0], 1, (0,)), (.37,), True),
        ("rzz", lambda p: rzz(p[0], 0, 1), (-.19,), False), ("rxx", lambda p: rxx(p[0], 0, 1), (.53,), False),
        ("u2", lambda p: u2(p[0], p[1], 0), (.17,-.29), False), ("u3", lambda p: u3(p[0], p[1], p[2], 0), (.21,-.33,.45), False),
    ),
)
def test_named_gate_values_and_parameter_shift_match_independent_float64_oracle(monkeypatch, name, factory, values, four_term):
    monkeypatch.setenv("WORLD_SIZE", "1"); monkeypatch.setenv("RANK", "0"); monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    layout = _Layout.explicit((0, 1), n_qubits=2, distributed_axes=0); spec = _ShardSpec.build(2, 1, 0, "vector", layout)
    state = _Pair(torch.tensor([[.5],[.5],[.5],[.5]], dtype=torch.float32), torch.tensor([[.1],[-.2],[.3],[-.4]], dtype=torch.float32))
    def native(point):
        leaves = tuple(torch.tensor(float(x), dtype=torch.float32) for x in point)
        plan = _GatePlanner(backend, layout, 2).plan(factory(leaves), 0)
        return float(_PairReducer(backend).expectation(_PairVectorKernel(backend).apply(state, plan, operation_index=0), spec, PauliString("XI", n_qubits=2)).detach())
    np.testing.assert_allclose(native(values), _float64_gate_objective(name, values), atol=2e-5, rtol=2e-5)
    oracle = psr4 if four_term else parameter_shift_gradient
    np.testing.assert_allclose(oracle(lambda point: _float64_gate_objective(name, point), np.asarray(values)), oracle(lambda point: native(point), np.asarray(values)), atol=2e-4, rtol=2e-4)


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

    values = parameters.detach().numpy().astype(np.float64)
    # Independent float64 parameter-shift oracle: this does not call the
    # paired-real kernel or any production gate-matrix helper.
    shifted = parameter_shift_gradient(
        lambda point: -np.sin(np.sum(point)), values
    )
    np.testing.assert_allclose(parameters.grad.detach().numpy(), shifted, atol=2e-4, rtol=2e-4)


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
    context = _AutogradExecutionContext()
    planner = _GatePlanner(backend, _Layout.explicit((0, 1), n_qubits=2, distributed_axes=1), 2, execution_context=context)
    theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)

    first = planner.plan(ry(theta, 1), instruction_index=0)
    second = _GatePlanner(backend, planner._layout, 2, execution_context=context).plan(ry(theta, 1), instruction_index=1)
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


def test_trainable_complex_custom_unitary_is_rejected_at_native_boundary(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    matrix = torch.eye(2, dtype=torch.complex64, requires_grad=True)

    with pytest.raises(TypeError, match="requires_grad complex unitary"):
        _GatePlanner(backend, _Layout.explicit((0,), n_qubits=1, distributed_axes=0), 1).plan(
            {"type": "unitary", "parameter": matrix, "n_qubits": 1}, instruction_index=0
        )


@pytest.mark.parametrize(
    "observable",
    (
        PauliString("Z", n_qubits=1),
        Hamiltonian([("Z", 0.7), ("X", -0.2)]),
        Observable("matrix", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64), metadata={"qubits": (0,)}),
    ),
    ids=("pauli", "hamiltonian", "dense"),
)
def test_pair_reducer_observable_values_and_gradients_match_parameter_shift(monkeypatch, observable):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)

    def objective(value, grad=False):
        theta = torch.tensor(float(value), dtype=torch.float32, requires_grad=grad)
        state = _Pair(torch.tensor([[0.8], [0.6]], dtype=torch.float32), torch.zeros((2, 1), dtype=torch.float32))
        plan = _GatePlanner(backend, layout, 1).plan(ry(theta, 0), 0)
        return _PairReducer(backend).expectation(_PairVectorKernel(backend).apply(state, plan, operation_index=0), spec, observable), theta

    value, theta = objective(0.31, grad=True)
    value.backward()
    shifted = parameter_shift_gradient(lambda values: float(objective(values[0])[0].detach()), np.array([0.31]))[0]
    assert np.isfinite(float(value.detach()))
    assert float(theta.grad) == pytest.approx(float(shifted), abs=1e-4)


def test_probability_jacobian_matches_parameter_shift_vjp_basis(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)

    def probabilities(value, grad=False):
        theta = torch.tensor(float(value), dtype=torch.float32, requires_grad=grad)
        state = _Pair(torch.tensor([[1.0], [0.0]], dtype=torch.float32), torch.zeros((2, 1), dtype=torch.float32))
        plan = _GatePlanner(backend, layout, 1).plan(ry(theta, 0), 0)
        return _PairReducer(backend).probabilities(_PairVectorKernel(backend).apply(state, plan, operation_index=0), spec), theta

    native = []
    for basis in torch.eye(2, dtype=torch.float32):
        result, theta = probabilities(0.31, grad=True)
        (result * basis).sum().backward()
        native.append(float(theta.grad))
    shifted = [
        parameter_shift_gradient(lambda values, index=index: float(probabilities(values[0])[0][index]), np.array([0.31]))[0]
        for index in range(2)
    ]
    np.testing.assert_allclose(native, shifted, atol=1e-4, rtol=1e-4)

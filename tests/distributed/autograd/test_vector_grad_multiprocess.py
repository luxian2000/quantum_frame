"""Cross-shard paired-real statevector gradient regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from aicir import Hamiltonian, PauliString
from aicir.ir import Observable
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, parameter_shift_gradient
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _float64_ry_reference(full, theta, axis, n_qubits):
    """Independent CPU float64 full-state reference for one RY and Z_0."""

    real = torch.tensor(full, dtype=torch.float64, requires_grad=True)
    imag = torch.zeros_like(real, requires_grad=True)
    t = torch.tensor(float(theta), dtype=torch.float64)
    gate = torch.stack((torch.stack((torch.cos(t / 2), -torch.sin(t / 2))), torch.stack((torch.sin(t / 2), torch.cos(t / 2)))) )
    unitary = torch.ones((1, 1), dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    for qubit in range(n_qubits):
        unitary = torch.kron(unitary, gate if qubit == axis else identity)
    out_r, out_i = unitary @ real, unitary @ imag
    signs = torch.tensor([1.0 if index < (1 << (n_qubits - 1)) else -1.0 for index in range(1 << n_qubits)], dtype=torch.float64).reshape(-1, 1)
    (signs * (out_r.square() + out_i.square())).sum().backward()
    return real.grad.numpy(), imag.grad.numpy()


def _float64_observable_reference(full, theta, axis, n_qubits, kind):
    """Independent CPU complex128 values for all distributed observable classes."""

    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    unitary = np.ones((1, 1), dtype=np.complex128)
    for qubit in range(n_qubits):
        unitary = np.kron(unitary, gate if qubit == axis else identity)
    state = np.asarray(full, dtype=np.complex128).reshape(-1, 1)
    evolved = unitary @ state
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z0 = np.kron(z, np.eye(1 << (n_qubits - 1), dtype=np.complex128))
    if kind == "pauli" or kind == "dense":
        observable = z0
    elif kind == "hamiltonian":
        observable = 0.7 * z0 - 0.2 * np.kron(x, np.eye(1 << (n_qubits - 1), dtype=np.complex128))
    else:
        raise AssertionError(kind)
    return float(np.real(np.conj(evolved).T @ observable @ evolved)[0, 0])


def _worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float32)
    full = full / np.linalg.norm(full)
    local = full[spec.global_start : spec.global_stop].reshape(-1, 1)
    observables = (
        ("pauli", PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)),
        ("hamiltonian", Hamiltonian([("Z" + "I" * (n_qubits - 1), 0.7), ("X" + "I" * (n_qubits - 1), -0.2)])),
        ("dense", Observable("matrix", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64), metadata={"qubits": (0,)})),
    )

    results = []
    for axis in range(layout.distributed_axes):
        def objective(value, observable, *, trainable=False, state_trainable=False):
            theta = torch.tensor(float(value), dtype=torch.float32, requires_grad=trainable)
            real = torch.tensor(local, dtype=torch.float32, requires_grad=state_trainable)
            imag = torch.zeros_like(real, requires_grad=state_trainable)
            pair = _Pair(real, imag)
            plan = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext()).plan(ry(theta, axis), axis)
            backend.communicator.clear_communication_records()
            evolved = _PairVectorKernel(backend).apply(pair, plan, operation_index=axis)
            value = _PairReducer(backend).expectation(evolved, spec, observable)
            return value, theta, real, imag

        # Pauli also carries the independent initial-state float64 reference;
        # Hamiltonian and dense observables reuse the same distributed axis.
        value, theta, real, imag = objective(0.31, observables[0][1], trainable=True, state_trainable=True)
        value.backward()
        records = backend.communicator.communication_records
        exchanges = [record for record in records if record["kind"] == "exchange"]
        forward = sum(record["tag"] % 8 < 4 for record in exchanges)
        backward = sum(record["tag"] % 8 >= 4 for record in exchanges)
        shifted = parameter_shift_gradient(
            lambda values: float(objective(values[0], observables[0][1])[0].detach()), np.array([0.31])
        )[0]
        assert abs(float(theta.grad) - float(shifted)) <= 1e-4, (
            f"rank={rank} axis={axis} native={float(theta.grad)} shifted={float(shifted)}"
        )
        assert forward > 0 and backward > 0
        assert real.grad is not None and imag.grad is not None
        gathered_real = [torch.zeros_like(real.grad) for _ in range(world_size)]
        gathered_imag = [torch.zeros_like(imag.grad) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_real, real.grad)
        torch.distributed.all_gather(gathered_imag, imag.grad)
        if rank == 0:
            reference_real, reference_imag = _float64_ry_reference(full.reshape(-1, 1), 0.31, axis, n_qubits)
            np.testing.assert_allclose(
                torch.cat(gathered_real).numpy(), reference_real, atol=2e-4, rtol=2e-4
            )
            np.testing.assert_allclose(
                torch.cat(gathered_imag).numpy(), reference_imag, atol=2e-4, rtol=2e-4
            )
        for kind, observable in observables:
            current_value, _, _, _ = objective(0.31, observable)
            if rank == 0:
                np.testing.assert_allclose(
                    float(current_value.detach()),
                    _float64_observable_reference(full, 0.31, axis, n_qubits, kind),
                    atol=2e-5,
                    rtol=2e-5,
                )
        for _, observable in observables[1:]:
            current, current_theta, _, _ = objective(0.31, observable, trainable=True)
            current.backward()
            shifted_observable = parameter_shift_gradient(
                lambda values, observable=observable: float(objective(values[0], observable)[0].detach()), np.array([0.31])
            )[0]
            assert abs(float(current_theta.grad) - float(shifted_observable)) <= 1e-4
        results.append({"axis": axis, "gradient": float(theta.grad), "forward": forward, "backward": backward})

    if rank == 0:
        Path(output_path).write_text(json.dumps(results), encoding="utf-8")
    torch.distributed.destroy_process_group()


def _probability_worker(rank, world_size, port, output_path):
    """Compare every distributed probability Jacobian row with parameter shift."""

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    dimension = 1 << n_qubits
    indices = np.arange(spec.global_start, spec.global_stop, dtype=np.float32)
    local = ((indices + 1.0) / np.sqrt(dimension * (dimension + 1) * (2 * dimension + 1) / 6.0)).reshape(-1, 1)
    errors = []

    def cpu_probability(global_component, theta, axis):
        bit = 1 << (n_qubits - 1 - axis)
        norm = np.sqrt(dimension * (dimension + 1) * (2 * dimension + 1) / 6.0)
        own = (global_component + 1.0) / norm
        partner = ((global_component ^ bit) + 1.0) / norm
        amplitude = np.cos(theta / 2.0) * own + (
            np.sin(theta / 2.0) if global_component & bit else -np.sin(theta / 2.0)
        ) * partner
        return float(amplitude * amplitude)

    def probabilities(value, axis, *, gradient=False):
        theta = torch.tensor(float(value), dtype=torch.float32, requires_grad=gradient)
        pair = _Pair(torch.tensor(local, dtype=torch.float32), torch.zeros_like(torch.tensor(local, dtype=torch.float32)))
        plan = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext()).plan(ry(theta, axis), axis)
        return _PairReducer(backend).probabilities(_PairVectorKernel(backend).apply(pair, plan, operation_index=axis), spec), theta

    for axis in range(layout.distributed_axes):
        for global_component in range(dimension):
            values, theta = probabilities(0.31, axis, gradient=True)
            loss = values[global_component - spec.global_start] if spec.global_start <= global_component < spec.global_stop else values.sum() * 0.0
            loss.backward()
            shifted = parameter_shift_gradient(
                lambda point: cpu_probability(global_component, point[0], axis),
                np.array([0.31]),
            )[0]
            errors.append(abs(float(theta.grad) - float(shifted)))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"max_abs_error": max(errors)}), encoding="utf-8")
    torch.distributed.destroy_process_group()


def _unnormalized_probability_vjp_worker(rank, world_size, port, output_path):
    """Check the normalization denominator adjoint for sharded state leaves."""

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    dimension = 1 << n_qubits
    global_indices = np.arange(dimension, dtype=np.float64)
    full_real = (0.2 + global_indices / 7.0).reshape(-1, 1)
    full_imag = ((-0.3 + (global_indices % 3) / 5.0)).reshape(-1, 1)
    weights = np.linspace(-0.7, 0.9, dimension, dtype=np.float64).reshape(-1, 1)
    real = torch.tensor(full_real[spec.global_start : spec.global_stop], dtype=torch.float32, requires_grad=True)
    imag = torch.tensor(full_imag[spec.global_start : spec.global_stop], dtype=torch.float32, requires_grad=True)
    probabilities = _PairReducer(backend).probabilities(_Pair(real, imag), spec)
    local_weights = torch.tensor(weights[spec.global_start : spec.global_stop], dtype=torch.float32)
    (probabilities.reshape(-1, 1) * local_weights).sum().backward()
    gathered_real = [torch.zeros_like(real.grad) for _ in range(world_size)]
    gathered_imag = [torch.zeros_like(imag.grad) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_real, real.grad)
    torch.distributed.all_gather(gathered_imag, imag.grad)
    if rank == 0:
        reference_real = torch.tensor(full_real, dtype=torch.float64, requires_grad=True)
        reference_imag = torch.tensor(full_imag, dtype=torch.float64, requires_grad=True)
        reference_probability = (reference_real.square() + reference_imag.square())
        reference_probability = reference_probability / reference_probability.sum()
        (reference_probability * torch.tensor(weights, dtype=torch.float64)).sum().backward()
        Path(output_path).write_text(json.dumps({
            "real_error": float(np.max(np.abs(torch.cat(gathered_real).numpy() - reference_real.grad.numpy()))),
            "imag_error": float(np.max(np.abs(torch.cat(gathered_imag).numpy() - reference_imag.grad.numpy()))),
        }), encoding="utf-8")
    torch.distributed.destroy_process_group()


def _custom_pair_reference(real, imag, *, n_qubits, axis):
    """Independent complex128 full-state oracle for the custom paired unitary."""

    matrix = real.astype(np.complex128) + 1j * imag.astype(np.complex128)
    full_matrix = np.ones((1, 1), dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    for qubit in range(n_qubits):
        full_matrix = np.kron(full_matrix, matrix if qubit == axis else identity)
    dimension = 1 << n_qubits
    state = np.arange(1, dimension + 1, dtype=np.float64).astype(np.complex128)
    state /= np.linalg.norm(state)
    evolved = full_matrix @ (full_matrix @ state)
    signs = np.array([1.0 if index < dimension // 2 else -1.0 for index in range(dimension)])
    return float(np.sum(signs * np.abs(evolved) ** 2))


def _custom_pair_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    dimension = 1 << n_qubits
    local_indices = np.arange(spec.global_start, spec.global_stop, dtype=np.float32)
    local = ((local_indices + 1.0) / np.sqrt(dimension * (dimension + 1) * (2 * dimension + 1) / 6.0)).reshape(-1, 1)
    c, s = np.cos(0.31 / 2.0), np.sin(0.31 / 2.0)
    initial_real = np.array([[c, 0.0], [0.0, c]], dtype=np.float32)
    initial_imag = np.array([[0.0, -s], [-s, 0.0]], dtype=np.float32)
    original_reduce = backend.communicator.all_reduce_sum_real
    reductions = 0

    def counted_reduce(value):
        nonlocal reductions
        reductions += 1
        return original_reduce(value)

    backend.communicator.all_reduce_sum_real = counted_reduce
    results = []
    for axis in range(layout.distributed_axes):
        real = torch.tensor(initial_real, dtype=torch.float32, requires_grad=True)
        imag = torch.tensor(initial_imag, dtype=torch.float32, requires_grad=True)
        context = _AutogradExecutionContext()
        pair_matrix = _Pair(real, imag)
        first = _GatePlanner(backend, layout, n_qubits, execution_context=context).plan_matrix(pair_matrix, (axis,), instruction_index=0)
        second = _GatePlanner(backend, layout, n_qubits, execution_context=context).plan_matrix(pair_matrix, (axis,), instruction_index=1)
        state = _Pair(torch.tensor(local, dtype=torch.float32), torch.zeros_like(torch.tensor(local, dtype=torch.float32)))
        evolved = _PairVectorKernel(backend).apply(state, first, operation_index=0)
        evolved = _PairVectorKernel(backend).apply(evolved, second, operation_index=1)
        value = _PairReducer(backend).expectation(evolved, spec, PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits))
        before_backward = reductions
        value.backward()
        parameter_reductions = reductions - before_backward - 1  # reducer's replicated scalar seed
        reference_real = np.zeros_like(initial_real, dtype=np.float64)
        reference_imag = np.zeros_like(initial_imag, dtype=np.float64)
        epsilon = 1e-5
        for component, reference in ((initial_real, reference_real), (initial_imag, reference_imag)):
            for row in range(2):
                for column in range(2):
                    plus, minus = component.astype(np.float64).copy(), component.astype(np.float64).copy()
                    plus[row, column] += epsilon
                    minus[row, column] -= epsilon
                    if component is initial_real:
                        numerator = _custom_pair_reference(plus, initial_imag, n_qubits=n_qubits, axis=axis) - _custom_pair_reference(minus, initial_imag, n_qubits=n_qubits, axis=axis)
                    else:
                        numerator = _custom_pair_reference(initial_real, plus, n_qubits=n_qubits, axis=axis) - _custom_pair_reference(initial_real, minus, n_qubits=n_qubits, axis=axis)
                    reference[row, column] = numerator / (2.0 * epsilon)
        results.append({
            "axis": axis,
            "real_error": float(np.max(np.abs(real.grad.numpy() - reference_real))),
            "imag_error": float(np.max(np.abs(imag.grad.numpy() - reference_imag))),
            "parameter_reductions": parameter_reductions,
        })
    if rank == 0:
        Path(output_path).write_text(json.dumps(results), encoding="utf-8")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_cross_shard_parameter_gradients_use_forward_and_backward_p2p(tmp_path, world_size):
    output = tmp_path / f"world-{world_size}.json"
    mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    results = json.loads(output.read_text(encoding="utf-8"))
    assert [item["axis"] for item in results] == list(range(int(np.log2(world_size))))
    assert all(item["forward"] > 0 and item["backward"] > 0 for item in results)


@pytest.mark.parametrize("world_size", (2, 4))
def test_cross_shard_probability_full_local_vector_jacobian_matches_parameter_shift(tmp_path, world_size):
    output = tmp_path / f"probability-world-{world_size}.json"
    mp.spawn(_probability_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    assert json.loads(output.read_text(encoding="utf-8"))["max_abs_error"] <= 1e-4


@pytest.mark.parametrize("world_size", (2, 4))
def test_unnormalized_state_probability_vjp_matches_float64_normalization_derivative(tmp_path, world_size):
    output = tmp_path / f"unnormalized-probability-world-{world_size}.json"
    mp.spawn(_unnormalized_probability_vjp_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    errors = json.loads(output.read_text(encoding="utf-8"))
    assert errors["real_error"] <= 1e-4
    assert errors["imag_error"] <= 1e-4


@pytest.mark.parametrize("world_size", (2, 4))
def test_trainable_custom_paired_unitary_matches_complex128_and_wraps_each_leaf_once(tmp_path, world_size):
    output = tmp_path / f"custom-pair-world-{world_size}.json"
    mp.spawn(_custom_pair_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    results = json.loads(output.read_text(encoding="utf-8"))
    assert [item["axis"] for item in results] == list(range(int(np.log2(world_size))))
    assert all(item["real_error"] <= 5e-4 and item["imag_error"] <= 5e-4 for item in results)
    assert all(item["parameter_reductions"] == 2 for item in results)

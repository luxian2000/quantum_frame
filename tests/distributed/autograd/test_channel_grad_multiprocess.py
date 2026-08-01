"""Multi-rank paired-real channel transport and gradient coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from aicir import PauliString
from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._parameters import StinespringParam
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState
from aicir.noise import AmplitudeDampingChannel, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits, distributed_axes = int(np.log2(world_size)) + 1, int(np.log2(world_size))
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=distributed_axes)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    local = np.arange(vector_spec.global_start + 1, vector_spec.global_stop + 1, dtype=np.float64)
    local /= np.sqrt(sum((np.arange(1, (1 << n_qubits) + 1, dtype=np.float64)) ** 2))
    state = DistState.from_pair(
        _Pair(
            torch.tensor(local.reshape(-1, 1), dtype=torch.float32, requires_grad=True),
            torch.zeros(vector_spec.local_shape, dtype=torch.float32, requires_grad=True),
        ),
        spec=vector_spec,
        backend=backend,
    )
    axes = tuple(logical for logical, storage in enumerate(layout.logical_to_storage) if storage < distributed_axes)
    transport, gradients = [], []
    for index, axis in enumerate(axes):
        probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
        backend.communicator.clear_communication_records()
        evolved = _PairMatrixKernel(backend).apply_channel(state, BitFlipChannel(axis, probability), instruction_index=70 + index)
        value = _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits))
        value.backward()
        records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
        tags = {record["tag"] % 8 for record in records}
        transport.append({0, 1, 4, 5}.issubset(tags) and all(record["dtype"] == "torch.float32" and record["bytes"] > 0 for record in records))
        gradients.append(float(probability.grad))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"logical_axes": axes, "storage_axes": [layout.logical_to_storage[axis] for axis in axes], "transport": transport, "gradient_finite": all(np.isfinite(gradients))}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_channel_gradients_cover_every_distributed_storage_axis_with_forward_and_backward_p2p(world_size, tmp_path):
    output = tmp_path / f"channels-{world_size}.json"
    mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert sorted(result["storage_axes"]) == list(range(int(np.log2(world_size))))
    assert all(result["transport"])
    assert result["gradient_finite"]


def _stinespring_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits, distributed_axes = int(np.log2(world_size)) + 1, int(np.log2(world_size))
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=distributed_axes)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    local = np.arange(vector_spec.global_start + 1, vector_spec.global_stop + 1, dtype=np.float64)
    local /= np.sqrt(sum((np.arange(1, (1 << n_qubits) + 1, dtype=np.float64)) ** 2))
    state = DistState.from_pair(
        _Pair(torch.tensor(local.reshape(-1, 1), dtype=torch.float32, requires_grad=True), torch.zeros(vector_spec.local_shape, dtype=torch.float32, requires_grad=True)),
        spec=vector_spec,
        backend=backend,
    )
    dimension = 1 << n_qubits
    raw_real_leaf = torch.linspace(-0.7, 0.9, 4 * dimension * dimension, dtype=torch.float32, requires_grad=True)
    raw_imag_leaf = torch.linspace(0.8, -0.6, 4 * dimension * dimension, dtype=torch.float32, requires_grad=True)
    raw_real = raw_real_leaf.reshape(2 * dimension, 2 * dimension)
    raw_imag = raw_imag_leaf.reshape(2 * dimension, 2 * dimension)
    backend.communicator.clear_communication_records()
    parameter = StinespringParam(dimension, dimension, 2, raw_real, raw_imag)
    evolved = _PairMatrixKernel(backend).apply_channel(state, parameter, instruction_index=91)
    _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)).backward()
    records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
    if rank == 0:
        Path(output_path).write_text(json.dumps({"axes": list(range(distributed_axes)), "tags": sorted({record["tag"] % 8 for record in records}), "dtypes": sorted({record["dtype"] for record in records}), "real_grad": bool(torch.isfinite(raw_real_leaf.grad).all()), "imag_grad": bool(torch.isfinite(raw_imag_leaf.grad).all())}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_stinespring_gradients_cover_every_distributed_storage_axis_with_forward_and_backward_p2p(world_size, tmp_path):
    output = tmp_path / f"stinespring-{world_size}.json"
    mp.spawn(_stinespring_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert result["axes"] == list(range(int(np.log2(world_size))))
    assert {0, 1, 4, 5}.issubset(result["tags"])
    assert result["dtypes"] == ["torch.float32"]
    assert result["real_grad"] and result["imag_grad"]


def _channel_failure_worker(rank, world_size, port, output_path, case):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=1)
    spec = _ShardSpec.build(2, world_size, rank, "vector", layout)
    state = DistState.from_pair(_Pair(torch.ones(spec.local_shape, dtype=torch.float32), torch.zeros(spec.local_shape, dtype=torch.float32)), spec=spec, backend=backend)
    if case in {"invalid_probability", "divergent_probability"}:
        channel = BitFlipChannel(1, 1.2 if (case == "invalid_probability" and rank == 0) else (0.2 + 0.1 * rank if case == "divergent_probability" else 0.2))
    elif case in {"invalid_target", "divergent_target"}:
        channel = BitFlipChannel(4 if (case == "invalid_target" and rank == 0) else (rank if case == "divergent_target" else 1), 0.2)
    else:
        raw = torch.ones((4, 4), dtype=torch.float32)
        channel = StinespringParam(2, 2, 2, raw, torch.zeros_like(raw), target_qubits=(1,))
        if case == "invalid_shape" and rank == 0:
            object.__setattr__(channel, "real", torch.ones((2, 2), dtype=torch.float32))
        if case == "invalid_dimension" and rank == 0:
            object.__setattr__(channel, "input_dim", 3)
    backend.communicator.clear_communication_records()
    try:
        _PairMatrixKernel(backend).apply_channel(state, channel, instruction_index=123)
    except ValueError as error:
        message = str(error)
    else:
        message = ""
    messages = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(messages, message)
    torch.distributed.barrier()
    if rank == 0:
        Path(output_path).write_text(json.dumps({"messages": messages, "exchange_records": [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("case", ("invalid_probability", "invalid_target", "invalid_shape", "invalid_dimension", "divergent_probability", "divergent_target"))
def test_channel_preflight_synchronizes_invalid_or_divergent_metadata_before_p2p(case, tmp_path):
    output = tmp_path / f"channel-preflight-{case}.json"
    mp.spawn(_channel_failure_worker, args=(2, _free_port(), str(output), case), nprocs=2, join=True)
    result = json.loads(output.read_text())
    assert result["messages"][0]
    assert result["messages"] == [result["messages"][0]] * 2
    assert not result["exchange_records"]


def _embedded(matrix, *, storage_axis, n_qubits):
    result = np.array([[1.0]], dtype=np.complex128)
    for axis in range(n_qubits):
        result = np.kron(result, matrix if axis == storage_axis else np.eye(2))
    return result


def _kraus(kind, p):
    identity = np.eye(2, dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    if kind == "bit":
        return (np.sqrt(1.0 - p) * identity, np.sqrt(p) * x), z
    if kind == "phase":
        return (np.sqrt(1.0 - p) * identity, np.sqrt(p) * z), x
    if kind == "depolarizing":
        return (np.sqrt(1.0 - p) * identity, np.sqrt(p / 3.0) * x, np.sqrt(p / 3.0) * y, np.sqrt(p / 3.0) * z), z
    return (
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=np.complex128),
        np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=np.complex128),
    ), z


def _channel_reference(full, *, kind, p, storage_axis, n_qubits, weights):
    matrices, observable = _kraus(kind, p)
    rho = np.outer(full, full.conj())
    out = sum(_embedded(matrix, storage_axis=storage_axis, n_qubits=n_qubits) @ rho @ _embedded(matrix, storage_axis=storage_axis, n_qubits=n_qubits).conj().T for matrix in matrices)
    return np.real(np.diag(out)), float(np.trace(out @ _embedded(observable, storage_axis=storage_axis, n_qubits=n_qubits)).real), float(np.real(np.diag(out)) @ weights)


def _builtin_reference_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits, distributed_axes = int(np.log2(world_size)) + 1, int(np.log2(world_size))
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=distributed_axes)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    matrix_spec = _ShardSpec.build(n_qubits, world_size, rank, "matrix", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) + 1j * np.arange(2, (1 << n_qubits) + 2, dtype=np.float64) / 7.0
    full /= np.linalg.norm(full)
    local = full[vector_spec.global_start:vector_spec.global_stop].reshape(-1, 1)
    weights = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) / (1 << n_qubits)
    factories = {"bit": BitFlipChannel, "phase": PhaseFlipChannel, "depolarizing": DepolarizingChannel, "amplitude": AmplitudeDampingChannel}
    axes = tuple(logical for logical, storage in enumerate(layout.logical_to_storage) if storage < distributed_axes)
    errors, p2p = [], []
    for logical_axis in axes:
        storage_axis = layout.logical_to_storage[logical_axis]
        for kind, factory in factories.items():
            probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
            state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32, requires_grad=True), torch.tensor(local.imag, dtype=torch.float32, requires_grad=True)), spec=vector_spec, backend=backend)
            backend.communicator.clear_communication_records()
            channel = factory(logical_axis, probability) if kind != "amplitude" else factory(logical_axis, probability)
            evolved = _PairMatrixKernel(backend).apply_channel(state, channel, instruction_index=300 + 16 * logical_axis + len(errors))
            matrices, observable = _kraus(kind, 0.23)
            labels = ["I"] * n_qubits; labels[logical_axis] = "X" if kind == "phase" else "Z"
            pauli = _PairReducer(backend).expectation(evolved._pair, matrix_spec, PauliString("".join(labels), n_qubits=n_qubits))
            pauli.backward()
            actual_probabilities = np.concatenate([part.cpu().numpy() for part in backend.communicator.all_gather_real(_PairReducer(backend).probabilities(evolved._pair, matrix_spec).detach())])
            expected_probabilities, expected_pauli, _ = _channel_reference(full, kind=kind, p=0.23, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)
            epsilon = 1e-6
            plus = _channel_reference(full, kind=kind, p=0.23 + epsilon, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)[1]
            minus = _channel_reference(full, kind=kind, p=0.23 - epsilon, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)[1]
            errors.extend((float(np.max(np.abs(actual_probabilities - expected_probabilities))), abs(float(pauli.detach()) - expected_pauli), abs(float(probability.grad) - (plus - minus) / (2 * epsilon))))
            records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
            p2p.append({0, 1, 4, 5}.issubset({record["tag"] % 8 for record in records}) and all(record["dtype"] == "torch.float32" and record["bytes"] > 0 for record in records))

            probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
            state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32, requires_grad=True), torch.tensor(local.imag, dtype=torch.float32, requires_grad=True)), spec=vector_spec, backend=backend)
            evolved = _PairMatrixKernel(backend).apply_channel(state, factory(logical_axis, probability), instruction_index=500 + 16 * logical_axis + len(errors))
            probabilities = _PairReducer(backend).probabilities(evolved._pair, matrix_spec)
            local_weights = torch.tensor(weights[matrix_spec.global_start:matrix_spec.global_stop], dtype=torch.float32)
            (probabilities * local_weights).sum().backward()
            actual_weighted = float(np.concatenate([part.cpu().numpy() for part in backend.communicator.all_gather_real(probabilities.detach())]) @ weights)
            _, _, weighted = _channel_reference(full, kind=kind, p=0.23, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)
            plus = _channel_reference(full, kind=kind, p=0.23 + epsilon, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)[2]
            minus = _channel_reference(full, kind=kind, p=0.23 - epsilon, storage_axis=storage_axis, n_qubits=n_qubits, weights=weights)[2]
            errors.extend((abs(actual_weighted - weighted), abs(float(probability.grad) - (plus - minus) / (2 * epsilon))))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"axes": sorted(layout.logical_to_storage[axis] for axis in axes), "max_error": max(errors), "p2p": all(p2p)}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_all_builtin_channels_match_complex128_probability_vjp_and_pauli_finite_differences_on_every_axis(world_size, tmp_path):
    output = tmp_path / f"builtin-reference-{world_size}.json"
    mp.spawn(_builtin_reference_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert result["axes"] == list(range(int(np.log2(world_size))))
    assert result["max_error"] <= 1e-4
    assert result["p2p"]


def _stinespring_reference_kraus(real, imag):
    current = np.eye(4, 2, dtype=np.complex128)
    for vector in real + 1j * imag:
        norm_squared = np.vdot(vector, vector).real
        if norm_squared != 0.0:
            current = current - 2.0 * np.outer(vector, vector.conj() @ current) / norm_squared
    return tuple(current[index * 2:(index + 1) * 2] for index in range(2))


def _stinespring_reference(full, real, imag, *, storage_axis, n_qubits):
    rho = np.outer(full, full.conj())
    out = sum(_embedded(matrix, storage_axis=storage_axis, n_qubits=n_qubits) @ rho @ _embedded(matrix, storage_axis=storage_axis, n_qubits=n_qubits).conj().T for matrix in _stinespring_reference_kraus(real, imag))
    z = _embedded(np.diag([1.0, -1.0]), storage_axis=storage_axis, n_qubits=n_qubits)
    return out, float(np.trace(out @ z).real)


def _stinespring_reference_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits, distributed_axes = int(np.log2(world_size)) + 1, int(np.log2(world_size))
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=distributed_axes)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    matrix_spec = _ShardSpec.build(n_qubits, world_size, rank, "matrix", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) + 1j * np.arange(2, (1 << n_qubits) + 2, dtype=np.float64) / 5.0
    full /= np.linalg.norm(full)
    local = full[vector_spec.global_start:vector_spec.global_stop].reshape(-1, 1)
    base_real = np.array([[0.2, -0.4, 0.1, 0.3], [0.5, 0.7, -0.2, 0.6], [-0.1, 0.2, 0.8, -0.3], [0.4, -0.5, 0.9, 0.1]], dtype=np.float64)
    base_imag = np.array([[0.1, 0.3, -0.6, 0.2], [-0.7, 0.4, 0.5, -0.1], [0.6, -0.2, 0.3, 0.8], [0.2, 0.1, -0.4, 0.7]], dtype=np.float64)
    axes = tuple(logical for logical, storage in enumerate(layout.logical_to_storage) if storage < distributed_axes)
    errors, physical, transport = [], [], []
    for logical_axis in axes:
        raw_real = torch.tensor(base_real, dtype=torch.float32, requires_grad=True)
        raw_imag = torch.tensor(base_imag, dtype=torch.float32, requires_grad=True)
        state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32, requires_grad=True), torch.tensor(local.imag, dtype=torch.float32, requires_grad=True)), spec=vector_spec, backend=backend)
        backend.communicator.clear_communication_records()
        evolved = _PairMatrixKernel(backend).apply_channel(state, StinespringParam(2, 2, 2, raw_real, raw_imag, target_qubits=(logical_axis,)), instruction_index=900 + logical_axis)
        labels = ["I"] * n_qubits; labels[logical_axis] = "Z"
        value = _PairReducer(backend).expectation(evolved._pair, matrix_spec, PauliString("".join(labels), n_qubits=n_qubits))
        value.backward()
        actual = np.concatenate([part.cpu().numpy() for part in backend.communicator.all_gather_real(evolved._pair.real.detach())], axis=0) + 1j * np.concatenate([part.cpu().numpy() for part in backend.communicator.all_gather_real(evolved._pair.imag.detach())], axis=0)
        expected, expected_value = _stinespring_reference(full, base_real, base_imag, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=n_qubits)
        epsilon = 1e-5
        real_plus, real_minus = base_real.copy(), base_real.copy(); real_plus[0, 0] += epsilon; real_minus[0, 0] -= epsilon
        imag_plus, imag_minus = base_imag.copy(), base_imag.copy(); imag_plus[0, 0] += epsilon; imag_minus[0, 0] -= epsilon
        real_fd = (_stinespring_reference(full, real_plus, base_imag, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=n_qubits)[1] - _stinespring_reference(full, real_minus, base_imag, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=n_qubits)[1]) / (2 * epsilon)
        imag_fd = (_stinespring_reference(full, base_real, imag_plus, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=n_qubits)[1] - _stinespring_reference(full, base_real, imag_minus, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=n_qubits)[1]) / (2 * epsilon)
        errors.extend((float(np.max(np.abs(actual - expected))), abs(float(value.detach()) - expected_value), abs(float(raw_real.grad[0, 0]) - real_fd), abs(float(raw_imag.grad[0, 0]) - imag_fd)))
        physical.extend((float(np.max(np.abs(actual - actual.conj().T))), max(0.0, -float(np.linalg.eigvalsh(actual).min())), abs(float(np.trace(actual).real) - 1.0)))
        records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
        transport.append({0, 1, 4, 5}.issubset({record["tag"] % 8 for record in records}) and all(record["dtype"] == "torch.float32" and record["bytes"] > 0 for record in records))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"axes": sorted(layout.logical_to_storage[axis] for axis in axes), "max_error": max(errors), "physical_error": max(physical), "transport": all(transport)}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_stinespring_nonzero_targets_match_float64_raw_gradients_on_every_distributed_axis(world_size, tmp_path):
    output = tmp_path / f"stinespring-reference-{world_size}.json"
    mp.spawn(_stinespring_reference_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert result["axes"] == list(range(int(np.log2(world_size))))
    assert result["max_error"] <= 1e-4
    assert result["physical_error"] <= 1e-5
    assert result["transport"]

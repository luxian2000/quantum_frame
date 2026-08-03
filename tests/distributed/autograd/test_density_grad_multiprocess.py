"""Independent complex128 distributed density-gradient regression coverage."""

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
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._density import _preflight_density_operation
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState
from aicir.ir import Observable


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _embedded(matrix, *, storage_axis, n_qubits):
    result = np.array([[1.0]], dtype=np.complex128)
    for axis in range(n_qubits):
        result = np.kron(result, matrix if axis == storage_axis else np.eye(2))
    return result


def _reference(full, theta, *, logical_axis, layout, observable=None, weights=None):
    gate = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)
    out = _embedded(gate, storage_axis=layout.logical_to_storage[logical_axis], n_qubits=layout.n_qubits) @ full
    density = np.outer(out, out.conj())
    probabilities = np.real(np.diag(density))
    if weights is not None:
        return probabilities, float(probabilities @ weights), density
    return probabilities, float(np.trace(density @ observable).real), density


def _worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    distributed_axes = int(np.log2(world_size))
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=distributed_axes)
    matrix_spec = _ShardSpec.build(n_qubits, world_size, rank, "matrix", layout)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) + 1j * np.arange(2, (1 << n_qubits) + 2, dtype=np.float64) / 7.0
    full /= np.linalg.norm(full)
    local = full[vector_spec.global_start:vector_spec.global_stop].reshape(-1, 1)
    logical_axes = tuple(logical for logical, storage in enumerate(layout.logical_to_storage) if storage < distributed_axes)
    storage_axes = {layout.logical_to_storage[logical] for logical in logical_axes}
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    local_logical = next(logical for logical, storage in enumerate(layout.logical_to_storage) if storage >= distributed_axes)
    observables = (
        ("pauli", PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits), _embedded(z, storage_axis=layout.logical_to_storage[0], n_qubits=n_qubits)),
        ("dense", Observable("matrix", x.astype(np.complex64), metadata={"qubits": (local_logical,)}), _embedded(x, storage_axis=layout.logical_to_storage[local_logical], n_qubits=n_qubits)),
    )
    errors, physical, transport = [], [], []
    for logical_axis in logical_axes:
        for name, observable, reference_observable in observables:
            theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
            state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32, requires_grad=True), torch.tensor(local.imag, dtype=torch.float32, requires_grad=True)), spec=vector_spec, backend=backend)
            kernel = _PairMatrixKernel(backend)
            backend.communicator.clear_communication_records()
            density = kernel.promote_vector(state)
            plan = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext()).plan(ry(theta, logical_axis), logical_axis)
            operation_index = 100 + logical_axis
            evolved = kernel.apply_unitary(density, plan, operation_index=operation_index)
            value = _PairReducer(backend).expectation(evolved._pair, matrix_spec, observable)
            value.backward()
            _, expected, reference_density = _reference(full, 0.31, logical_axis=logical_axis, layout=layout, observable=reference_observable)
            epsilon = 1e-6
            gradient = (_reference(full, 0.31 + epsilon, logical_axis=logical_axis, layout=layout, observable=reference_observable)[1] - _reference(full, 0.31 - epsilon, logical_axis=logical_axis, layout=layout, observable=reference_observable)[1]) / (2 * epsilon)
            errors.extend((abs(float(value.detach()) - expected), abs(float(theta.grad) - gradient)))
            real_parts = backend.communicator.all_gather_real(evolved._pair.real.detach())
            imag_parts = backend.communicator.all_gather_real(evolved._pair.imag.detach())
            actual_density = np.concatenate([part.cpu().numpy() for part in real_parts], axis=0) + 1j * np.concatenate([part.cpu().numpy() for part in imag_parts], axis=0)
            physical.extend((float(np.max(np.abs(actual_density - reference_density))), float(np.max(np.abs(actual_density - actual_density.conj().T))), max(0.0, -float(np.linalg.eigvalsh(actual_density).min())), abs(float(np.trace(actual_density).real) - 1.0)))
            base = operation_index * world_size + 1
            records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange" and record["tag"] // 8 == base]
            transport.append(bool(records) and {record["tag"] % 8 for record in records} >= {0, 1, 4, 5} and all(record["dtype"] == "torch.float32" and record["peer"] is not None and record["peer"] != rank and 0 <= record["peer"] < world_size and record["bytes"] > 0 for record in records))

        # Probability-vector VJP: each rank contributes weights for its local
        # diagonal, while replicated-parameter backward produces the global VJP.
        theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
        state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32, requires_grad=True), torch.tensor(local.imag, dtype=torch.float32, requires_grad=True)), spec=vector_spec, backend=backend)
        kernel = _PairMatrixKernel(backend)
        density = kernel.promote_vector(state)
        plan = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext()).plan(ry(theta, logical_axis), logical_axis)
        evolved = kernel.apply_unitary(density, plan, operation_index=300 + logical_axis)
        probabilities = _PairReducer(backend).probabilities(evolved._pair, matrix_spec)
        weights = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) / float(1 << n_qubits)
        local_weights = torch.tensor(weights[matrix_spec.global_start:matrix_spec.global_stop], dtype=torch.float32)
        (probabilities * local_weights).sum().backward()
        gathered = backend.communicator.all_gather_real(probabilities.detach())
        actual_probabilities = np.concatenate([value.cpu().numpy() for value in gathered])
        expected_probabilities, _, _ = _reference(full, 0.31, logical_axis=logical_axis, layout=layout, weights=weights)
        epsilon = 1e-6
        probability_gradient = (_reference(full, 0.31 + epsilon, logical_axis=logical_axis, layout=layout, weights=weights)[1] - _reference(full, 0.31 - epsilon, logical_axis=logical_axis, layout=layout, weights=weights)[1]) / (2 * epsilon)
        errors.extend((float(np.max(np.abs(actual_probabilities - expected_probabilities))), abs(float(theta.grad) - probability_gradient)))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"logical_axes": logical_axes, "storage_axes": sorted(storage_axes), "max_error": max(errors), "max_physical_error": max(physical), "transport": all(transport)}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_density_gradients_cover_every_distributed_storage_axis(world_size, tmp_path):
    output = tmp_path / f"density-{world_size}.json"
    mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert result["storage_axes"] == list(range(int(np.log2(world_size))))
    assert len(result["logical_axes"]) == int(np.log2(world_size))
    assert result["max_error"] <= 1e-4
    assert result["max_physical_error"] <= 1e-5
    assert result["transport"]


def _failure_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=1)
    spec = _ShardSpec.build(2, world_size, rank, "matrix", layout)
    state = DistState.from_pair(_Pair(torch.ones(spec.local_shape, dtype=torch.float32), torch.zeros(spec.local_shape, dtype=torch.float32)), spec=spec, backend=backend)
    # Deliberately disagree only in metadata.  The preflight must reject before
    # any density data-plane P2P operation is possible.
    plan = _GatePlanner(backend, layout, 2).plan(ry(torch.tensor(0.2), rank), rank)
    backend.communicator.clear_communication_records()
    try:
        _preflight_density_operation(state, plan, operation_index=7)
    except ValueError as error:
        message = str(error)
    else:
        message = ""
    messages = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(messages, message)
    torch.distributed.barrier()
    if rank == 0:
        Path(output_path).write_text(json.dumps({"messages": messages, "records": backend.communicator.communication_records}))
    torch.distributed.destroy_process_group()


def test_density_preflight_rejects_rank_mismatched_axis_before_p2p(tmp_path):
    output = tmp_path / "density-failure.json"
    mp.spawn(_failure_worker, args=(2, _free_port(), str(output)), nprocs=2, join=True)
    result = json.loads(output.read_text())
    assert result["messages"] == ["分布式 autograd collective 参数不一致: density plan", "分布式 autograd collective 参数不一致: density plan"]
    assert not [record for record in result["records"] if record["kind"] == "exchange"]

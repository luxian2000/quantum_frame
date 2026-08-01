"""Distributed paired-real density backward coverage for every shard axis."""

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
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    matrix_spec = _ShardSpec.build(n_qubits, world_size, rank, "matrix", layout)
    vector_spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float64) + 1j * np.arange(2, (1 << n_qubits) + 2, dtype=np.float64) / 7.0
    full /= np.linalg.norm(full)
    local = full[vector_spec.global_start:vector_spec.global_stop].reshape(-1, 1)
    errors, records_ok = [], []
    for axis in range(layout.distributed_axes):
        theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
        state = DistState.from_pair(_Pair(torch.tensor(local.real, dtype=torch.float32), torch.tensor(local.imag, dtype=torch.float32)), spec=vector_spec, backend=backend)
        kernel = _PairMatrixKernel(backend)
        density = kernel.promote_vector(state)
        planner = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext())
        backend.communicator.clear_communication_records()
        evolved = kernel.apply_unitary(density, planner.plan(ry(theta, axis), axis), operation_index=axis + 1)
        value = _PairReducer(backend).expectation(evolved._pair, matrix_spec, PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits))
        value.backward()
        eps = 1e-6
        def reference(t):
            gate = np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=np.complex128)
            unitary = np.array([[1.0]], dtype=np.complex128)
            # logical axis maps to storage axis; the local shard representation is storage order.
            storage_axis = layout.logical_to_storage[axis]
            for qubit in range(n_qubits): unitary = np.kron(unitary, gate if qubit == storage_axis else np.eye(2))
            out = unitary @ full.reshape(-1, 1)
            observable = np.array([[1.0]], dtype=np.complex128)
            observable_axis = layout.logical_to_storage[0]
            for qubit in range(n_qubits):
                observable = np.kron(
                    observable,
                    np.diag([1.0, -1.0]) if qubit == observable_axis else np.eye(2),
                )
            return float(np.real(np.conj(out).T @ observable @ out)[0, 0])
        expected = (reference(0.31 + eps) - reference(0.31 - eps)) / (2 * eps)
        errors.append(abs(float(theta.grad) - expected))
        records_ok.append(all(record["dtype"] == "torch.float32" for record in backend.communicator.communication_records))
    if rank == 0:
        Path(output_path).write_text(json.dumps({"max_error": max(errors), "records_ok": all(records_ok)}))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_density_gradients_cover_every_distributed_axis(world_size, tmp_path):
    output = tmp_path / f"density-{world_size}.json"
    mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    result = json.loads(output.read_text())
    assert result["max_error"] <= 1e-4
    assert result["records_ok"]

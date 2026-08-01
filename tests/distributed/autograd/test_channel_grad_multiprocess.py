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
from aicir.noise import BitFlipChannel


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

"""Initial-state paired-real gradients use collective-safe preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.multiprocessing as mp

from aicir import Circuit, PauliString
from aicir.distributed import DistNPUBackend, DistSimulator, DistState, PureStateParam
from aicir.distributed._contracts import AUTOGRAD_ERROR
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join_spawn_context(context, *, timeout=30):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "distributed initial-state test timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5)
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode == 0 for process in context.processes)


def _worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    simulator = DistSimulator(backend)
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=1)
    spec = _ShardSpec.build(2, world_size, rank, "vector", layout)
    observable = PauliString("ZI", n_qubits=2)
    try:
        root_real = torch.arange(1, 5, dtype=torch.float32, requires_grad=True) if rank == 0 else None
        root_imag = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float32, requires_grad=True) if rank == 0 else None
        root_state = simulator._prepare_initial_state(
            n_qubits=2, layout=layout,
            initial_state=PureStateParam(root_real, root_imag) if rank == 0 else None,
            initial_density_matrix=None,
        )
        root_value = _PairReducer(backend).expectation(root_state._pair, spec, observable)
        root_value.backward()

        local_real = torch.full(spec.local_shape, float(rank + 1), dtype=torch.float32, requires_grad=True)
        local_imag = torch.full(spec.local_shape, float(-rank), dtype=torch.float32, requires_grad=True)
        sharded = DistState.from_pair(_Pair(local_real, local_imag), spec=spec, backend=backend)
        sharded_state = simulator._prepare_initial_state(
            n_qubits=2, layout=layout, initial_state=sharded, initial_density_matrix=None,
        )
        _PairReducer(backend).probabilities(sharded_state._pair, spec).sum().backward()

        mismatch = DistState.from_pair(
            _Pair(
                torch.ones(spec.local_shape, dtype=torch.float32, requires_grad=rank == 0),
                torch.zeros(spec.local_shape, dtype=torch.float32, requires_grad=rank == 0),
            ), spec=spec, backend=backend,
        )
        try:
            simulator._prepare_initial_state(n_qubits=2, layout=layout, initial_state=mismatch, initial_density_matrix=None)
        except ValueError as error:
            mismatch_error = str(error)
        else:
            mismatch_error = "NO_ERROR"
        torch.distributed.barrier()

        try:
            simulator._prepare_initial_state(
                n_qubits=2, layout=layout,
                initial_state=(torch.tensor([1, 0, 0, 0], dtype=torch.complex64, requires_grad=True) if rank == 0 else None),
                initial_density_matrix=None,
            )
        except ValueError as error:
            complex_error = str(error)
        else:
            complex_error = "NO_ERROR"
        torch.distributed.barrier()

        try:
            simulator.run(Circuit(n_qubits=2), initial_state=(PureStateParam(root_real, root_imag) if rank == 0 else None), shots=1, collapse=True)
        except ValueError as error:
            sample_error = str(error)
        else:
            sample_error = "NO_ERROR"
        torch.distributed.barrier()
        if rank == 0:
            Path(output_path).write_text(json.dumps({
                "root_grad": root_real.grad is not None and root_imag.grad is not None,
                "local_grad": local_real.grad is not None and local_imag.grad is not None,
                "mismatch_error": mismatch_error,
                "complex_error": complex_error,
                "sample_error": sample_error,
            }), encoding="utf-8")
    finally:
        torch.distributed.destroy_process_group()


def test_paired_initial_states_keep_owner_gradients_and_recover_from_errors(tmp_path):
    context = mp.spawn(_worker, args=(2, _free_port(), str(tmp_path / "result.json")), nprocs=2, join=False)
    _join_spawn_context(context)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["root_grad"]
    assert result["local_grad"]
    assert result["mismatch_error"] == "DistState paired-real requires_grad 在各 rank 间不一致"
    assert result["complex_error"] == "原生 distributed autograd 不接受 requires_grad complex initial_state；请使用 PureStateParam(real, imag)"
    assert result["sample_error"] == AUTOGRAD_ERROR

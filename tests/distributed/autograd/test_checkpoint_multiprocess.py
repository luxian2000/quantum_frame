"""Multi-rank deterministic paired-real checkpoint replay coverage."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from aicir import Circuit, PauliString
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, DistSimulator
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join(context, timeout=45):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "checkpoint multiprocess test timed out"
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
    assert all(process.exitcode == 0 for process in context.processes)


def _state(backend, layout, *, kind):
    n_qubits = layout.n_qubits
    spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, kind, layout)
    vector = np.arange(1, (1 << n_qubits) + 1, dtype=np.float32) + 1j * np.arange(2, (1 << n_qubits) + 2, dtype=np.float32) / 9.0
    vector /= np.linalg.norm(vector)
    if kind == "matrix":
        full = np.outer(vector, vector.conj())
        local = full[spec.global_start:spec.global_stop]
    else:
        local = vector[spec.global_start:spec.global_stop, None]
    return DistState.from_pair(
        _Pair(torch.tensor(local.real, dtype=torch.float32), torch.tensor(local.imag, dtype=torch.float32)),
        spec=spec,
        backend=backend,
    )


def _noncontiguous_density_state(backend, layout):
    state = _state(backend, layout, kind="matrix")
    # A transposed paired view models the density right-action output.  Replay
    # must transport it through the existing P2P boundary without a full-state
    # copy in the density kernel.
    return DistState.from_pair(
        _Pair(
            state._pair.real.transpose(0, 1).contiguous().transpose(0, 1),
            state._pair.imag.transpose(0, 1).contiguous().transpose(0, 1),
        ),
        spec=state.spec,
        backend=backend,
    )


def _evaluate(simulator, backend, layout, kind, policy):
    theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(n_qubits=layout.n_qubits)
    # A distributed logical axis under this non-identity layout forces paired
    # P2P traffic; a local axis checks that checkpoint boundaries do not alter
    # plan/index sequencing in mixed layouts.
    for index in range(5):
        circuit.append(ry(theta, index % layout.n_qubits))
    backend.communicator.clear_communication_records()
    evolved, metrics = simulator._run_paired_real(
        circuit,
        initial_state=_state(backend, layout, kind=kind),
        layout=layout,
        grad_checkpoint=policy,
        available_memory_bytes=1 << 30,
    )
    value = _PairReducer(backend).expectation(
        evolved._pair,
        evolved.spec,
        PauliString("Z" + "I" * (layout.n_qubits - 1), n_qubits=layout.n_qubits),
    )
    value.backward()
    records = [
        (record["peer"], record["tag"])
        for record in backend.communicator.communication_records
        if record["kind"] == "exchange"
    ]
    return {
        "value": float(value.detach().cpu()),
        "grad": float(theta.grad.detach().cpu()),
        "interval": metrics.interval,
        "saved": metrics.saved_state_count,
        "recomputed": metrics.recomputed_gate_count,
        "shape": tuple(evolved.local_shape),
        "records": records,
        "memory_source": metrics.memory_source,
    }


def _worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    simulator = DistSimulator(backend)
    n_qubits = int(math.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(reversed(range(n_qubits))), n_qubits=n_qubits, distributed_axes=int(math.log2(world_size)))
    results = {
        kind: {str(policy): _evaluate(simulator, backend, layout, kind, policy) for policy in ("none", "auto", 1, 4, 16)}
        for kind in ("vector", "matrix")
    }
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, results)
    if rank == 0:
        Path(output_path).write_text(json.dumps(gathered), encoding="utf-8")
    torch.distributed.destroy_process_group()


def _noncontiguous_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=1)
    theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(n_qubits=2)
    circuit.append(ry(theta, 1))
    circuit.append(ry(theta, 1))
    circuit.append(ry(theta, 1))
    evolved, metrics = DistSimulator(backend)._run_paired_real(
        circuit, initial_state=_noncontiguous_density_state(backend, layout), layout=layout, grad_checkpoint=1,
    )
    value = _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("ZI", n_qubits=2))
    value.backward()
    if rank == 0:
        Path(output_path).write_text(json.dumps({"grad": float(theta.grad), "recomputed": metrics.recomputed_gate_count, "contiguous": evolved._pair.real.is_contiguous()}))
    torch.distributed.destroy_process_group()


def _mismatch_worker(rank, world_size, port, output_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=1)
    circuit = Circuit(n_qubits=2)
    circuit.append(ry(torch.tensor(0.2), 1))
    backend.communicator.clear_communication_records()
    try:
        DistSimulator(backend)._run_paired_real(circuit, initial_state=_state(backend, layout, kind="vector"), layout=layout, grad_checkpoint=1 if rank == 0 else 4)
    except ValueError as error:
        message = str(error)
    else:
        message = "NO_ERROR"
    messages = [None] * world_size
    torch.distributed.all_gather_object(messages, message)
    torch.distributed.barrier()
    if rank == 0:
        Path(output_path).write_text(json.dumps({"messages": messages, "records": backend.communicator.communication_records}), encoding="utf-8")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_checkpoint_intervals_agree_and_preserve_vector_density_gradients(world_size, tmp_path):
    output = tmp_path / f"checkpoint-{world_size}.json"
    context = mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=False)
    _join(context)
    ranks = json.loads(output.read_text(encoding="utf-8"))
    for kind in ("vector", "matrix"):
        reference = ranks[0][kind]["none"]
        for rank in ranks:
            assert rank[kind]["none"]["value"] == pytest.approx(reference["value"], abs=1e-6)
            assert rank[kind]["none"]["grad"] == pytest.approx(reference["grad"], abs=1e-6)
        for policy in ("auto", "1", "4", "16"):
            intervals = {rank[kind][policy]["interval"] for rank in ranks}
            assert len(intervals) == 1
            assert len({rank[kind][policy]["memory_source"] for rank in ranks}) == 1
            actual = ranks[0][kind][policy]
            assert actual["value"] == pytest.approx(reference["value"], abs=1e-6)
            assert actual["grad"] == pytest.approx(reference["grad"], abs=1e-5)
            assert actual["saved"] == (6 if policy in {"auto", "1"} else 3 if policy == "4" else 2)
            assert actual["recomputed"] > 0
            # Replay repeats the original forward P2P order; it may add those
            # records, but never changes peers/tags or operation identities.
            assert {tuple(item) for item in actual["records"]} >= {tuple(item) for item in reference["records"]}
        assert reference["saved"] == 6
        assert reference["recomputed"] == 0


def test_checkpoint_replays_noncontiguous_density_pair_at_transport_boundary(tmp_path):
    output = tmp_path / "noncontiguous.json"
    context = mp.spawn(_noncontiguous_worker, args=(2, _free_port(), str(output)), nprocs=2, join=False)
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert np.isfinite(result["grad"])
    assert result["recomputed"] == 3


def test_checkpoint_interval_disagreement_fails_before_data_p2p_and_recovers(tmp_path):
    output = tmp_path / "mismatch.json"
    context = mp.spawn(_mismatch_worker, args=(2, _free_port(), str(output)), nprocs=2, join=False)
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["messages"] == ["各 rank 的梯度检查点间隔不一致"] * 2
    assert not [record for record in result["records"] if record["kind"] == "exchange"]

"""Public DistSimulator autograd routing contracts."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import socket

import pytest
import torch
import torch.multiprocessing as mp

from aicir import Circuit, PauliString, ry
from aicir.distributed import DistNPUBackend, DistSimulator, DistState, PureStateParam
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _simulator(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistSimulator(
        DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)
    )


def test_non_trainable_run_keeps_forward_engine(monkeypatch):
    """Routing must not perturb the existing complex64 forward path."""

    simulator = _simulator(monkeypatch)
    called = False
    original = simulator._run_paired_real

    def paired(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(simulator, "_run_paired_real", paired)
    result = simulator.run(Circuit(ry(0.31, 0), n_qubits=1))

    assert not called
    assert result.state._pair is None
    assert not result.local_probabilities.requires_grad


def test_trainable_gate_routes_to_paired_real_engine(monkeypatch):
    """A real float32 gate leaf opens the native paired-real route."""

    simulator = _simulator(monkeypatch)
    theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    result = simulator.run(
        Circuit(ry(theta, 0), n_qubits=1),
        observables={"z": PauliString("Z", n_qubits=1)},
        grad_checkpoint="none",
    )

    assert result.state._pair is not None
    assert result.is_differentiable
    assert result.local_probabilities.requires_grad
    assert result.expectations["z"].requires_grad
    result.expectations["z"].backward()
    assert theta.grad is not None
    assert abs(float(theta.grad)) > 1e-5


@pytest.mark.parametrize("kind", ("root", "sharded"))
def test_trainable_initial_states_route_to_paired_real_engine(monkeypatch, kind):
    """Root-owned and sharded paired-real initial states use one public route."""

    simulator = _simulator(monkeypatch)
    if kind == "root":
        real = torch.tensor([1.0, 0.0], dtype=torch.float32, requires_grad=True)
        imag = torch.zeros(2, dtype=torch.float32, requires_grad=True)
        initial_state = PureStateParam(real, imag)
    else:
        layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
        spec = _ShardSpec.build(1, 1, 0, "vector", layout)
        real = torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True)
        imag = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
        initial_state = DistState.from_pair(
            _Pair(real, imag), spec=spec, backend=simulator.backend
        )

    result = simulator.run(
        Circuit(n_qubits=1),
        initial_state=initial_state,
        observables={"z": PauliString("Z", n_qubits=1)},
    )

    assert result.state._pair is not None
    result.expectations["z"].backward()
    assert real.grad is not None
    assert imag.grad is not None


def _two_rank_route_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True, process_group_backend="gloo"
    )
    theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    try:
        result = DistSimulator(backend).run(
            Circuit(ry(theta, 0), n_qubits=2),
            observables={"z": PauliString("ZI", n_qubits=2)},
        )
        result.expectations["z"].backward()
        records = backend.communicator.all_gather_real(
            torch.tensor(
                [
                    float(result.state._pair is not None),
                    float(theta.grad.detach().cpu()),
                ],
                dtype=torch.float32,
                device=backend._device,
            )
        )
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(
                    [
                        {"pair": bool(item[0]), "gradient": float(item[1])}
                        for item in (record.detach().cpu().tolist() for record in records)
                    ]
                ),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_trainable_gate_routes_collectively(tmp_path):
    """Every rank selects the paired-real graph before sharded transport."""

    output = tmp_path / "route.json"
    mp.spawn(
        _two_rank_route_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    records = json.loads(output.read_text(encoding="utf-8"))
    assert all(record["pair"] for record in records)
    assert records[0]["gradient"] == pytest.approx(records[1]["gradient"], abs=1e-6)


def _two_rank_route_mismatch_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True, process_group_backend="gloo"
    )
    try:
        parameter = torch.tensor(
            0.1, dtype=torch.float32, requires_grad=rank == 0
        )
        try:
            DistSimulator(backend).run(Circuit(ry(parameter, 0), n_qubits=2))
        except ValueError as error:
            payload = f"{type(error).__name__}:{error}"
        else:
            payload = "NO_ERROR"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        gathered = backend.communicator.all_gather_real(
            torch.tensor(
                [float(value) for value in digest],
                dtype=torch.float32,
                device=backend._device,
            )
        )
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "payload": payload,
                        "digests": [
                            [int(value) for value in item.detach().cpu().tolist()]
                            for item in gathered
                        ],
                    }
                ),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_one_rank_trainable_mismatch_fails_before_state_transport(tmp_path):
    """A collective routing disagreement has one exact all-rank error."""

    output = tmp_path / "route-mismatch.json"
    mp.spawn(
        _two_rank_route_mismatch_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["payload"] == "ValueError:各 rank 的自动微分路由不一致"
    assert len({tuple(digest) for digest in result["digests"]}) == 1

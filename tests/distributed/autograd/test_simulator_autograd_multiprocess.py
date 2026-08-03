"""Public DistSimulator autograd routing contracts."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from aicir import Circuit, Observable, PauliString, ry
from aicir.distributed import (
    DensityParam,
    DistNPUBackend,
    DistSimulator,
    DistState,
    PureStateParam,
)
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.noise import DepolarizingChannel, NoiseModel


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


def test_trainable_noise_leaf_routes_and_receives_runtime_gradient(monkeypatch):
    simulator = _simulator(monkeypatch)
    probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(ry(0.41, 0), n_qubits=1)
    circuit.noise_model = NoiseModel().add_channel(
        DepolarizingChannel(0, probability),
        after_gates=("ry",),
    )

    result = simulator.run(
        circuit,
        observables={"z": PauliString("Z", n_qubits=1)},
        grad_checkpoint="none",
    )
    result.expectations["z"].backward()

    assert result.is_differentiable
    assert probability.grad is not None
    assert torch.isfinite(probability.grad)
    assert abs(float(probability.grad)) > 1e-5


@pytest.mark.parametrize("kind", ("root", "sharded"))
def test_frozen_paired_inputs_keep_prior_exact_rejection(monkeypatch, kind):
    simulator = _simulator(monkeypatch)
    if kind == "root":
        initial_state = PureStateParam(
            torch.tensor([1.0, 0.25], dtype=torch.float32),
            torch.tensor([0.0, -0.5], dtype=torch.float32),
        )
    else:
        layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
        spec = _ShardSpec.build(1, 1, 0, "vector", layout)
        initial_state = DistState.from_pair(
            _Pair(
                torch.tensor([[1.0], [0.25]], dtype=torch.float32),
                torch.tensor([[0.0], [-0.5]], dtype=torch.float32),
            ),
            spec=spec,
            backend=simulator.backend,
        )

    with pytest.raises(
        ValueError,
        match="^DistSimulator 首期仅支持前向模拟，不支持自动微分$",
    ):
        simulator.run(
            Circuit(ry(0.31, 0), n_qubits=1),
            initial_state=initial_state,
            observables={"z": PauliString("Z", n_qubits=1)},
        )


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


def _two_rank_density_worker(rank, world_size, port, output_path):
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
        real = (
            torch.tensor(
                [[1.0, 0.2, -0.1, 0.3], [0.1, 0.7, 0.2, -0.2],
                 [0.4, -0.3, 0.8, 0.1], [0.2, 0.1, -0.4, 0.9]],
                dtype=torch.float32,
                requires_grad=True,
            )
            if rank == 0
            else None
        )
        imag = (
            torch.tensor(
                [[0.0, 0.1, 0.2, -0.1], [-0.2, 0.0, 0.1, 0.3],
                 [0.1, -0.2, 0.0, 0.1], [0.3, 0.1, -0.1, 0.0]],
                dtype=torch.float32,
                requires_grad=True,
            )
            if rank == 0
            else None
        )
        result = DistSimulator(backend).run(
            Circuit(ry(0.27, 0), n_qubits=2),
            initial_density_matrix=(
                DensityParam(real, imag) if rank == 0 else None
            ),
            observables={"z": PauliString("ZI", n_qubits=2)},
            grad_checkpoint="none",
        )
        result.expectations["z"].backward()
        records = backend.communicator.all_gather_real(
            torch.tensor(
                [
                    float(result.is_differentiable),
                    float(
                        rank != 0
                        or (
                            real.grad is not None
                            and imag.grad is not None
                            and torch.isfinite(real.grad).all()
                            and torch.isfinite(imag.grad).all()
                            and real.grad.abs().max() > 1e-6
                        )
                    ),
                ],
                dtype=torch.float32,
                device=backend._device,
            )
        )
        if rank == 0:
            Path(output_path).write_text(
                json.dumps([item.detach().cpu().tolist() for item in records]),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_root_density_param_keeps_owner_gradient(tmp_path):
    output = tmp_path / "density.json"
    mp.spawn(
        _two_rank_density_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    records = json.loads(output.read_text(encoding="utf-8"))
    assert all(record[0] == 1.0 for record in records)
    assert all(record[1] == 1.0 for record in records)


def _two_rank_preflight_worker(rank, world_size, port, output_path, case):
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
        theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
        circuit = Circuit(ry(theta, 0), n_qubits=2)
        kwargs = {"observables": {"z": PauliString("ZI", n_qubits=2)}}
        if case == "checkpoint":
            kwargs["grad_checkpoint"] = "invalid" if rank == 0 else "auto"
        elif case == "observable":
            matrix = np.eye(2, dtype=np.complex64)
            if rank == 1:
                matrix[0, 1] = matrix[1, 0] = 0.25
            kwargs["observables"] = {
                "z": Observable.matrix(
                    matrix,
                    metadata={"qubits": (0,)},
                )
            }
        elif case == "noise":
            circuit.noise_model = NoiseModel().add_channel(
                DepolarizingChannel(
                    0,
                    torch.tensor(
                        0.2 if rank == 0 else 0.3,
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                ),
                after_gates=("ry",),
            )
        try:
            DistSimulator(backend).run(circuit, **kwargs)
        except Exception as error:  # noqa: BLE001 - exact public contract
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
                        "unique": len(
                            {
                                tuple(int(value) for value in item.detach().cpu())
                                for item in gathered
                            }
                        ),
                    }
                ),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "checkpoint",
            "ValueError:grad_checkpoint 必须是 'none'、'auto' 或正整数",
        ),
        ("observable", "ValueError:各 rank 的 observable schema 不一致"),
        ("noise", "ValueError:各 rank 的线路、噪声模型或参数内容不一致"),
    ),
)
def test_one_rank_invalid_preflight_is_exact_and_collective_safe(
    tmp_path, case, expected
):
    output = tmp_path / f"{case}.json"
    mp.spawn(
        _two_rank_preflight_worker,
        args=(2, _free_port(), str(output), case),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result == {"payload": expected, "unique": 1}


def _two_rank_capability_before_planning_worker(
    rank, world_size, port, output_path, case
):
    """Record all data-plane entry points before the expected rejection."""

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
    original_plan = _GatePlanner.plan
    original_scatter = backend.communicator.scatter_from_root
    original_real_scatter = backend.communicator.scatter_from_root_real
    evidence = {"plans": 0, "scatters": 0}

    def record_plan(self, gate, instruction_index):
        evidence["plans"] += 1
        return original_plan(self, gate, instruction_index)

    def record_scatter(*args, **kwargs):
        evidence["scatters"] += 1
        return original_scatter(*args, **kwargs)

    def record_real_scatter(*args, **kwargs):
        evidence["scatters"] += 1
        return original_real_scatter(*args, **kwargs)

    _GatePlanner.plan = record_plan
    backend.communicator.scatter_from_root = record_scatter
    backend.communicator.scatter_from_root_real = record_real_scatter
    try:
        theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
        circuit = Circuit(ry(theta, 0), n_qubits=2)
        kwargs = {}
        if case == "frozen_complex_initial_state":
            kwargs["initial_state"] = (
                torch.tensor(
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    dtype=torch.complex64,
                )
                if rank == 0
                else None
            )
        elif case == "unsupported_gate":
            circuit = Circuit(
                {"type": "unsupported", "target_qubit": 0, "parameter": theta},
                n_qubits=2,
            )
        elif case == "unsupported_channel":
            circuit.noise_model = NoiseModel().add_channel(
                object(), after_gates=("ry",)
            )
        elif case == "unsupported_observable":
            kwargs["observables"] = {"bad": object()}
        else:
            raise AssertionError(f"unknown preflight case: {case}")

        try:
            DistSimulator(backend).run(circuit, **kwargs)
        except Exception as error:  # noqa: BLE001 - exact public contract
            payload = f"{type(error).__name__}:{error}"
        else:
            payload = "NO_ERROR"
        snapshot = torch.tensor(
            [evidence["plans"], evidence["scatters"], len(backend.communicator.communication_records)],
            dtype=torch.int64,
            device=backend._device,
        )
        gathered = [torch.empty_like(snapshot) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, snapshot)
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "payload": payload,
                        "records": [item.detach().cpu().tolist() for item in gathered],
                    }
                ),
                encoding="utf-8",
            )
    finally:
        _GatePlanner.plan = original_plan
        backend.communicator.scatter_from_root = original_scatter
        backend.communicator.scatter_from_root_real = original_real_scatter
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "frozen_complex_initial_state",
            "TypeError:自动微分模式的初态必须是 PureStateParam、DensityParam 或 paired-real DistState",
        ),
        (
            "unsupported_gate",
            "ValueError:指令 'unsupported' 没有可用于分布式执行的局部门矩阵",
        ),
        (
            "unsupported_channel",
            "TypeError:自动微分模式不支持噪声通道 object",
        ),
        (
            "unsupported_observable",
            "TypeError:自动微分模式不支持 observable 'bad'",
        ),
    ),
)
def test_trainable_capability_failures_happen_before_planning_or_transport(
    tmp_path, case, expected
):
    """Invalid native requests must never enter planner or state transport."""

    output = tmp_path / f"{case}.json"
    mp.spawn(
        _two_rank_capability_before_planning_worker,
        args=(2, _free_port(), str(output), case),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["payload"] == expected
    assert result["records"] == [[0, 0, 0], [0, 0, 0]]


def _two_rank_reversed_unsupported_observables_worker(
    rank, world_size, port, output_path
):
    """Capability rejection must not depend on each rank's mapping insertion order."""

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
    original_plan = _GatePlanner.plan
    original_scatter = backend.communicator.scatter_from_root
    original_real_scatter = backend.communicator.scatter_from_root_real
    evidence = {"plans": 0, "scatters": 0}

    def record_plan(self, gate, instruction_index):
        evidence["plans"] += 1
        return original_plan(self, gate, instruction_index)

    def record_scatter(*args, **kwargs):
        evidence["scatters"] += 1
        return original_scatter(*args, **kwargs)

    def record_real_scatter(*args, **kwargs):
        evidence["scatters"] += 1
        return original_real_scatter(*args, **kwargs)

    _GatePlanner.plan = record_plan
    backend.communicator.scatter_from_root = record_scatter
    backend.communicator.scatter_from_root_real = record_real_scatter
    try:
        theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
        entries = [("second", object()), ("first", object())]
        observables = dict(entries if rank == 0 else reversed(entries))
        try:
            DistSimulator(backend).run(
                Circuit(ry(theta, 0), n_qubits=2), observables=observables
            )
        except Exception as error:  # noqa: BLE001 - exact public contract
            payload = f"{type(error).__name__}:{error}"
        else:
            payload = "NO_ERROR"
        snapshots = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            snapshots,
            {
                "payload": payload,
                "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "plans": evidence["plans"],
                "scatters": evidence["scatters"],
                "records": len(backend.communicator.communication_records),
            },
        )
        if rank == 0:
            Path(output_path).write_text(json.dumps(snapshots), encoding="utf-8")
    finally:
        _GatePlanner.plan = original_plan
        backend.communicator.scatter_from_root = original_scatter
        backend.communicator.scatter_from_root_real = original_real_scatter
        torch.distributed.destroy_process_group()


def test_unsupported_observable_rejection_is_rank_deterministic_before_transport(tmp_path):
    output = tmp_path / "reversed-unsupported-observables.json"
    mp.spawn(
        _two_rank_reversed_unsupported_observables_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert [item["payload"] for item in result] == [
        "TypeError:自动微分模式不支持 observable 'first'"
    ] * 2
    assert len({item["digest"] for item in result}) == 1
    assert all(
        item["plans"] == item["scatters"] == item["records"] == 0
        for item in result
    )

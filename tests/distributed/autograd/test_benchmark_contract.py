import json
import os
from pathlib import Path
import socket
import time

import pytest
import scripts.npu.distributed_autograd_benchmark as benchmark
import scripts.npu.distributed_autograd_probe as probe
import torch
import torch.multiprocessing as mp

from aicir.distributed import DistNPUBackend

from scripts.npu.distributed_autograd_benchmark import _validate_benchmark_report


def _report(**overrides):
    result = {
        "communication_mode": "baseline", "gradient_method": "native", "path": "statevector",
        "world_size": 2, "n_qubits": 24, "depth": 64, "parameters": 32, "warmups": 5, "runs": 30,
        "forward_ms_median": 0.0, "backward_ms_median": 0.0, "gradient_ms_median": 0.0,
        "gradient_ms_p95": 0.0, "peak_memory_bytes": 1, "p2p_bytes": 1, "wait_ms": 0.0,
        "buffer_reuse_count": 0, "fallback_to_cpu": False,
    }
    result.update(overrides)
    return result


def test_benchmark_contract_accepts_frozen_baseline_schema():
    _validate_benchmark_report(_report())


@pytest.mark.parametrize("overrides", ({"runs": 0}, {"gradient_ms_p95": -1}, {"fallback_to_cpu": True}, {"path": "density", "gradient_method": "parameter_shift"}))
def test_benchmark_contract_rejects_invalid_reports(overrides):
    with pytest.raises(ValueError):
        _validate_benchmark_report(_report(**overrides))


def test_cli_honors_counts_and_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "benchmark.json"
    observed = {}

    def run(args):
        observed.update(warmups=args.warmups, runs=args.runs)
        return _report(warmups=args.warmups, runs=args.runs)

    monkeypatch.setattr(benchmark, "_run_benchmark", run)
    monkeypatch.setattr("sys.argv", [
        "benchmark", "--communication-mode", "reuse",
        "--gradient-method", "native", "--path", "statevector",
        "--n-qubits", "2", "--depth", "1", "--parameters", "1",
        "--warmups", "6", "--runs", "31", "--output-json", str(output),
    ])
    assert benchmark.main() == 0
    assert observed == {"warmups": 6, "runs": 31}
    assert json.loads(output.read_text()) == _report(warmups=6, runs=31)


def test_timing_synchronizes_each_forward_and_gradient_boundary(monkeypatch):
    calls = []

    backend = type("Backend", (), {"_device": "cpu"})()
    monkeypatch.setattr(probe, "_synchronize_npu", lambda backend: calls.append(1))
    probe._timed_benchmark_iteration(backend, lambda: "state", lambda: "gradient")
    assert len(calls) == 4


@pytest.mark.parametrize(
    ("path", "method"),
    (
        ("statevector", "native"),
        ("statevector", "parameter_shift"),
        ("density", "native"),
        ("density", "finite_difference"),
        ("noise", "native"),
        ("noise", "finite_difference"),
        ("stinespring", "native"),
        ("stinespring", "finite_difference"),
    ),
)
def test_shared_workload_dispatch_accepts_only_honest_path_method_pairs(path, method):
    probe._validate_benchmark_workload_config(
        path=path,
        gradient_method=method,
        n_qubits=3,
        depth=2,
        parameters=2,
        world_size=4,
    )


@pytest.mark.parametrize(
    ("path", "method", "n_qubits"),
    (
        ("density", "parameter_shift", 3),
        ("stinespring", "parameter_shift", 3),
        ("statevector", "native", 1),
    ),
)
def test_shared_workload_dispatch_rejects_unimplemented_or_non_sharded_combinations(path, method, n_qubits):
    with pytest.raises(ValueError):
        probe._validate_benchmark_workload_config(
            path=path,
            gradient_method=method,
            n_qubits=n_qubits,
            depth=2,
            parameters=2,
            world_size=4,
        )


@pytest.mark.parametrize(
    ("path", "method", "expected_family"),
    (
        ("statevector", "parameter_shift", "gate_angle"),
        ("statevector", "finite_difference", "raw_state"),
        ("density", "finite_difference", "density_factor"),
        ("noise", "finite_difference", "channel_logit"),
        ("stinespring", "finite_difference", "stinespring_factor"),
    ),
)
def test_benchmark_parameter_family_prevents_oracle_relabelling(path, method, expected_family):
    assert probe._resolve_benchmark_parameter_family(path, method) == expected_family


def test_raw_state_finite_difference_is_a_valid_distinct_statevector_workload():
    probe._validate_benchmark_workload_config(
        path="statevector",
        gradient_method="finite_difference",
        parameter_family="raw_state",
        n_qubits=3,
        depth=1,
        parameters=1,
        world_size=4,
    )
    with pytest.raises(ValueError, match="raw_state"):
        probe._validate_benchmark_workload_config(
            path="statevector",
            gradient_method="finite_difference",
            parameter_family="gate_angle",
            n_qubits=3,
            depth=1,
            parameters=1,
            world_size=4,
        )


def test_density_factor_workload_materializes_full_factor_only_on_root(monkeypatch):
    layout = probe._Layout.explicit((0, 1), n_qubits=2, distributed_axes=1)
    captures = []

    def scatter(pair, **kwargs):
        captures.append(pair)
        return probe._Pair(pair.real[0], pair.imag[0]) if pair is not None else probe._Pair(
            torch.zeros(kwargs["local_shape"], dtype=torch.float32),
            torch.zeros(kwargs["local_shape"], dtype=torch.float32),
        )

    monkeypatch.setattr(probe, "_scatter_root_pair", scatter)
    for rank in (0, 1):
        backend = type("Backend", (), {
            "rank": rank,
            "world_size": 2,
            "_device": torch.device("cpu"),
            "communicator": object(),
        })()
        vector_spec = probe._ShardSpec.build(2, 2, rank, "vector", layout)
        state = probe._benchmark_density_factor_state(
            backend,
            vector_spec,
            (torch.tensor(0.19, dtype=torch.float32, requires_grad=rank == 0),),
        )
        assert state.local_shape == (2, 4)
    assert captures[0] is not None
    assert captures[0].real.shape == (2, 2, 4)
    assert captures[1] is None


def test_performance_records_every_native_oracle_pair_and_communication_mode(monkeypatch):
    calls = []
    backend = type("Backend", (), {
        "world_size": 2,
        "communicator": type("C", (), {"set_autograd_communication_mode": lambda self, mode: None})(),
    })()

    def run(_backend, **kwargs):
        calls.append(kwargs)
        return {
            "parameter_family": probe._resolve_benchmark_parameter_family(
                kwargs["path"], kwargs["gradient_method"], kwargs.get("parameter_family"),
            ),
            "state_max_abs_error": 0.0,
            "gradient_max_abs_error": 0.0,
            "p2p_bytes": 8,
            "wait_ms": 0.0,
            "buffer_reuse_count": 0 if kwargs["communication_mode"] == "baseline" else 1,
            "all_handles_complete": True,
            "fallback_to_cpu": False,
        }

    monkeypatch.setattr(probe, "run_benchmark_workload", run)
    performance = probe._performance_section(backend)
    expected_pairs = {
        ("statevector", "gate_angle", "native"),
        ("statevector", "gate_angle", "parameter_shift"),
        ("statevector", "raw_state", "native"),
        ("statevector", "raw_state", "finite_difference"),
        ("density", "density_factor", "native"),
        ("density", "density_factor", "finite_difference"),
        ("noise", "channel_logit", "native"),
        ("noise", "channel_logit", "finite_difference"),
        ("stinespring", "stinespring_factor", "native"),
        ("stinespring", "stinespring_factor", "finite_difference"),
    }
    assert {
        (item["path"], item["parameter_family"], item["gradient_method"])
        for item in calls
    } == expected_pairs
    assert {item["communication_mode"] for item in calls} == {"baseline", "reuse", "overlap"}
    assert len(calls) == 3 * len(expected_pairs)
    assert performance["passed"] is True


def test_shared_workload_rejects_a_single_rank_non_p2p_benchmark():
    with pytest.raises(ValueError, match="multi-rank"):
        probe._validate_benchmark_workload_config(
            path="statevector",
            gradient_method="native",
            n_qubits=1,
            depth=1,
            parameters=1,
            world_size=1,
        )


def test_cli_delegates_all_workload_fields_to_shared_runner(monkeypatch):
    calls = []
    backend = type("Backend", (), {"world_size": 2, "_device": "cpu", "communicator": type("C", (), {
        "set_autograd_communication_mode": lambda self, mode: None,
        "all_gather_real": lambda self, payload: [payload.clone(), payload.clone()],
    })()})()
    monkeypatch.setattr(benchmark, "_strict_backend", lambda **_: backend)
    monkeypatch.setattr(benchmark.torch, "npu", type("Npu", (), {"reset_peak_memory_stats": lambda *args: None, "max_memory_allocated": lambda *args: 17})(), raising=False)
    monkeypatch.setattr(benchmark.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(benchmark, "run_benchmark_workload", lambda bk, **kwargs: calls.append((bk, kwargs)) or {
        "forward_ms_median": 1.0, "backward_ms_median": 2.0, "gradient_ms_median": 3.0,
        "gradient_ms_p95": 4.0, "p2p_bytes": 5, "wait_ms": 6.0, "buffer_reuse_count": 7,
        "state_max_abs_error": 0.0, "gradient_max_abs_error": 0.0, "all_handles_complete": True,
    })
    args = type("Args", (), {
        "communication_mode": "overlap", "gradient_method": "native", "path": "statevector",
        "n_qubits": 2, "depth": 3, "parameters": 4, "warmups": 5, "runs": 6,
    })()
    report = benchmark._run_benchmark(args)
    assert calls == [(backend, {"communication_mode": "overlap", "path": "statevector", "gradient_method": "native", "n_qubits": 2, "depth": 3, "parameters": 4, "warmups": 5, "runs": 6})]
    assert report["forward_ms_median"] == 1.0


def test_run_benchmark_accepts_raw_state_finite_difference_report(monkeypatch):
    backend = type("Backend", (), {
        "rank": 0, "world_size": 1, "_device": "cpu",
        "communicator": type("C", (), {"set_autograd_communication_mode": lambda self, mode: None})(),
    })()
    monkeypatch.setattr(benchmark, "_strict_backend", lambda **_: backend)
    monkeypatch.setattr(
        benchmark.torch,
        "npu",
        type("Npu", (), {"reset_peak_memory_stats": lambda *args: None, "max_memory_allocated": lambda *args: 17})(),
        raising=False,
    )
    monkeypatch.setattr(benchmark.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(benchmark, "run_benchmark_workload", lambda *_args, **_kwargs: _runner_metrics())
    args = type("Args", (), {
        "communication_mode": "baseline", "gradient_method": "finite_difference", "path": "statevector",
        "n_qubits": 2, "depth": 1, "parameters": 1, "warmups": 1, "runs": 1,
    })()

    report = benchmark._run_benchmark(args)

    assert report["path"] == "statevector"
    assert report["gradient_method"] == "finite_difference"


def _runner_metrics(**overrides):
    result = {
        "forward_ms_median": 1.0, "backward_ms_median": 2.0,
        "gradient_ms_median": 3.0, "gradient_ms_p95": 4.0,
        "p2p_bytes": 5, "wait_ms": 6.0, "buffer_reuse_count": 7,
        "state_max_abs_error": 0.0, "gradient_max_abs_error": 0.0,
        "all_handles_complete": True, "fallback_to_cpu": False,
    }
    result.update(overrides)
    return result


def test_collective_report_uses_float32_payload_and_rejects_a_failed_rank():
    class Communicator:
        def __init__(self):
            self.payloads = []

        def all_gather_real(self, payload):
            self.payloads.append(payload)
            return [payload.clone(), torch.zeros_like(payload)]

    communicator = Communicator()
    backend = type("Backend", (), {"rank": 0, "world_size": 2, "_device": "cpu", "communicator": communicator})()
    with pytest.raises(RuntimeError, match="another rank"):
        benchmark._collective_benchmark_metrics(backend, _runner_metrics(), peak_memory_bytes=8)
    assert len(communicator.payloads) == 1
    assert communicator.payloads[0].dtype == torch.float32
    assert communicator.payloads[0].is_contiguous()


@pytest.mark.parametrize("overrides", ({"all_handles_complete": False}, {"fallback_to_cpu": True}))
def test_run_benchmark_rejects_incomplete_handles_or_runner_fallback(monkeypatch, overrides):
    backend = type("Backend", (), {
        "rank": 0, "world_size": 1, "_device": "cpu",
        "communicator": type("C", (), {"set_autograd_communication_mode": lambda self, mode: None})(),
    })()
    monkeypatch.setattr(benchmark, "_strict_backend", lambda **_: backend)
    monkeypatch.setattr(benchmark.torch, "npu", type("Npu", (), {"reset_peak_memory_stats": lambda *args: None, "max_memory_allocated": lambda *args: 17})(), raising=False)
    monkeypatch.setattr(benchmark.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(benchmark, "run_benchmark_workload", lambda *_args, **_kwargs: _runner_metrics(**overrides))
    args = type("Args", (), {
        "communication_mode": "baseline", "gradient_method": "native", "path": "statevector",
        "n_qubits": 2, "depth": 1, "parameters": 1, "warmups": 1, "runs": 1,
    })()
    with pytest.raises(RuntimeError, match="benchmark runner"):
        benchmark._run_benchmark(args)


def test_main_only_rank_zero_atomically_writes_the_valid_report(monkeypatch, tmp_path):
    output = tmp_path / "nested" / "benchmark.json"
    calls = []
    monkeypatch.setattr(benchmark, "_run_benchmark", lambda args: calls.append(1) or _report())
    argv = [
        "benchmark", "--communication-mode", "baseline", "--gradient-method", "native",
        "--path", "statevector", "--n-qubits", "2", "--depth", "1", "--parameters", "1",
        "--output-json", str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setenv("RANK", "1")
    assert benchmark.main() == 0
    assert not output.exists()
    monkeypatch.setenv("RANK", "0")
    assert benchmark.main() == 0
    assert json.loads(output.read_text()) == _report()
    assert not list(output.parent.glob("*.tmp"))
    assert calls == [1, 1]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join(context, *, timeout=90):
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "benchmark workload worker timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert all(process.exitcode == 0 for process in context.processes)


def _shared_workload_worker(rank, world_size, port, output):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        probe._synchronize_npu = lambda _: None
        metrics = {
            mode: probe.run_benchmark_workload(
                backend,
                communication_mode=mode,
                path="statevector",
                gradient_method="native",
                n_qubits=int(world_size.bit_length()),
                depth=1,
                parameters=1,
                warmups=1,
                runs=1,
            )
            for mode in ("baseline", "reuse", "overlap")
        }
        assert all(item["state_max_abs_error"] <= 1e-6 for item in metrics.values())
        assert all(item["gradient_max_abs_error"] <= 1e-4 for item in metrics.values())
        assert all(item["all_handles_complete"] for item in metrics.values())
        if world_size == 2:
            expanded = probe.run_benchmark_workload(
                backend,
                communication_mode="baseline",
                path="statevector",
                gradient_method="native",
                n_qubits=3,
                depth=2,
                parameters=2,
                warmups=1,
                runs=1,
            )
            assert expanded["p2p_bytes"] > metrics["baseline"]["p2p_bytes"]
            assert expanded["state_max_abs_error"] <= 1e-6
            metrics["expanded_config"] = expanded
        # W2 and W4 execute every real path and every supported numerical
        # method, not merely the statevector schedule.  n_qubits grows with
        # the shard selector axes so each case includes cross-shard P2P.
        for path, parameter_family, gradient_method in (
            ("statevector", "gate_angle", "parameter_shift"),
            ("statevector", "raw_state", "native"),
            ("statevector", "raw_state", "finite_difference"),
            ("density", "density_factor", "native"),
            ("density", "density_factor", "finite_difference"),
            ("noise", "channel_logit", "native"),
            ("noise", "channel_logit", "finite_difference"),
            ("stinespring", "stinespring_factor", "native"),
            ("stinespring", "stinespring_factor", "finite_difference"),
        ):
            oracle = probe.run_benchmark_workload(
                backend,
                communication_mode="baseline",
                path=path,
                parameter_family=parameter_family,
                gradient_method=gradient_method,
                n_qubits=int(world_size.bit_length()),
                depth=1,
                parameters=1,
                warmups=1,
                runs=1,
            )
            assert oracle["state_max_abs_error"] <= 1e-6
            assert oracle["gradient_max_abs_error"] <= 1e-4, (path, gradient_method, oracle)
            assert oracle["p2p_bytes"] > 0
            assert oracle["all_handles_complete"]
            assert oracle["parameter_family"] == parameter_family
        if rank == 0:
            Path(output).write_text(json.dumps(metrics), encoding="utf-8")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_shared_statevector_workload_exercises_cross_shard_modes(world_size, tmp_path):
    output = tmp_path / f"benchmark-{world_size}.json"
    context = mp.spawn(_shared_workload_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=False)
    _join(context)
    metrics = json.loads(output.read_text())
    assert metrics["baseline"]["p2p_bytes"] > 0
    assert metrics["reuse"]["buffer_reuse_count"] > 0
    assert metrics["overlap"]["buffer_reuse_count"] > 0
    if world_size == 2:
        assert metrics["expanded_config"]["p2p_bytes"] > metrics["baseline"]["p2p_bytes"]

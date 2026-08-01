import json

import pytest
import scripts.npu.distributed_autograd_benchmark as benchmark
import scripts.npu.distributed_autograd_probe as probe
import torch

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


def test_timing_synchronizes_each_forward_and_backward_boundary(monkeypatch):
    calls = []

    class Communicator:
        communication_counters = {"bytes": 0, "p2p_wait_ms": 0.0}
        work_handle_status = {"all_handles_complete": True}
        def set_autograd_communication_mode(self, mode): pass
        def clear_communication_records(self): pass

    backend = type("Backend", (), {"rank": 0, "_device": "cpu", "communicator": Communicator()})()
    monkeypatch.setattr(probe, "_synchronize_npu", lambda backend: calls.append(1))
    monkeypatch.setattr(probe, "_exchange_pair", lambda pair, **kwargs: pair)
    probe._performance_exchange_case(backend, "baseline", warmups=2, runs=3)
    assert len(calls) == 4 * (2 + 3)

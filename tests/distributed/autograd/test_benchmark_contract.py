import pytest

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

from pathlib import Path


PROBE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "npu"
    / "distributed_state_probe.py"
)


def test_probe_is_strict_and_covers_distributed_contract():
    source = PROBE.read_text(encoding="utf-8")

    assert "fallback_to_cpu=False" in source
    assert "local_rank" in source
    assert 'device.type != "npu"' in source
    assert "local_gate" in source
    assert "communicating_gate" in source
    assert "density" in source
    assert "AmplitudeDampingChannel" in source
    assert "expectation" in source
    assert "shots=" in source
    assert "local_tensor_sizes" in source
    assert "if backend.rank == 0" in source
    assert "json.dumps" in source

"""Static contracts for the strict distributed-native-autograd probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_PROBE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "npu"
    / "distributed_autograd_probe.py"
)


def _probe_module():
    spec = importlib.util.spec_from_file_location("distributed_autograd_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_declares_every_section_complete_and_routable():
    probe = _probe_module()

    assert len(probe.SECTIONS) == 13
    assert set(probe.SECTIONS) == set(probe.SECTION_RUNNERS)
    assert "BLOCKED" not in _PROBE_PATH.read_text(encoding="utf-8")


def test_probe_report_has_release_gate_top_level_contract():
    probe = _probe_module()
    report = probe._report_contract(
        commit="a" * 40,
        world_size=2,
        sections={
            name: {
                "status": "PASS",
                "passed": True,
                "metrics": {},
                "failed_invariants": [],
            }
            for name in probe.SECTIONS
        },
    )

    assert report == {
        "commit": "a" * 40,
        "world_size": 2,
        "backend": "hccl",
        "fallback_to_cpu": False,
        "passed": True,
        "failed_invariants": [],
        "sections": report["sections"],
    }
    assert all(
        set(section) >= {"status", "passed", "metrics", "failed_invariants"}
        for section in report["sections"].values()
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda probe: probe._require_hccl_backend("gloo"), "严格 distributed autograd 探针要求 HCCL process group"),
        (lambda probe: probe._require_supported_channel(object()), "自动微分模式不支持噪声通道 object"),
        (lambda probe: probe._validate_tag_phases((7,), (7,)), "forward/backward P2P tag 不匹配"),
    ),
)
def test_probe_explicit_error_contracts(call, message):
    probe = _probe_module()

    with pytest.raises(ValueError, match=f"^{message}$"):
        call(probe)

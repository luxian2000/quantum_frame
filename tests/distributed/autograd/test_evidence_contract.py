"""Contract tests for archived distributed-autograd release evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_SCRIPT = REPO_ROOT / "scripts/npu/distributed_autograd_evidence.py"
SPEC = importlib.util.spec_from_file_location("distributed_autograd_evidence", EVIDENCE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


COMMIT = "a" * 40


def _raw_sha256(report: dict) -> str:
    payload = {key: value for key, value in report.items() if key != "raw_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _passing_sections() -> dict:
    sections = {
        name: {"status": "PASS", "passed": True, "metrics": {}, "failed_invariants": []}
        for name in evidence.SECTIONS
    }
    sections["statevector"]["metrics"] = {"gradient_max_abs_error": 5e-5}
    sections["density"]["metrics"] = {"gradient_max_abs_error": 5e-5}
    sections["communication"]["metrics"] = {
        "rank_disagreement": 5e-7,
        "all_handles_complete": True,
    }
    sections["memory"]["metrics"] = {"memory_growth_percent": 0.5}
    sections["performance"]["metrics"] = {
        "workloads": [
            {
                "path": "statevector",
                "parameter_family": "gate_angle",
                "gradient_method": "native",
                "modes": {
                    "baseline": {"gradient_ms_median": 1.0, "all_handles_complete": True},
                    "reuse": {"gradient_ms_median": 1.0, "all_handles_complete": True},
                    "overlap": {"gradient_ms_median": 1.0, "all_handles_complete": True},
                },
            },
            {
                "path": "statevector",
                "parameter_family": "gate_angle",
                "gradient_method": "parameter_shift",
                "modes": {"baseline": {"gradient_ms_median": 2.0, "all_handles_complete": True}},
            },
            {
                "path": "density",
                "parameter_family": "density_factor",
                "gradient_method": "native",
                "modes": {"baseline": {"gradient_ms_median": 1.0, "all_handles_complete": True}},
            },
            {
                "path": "density",
                "parameter_family": "density_factor",
                "gradient_method": "finite_difference",
                "modes": {"baseline": {"gradient_ms_median": 2.0, "all_handles_complete": True}},
            },
            {
                "path": "statevector",
                "parameter_family": "raw_state",
                "gradient_method": "native",
                "modes": {"baseline": {"gradient_ms_median": 1.0, "all_handles_complete": True}},
            },
            {
                "path": "statevector",
                "parameter_family": "raw_state",
                "gradient_method": "finite_difference",
                "modes": {"baseline": {"gradient_ms_median": 2.0, "all_handles_complete": True}},
            },
            {
                "path": "noise",
                "parameter_family": "channel_logit",
                "gradient_method": "native",
                "modes": {"baseline": {"gradient_ms_median": 1.0, "all_handles_complete": True}},
            },
            {
                "path": "noise",
                "parameter_family": "channel_logit",
                "gradient_method": "finite_difference",
                "modes": {"baseline": {"gradient_ms_median": 2.0, "all_handles_complete": True}},
            },
            {
                "path": "stinespring",
                "parameter_family": "stinespring_factor",
                "gradient_method": "native",
                "modes": {"baseline": {"gradient_ms_median": 1.0, "all_handles_complete": True}},
            },
            {
                "path": "stinespring",
                "parameter_family": "stinespring_factor",
                "gradient_method": "finite_difference",
                "modes": {"baseline": {"gradient_ms_median": 2.0, "all_handles_complete": True}},
            },
        ]
    }
    return sections


def _report(world_size: int, *, commit: str = COMMIT) -> dict:
    result = {
        "commit": commit,
        "command": "torchrun --nproc-per-node=%d scripts/npu/distributed_autograd_probe.py --section all" % world_size,
        "exit_code": 0,
        "world_size": world_size,
        "rank_devices": ["npu:%d" % index for index in range(world_size)],
        "torch_version": "2.6.0",
        "torch_npu_version": "2.6.0",
        "cann_version": "unknown",
        "backend": "hccl",
        "fallback_to_cpu": False,
        "passed": True,
        "failed_invariants": [],
        "sections": _passing_sections(),
    }
    result["raw_sha256"] = _raw_sha256(result)
    return result


def _write_report(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / ("world%d.json" % report["world_size"])
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _aggregate(tmp_path: Path, *reports: dict) -> tuple[subprocess.CompletedProcess[str], Path]:
    paths = [_write_report(tmp_path, report) for report in reports]
    output = tmp_path / "manifest.json"
    result = subprocess.run(
        [sys.executable, str(EVIDENCE_SCRIPT), "aggregate", *(str(path) for path in paths), "--output", str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output


def test_validate_run_accepts_complete_hccl_report(tmp_path: Path) -> None:
    report = _report(2)
    path = _write_report(tmp_path, report)

    result = subprocess.run(
        [sys.executable, str(EVIDENCE_SCRIPT), "validate-run", str(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_aggregate_requires_independent_2_4_8_hccl_passes(tmp_path: Path) -> None:
    result, output = _aggregate(tmp_path, _report(2), _report(4), _report(8))

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release_gate"] == "PASS"
    assert manifest["commit"] == COMMIT


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda reports: reports.pop(), "missing world sizes: [8]"),
        (lambda reports: reports.__setitem__(2, _report(4)), "duplicate world size: 4"),
        (lambda reports: reports[1].__setitem__("commit", "b" * 40), "commit mismatch"),
        (lambda reports: reports[0]["sections"]["gates"].update(status="FAIL", passed=False), "sections.gates"),
        (lambda reports: reports[0].__setitem__("fallback_to_cpu", True), "fallback_to_cpu"),
        (lambda reports: reports[0]["sections"]["statevector"]["metrics"].__setitem__("gradient_max_abs_error", 2e-4), "gradient_max_abs_error"),
        (lambda reports: reports[0]["sections"]["communication"]["metrics"].__setitem__("rank_disagreement", 2e-6), "rank_disagreement"),
        (lambda reports: reports[0]["sections"]["performance"]["metrics"]["workloads"][0]["modes"]["baseline"].__setitem__("gradient_ms_median", 2.0), "native median"),
        (lambda reports: reports[0]["sections"]["memory"]["metrics"].__setitem__("memory_growth_percent", 1.1), "memory_growth_percent"),
        (lambda reports: reports[0]["sections"]["communication"]["metrics"].__setitem__("all_handles_complete", False), "all_handles_complete"),
        (lambda reports: reports[0].__setitem__("raw_sha256", "0" * 64), "raw_sha256"),
    ],
)
def test_aggregate_blocks_each_invalid_release_condition(tmp_path: Path, mutator, expected: str) -> None:
    reports = [_report(2), _report(4), _report(8)]
    mutator(reports)
    for report in reports:
        if report["raw_sha256"] != "0" * 64:
            report["raw_sha256"] = _raw_sha256(report)

    result, output = _aggregate(tmp_path, *reports)

    assert result.returncode != 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release_gate"] == "BLOCKED"
    assert "SKIPPED" not in json.dumps(manifest)
    assert expected in "\n".join(manifest["failed_conditions"])


def test_aggregate_blocks_missing_native_performance_family(tmp_path: Path) -> None:
    reports = [_report(2), _report(4), _report(8)]
    for report in reports:
        workloads = report["sections"]["performance"]["metrics"]["workloads"]
        report["sections"]["performance"]["metrics"]["workloads"] = [
            workload
            for workload in workloads
            if not (
                workload["path"] == "noise"
                and workload["gradient_method"] == "native"
            )
        ]
        report["raw_sha256"] = _raw_sha256(report)

    result, output = _aggregate(tmp_path, *reports)

    assert result.returncode != 0
    assert "required performance workload is missing: noise/channel_logit/native" in "\n".join(
        json.loads(output.read_text(encoding="utf-8"))["failed_conditions"]
    )

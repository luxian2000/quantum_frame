#!/usr/bin/env python3
"""Validate and aggregate immutable 2/4/8-NPU autograd release evidence.

The program deliberately accepts only completed, strict-HCCL rank-0 reports.
It does not run a workload and cannot turn a CPU/Gloo result into release
evidence.  A missing required world size is a blocked release, never a skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


SECTIONS = (
    "environment",
    "statevector",
    "density",
    "gates",
    "probability",
    "observable",
    "noise",
    "stinespring",
    "communication",
    "optimizer",
    "performance",
    "memory",
    "contract",
)
WORLD_SIZES = frozenset({2, 4, 8})
GRADIENT_ERROR_LIMIT = 1e-4
RANK_DISAGREEMENT_LIMIT = 1e-6
MEMORY_GROWTH_PERCENT_LIMIT = 1.0
REQUIRED_PERFORMANCE_WORKLOADS = (
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
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    """Hash the report payload without its self-referential digest field."""

    payload = {key: item for key, item in value.items() if key != "raw_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _validate_sections(report: dict[str, Any], failures: list[str]) -> None:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        failures.append("sections must be an object")
        return
    names = set(sections)
    if names != set(SECTIONS):
        failures.append(
            "sections must contain exactly the 13 required names; "
            f"missing={sorted(set(SECTIONS) - names)}, extra={sorted(names - set(SECTIONS))}"
        )
        return
    for name in SECTIONS:
        section = sections[name]
        if not isinstance(section, dict):
            failures.append(f"sections.{name} must be an object")
            continue
        if set(("status", "passed", "metrics", "failed_invariants")) - set(section):
            failures.append(f"sections.{name} lacks the required section contract")
            continue
        if section["status"] != "PASS" or section["passed"] is not True:
            failures.append(f"sections.{name} must be PASS with passed=true")
        if not isinstance(section["metrics"], dict):
            failures.append(f"sections.{name}.metrics must be an object")
        if not isinstance(section["failed_invariants"], list) or section["failed_invariants"]:
            failures.append(f"sections.{name}.failed_invariants must be empty")


def _validate_common_contract(report: dict[str, Any], failures: list[str]) -> None:
    required = {
        "commit", "command", "exit_code", "world_size", "rank_devices",
        "torch_version", "torch_npu_version", "cann_version", "backend",
        "fallback_to_cpu", "passed", "failed_invariants", "sections", "raw_sha256",
    }
    missing = sorted(required - set(report))
    if missing:
        failures.append(f"missing required fields: {missing}")
    if not _is_commit(report.get("commit")):
        failures.append("commit must be a full lowercase 40-character SHA")
    if not isinstance(report.get("command"), str) or not report["command"].strip():
        failures.append("command must record the executed torchrun command")
    if report.get("exit_code") != 0:
        failures.append("exit_code must be 0")
    world_size = report.get("world_size")
    if world_size not in WORLD_SIZES:
        failures.append("world_size must be one of 2, 4, or 8")
    devices = report.get("rank_devices")
    if not isinstance(devices, list) or not isinstance(world_size, int) or devices != [
        f"npu:{rank}" for rank in range(world_size)
    ]:
        failures.append("rank_devices must exactly map every rank to npu:LOCAL_RANK")
    for field in ("torch_version", "torch_npu_version", "cann_version"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            failures.append(f"{field} must be a non-empty recorded runtime identity")
    if report.get("backend") != "hccl":
        failures.append("backend must be hccl")
    if report.get("fallback_to_cpu") is not False:
        failures.append("fallback_to_cpu must be false")
    if report.get("passed") is not True:
        failures.append("top-level passed must be true")
    if not isinstance(report.get("failed_invariants"), list) or report["failed_invariants"]:
        failures.append("top-level failed_invariants must be empty")
    if not _is_sha256(report.get("raw_sha256")):
        failures.append("raw_sha256 must be a lowercase SHA-256 digest")
    elif report["raw_sha256"] != _canonical_sha256(report):
        failures.append("raw_sha256 does not match the canonical raw report payload")
    _validate_sections(report, failures)


def _validate_correctness(report: dict[str, Any], failures: list[str]) -> None:
    metrics = report.get("sections", {})
    leaves = list(_walk(metrics))
    gradient_errors = [
        (path, number)
        for path, value in leaves
        if (number := _number(value)) is not None
        and "gradient" in path.lower()
        and "error" in path.lower()
    ]
    if not gradient_errors:
        failures.append("correctness evidence lacks a gradient error metric")
    for path, value in gradient_errors:
        if value > GRADIENT_ERROR_LIMIT:
            failures.append(f"{path}={value:g} exceeds {GRADIENT_ERROR_LIMIT:g}")
    disagreements = [
        (path, number)
        for path, value in leaves
        if (number := _number(value)) is not None and "rank_disagreement" in path.lower()
    ]
    if not disagreements:
        failures.append("correctness evidence lacks a rank_disagreement metric")
    for path, value in disagreements:
        if value > RANK_DISAGREEMENT_LIMIT:
            failures.append(f"{path}={value:g} exceeds {RANK_DISAGREEMENT_LIMIT:g}")


def _performance_median(record: dict[str, Any]) -> float | None:
    modes = record.get("modes")
    if not isinstance(modes, dict) or "baseline" not in modes or not isinstance(modes["baseline"], dict):
        return None
    return _number(modes["baseline"].get("gradient_ms_median"))


def _validate_performance(report: dict[str, Any], failures: list[str]) -> None:
    performance = report.get("sections", {}).get("performance", {})
    metrics = performance.get("metrics") if isinstance(performance, dict) else None
    workloads = metrics.get("workloads") if isinstance(metrics, dict) else None
    if not isinstance(workloads, list) or not workloads:
        failures.append("performance evidence lacks workloads")
        return
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in workloads:
        if not isinstance(record, dict):
            failures.append("performance workload must be an object")
            continue
        key = (str(record.get("path")), str(record.get("parameter_family")), str(record.get("gradient_method")))
        records[key] = record
    for required in REQUIRED_PERFORMANCE_WORKLOADS:
        if required not in records:
            failures.append(
                "required performance workload is missing: " + "/".join(required)
            )
    native_records = [record for key, record in records.items() if key[2] == "native"]
    if not native_records:
        failures.append("performance evidence lacks native workload records")
    for native in native_records:
        path = str(native.get("path"))
        family = str(native.get("parameter_family"))
        oracle_method = "parameter_shift" if path == "statevector" and family == "gate_angle" else "finite_difference"
        oracle = records.get((path, family, oracle_method))
        native_median = _performance_median(native)
        oracle_median = _performance_median(oracle) if oracle is not None else None
        if native_median is None or oracle_median is None:
            failures.append(
                f"performance {path}/{family} needs baseline native and {oracle_method} gradient_ms_median"
            )
        elif native_median >= oracle_median:
            failures.append(
                f"native median {native_median:g} is not below {oracle_method} median {oracle_median:g} for {path}/{family}"
            )


def _validate_stability(report: dict[str, Any], failures: list[str]) -> None:
    leaves = list(_walk(report.get("sections", {})))
    growth = [
        (path, number)
        for path, value in leaves
        if (number := _number(value)) is not None and "memory_growth_percent" in path.lower()
    ]
    if not growth:
        failures.append("stability evidence lacks memory_growth_percent")
    for path, value in growth:
        if value > MEMORY_GROWTH_PERCENT_LIMIT:
            failures.append(f"{path}={value:g} exceeds {MEMORY_GROWTH_PERCENT_LIMIT:g}%")
    handles = [(path, value) for path, value in leaves if path.lower().endswith("all_handles_complete")]
    if not handles:
        failures.append("stability evidence lacks all_handles_complete")
    for path, value in handles:
        if value is not True:
            failures.append(f"{path} must be true")


def validate_report(report: Any) -> list[str]:
    """Return every release-blocking contract violation for one rank-0 report."""

    if not isinstance(report, dict):
        return ["report root must be a JSON object"]
    failures: list[str] = []
    _validate_common_contract(report, failures)
    if not failures:
        _validate_correctness(report, failures)
        _validate_performance(report, failures)
        _validate_stability(report, failures)
    return failures


def _read_report(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{path}: cannot read JSON: {error}"]
    failures = [f"{path}: {failure}" for failure in validate_report(value)]
    return value if isinstance(value, dict) else None, failures


def aggregate(paths: Iterable[Path]) -> dict[str, Any]:
    """Build a deterministic manifest; any absence or violation blocks release."""

    reports: list[tuple[Path, dict[str, Any]]] = []
    failures: list[str] = []
    by_world_size: dict[int, Path] = {}
    for path in paths:
        report, report_failures = _read_report(path)
        failures.extend(report_failures)
        if report is None:
            continue
        world_size = report.get("world_size")
        if isinstance(world_size, int):
            if world_size in by_world_size:
                failures.append(f"duplicate world size: {world_size}")
            else:
                by_world_size[world_size] = path
        reports.append((path, report))
    missing = sorted(WORLD_SIZES - set(by_world_size))
    extra = sorted(set(by_world_size) - WORLD_SIZES)
    if missing:
        failures.append(f"missing world sizes: {missing}")
    if extra:
        failures.append(f"unexpected world sizes: {extra}")
    commits = {report.get("commit") for _, report in reports if _is_commit(report.get("commit"))}
    if len(commits) != 1:
        failures.append("commit mismatch across 2/4/8 reports")
    commit = next(iter(commits)) if len(commits) == 1 else None
    manifest_reports = [
        {
            "world_size": report.get("world_size"),
            "path": str(path),
            "raw_sha256": report.get("raw_sha256"),
        }
        for path, report in sorted(reports, key=lambda item: (item[1].get("world_size", 0), str(item[0])))
    ]
    return {
        "commit": commit,
        "required_world_sizes": sorted(WORLD_SIZES),
        "reports": manifest_reports,
        "failed_conditions": failures,
        "release_gate": "PASS" if not failures else "BLOCKED",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    validate = subparsers.add_parser("validate-run", help="validate one completed rank-0 report")
    validate.add_argument("report", type=Path)
    aggregate_parser = subparsers.add_parser("aggregate", help="aggregate independent 2/4/8 reports")
    aggregate_parser.add_argument("reports", nargs="*", type=Path)
    aggregate_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command_name == "validate-run":
        report, failures = _read_report(args.report)
        result = {"report": str(args.report), "valid": report is not None and not failures, "failed_conditions": failures}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    manifest = aggregate(args.reports)
    _write_json(args.output, manifest)
    print(json.dumps({"release_gate": manifest["release_gate"], "output": str(args.output)}, sort_keys=True))
    return 0 if manifest["release_gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

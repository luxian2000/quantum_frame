#!/usr/bin/env python3
"""Validate and atomically aggregate immutable 2/4/8-NPU evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Any, Iterable
import uuid


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
MODES = ("baseline", "reuse", "overlap")
GRADIENT_ERROR_LIMIT = 1e-4
RANK_DISAGREEMENT_LIMIT = 1e-6
MEMORY_GROWTH_PERCENT_LIMIT = 1.0
RAW_SHA256_PLACEHOLDER = "0" * 64
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
TOP_LEVEL_FIELDS = frozenset(
    {
        "commit",
        "command",
        "exit_code",
        "world_size",
        "rank_devices",
        "torch_version",
        "torch_npu_version",
        "cann_version",
        "backend",
        "fallback_to_cpu",
        "run_id",
        "started_at",
        "finished_at",
        "source_clean",
        "passed",
        "failed_invariants",
        "sections",
        "raw_sha256",
    }
)


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _is_world_size(value: Any) -> bool:
    return type(value) is int and value in WORLD_SIZES


def _zero_unfinished_handles(value: Any) -> bool:
    if type(value) is int:
        return value == 0
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item == 0 for item in value)
    )


def _complete_handles(value: Any) -> bool:
    if value is True:
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and all(item is True for item in value)
    )


def _validate_nested_runtime_guards(
    report: dict[str, Any], failures: list[str]
) -> None:
    """Fail closed on every producer-emitted nested runtime guard."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                if key == "fallback_to_cpu" and item is not False:
                    failures.append(f"{child} must be false")
                elif (
                    key == "unfinished_work_handles"
                    and not _zero_unfinished_handles(item)
                ):
                    failures.append(
                        f"{child} must contain only exact integer zero values"
                    )
                elif (
                    key == "all_handles_complete"
                    and not _complete_handles(item)
                ):
                    failures.append(f"{child} must contain only true values")
                visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(report.get("sections"), "sections")


def _validate_exact_byte_digest(
    raw: bytes, report: dict[str, Any], failures: list[str]
) -> None:
    field_marker = b'"raw_sha256"'
    if raw.count(field_marker) != 1:
        failures.append("raw bytes must contain exactly one raw_sha256 field occurrence")
        return
    digest = report.get("raw_sha256")
    if not _is_sha256(digest):
        failures.append("raw_sha256 must be a lowercase SHA-256 digest")
        return
    value_marker = b'"raw_sha256":"' + digest.encode("ascii") + b'"'
    if raw.count(value_marker) != 1:
        failures.append(
            "raw_sha256 cannot validate the exact report bytes; "
            "the producer's fixed-width digest field is required"
        )
        return
    placeholder = (
        b'"raw_sha256":"' + RAW_SHA256_PLACEHOLDER.encode("ascii") + b'"'
    )
    bound_bytes = raw.replace(value_marker, placeholder, 1)
    if hashlib.sha256(bound_bytes).hexdigest() != digest:
        failures.append("raw_sha256 does not match the exact report bytes")


def _validate_sections(report: dict[str, Any], failures: list[str]) -> None:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        failures.append("sections must be an object")
        return
    names = set(sections)
    if names != set(SECTIONS):
        failures.append(
            "sections must contain exactly the 13 required names; "
            f"missing={sorted(set(SECTIONS) - names)}, "
            f"extra={sorted(names - set(SECTIONS))}"
        )
        return
    for name in SECTIONS:
        section = sections[name]
        if not isinstance(section, dict):
            failures.append(f"sections.{name} must be an object")
            continue
        if set(section) != {
            "status",
            "passed",
            "metrics",
            "failed_invariants",
        }:
            failures.append(
                f"sections.{name} must contain exactly status, passed, metrics, and failed_invariants"
            )
            continue
        if section["status"] != "PASS" or section["passed"] is not True:
            failures.append(f"sections.{name} must be PASS with passed=true")
        if not isinstance(section["metrics"], dict):
            failures.append(f"sections.{name}.metrics must be an object")
        if (
            not isinstance(section["failed_invariants"], list)
            or section["failed_invariants"]
        ):
            failures.append(f"sections.{name}.failed_invariants must be empty")


def _parse_utc_timestamp(value: Any, field: str, failures: list[str]):
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        value,
    ) is None:
        failures.append(
            f"{field} must use the producer's canonical "
            "ISO-8601 UTC microsecond format"
        )
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        failures.append(f"{field} must be a valid ISO-8601 UTC timestamp")
        return None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        failures.append(f"{field} must be UTC")
        return None
    return parsed


def _validate_command(
    command: Any, world_size: Any, failures: list[str]
) -> None:
    if not isinstance(command, str):
        failures.append("command must be the canonical torchrun command")
        return
    try:
        tokens = shlex.split(command)
    except ValueError:
        failures.append("command must be the canonical torchrun command")
        return
    expected_prefix = [
        "torchrun",
        f"--nproc-per-node={world_size}",
        "scripts/npu/distributed_autograd_probe.py",
        "--section",
        "all",
        "--output-json",
    ]
    if (
        len(tokens) != 7
        or tokens[:6] != expected_prefix
        or not isinstance(world_size, int)
        or Path(tokens[6]).name != f"world{world_size}.json"
        or command != shlex.join((*expected_prefix, tokens[6]))
    ):
        failures.append(
            "command must be the canonical torchrun command for the exact "
            "autograd probe, --section all, and matching process count"
        )


def _validate_common_contract(
    report: dict[str, Any], failures: list[str]
) -> None:
    fields = set(report)
    if fields != TOP_LEVEL_FIELDS:
        failures.append(
            "top-level fields must exactly match the producer contract; "
            f"missing={sorted(TOP_LEVEL_FIELDS - fields)}, "
            f"extra={sorted(fields - TOP_LEVEL_FIELDS)}"
        )
    if not _is_commit(report.get("commit")):
        failures.append("commit must be a full lowercase 40-character SHA")
    world_size = report.get("world_size")
    if not _is_world_size(world_size):
        failures.append("world_size must be an exact integer 2, 4, or 8")
    _validate_command(report.get("command"), world_size, failures)
    if type(report.get("exit_code")) is not int or report.get("exit_code") != 0:
        failures.append("exit_code must be 0")
    devices = report.get("rank_devices")
    if (
        not isinstance(devices, list)
        or not _is_world_size(world_size)
        or devices != [f"npu:{rank}" for rank in range(world_size)]
    ):
        failures.append("rank_devices must exactly map every rank to npu:LOCAL_RANK")
    for field in ("torch_version", "torch_npu_version", "cann_version"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            failures.append(f"{field} must be a non-empty runtime identity")
    if report.get("backend") != "hccl":
        failures.append("backend must be hccl")
    if report.get("fallback_to_cpu") is not False:
        failures.append("fallback_to_cpu must be false")
    if report.get("source_clean") is not True:
        failures.append("source_clean must be true for release evidence")
    run_id = report.get("run_id")
    try:
        parsed_run_id = uuid.UUID(run_id) if isinstance(run_id, str) else None
    except ValueError:
        parsed_run_id = None
    if (
        parsed_run_id is None
        or parsed_run_id.version != 4
        or str(parsed_run_id) != run_id
    ):
        failures.append("run_id must be a canonical UUIDv4")
    started = _parse_utc_timestamp(report.get("started_at"), "started_at", failures)
    finished = _parse_utc_timestamp(
        report.get("finished_at"), "finished_at", failures
    )
    if started is not None and finished is not None and finished <= started:
        failures.append("finished_at must be strictly after started_at")
    if report.get("passed") is not True:
        failures.append("top-level passed must be true")
    if (
        not isinstance(report.get("failed_invariants"), list)
        or report["failed_invariants"]
    ):
        failures.append("top-level failed_invariants must be empty")
    _validate_sections(report, failures)


def _validate_finite_numbers(
    report: dict[str, Any], failures: list[str]
) -> None:
    for path, value in _walk(report):
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                failures.append(
                    f"{path} is a non-finite numeric metric "
                    "(out of finite numeric range)"
                )


def _validate_correctness(
    report: dict[str, Any], failures: list[str]
) -> None:
    leaves = list(_walk(report.get("sections", {})))
    gradient_errors = [
        (path, value)
        for path, value in leaves
        if "gradient" in path.lower() and "error" in path.lower()
    ]
    if not gradient_errors:
        failures.append("correctness evidence lacks a gradient error metric")
    for path, raw_value in gradient_errors:
        value = _number(raw_value)
        if value is None:
            failures.append(f"{path} must be a finite numeric gradient error")
        elif value < 0:
            failures.append(f"{path} must be a non-negative gradient error")
        elif value > GRADIENT_ERROR_LIMIT:
            failures.append(f"{path}={value:g} exceeds {GRADIENT_ERROR_LIMIT:g}")
    disagreements = [
        (path, value)
        for path, value in leaves
        if "rank_disagreement" in path.lower()
    ]
    if not disagreements:
        failures.append("correctness evidence lacks a rank_disagreement metric")
    for path, raw_value in disagreements:
        value = _number(raw_value)
        if value is None:
            failures.append(f"{path} must be a finite numeric rank_disagreement")
        elif value < 0:
            failures.append(f"{path} must be a non-negative rank_disagreement")
        elif value > RANK_DISAGREEMENT_LIMIT:
            failures.append(f"{path}={value:g} exceeds {RANK_DISAGREEMENT_LIMIT:g}")


def _performance_median(
    record: dict[str, Any], mode: str
) -> float | None:
    modes = record.get("modes")
    if not isinstance(modes, dict) or not isinstance(modes.get(mode), dict):
        return None
    return _number(modes[mode].get("gradient_ms_median"))


def _section_metrics(
    report: dict[str, Any], section_name: str
) -> dict[str, Any] | None:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return None
    section = sections.get(section_name)
    if not isinstance(section, dict):
        return None
    metrics = section.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _validate_performance(
    report: dict[str, Any], failures: list[str]
) -> None:
    metrics = _section_metrics(report, "performance")
    workloads = metrics.get("workloads") if isinstance(metrics, dict) else None
    if not isinstance(workloads, list):
        failures.append("performance evidence lacks workloads")
        return
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    keys = []
    for index, record in enumerate(workloads):
        if not isinstance(record, dict):
            failures.append(f"performance workload {index} must be an object")
            continue
        key = (
            record.get("path"),
            record.get("parameter_family"),
            record.get("gradient_method"),
        )
        if not all(isinstance(value, str) for value in key):
            failures.append(f"performance workload {index} has an invalid key")
            continue
        keys.append(key)
        records.setdefault(key, record)
        modes = record.get("modes")
        if not isinstance(modes, dict) or set(modes) != set(MODES):
            failures.append(
                "performance workload "
                + "/".join(key)
                + " must contain exactly baseline, reuse, and overlap modes"
            )
            continue
        for mode in MODES:
            mode_metrics = modes[mode]
            if not isinstance(mode_metrics, dict):
                failures.append(
                    f"performance {'/'.join(key)}/{mode} must be an object"
                )
                continue
            median = _number(mode_metrics.get("gradient_ms_median"))
            if median is None or median <= 0:
                failures.append(
                    f"performance {'/'.join(key)}/{mode} needs a finite positive gradient_ms_median"
                )
            disagreement = _number(mode_metrics.get("rank_disagreement"))
            if disagreement is None:
                failures.append(
                    f"performance {'/'.join(key)}/{mode} needs finite rank_disagreement"
                )
            elif disagreement < 0:
                failures.append(
                    f"performance {'/'.join(key)}/{mode} "
                    "rank_disagreement must be non-negative"
                )
            elif disagreement > RANK_DISAGREEMENT_LIMIT:
                failures.append(
                    f"performance {'/'.join(key)}/{mode} rank_disagreement="
                    f"{disagreement:g} exceeds {RANK_DISAGREEMENT_LIMIT:g}"
                )
            if mode_metrics.get("all_handles_complete") is not True:
                failures.append(
                    f"performance {'/'.join(key)}/{mode} all_handles_complete must be true"
                )

    counts = Counter(keys)
    required = set(REQUIRED_PERFORMANCE_WORKLOADS)
    for key, count in counts.items():
        if count > 1:
            failures.append(
                f"duplicate performance workload: {'/'.join(key)} appears {count} times"
            )
        if key not in required:
            failures.append(f"unexpected performance workload: {'/'.join(key)}")
    for key in REQUIRED_PERFORMANCE_WORKLOADS:
        if counts[key] == 0:
            failures.append(
                "required performance workload is missing: " + "/".join(key)
            )
    if len(workloads) != len(REQUIRED_PERFORMANCE_WORKLOADS):
        failures.append(
            "performance workloads must contain exactly the 10 required records"
        )

    native_keys = tuple(
        key for key in REQUIRED_PERFORMANCE_WORKLOADS if key[2] == "native"
    )
    for path, family, _ in native_keys:
        oracle_method = (
            "parameter_shift"
            if path == "statevector" and family == "gate_angle"
            else "finite_difference"
        )
        native = records.get((path, family, "native"))
        oracle = records.get((path, family, oracle_method))
        if native is None or oracle is None:
            continue
        for mode in MODES:
            native_median = _performance_median(native, mode)
            oracle_median = _performance_median(oracle, mode)
            if native_median is None or oracle_median is None:
                continue
            if native_median >= oracle_median:
                failures.append(
                    f"native median {native_median:g} is not below "
                    f"{oracle_method} median {oracle_median:g} for "
                    f"{path}/{family}/{mode}"
                )


def _validate_stability(
    report: dict[str, Any], failures: list[str]
) -> None:
    metrics = _section_metrics(report, "memory")
    if not isinstance(metrics, dict):
        failures.append("stability evidence lacks memory metrics")
        return
    repeated = metrics.get("repeated_measurements")
    if not isinstance(repeated, int) or isinstance(repeated, bool) or repeated < 2:
        failures.append("memory evidence requires at least two repeated_measurements")
    if metrics.get("repeated_policy") != "each":
        failures.append("memory repeated_policy must be 'each'")
    policies = metrics.get("policies")
    expected_policies = {"none", "auto", "16"}
    if not isinstance(policies, dict) or set(policies) != expected_policies:
        failures.append(
            "memory policies must contain exactly none, auto, and 16"
        )
        policies = {}
    policy_growth = []
    for policy_name in sorted(expected_policies):
        policy = policies.get(policy_name)
        path = f"sections.memory.metrics.policies.{policy_name}"
        if not isinstance(policy, dict):
            failures.append(f"{path} must be an object")
            continue
        samples = policy.get("peak_allocation_bytes")
        policy_repeated = policy.get("repeated_measurements")
        numeric_samples = (
            [_number(value) for value in samples]
            if isinstance(samples, list)
            else []
        )
        if (
            not isinstance(samples, list)
            or len(samples) < 2
            or any(value is None or value < 0 for value in numeric_samples)
        ):
            failures.append(
                f"{path}.peak_allocation_bytes needs at least two finite "
                "non-negative measurements"
            )
            continue
        if (
            not isinstance(policy_repeated, int)
            or isinstance(policy_repeated, bool)
            or policy_repeated != len(samples)
            or policy_repeated != repeated
        ):
            failures.append(
                f"{path}.repeated_measurements must equal the sample count "
                "and top-level repeated_measurements"
            )
        measured_growth = max(
            0.0,
            (numeric_samples[-1] - numeric_samples[0])
            / max(numeric_samples[0], 1.0)
            * 100.0,
        )
        claimed_growth = _number(policy.get("memory_growth_percent"))
        if claimed_growth is None or not math.isclose(
            claimed_growth,
            measured_growth,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            failures.append(
                f"{path}.memory_growth_percent must be recomputed from "
                "the repeated peak_allocation_bytes samples"
            )
            continue
        policy_growth.append(claimed_growth)
        if claimed_growth > MEMORY_GROWTH_PERCENT_LIMIT:
            failures.append(
                f"{path}.memory_growth_percent={claimed_growth:g} exceeds "
                f"{MEMORY_GROWTH_PERCENT_LIMIT:g}%"
            )
    top_growth = _number(metrics.get("memory_growth_percent"))
    if (
        len(policy_growth) != len(expected_policies)
        or top_growth is None
        or not math.isclose(
            top_growth,
            max(policy_growth, default=math.inf),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        failures.append(
            "sections.memory.metrics.memory_growth_percent must equal the "
            "maximum recomputed policy growth"
        )
    handles = [
        (path, value)
        for path, value in _walk(report.get("sections", {}))
        if path.lower().endswith("all_handles_complete")
    ]
    if not handles:
        failures.append("stability evidence lacks all_handles_complete")
    for path, value in handles:
        if value is not True:
            failures.append(f"{path} must be true")


def validate_report(
    report: Any, *, raw_bytes: bytes | None = None
) -> list[str]:
    """Return every release-blocking violation for one rank-0 report."""

    if not isinstance(report, dict):
        return ["report root must be a JSON object"]
    failures: list[str] = []
    if raw_bytes is None:
        failures.append("validator requires the exact report bytes")
    else:
        _validate_exact_byte_digest(raw_bytes, report, failures)
    _validate_finite_numbers(report, failures)
    _validate_common_contract(report, failures)
    _validate_nested_runtime_guards(report, failures)
    _validate_correctness(report, failures)
    _validate_performance(report, failures)
    _validate_stability(report, failures)
    return failures


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value} is forbidden")


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _read_report(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return None, [f"{path}: cannot read strict JSON: {error}"]
    try:
        failures = [
            f"{path}: {failure}"
            for failure in validate_report(value, raw_bytes=raw)
        ]
    except Exception as error:  # noqa: BLE001 - untrusted report boundary
        failures = [
            f"{path}: malformed report could not be validated: "
            f"{type(error).__name__}: {error}"
        ]
    return value if isinstance(value, dict) else None, failures


def aggregate(paths: Iterable[Path]) -> dict[str, Any]:
    """Build a deterministic manifest; any absence or violation blocks release."""

    valid_reports: list[tuple[Path, dict[str, Any]]] = []
    manifest_entries: list[tuple[int, int, str, dict[str, Any]]] = []
    failures: list[str] = []
    by_world_size: dict[int, Path] = {}
    for path in paths:
        report, report_failures = _read_report(path)
        failures.extend(report_failures)
        if report is None or report_failures:
            manifest_entries.append(
                (1, 0, str(path), {"path": str(path), "valid": False})
            )
            continue
        world_size = report["world_size"]
        if world_size in by_world_size:
            failures.append(f"duplicate world size: {world_size}")
        else:
            by_world_size[world_size] = path
        valid_reports.append((path, report))
        manifest_entries.append(
            (
                0,
                world_size,
                str(path),
                {
                    "world_size": world_size,
                    "path": str(path),
                    "raw_sha256": report["raw_sha256"],
                    "run_id": report["run_id"],
                    "started_at": report["started_at"],
                    "finished_at": report["finished_at"],
                },
            )
        )
    missing = sorted(WORLD_SIZES - set(by_world_size))
    extra = sorted(set(by_world_size) - WORLD_SIZES)
    if missing:
        failures.append(f"missing world sizes: {missing}")
    if extra:
        failures.append(f"unexpected world sizes: {extra}")
    commits = {report["commit"] for _, report in valid_reports}
    if len(commits) != 1:
        failures.append("commit mismatch across 2/4/8 reports")
    commit = next(iter(commits)) if len(commits) == 1 else None
    for field in ("run_id", "started_at", "finished_at"):
        values = [report[field] for _, report in valid_reports]
        duplicates = sorted(
            value for value, count in Counter(values).items() if count > 1
        )
        if duplicates:
            failures.append(f"duplicate {field} across independent runs: {duplicates}")
    manifest_reports = [
        summary for _, _, _, summary in sorted(manifest_entries)
    ]
    return {
        "commit": commit,
        "required_world_sizes": sorted(WORLD_SIZES),
        "reports": manifest_reports,
        "failed_conditions": failures,
        "release_gate": "PASS" if not failures else "BLOCKED",
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    validate = subparsers.add_parser(
        "validate-run", help="validate one completed rank-0 report"
    )
    validate.add_argument("report", type=Path)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="aggregate independent 2/4/8 reports"
    )
    aggregate_parser.add_argument("reports", nargs="*", type=Path)
    aggregate_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command_name == "validate-run":
        report, failures = _read_report(args.report)
        result = {
            "report": str(args.report),
            "valid": report is not None and not failures,
            "failed_conditions": failures,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    output = args.output.resolve()
    collisions = [
        path for path in args.reports if path.resolve() == output
    ]
    if collisions:
        result = {
            "release_gate": "BLOCKED",
            "failed_conditions": [
                f"output {args.output} collides with an input report"
            ],
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    manifest = aggregate(args.reports)
    _write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "release_gate": manifest["release_gate"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["release_gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Producer-compatible contract tests for distributed-autograd evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_SCRIPT = REPO_ROOT / "scripts/npu/distributed_autograd_evidence.py"
PROBE_SCRIPT = REPO_ROOT / "scripts/npu/distributed_autograd_probe.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = _load("distributed_autograd_evidence", EVIDENCE_SCRIPT)
probe = _load("distributed_autograd_probe", PROBE_SCRIPT)

COMMIT = "a" * 40
MODES = ("baseline", "reuse", "overlap")


def _performance_workloads() -> list[dict]:
    records = []
    for path, family, method in evidence.REQUIRED_PERFORMANCE_WORKLOADS:
        median = 1.0 if method == "native" else 2.0
        records.append(
            {
                "path": path,
                "parameter_family": family,
                "gradient_method": method,
                "modes": {
                    mode: {
                        "gradient_ms_median": median,
                        "rank_disagreement": 5e-7,
                        "unfinished_work_handles": 0,
                        "all_handles_complete": True,
                        "fallback_to_cpu": False,
                    }
                    for mode in MODES
                },
            }
        )
    return records


def _passing_probe_sections() -> dict:
    sections = {
        name: {"status": "PASS", "passed": True}
        for name in probe.SECTIONS
    }
    sections["statevector"]["gradient_max_abs_error"] = 5e-5
    sections["density"]["gradient_max_abs_error"] = 5e-5
    sections["communication"].update(
        rank_disagreement=5e-7,
        unfinished_work_handles=0,
        all_handles_complete=True,
    )
    sections["optimizer"].update(
        cases={
            "sgd-32": {
                "unfinished_work_handles": 0,
                "all_handles_complete": True,
            }
        },
        integrated_private_path={
            "unfinished_work_handles": [0, 0],
            "all_handles_complete": [True, True],
        },
        all_handles_complete=True,
    )
    sections["memory"].update(
        memory_growth_percent=0.5,
        repeated_policy="each",
        repeated_measurements=2,
        policies={
            "none": {
                "peak_allocation_bytes": [100, 100],
                "memory_growth_percent": 0.0,
                "repeated_measurements": 2,
            },
            "auto": {
                "peak_allocation_bytes": [200, 201],
                "memory_growth_percent": 0.5,
                "repeated_measurements": 2,
            },
            "16": {
                "peak_allocation_bytes": [300, 300],
                "memory_growth_percent": 0.0,
                "repeated_measurements": 2,
            },
        },
    )
    sections["performance"].update(
        workloads=_performance_workloads(),
        all_handles_complete=True,
    )
    return sections


def _report(
    world_size: int,
    *,
    commit: str = COMMIT,
    run_suffix: int | None = None,
) -> dict:
    suffix = world_size if run_suffix is None else run_suffix
    started = f"2026-08-02T00:00:{world_size:02d}.000000Z"
    finished = f"2026-08-02T00:00:{world_size + 1:02d}.000000Z"
    report = probe._report_contract(
        commit=commit,
        command=probe._canonical_probe_command(
            world_size, Path(f"world{world_size}.json")
        ),
        exit_code=0,
        world_size=world_size,
        rank_devices=[f"npu:{rank}" for rank in range(world_size)],
        torch_version="2.6.0",
        torch_npu_version="2.6.0",
        cann_version="unknown",
        run_id=f"00000000-0000-4000-8000-{suffix:012d}",
        started_at=started,
        finished_at=finished,
        source_clean=True,
        sections=_passing_probe_sections(),
    )
    return report


def _write_report(path: Path, report: dict) -> Path:
    probe._write_report(path, report)
    return path


def _validate(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EVIDENCE_SCRIPT), "validate-run", str(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _aggregate(
    tmp_path: Path, reports: list[dict], *, output: Path | None = None
) -> tuple[subprocess.CompletedProcess[str], Path, list[Path]]:
    paths = [
        _write_report(tmp_path / f"input-{index}-world{report['world_size']}.json", report)
        for index, report in enumerate(reports)
    ]
    output = output or tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_SCRIPT),
            "aggregate",
            *(str(path) for path in paths),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output, paths


def _exact_bytes_with_digest(report: dict, *, allow_nan: bool = False) -> bytes:
    report = copy.deepcopy(report)
    report["raw_sha256"] = "0" * 64
    raw = (
        json.dumps(
            report,
            allow_nan=allow_nan,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest().encode("ascii")
    marker = b'"raw_sha256":"' + b"0" * 64 + b'"'
    assert raw.count(marker) == 1
    return raw.replace(
        marker,
        b'"raw_sha256":"' + digest + b'"',
        1,
    )


def _bind_exact_raw_bytes(raw: bytes) -> bytes:
    marker_start = b'"raw_sha256":"'
    start = raw.index(marker_start) + len(marker_start)
    assert raw[start + 64 : start + 65] == b'"'
    placeholder = raw[:start] + b"0" * 64 + raw[start + 64 :]
    digest = hashlib.sha256(placeholder).hexdigest().encode("ascii")
    return placeholder[:start] + digest + placeholder[start + 64 :]


def _rewrite(path: Path, report: dict) -> None:
    _write_report(path, report)


def test_actual_probe_contract_and_writer_validate(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "world2.json", _report(2))

    result = _validate(path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"] is True
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b'"raw_sha256":') == 1


@pytest.mark.parametrize("mode", MODES)
def test_actual_producer_report_rejects_nested_performance_fallback(
    tmp_path: Path,
    mode: str,
) -> None:
    report = _report(2)
    report["sections"]["performance"]["metrics"]["workloads"][0]["modes"][
        mode
    ]["fallback_to_cpu"] = True
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "fallback_to_cpu" in "\n".join(payload["failed_conditions"])
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("section", "nested_key", "value"),
    [
        ("communication", None, 1),
        ("optimizer", "cases", 1),
        ("optimizer", "integrated_private_path", [0, 1]),
        ("performance", "mode", 1),
    ],
)
def test_actual_producer_report_rejects_any_nested_unfinished_work(
    tmp_path: Path,
    section: str,
    nested_key: str | None,
    value,
) -> None:
    report = _report(2)
    metrics = report["sections"][section]["metrics"]
    if nested_key is None:
        metrics["unfinished_work_handles"] = value
    elif nested_key == "cases":
        metrics["cases"]["sgd-32"]["unfinished_work_handles"] = value
    elif nested_key == "integrated_private_path":
        metrics["integrated_private_path"]["unfinished_work_handles"] = value
    else:
        metrics["workloads"][0]["modes"]["baseline"][
            "unfinished_work_handles"
        ] = value
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "unfinished_work_handles" in "\n".join(payload["failed_conditions"])
    assert result.stderr == ""


def test_exact_byte_digest_rejects_reformat_even_when_object_is_unchanged(
    tmp_path: Path,
) -> None:
    path = _write_report(tmp_path / "world2.json", _report(2))
    value = json.loads(path.read_bytes())
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _validate(path)

    assert result.returncode != 0
    assert "exact report bytes" in result.stdout


def test_exact_byte_digest_rejects_one_byte_mutation(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "world2.json", _report(2))
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"torch_version":"2.6.0"', b'"torch_version":"2.6.1"'))

    result = _validate(path)

    assert result.returncode != 0
    assert "exact report bytes" in result.stdout


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_non_finite_constants_are_rejected_even_with_matching_digest(
    tmp_path: Path, constant: str
) -> None:
    report = _report(2)
    report["sections"]["memory"]["metrics"]["memory_growth_percent"] = float(
        constant.replace("Infinity", "inf").lower()
    )
    path = tmp_path / "world2.json"
    path.write_bytes(_exact_bytes_with_digest(report, allow_nan=True))

    result = _validate(path)

    assert result.returncode != 0
    assert "non-finite JSON constant" in result.stdout


def test_overflow_to_non_finite_number_is_rejected(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "world2.json", _report(2))
    raw = path.read_bytes().replace(
        b'"memory_growth_percent":0.5',
        b'"memory_growth_percent":1e999',
    )
    path.write_bytes(_bind_exact_raw_bytes(raw))

    result = _validate(path)

    assert result.returncode != 0
    assert "non-finite numeric metric" in result.stdout


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (
            lambda report: report["sections"]["performance"]["metrics"][
                "workloads"
            ].append(
                copy.deepcopy(
                    report["sections"]["performance"]["metrics"]["workloads"][0]
                )
            ),
            "duplicate performance workload",
        ),
        (
            lambda report: report["sections"]["performance"]["metrics"][
                "workloads"
            ].append(
                {
                    "path": "extra",
                    "parameter_family": "extra",
                    "gradient_method": "native",
                    "modes": {},
                }
            ),
            "unexpected performance workload",
        ),
        (
            lambda report: report["sections"]["performance"]["metrics"][
                "workloads"
            ].pop(),
            "required performance workload is missing",
        ),
    ],
)
def test_workload_list_is_an_exact_multiset(
    tmp_path: Path, mutator, expected: str
) -> None:
    report = _report(2)
    mutator(report)
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert expected in result.stdout


def test_every_workload_mode_requires_measured_rank_disagreement(
    tmp_path: Path,
) -> None:
    report = _report(2)
    del report["sections"]["performance"]["metrics"]["workloads"][3]["modes"][
        "reuse"
    ]["rank_disagreement"]
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert "rank_disagreement" in result.stdout


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            ("statevector", "gradient_max_abs_error"),
            None,
            "gradient_max_abs_error",
        ),
        (
            ("communication", "rank_disagreement"),
            False,
            "rank_disagreement",
        ),
        (
            ("statevector", "gradient_max_abs_error"),
            -1.0,
            "gradient_max_abs_error",
        ),
        (
            ("communication", "rank_disagreement"),
            -1.0,
            "rank_disagreement",
        ),
    ],
)
def test_correctness_metrics_must_be_finite_numbers(
    tmp_path: Path, path: tuple[str, str], value, expected: str
) -> None:
    report = _report(2)
    section, metric = path
    report["sections"][section]["metrics"][metric] = value
    report_path = _write_report(tmp_path / "world2.json", report)

    result = _validate(report_path)

    assert result.returncode != 0
    assert expected in result.stdout


def test_native_median_must_be_strictly_below_exact_oracle_pair(
    tmp_path: Path,
) -> None:
    report = _report(2)
    workloads = report["sections"]["performance"]["metrics"]["workloads"]
    workloads[0]["modes"]["baseline"]["gradient_ms_median"] = 2.0
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert "native median" in result.stdout


@pytest.mark.parametrize(
    ("native_median", "oracle_median"),
    [(2.0, 1.0), (0.0, 1.0)],
)
def test_probe_performance_section_requires_positive_strictly_faster_native(
    monkeypatch: pytest.MonkeyPatch,
    native_median: float,
    oracle_median: float,
) -> None:
    def measured_workload(
        backend,
        *,
        communication_mode,
        path,
        gradient_method,
        parameter_family,
        **kwargs,
    ):
        del backend, path, kwargs
        return {
            "parameter_family": parameter_family,
            "gradient_ms_median": (
                native_median if gradient_method == "native" else oracle_median
            ),
            "state_max_abs_error": 0.0,
            "gradient_max_abs_error": 0.0,
            "rank_disagreement": 0.0,
            "p2p_bytes": 1,
            "all_handles_complete": True,
            "fallback_to_cpu": False,
            "buffer_reuse_count": 0 if communication_mode == "baseline" else 1,
        }

    class Communicator:
        def set_autograd_communication_mode(self, mode):
            self.mode = mode

    backend = type(
        "Backend",
        (),
        {"world_size": 2, "communicator": Communicator()},
    )()
    monkeypatch.setattr(probe, "run_benchmark_workload", measured_workload)

    section = probe._performance_section(backend)

    assert section["status"] == "FAIL"
    assert section["passed"] is False


@pytest.mark.parametrize(
    "command",
    [
        "torchrun --nproc-per-node=4 scripts/npu/distributed_autograd_probe.py --section all --output-json world2.json",
        "torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py --section all --output-json world2.json",
        "torchrun --nproc-per-node=2 scripts/npu/distributed_autograd_probe.py --section memory --output-json world2.json",
        "torchrun --nproc-per-node=2 scripts/npu/distributed_autograd_probe.py --section all --output-json world2.json --extra",
        "torchrun --nproc-per-node=2 scripts/npu/distributed_autograd_probe.py --section all --output-json 'world2.json'",
    ],
)
def test_command_must_be_exact_probe_all_process_contract(
    tmp_path: Path, command: str
) -> None:
    report = _report(2)
    report["command"] = command
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert "canonical torchrun command" in result.stdout


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("source_clean", False, "source_clean"),
        ("exit_code", False, "exit_code"),
        ("started_at", "not-a-time", "started_at"),
        ("started_at", "2026-08-02T00:00:02Z", "started_at"),
        ("finished_at", "2026-08-01T23:59:59.000000Z", "finished_at"),
        ("run_id", "same-run", "run_id"),
    ],
)
def test_source_and_run_provenance_is_strict(
    tmp_path: Path, field: str, value, expected: str
) -> None:
    report = _report(2)
    report[field] = value
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert expected in result.stdout


@pytest.mark.parametrize("world_size", [None, [], True, "2", 2.0])
def test_validate_run_rejects_abnormal_world_size_without_crashing(
    tmp_path: Path,
    world_size,
) -> None:
    report = _report(2)
    report["world_size"] = world_size
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "world_size" in "\n".join(payload["failed_conditions"])
    assert result.stderr == ""


def test_aggregate_accepts_unique_independent_2_4_8_runs(tmp_path: Path) -> None:
    result, output, _ = _aggregate(
        tmp_path, [_report(2), _report(4), _report(8)]
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release_gate"] == "PASS"
    assert manifest["commit"] == COMMIT


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report.__setitem__("world_size", []),
        lambda report: report.__setitem__("sections", []),
        lambda report: report["sections"]["memory"]["metrics"]["policies"][
            "auto"
        ].__setitem__("peak_allocation_bytes", [100, 10**400]),
        lambda report: report["sections"]["performance"]["metrics"]["workloads"][
            0
        ]["modes"]["baseline"].__setitem__("fallback_to_cpu", True),
    ],
)
def test_aggregate_atomically_writes_blocked_for_any_invalid_report(
    tmp_path: Path,
    mutator,
) -> None:
    reports = [_report(2), _report(4), _report(8)]
    reports[0]["untrusted_manifest_field"] = "must-not-be-copied"
    mutator(reports[0])
    output = tmp_path / "manifest.json"
    output.write_text('{"release_gate":"STALE"}\n', encoding="utf-8")

    result, output, _ = _aggregate(tmp_path, reports, output=output)

    assert result.returncode != 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release_gate"] == "BLOCKED"
    assert "must-not-be-copied" not in json.dumps(manifest)
    assert any(entry.get("valid") is False for entry in manifest["reports"])
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda reports: reports.pop(), "missing world sizes: [8]"),
        (lambda reports: reports.__setitem__(2, _report(4, run_suffix=8)), "duplicate world size: 4"),
        (lambda reports: reports[1].__setitem__("commit", "b" * 40), "commit mismatch"),
        (lambda reports: reports[1].__setitem__("run_id", reports[0]["run_id"]), "duplicate run_id"),
        (
            lambda reports: reports[1].__setitem__(
                "started_at", reports[0]["started_at"]
            ),
            "duplicate started_at",
        ),
        (
            lambda reports: reports[1].update(
                started_at="2026-08-02T00:00:02.500000Z",
                finished_at=reports[0]["finished_at"],
            ),
            "duplicate finished_at",
        ),
        (
            lambda reports: reports[0]["sections"]["gates"].update(
                status="FAIL", passed=False
            ),
            "sections.gates",
        ),
        (lambda reports: reports[0].__setitem__("fallback_to_cpu", True), "fallback_to_cpu"),
        (
            lambda reports: reports[0]["sections"]["statevector"]["metrics"].__setitem__(
                "gradient_max_abs_error", 2e-4
            ),
            "gradient_max_abs_error",
        ),
        (
            lambda reports: reports[0]["sections"]["performance"]["metrics"][
                "workloads"
            ][0]["modes"]["overlap"].__setitem__("rank_disagreement", 2e-6),
            "rank_disagreement",
        ),
        (
            lambda reports: reports[0]["sections"]["memory"]["metrics"].__setitem__(
                "memory_growth_percent", 1.1
            ),
            "memory_growth_percent",
        ),
        (
            lambda reports: reports[0]["sections"]["communication"]["metrics"].__setitem__(
                "all_handles_complete", False
            ),
            "all_handles_complete",
        ),
    ],
)
def test_aggregate_blocks_each_release_condition(
    tmp_path: Path, mutator, expected: str
) -> None:
    reports = [_report(2), _report(4), _report(8)]
    mutator(reports)

    result, output, _ = _aggregate(tmp_path, reports)

    assert result.returncode != 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["release_gate"] == "BLOCKED"
    assert "SKIPPED" not in json.dumps(manifest)
    assert expected in "\n".join(manifest["failed_conditions"])


def test_aggregate_rejects_resolved_output_input_collision_without_overwrite(
    tmp_path: Path,
) -> None:
    reports = [_report(2), _report(4), _report(8)]
    paths = [
        _write_report(tmp_path / f"world{report['world_size']}.json", report)
        for report in reports
    ]
    before = paths[0].read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_SCRIPT),
            "aggregate",
            *(str(path) for path in paths),
            "--output",
            str(tmp_path / "." / "world2.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "collides with an input report" in (result.stdout + result.stderr)
    assert paths[0].read_bytes() == before


def test_probe_writer_replaces_from_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real_replace = probe.os.replace

    def observed_replace(source, target):
        calls.append((Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(probe.os, "replace", observed_replace)
    target = tmp_path / "world2.json"

    _write_report(target, _report(2))

    assert len(calls) == 1
    assert calls[0][0].parent == target.parent
    assert calls[0][1] == target
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_manifest_writer_replaces_from_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real_replace = evidence.os.replace

    def observed_replace(source, target):
        calls.append((Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(evidence.os, "replace", observed_replace)
    target = tmp_path / "manifest.json"

    evidence._write_json_atomic(target, {"release_gate": "BLOCKED"})

    assert len(calls) == 1
    assert calls[0][0].parent == target.parent
    assert calls[0][1] == target
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_rank_devices_rejects_duplicate_local_rank_binding() -> None:
    class Communicator:
        def all_gather_real(self, payload):
            del payload
            return (
                torch.tensor([0.0, 0.0], dtype=torch.float32),
                torch.tensor([1.0, 0.0], dtype=torch.float32),
            )

    backend = type(
        "Backend",
        (),
        {
            "_device": torch.device("cpu"),
            "rank": 0,
            "local_rank": 0,
            "world_size": 2,
            "communicator": Communicator(),
        },
    )()

    with pytest.raises(RuntimeError, match="rank device provenance"):
        probe._rank_devices(backend)


def test_rank_disagreement_uses_fixed_float32_all_rank_exchange() -> None:
    seen = []

    class Communicator:
        def all_gather_real(self, payload):
            seen.append(payload)
            return (
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                torch.tensor([1.0 + 4e-7, 2.0 - 2e-7], dtype=torch.float32),
            )

    backend = type(
        "Backend",
        (),
        {"_device": torch.device("cpu"), "communicator": Communicator()},
    )()

    disagreement = probe._rank_disagreement_float32([1.0, 2.0], backend)

    assert seen[0].dtype == torch.float32
    assert seen[0].shape == (2,)
    assert math.isclose(disagreement, 4e-7, rel_tol=0, abs_tol=1e-7)


@pytest.mark.parametrize(
    ("measurements", "expected"),
    [
        ([100, 101], 1.0),
        ([100, 99], 0.0),
        ([0, 0], 0.0),
    ],
)
def test_memory_growth_is_computed_from_repeated_measurements(
    measurements: list[int], expected: float
) -> None:
    growth = probe._memory_growth_percent(measurements)

    assert math.isfinite(growth)
    assert growth == expected


@pytest.mark.parametrize(
    "mutator",
    [
        lambda policy: policy.__setitem__("peak_allocation_bytes", [100]),
        lambda policy: policy.__setitem__(
            "peak_allocation_bytes", [100, float("inf")]
        ),
        lambda policy: policy.__setitem__("repeated_measurements", 3),
        lambda policy: policy.__setitem__("memory_growth_percent", 0.75),
    ],
)
def test_memory_growth_must_be_recomputed_from_finite_repeated_samples(
    tmp_path: Path, mutator
) -> None:
    report = _report(2)
    mutator(report["sections"]["memory"]["metrics"]["policies"]["auto"])
    path = tmp_path / "world2.json"
    path.write_bytes(_exact_bytes_with_digest(report, allow_nan=True))

    result = _validate(path)

    assert result.returncode != 0
    assert "memory" in result.stdout


def test_out_of_float_range_json_integer_is_blocked_without_validator_crash(
    tmp_path: Path,
) -> None:
    report = _report(2)
    report["sections"]["performance"]["metrics"]["workloads"][0]["modes"][
        "baseline"
    ]["gradient_ms_median"] = 10**400
    path = _write_report(tmp_path / "world2.json", report)

    result = _validate(path)

    assert result.returncode != 0
    assert "out of finite numeric range" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "sample",
    [10**400, float("inf"), True, None],
)
def test_memory_peak_samples_fail_closed_without_validator_crash(
    tmp_path: Path,
    sample,
) -> None:
    report = _report(2)
    report["sections"]["memory"]["metrics"]["policies"]["auto"][
        "peak_allocation_bytes"
    ] = [100, sample]
    path = tmp_path / "world2.json"
    path.write_bytes(
        _exact_bytes_with_digest(
            report,
            allow_nan=isinstance(sample, float) and not math.isfinite(sample),
        )
    )

    result = _validate(path)

    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "memory" in "\n".join(payload["failed_conditions"]).lower()
    assert result.stderr == ""


@pytest.mark.parametrize("sample", [10**400, float("inf"), True, None])
def test_producer_memory_growth_rejects_invalid_numeric_samples(sample) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        probe._memory_growth_percent([100, sample])

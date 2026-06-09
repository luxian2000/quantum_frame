"""Compare VQE-QAS v3 screening labels against complex128 audit labels.

The intended comparison is c64-multi screening versus CPU-numpy-c128-multi
audit on the same candidate subset. The report emphasizes top-K overlap and
near-neighbor inversions instead of global Spearman, because far-apart
candidates can make global rank correlation look falsely reassuring.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    name: str
    family: str
    energy: float
    delta_ref: float
    rank: int


def _load_rows(path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, item in enumerate(reader, start=1):
            rows.append(
                AuditRow(
                    name=str(item["name"]),
                    family=str(item.get("family") or ""),
                    energy=float(item["energy"]),
                    delta_ref=float(item["delta_ref"]),
                    rank=int(item.get("rank") or item.get("candidate_index") or index),
                )
            )
    return sorted(rows, key=lambda row: (row.energy, row.name))


def _top_names(rows: list[AuditRow], k: int) -> set[str]:
    return {row.name for row in rows[: min(max(0, int(k)), len(rows))]}


def _overlap(left: list[AuditRow], right: list[AuditRow], k: int) -> float | None:
    left_names = _top_names(left, k)
    right_names = _top_names(right, k)
    denom = min(len(left_names), len(right_names))
    if denom <= 0:
        return None
    return len(left_names & right_names) / denom


def _near_neighbor_inversions(
    screening_rows: list[AuditRow],
    audit_by_name: dict[str, AuditRow],
    top_k: int,
    window_mha: float,
) -> dict[str, object]:
    pool = [row for row in screening_rows[: min(int(top_k), len(screening_rows))] if row.name in audit_by_name]
    threshold = float(window_mha) / 1000.0
    pairs: list[tuple[str, str]] = []
    inversions = 0
    for left_index, left in enumerate(pool):
        for right in pool[left_index + 1 :]:
            if abs(left.delta_ref - right.delta_ref) > threshold:
                continue
            pairs.append((left.name, right.name))
            screen_sign = left.delta_ref - right.delta_ref
            audit_sign = audit_by_name[left.name].delta_ref - audit_by_name[right.name].delta_ref
            if screen_sign == 0.0 or audit_sign == 0.0:
                continue
            if screen_sign * audit_sign < 0.0:
                inversions += 1
    pair_count = len(pairs)
    return {
        "top_k": int(top_k),
        "window_mha": float(window_mha),
        "pair_count": pair_count,
        "inversions": inversions,
        "inversion_rate": None if pair_count == 0 else inversions / pair_count,
    }


def analyze(
    screening_csv: Path,
    audit_csv: Path,
    output_dir: Path,
    top10_threshold: float,
    top20_threshold: float,
    inversion_threshold: float,
    max_delta_error_mha: float,
    neighbor_top_k: int,
    neighbor_window_mha: float,
) -> dict[str, object]:
    screening = _load_rows(screening_csv)
    audit = _load_rows(audit_csv)
    audit_by_name = {row.name: row for row in audit}
    common_names = sorted({row.name for row in screening} & set(audit_by_name))
    if not common_names:
        raise ValueError("No overlapping candidate names between screening and audit CSVs")

    screening_common = [row for row in screening if row.name in audit_by_name]
    audit_common = sorted([audit_by_name[name] for name in common_names], key=lambda row: (row.energy, row.name))
    delta_errors = [
        abs(row.delta_ref - audit_by_name[row.name].delta_ref)
        for row in screening_common
    ]
    max_error = max(delta_errors) if delta_errors else 0.0
    near = _near_neighbor_inversions(
        screening_common,
        audit_by_name,
        top_k=neighbor_top_k,
        window_mha=neighbor_window_mha,
    )
    top10 = _overlap(screening_common, audit_common, 10)
    top20 = _overlap(screening_common, audit_common, 20)
    inversion_rate = near["inversion_rate"]
    passed = (
        top10 is not None
        and top20 is not None
        and inversion_rate is not None
        and top10 >= float(top10_threshold)
        and top20 >= float(top20_threshold)
        and inversion_rate <= float(inversion_threshold)
        and max_error <= float(max_delta_error_mha) / 1000.0
    )
    report = {
        "passed": bool(passed),
        "screening_csv": str(screening_csv),
        "audit_csv": str(audit_csv),
        "common_candidates": len(common_names),
        "top10_overlap": top10,
        "top20_overlap": top20,
        "near_neighbor": near,
        "max_abs_delta_ref_error_mha": max_error * 1000.0,
        "thresholds": {
            "top10_overlap": float(top10_threshold),
            "top20_overlap": float(top20_threshold),
            "near_neighbor_inversion_rate": float(inversion_threshold),
            "max_abs_delta_ref_error_mha": float(max_delta_error_mha),
        },
        "screening_top20": [row.name for row in screening_common[: min(20, len(screening_common))]],
        "audit_top20": [row.name for row in audit_common[: min(20, len(audit_common))]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dtype_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "VQE-QAS v3 dtype audit",
        f"passed: {report['passed']}",
        f"common_candidates: {report['common_candidates']}",
        f"top10_overlap: {top10 if top10 is not None else 'NA'}",
        f"top20_overlap: {top20 if top20 is not None else 'NA'}",
        f"near_neighbor_pairs: {near['pair_count']}",
        f"near_neighbor_inversions: {near['inversions']}",
        f"near_neighbor_inversion_rate: {inversion_rate if inversion_rate is not None else 'NA'}",
        f"max_abs_delta_ref_error_mha: {report['max_abs_delta_ref_error_mha']:.6f}",
        "",
        "Pass thresholds",
        f"top10_overlap >= {top10_threshold}",
        f"top20_overlap >= {top20_threshold}",
        f"near_neighbor_inversion_rate <= {inversion_threshold}",
        f"max_abs_delta_ref_error_mha <= {max_delta_error_mha}",
    ]
    (output_dir / "dtype_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare c64 screening labels against c128 audit labels")
    parser.add_argument("--screening-csv", required=True, help="Usually c64-multi CSV")
    parser.add_argument("--audit-csv", required=True, help="Usually CPU-numpy-c128-multi CSV")
    parser.add_argument("--output-dir", default="outputs/vqe_tfim_v3_dtype_audit")
    parser.add_argument("--top10-threshold", type=float, default=0.5)
    parser.add_argument("--top20-threshold", type=float, default=0.7)
    parser.add_argument("--inversion-threshold", type=float, default=0.25)
    parser.add_argument("--max-delta-error-mha", type=float, default=2.0)
    parser.add_argument("--neighbor-top-k", type=int, default=20)
    parser.add_argument("--neighbor-window-mha", type=float, default=5.0)
    args = parser.parse_args()

    report = analyze(
        screening_csv=Path(args.screening_csv),
        audit_csv=Path(args.audit_csv),
        output_dir=Path(args.output_dir),
        top10_threshold=args.top10_threshold,
        top20_threshold=args.top20_threshold,
        inversion_threshold=args.inversion_threshold,
        max_delta_error_mha=args.max_delta_error_mha,
        neighbor_top_k=args.neighbor_top_k,
        neighbor_window_mha=args.neighbor_window_mha,
    )
    print((Path(args.output_dir) / "dtype_audit.md").read_text(encoding="utf-8"), end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

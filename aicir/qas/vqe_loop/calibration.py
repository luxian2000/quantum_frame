"""Calibrate the trust-region retrieval oracle from fair-VQE labels.

The report checks whether holdout rows inside the trust region have lower MAE
than out-of-region rows and whether sparse rows trigger abstain as intended.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter
from statistics import median
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_loop.protocol import (
    DEFAULT_TRUST_REGION_RULES,
    LabelSource,
    LabelStatus,
)
from aicir.qas.vqe_loop.geometry import (
    CandidateRecord,
    DistanceScales,
    fit_distance_scales,
    parse_hamiltonian_features,
    task_aware_composite_distance,
)


HOLDOUT_SOURCES = {
    LabelSource.HOLDOUT_ID.value,
    LabelSource.HOLDOUT_BOUNDARY.value,
    LabelSource.HOLDOUT_SPARSE.value,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _row_to_candidate(row: dict[str, str]) -> CandidateRecord:
    hamiltonian_class = row.get("hamiltonian_class", "")
    n_qubits = int(row.get("n_qubits") or 0)
    hamiltonian_id = row.get("hamiltonian_id", "") or (f"{hamiltonian_class}_{n_qubits}q" if hamiltonian_class and n_qubits else "")
    return CandidateRecord(
        architecture_id=row["architecture_id"],
        canonical_arch_hash=row.get("canonical_arch_hash", row["architecture_id"]),
        family=row.get("family", ""),
        entangler_type=row.get("entangler_type", ""),
        topology=row.get("topology", ""),
        depth_group=row.get("depth_group", ""),
        n_params=float(row.get("n_params") or 0),
        two_q_count=float(row.get("two_q_count") or 0),
        hamiltonian_id=hamiltonian_id,
        hamiltonian_class=hamiltonian_class,
        hamiltonian_coverage=float(row.get("hamiltonian_coverage") or 0),
        hamiltonian_features=parse_hamiltonian_features(row.get("hamiltonian_coverage_features", "")),
        metadata={"n_qubits": n_qubits, "source": row.get("source", ""), "hamiltonian_id": hamiltonian_id},
    )


def _completed_rows(rows: list[dict[str, str]], protocol_version: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("label_status") == LabelStatus.COMPLETED.value
        and row.get("protocol_version") == protocol_version
        and row.get("fair_best_energy") not in {"", None}
    ]


def _nearest(
    query: CandidateRecord,
    pool: list[tuple[CandidateRecord, float]],
    scales: DistanceScales,
) -> list[tuple[float, CandidateRecord, float]]:
    scored = [
        (task_aware_composite_distance(query, candidate, scales), candidate, energy)
        for candidate, energy in pool
        if candidate.architecture_id != query.architecture_id
    ]
    return sorted(scored, key=lambda item: (item[0], item[1].architecture_id))


def _predict(neighbors: list[tuple[float, CandidateRecord, float]], k: int) -> float | None:
    usable = neighbors[: max(1, int(k))]
    if not usable:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, _candidate, energy in usable:
        weight = 1.0 / max(float(distance), 1e-9)
        weighted_sum += weight * float(energy)
        weight_total += weight
    return weighted_sum / weight_total


def _mae(items: list[float]) -> float | None:
    if not items:
        return None
    return sum(abs(value) for value in items) / len(items)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _boundary_holdout_by_n_qubits(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(
        f"{row.get('n_qubits', '')}q"
        for row in rows
        if row.get("source") == LabelSource.HOLDOUT_BOUNDARY.value
    )
    return dict(sorted(counts.items()))


def _boundary_warnings(boundary_counts: dict[str, int], *, minimum: int = 3) -> list[str]:
    warnings = []
    for n_qubits, count in sorted(boundary_counts.items()):
        if int(count) < int(minimum):
            warnings.append(
                f"{n_qubits}: boundary holdout count {count} < {minimum}; "
                "TR boundary calibration for this scale is low-confidence."
            )
    return warnings


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Trust-region calibration",
        "",
        f"protocol_version: {report['protocol_version']}",
        f"n_completed: {report['n_completed']}",
        f"n_train: {report['n_train']}",
        f"n_holdout: {report['n_holdout']}",
        "",
        "## boundary_holdout_by_n_qubits",
    ]
    boundary_counts = report.get("boundary_holdout_by_n_qubits", {})
    if boundary_counts:
        for n_qubits, count in boundary_counts.items():
            suffix = "  <-- low confidence" if int(count) < 3 else ""
            lines.append(f"- {n_qubits}: {count}{suffix}")
    else:
        lines.append("- none: 0")
    warnings = report.get("warnings", [])
    lines.extend(["", "## warnings"])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## metrics",
            f"- k_min: {report['k_min']}",
            f"- d_max: {report['d_max']}",
            f"- tr_in_mae: {report['tr_in_mae']}",
            f"- tr_out_mae: {report['tr_out_mae']}",
            f"- sparse_abstain_rate: {report['sparse_abstain_rate']}",
            f"- overall_pass: {report['passes']['overall']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a trust-region kNN oracle")
    parser.add_argument("--benchmark-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--k-min", type=int, default=int(DEFAULT_TRUST_REGION_RULES["k_min"]))
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument("--target-ratio", type=float, default=float(DEFAULT_TRUST_REGION_RULES["target_tr_in_mae_ratio"]))
    parser.add_argument("--target-sparse-abstain", type=float, default=float(DEFAULT_TRUST_REGION_RULES["target_sparse_abstain_rate"]))
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    all_rows = _read_csv(Path(args.benchmark_table))
    rows = _completed_rows(all_rows, args.protocol_version)
    candidates = [_row_to_candidate(row) for row in rows]
    scales = fit_distance_scales(candidates) if candidates else DistanceScales(
        n_params=1.0,
        two_q_count=1.0,
        hamiltonian_coverage=1.0,
    )
    indexed = [
        (_row_to_candidate(row), float(row["fair_best_energy"]))
        for row in rows
    ]
    train = [
        item for item, row in zip(indexed, rows)
        if row.get("source") not in HOLDOUT_SOURCES
    ]
    holdout = [
        (item, row) for item, row in zip(indexed, rows)
        if row.get("source") in HOLDOUT_SOURCES
    ]

    if args.d_max is None and train:
        nearest_distances = []
        for candidate, _energy in train:
            neighbors = _nearest(candidate, train, scales)
            if neighbors:
                nearest_distances.append(neighbors[min(len(neighbors), int(args.k_min)) - 1][0])
        d_max = 1.5 * median(nearest_distances) if nearest_distances else 0.0
    else:
        d_max = float(args.d_max or 0.0)

    tr_errors: list[float] = []
    out_errors: list[float] = []
    source_counts: dict[str, int] = {}
    sparse_abstain = 0
    sparse_total = 0
    evaluated: list[dict[str, Any]] = []
    for (candidate, actual), row in holdout:
        neighbors = _nearest(candidate, train, scales)
        neighbor_count = sum(1 for distance, _candidate, _energy in neighbors if distance <= d_max)
        in_tr = neighbor_count >= int(args.k_min)
        prediction = _predict(neighbors, int(args.k_min))
        source = row.get("source", "")
        source_counts[source] = source_counts.get(source, 0) + 1
        if source == LabelSource.HOLDOUT_SPARSE.value:
            sparse_total += 1
            sparse_abstain += 0 if in_tr else 1
        if prediction is not None:
            error = float(prediction - actual)
            (tr_errors if in_tr else out_errors).append(error)
        evaluated.append(
            {
                "architecture_id": candidate.architecture_id,
                "source": source,
                "in_trust_region": in_tr,
                "neighbor_count": neighbor_count,
                "prediction": prediction,
                "actual": actual,
                "error": None if prediction is None else prediction - actual,
            }
        )

    tr_mae = _mae(tr_errors)
    out_mae = _mae(out_errors)
    sparse_abstain_rate = (sparse_abstain / sparse_total) if sparse_total else None
    ratio_pass = tr_mae is not None and out_mae not in {None, 0.0} and tr_mae <= float(args.target_ratio) * out_mae
    sparse_pass = sparse_abstain_rate is not None and sparse_abstain_rate >= float(args.target_sparse_abstain)
    report = {
        "protocol_version": args.protocol_version,
        "n_completed": len(rows),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "holdout_source_counts": source_counts,
        "boundary_holdout_by_n_qubits": _boundary_holdout_by_n_qubits(all_rows),
        "warnings": _boundary_warnings(_boundary_holdout_by_n_qubits(all_rows)),
        "distance_scales": scales.__dict__,
        "k_min": int(args.k_min),
        "d_max": d_max,
        "tr_in_mae": tr_mae,
        "tr_out_mae": out_mae,
        "tr_in_count": len(tr_errors),
        "tr_out_count": len(out_errors),
        "sparse_abstain_rate": sparse_abstain_rate,
        "passes": {
            "mae_ratio": ratio_pass,
            "sparse_abstain": sparse_pass,
            "overall": bool(ratio_pass and sparse_pass),
        },
        "evaluated_holdout": evaluated,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_output:
        _write_markdown_report(Path(args.markdown_output), _jsonable(report))
    print(json.dumps(_jsonable({key: report[key] for key in report if key != "evaluated_holdout"}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

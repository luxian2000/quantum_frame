"""Compare source-only, target-only, and source+target local oracle transfer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from typing import Any, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_qas_protocol import (
    CandidateRecord,
    DistanceScales,
    LabelSource,
    LabelStatus,
    fit_distance_scales,
    parse_hamiltonian_features,
    task_aware_composite_distance,
)


TARGET_HOLDOUT_SOURCES = {
    LabelSource.TARGET_HOLDOUT_ID.value,
    LabelSource.TARGET_HOLDOUT_BOUNDARY.value,
    LabelSource.TARGET_HOLDOUT_SPARSE.value,
}

TARGET_TRAIN_SOURCES = {
    LabelSource.TARGET_FEWSHOT_TRAIN.value,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _row_hamiltonian_id(row: dict[str, str]) -> str:
    hamiltonian_id = row.get("hamiltonian_id", "")
    if hamiltonian_id:
        return hamiltonian_id
    hamiltonian_class = row.get("hamiltonian_class", "")
    n_qubits = row.get("n_qubits", "")
    return f"{hamiltonian_class}_{n_qubits}q" if hamiltonian_class and n_qubits else ""


def _row_to_candidate(row: dict[str, str]) -> CandidateRecord:
    hamiltonian_id = _row_hamiltonian_id(row)
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
        hamiltonian_class=row.get("hamiltonian_class", ""),
        hamiltonian_coverage=float(row.get("hamiltonian_coverage") or 0),
        hamiltonian_features=parse_hamiltonian_features(row.get("hamiltonian_coverage_features", "")),
        metadata={
            "n_qubits": int(row.get("n_qubits") or 0),
            "source": row.get("source", ""),
            "hamiltonian_id": hamiltonian_id,
        },
    )


def _completed_rows(rows: Sequence[dict[str, str]], protocol_version: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("label_status") == LabelStatus.COMPLETED.value
        and row.get("protocol_version") == protocol_version
        and row.get("fair_best_energy") not in {"", None}
    ]


def _indexed(rows: Sequence[dict[str, str]]) -> list[tuple[CandidateRecord, float]]:
    return [(_row_to_candidate(row), float(row["fair_best_energy"])) for row in rows]


def _nearest(
    query: CandidateRecord,
    train: Sequence[tuple[CandidateRecord, float]],
    scales: DistanceScales,
) -> list[tuple[float, CandidateRecord, float]]:
    scored = [
        (task_aware_composite_distance(query, candidate, scales), candidate, energy)
        for candidate, energy in train
        if not (
            candidate.architecture_id == query.architecture_id
            and str(candidate.hamiltonian_id) == str(query.hamiltonian_id)
        )
    ]
    return sorted(scored, key=lambda item: (item[0], item[1].architecture_id))


def _predict(neighbors: Sequence[tuple[float, CandidateRecord, float]], k_min: int) -> float | None:
    usable = list(neighbors[: max(1, int(k_min))])
    if len(usable) < max(1, int(k_min)):
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, _candidate, energy in usable:
        weight = 1.0 / max(float(distance), 1e-9)
        weighted_sum += weight * float(energy)
        weight_total += weight
    return weighted_sum / weight_total


def _derive_d_max(train: Sequence[tuple[CandidateRecord, float]], scales: DistanceScales, k_min: int) -> float:
    kth_distances: list[float] = []
    for candidate, _energy in train:
        neighbors = _nearest(candidate, train, scales)
        if len(neighbors) >= max(1, int(k_min)):
            kth_distances.append(neighbors[int(k_min) - 1][0])
    return 1.5 * median(kth_distances) if kth_distances else 0.0


def _evaluate_scenario(
    name: str,
    train: Sequence[tuple[CandidateRecord, float]],
    holdout: Sequence[tuple[CandidateRecord, float]],
    scales: DistanceScales,
    *,
    k_min: int,
    d_max: float | None,
) -> dict[str, Any]:
    if not train or not holdout or len(train) < max(1, int(k_min)):
        return {
            "name": name,
            "usable": False,
            "n_train": len(train),
            "n_holdout": len(holdout),
            "mae": None,
            "abstain_rate": None,
            "in_trust_region": 0,
            "predicted": 0,
        }
    scenario_d_max = float(d_max) if d_max is not None else _derive_d_max(train, scales, int(k_min))
    errors: list[float] = []
    in_tr_count = 0
    evaluated: list[dict[str, Any]] = []
    for candidate, actual in holdout:
        neighbors = _nearest(candidate, train, scales)
        neighbor_count = sum(1 for distance, _candidate, _energy in neighbors if distance <= scenario_d_max)
        in_tr = neighbor_count >= int(k_min)
        prediction = _predict(neighbors, int(k_min)) if in_tr else None
        if in_tr:
            in_tr_count += 1
        if prediction is not None:
            errors.append(float(prediction - actual))
        evaluated.append(
            {
                "architecture_id": candidate.architecture_id,
                "actual": actual,
                "prediction": prediction,
                "neighbor_count": neighbor_count,
                "in_trust_region": in_tr,
                "error": None if prediction is None else prediction - actual,
            }
        )
    abstain_count = len(holdout) - in_tr_count
    return {
        "name": name,
        "usable": True,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "k_min": int(k_min),
        "d_max": scenario_d_max,
        "mae": None if not errors else sum(abs(error) for error in errors) / len(errors),
        "abstain_rate": abstain_count / float(len(holdout)) if holdout else None,
        "in_trust_region": in_tr_count,
        "predicted": len(errors),
        "evaluated_holdout": evaluated,
    }


def compare_transfer_oracles(
    rows: Sequence[dict[str, str]],
    *,
    target_hamiltonian_id: str,
    protocol_version: str = "fair_vqe_protocol_v1",
    k_min: int = 5,
    d_max: float | None = None,
) -> dict[str, Any]:
    """Compare transfer settings on completed target holdout labels."""

    completed = _completed_rows(rows, protocol_version)
    source_rows = [
        row for row in completed
        if _row_hamiltonian_id(row) != str(target_hamiltonian_id)
    ]
    target_rows = [
        row for row in completed
        if _row_hamiltonian_id(row) == str(target_hamiltonian_id)
    ]
    target_train_rows = [
        row for row in target_rows
        if row.get("source") in TARGET_TRAIN_SOURCES
    ]
    target_holdout_rows = [
        row for row in target_rows
        if row.get("source") in TARGET_HOLDOUT_SOURCES
    ]
    source_train = _indexed(source_rows)
    target_train = _indexed(target_train_rows)
    target_holdout = _indexed(target_holdout_rows)
    all_candidates = [item[0] for item in source_train + target_train + target_holdout]
    scales = fit_distance_scales(all_candidates) if all_candidates else DistanceScales(1.0, 1.0, 1.0)
    scenarios = {
        "source_only": _evaluate_scenario(
            "source_only",
            source_train,
            target_holdout,
            scales,
            k_min=int(k_min),
            d_max=d_max,
        ),
        "target_only": _evaluate_scenario(
            "target_only",
            target_train,
            target_holdout,
            scales,
            k_min=int(k_min),
            d_max=d_max,
        ),
        "source_plus_target": _evaluate_scenario(
            "source_plus_target",
            source_train + target_train,
            target_holdout,
            scales,
            k_min=int(k_min),
            d_max=d_max,
        ),
    }
    return {
        "target_hamiltonian_id": target_hamiltonian_id,
        "protocol_version": protocol_version,
        "n_completed": len(completed),
        "n_source_train": len(source_train),
        "n_target_train": len(target_train),
        "n_target_holdout": len(target_holdout),
        "distance_scales": scales.__dict__,
        "scenarios": scenarios,
        "note": "Positive transfer means source_plus_target improves target_only under the same target-label budget.",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source/target local oracle transfer")
    parser.add_argument("--benchmark-table", required=True)
    parser.add_argument("--target-hamiltonian-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v1")
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--d-max", type=float, default=None)
    args = parser.parse_args()

    rows = _read_csv(Path(args.benchmark_table))
    report = compare_transfer_oracles(
        rows,
        target_hamiltonian_id=args.target_hamiltonian_id,
        protocol_version=args.protocol_version,
        k_min=int(args.k_min),
        d_max=args.d_max,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        key: report[key]
        for key in [
            "target_hamiltonian_id",
            "protocol_version",
            "n_completed",
            "n_source_train",
            "n_target_train",
            "n_target_holdout",
            "note",
        ]
    }
    summary["scenarios"] = {
        name: {
            "usable": item["usable"],
            "n_train": item["n_train"],
            "n_holdout": item["n_holdout"],
            "mae": item["mae"],
            "abstain_rate": item["abstain_rate"],
            "in_trust_region": item["in_trust_region"],
        }
        for name, item in report["scenarios"].items()
    }
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Plan the next fair-label batch for trust-region VQE-QAS.

Track A proposes MoG-EA/local candidates and lets the oracle rank only inside
the trust region.  Abstained, boundary, sparse, and optional supernet-priority
candidates are routed into expansion labels instead of being discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_loop.protocol import (
    BENCHMARK_TABLE_FIELDS,
    DEFAULT_BATCH_QUOTAS,
    DEFAULT_TRUST_REGION_RULES,
    LabelSource,
    LabelStatus,
)
from aicir.qas.vqe_loop.geometry import (
    CandidateRecord,
    fit_distance_scales,
    min_distance_to_set,
    parse_pauli_hamiltonian_terms,
    parse_hamiltonian_features,
    task_aware_composite_distance,
)
from aicir.qas.vqe_loop.selection_ops import (
    compute_abstain_rate,
    select_farthest_first,
    select_mog_ea_candidates,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})


def _load_supernet_sidecar(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    records = raw.get("records", raw if isinstance(raw, list) else [])
    sidecar: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        architecture_id = str(record.get("architecture_id", "")).strip()
        if architecture_id:
            sidecar[architecture_id] = record
    return sidecar


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower() if str(value).strip() else ""


def _attach_supernet_sidecar(row: dict[str, str], sidecar: dict[str, dict[str, Any]]) -> dict[str, str]:
    enriched = dict(row)
    record = sidecar.get(str(row.get("architecture_id", "")).strip())
    if not record:
        return enriched
    for field in ("supernet_rank_score", "supernet_init_params_ref", "screening_energy"):
        value = record.get(field, "")
        if value not in {"", None}:
            enriched[field] = str(value)
    if "screening_energy_is_final_label" in record:
        enriched["screening_energy_is_final_label"] = _bool_text(record["screening_energy_is_final_label"])
    return enriched


def _supernet_priority(row: dict[str, Any]) -> float:
    raw = row.get("supernet_rank_score", "")
    if raw in {"", None}:
        return float("inf")
    try:
        # Lower shared-weight energy/loss is better, so use negative score for
        # descending priority tie-breaks.
        return float(raw)
    except (TypeError, ValueError):
        return float("inf")


def _row_to_candidate(row: dict[str, str]) -> CandidateRecord:
    hamiltonian_class = row.get("hamiltonian_class", "tfim")
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
        metadata={
            "n_qubits": n_qubits,
            "zero_cost_status": row.get("zero_cost_status", ""),
            "source": row.get("source", ""),
            "hamiltonian_id": hamiltonian_id,
            "hea_mask": row.get("hea_mask", ""),
            "ansatz_gene": row.get("ansatz_gene", ""),
        },
    )


def _completed_rows(rows: list[dict[str, str]], protocol_version: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("label_status") == LabelStatus.COMPLETED.value
        and row.get("protocol_version") == protocol_version
        and row.get("fair_best_energy") not in {"", None}
    ]


def _labeled_ids(rows: list[dict[str, str]], protocol_version: str) -> set[str]:
    active_statuses = {
        LabelStatus.PENDING.value,
        LabelStatus.RUNNING.value,
        LabelStatus.COMPLETED.value,
        LabelStatus.FAILED_NONRETRYABLE.value,
        LabelStatus.EXCLUDED_INVALID_CANDIDATE.value,
        LabelStatus.EXCLUDED_DUPLICATE.value,
    }
    return {
        row["architecture_id"]
        for row in rows
        if row.get("protocol_version") in {"", protocol_version}
        and row.get("architecture_id")
        and row.get("label_status") in active_statuses
    }


def _nearest(
    query: CandidateRecord,
    labeled: list[tuple[CandidateRecord, float]],
    scales,
) -> list[tuple[float, CandidateRecord, float]]:
    scored = [
        (task_aware_composite_distance(query, candidate, scales), candidate, energy)
        for candidate, energy in labeled
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


def _derive_task_context(benchmark_rows: list[dict[str, str]], protocol_version: str) -> dict[str, str]:
    """Return the active Hamiltonian task fields that Stage-2 queues must preserve."""

    task_fields = (
        "n_qubits",
        "hamiltonian_id",
        "hamiltonian_class",
        "hamiltonian_terms",
        "reference_energy",
    )
    completed = _completed_rows(benchmark_rows, protocol_version)
    source_rows = completed or [
        row for row in benchmark_rows
        if row.get("protocol_version") in {"", protocol_version}
    ]
    for row in source_rows:
        if row.get("hamiltonian_id") or row.get("hamiltonian_terms"):
            return {field: row.get(field, "") for field in task_fields}
    return {}


def _task_hamiltonian_terms(task_context: dict[str, str]) -> tuple[tuple[float, str], ...]:
    raw = str(task_context.get("hamiltonian_terms", "") or "").strip()
    if not raw:
        return ()
    return parse_pauli_hamiltonian_terms(json.loads(raw))


def _derive_d_max(labeled: list[tuple[CandidateRecord, float]], scales, k_min: int) -> float:
    kth_distances: list[float] = []
    for candidate, _energy in labeled:
        neighbors = _nearest(candidate, labeled, scales)
        if len(neighbors) >= max(1, int(k_min)):
            kth_distances.append(neighbors[int(k_min) - 1][0])
    return 1.5 * median(kth_distances) if kth_distances else 0.0


def _queue_row(
    candidate_row: dict[str, str],
    *,
    source: LabelSource,
    protocol_version: str,
    batch_id: str,
    task_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update({field: candidate_row.get(field, "") for field in row})
    if task_context:
        row.update({field: value for field, value in task_context.items() if value not in {"", None}})
    row.update(
        {
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": source.value,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "hamiltonian_id": row.get("hamiltonian_id")
            or f"{row.get('hamiltonian_class', 'tfim')}_{row.get('n_qubits', '')}q",
            "hamiltonian_coverage_features": row.get("hamiltonian_coverage_features")
            or row.get("hamiltonian_coverage", ""),
        }
    )
    return row


def _take_unique(
    selected: list[tuple[dict[str, str], LabelSource]],
    candidates: list[dict[str, str]],
    source: LabelSource,
    count: int,
) -> None:
    selected_ids = {row["architecture_id"] for row, _source in selected}
    for row in candidates:
        if len([1 for _row, item_source in selected if item_source == source]) >= count:
            return
        if row["architecture_id"] not in selected_ids:
            selected.append((row, source))
            selected_ids.add(row["architecture_id"])


def _priority_boundary_anchors(
    completed: list[tuple[CandidateRecord, float]],
    *,
    limit: int = 8,
) -> list[CandidateRecord]:
    preferred_sources = {
        LabelSource.TRACKB_SPARSE.value,
        LabelSource.TRACKB_SUPERNET.value,
        LabelSource.INITIAL_TRAIN.value,
    }
    preferred = [
        (record, energy)
        for record, energy in completed
        if str(record.metadata.get("source", "")) in preferred_sources
    ]
    pool = preferred or completed
    return [
        record
        for record, _energy in sorted(pool, key=lambda item: (item[1], item[0].architecture_id))[: max(1, int(limit))]
    ]


def _rank_boundary_records(
    boundary_pool: list[CandidateRecord],
    *,
    completed: list[tuple[CandidateRecord, float]],
    scales,
    rows_by_id: dict[str, dict[str, Any]],
    count: int,
) -> list[CandidateRecord]:
    if int(count) <= 0 or not boundary_pool:
        return []
    anchors = _priority_boundary_anchors(completed)
    if not anchors:
        return sorted(
            boundary_pool,
            key=lambda record: (_supernet_priority(rows_by_id.get(record.architecture_id, {})), record.architecture_id),
        )[: int(count)]

    def nearest_priority_distance(record: CandidateRecord) -> float:
        return min(task_aware_composite_distance(record, anchor, scales) for anchor in anchors)

    return sorted(
        boundary_pool,
        key=lambda record: (
            nearest_priority_distance(record),
            _supernet_priority(rows_by_id.get(record.architecture_id, {})),
            record.architecture_id,
        ),
    )[: int(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan the next trust-region VQE-QAS label batch")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--benchmark-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--batch-id", default="batch")
    parser.add_argument("--k-min", type=int, default=int(DEFAULT_TRUST_REGION_RULES["k_min"]))
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument("--local", type=int, default=DEFAULT_BATCH_QUOTAS["local"])
    parser.add_argument("--boundary", type=int, default=DEFAULT_BATCH_QUOTAS["boundary"])
    parser.add_argument("--sparse", type=int, default=DEFAULT_BATCH_QUOTAS["sparse"])
    parser.add_argument("--control", type=int, default=DEFAULT_BATCH_QUOTAS["control"])
    parser.add_argument("--ea-population", type=int, default=48)
    parser.add_argument("--ea-generations", type=int, default=12)
    parser.add_argument("--ea-mutation-rate", type=float, default=0.20)
    parser.add_argument("--ea-crossover-rate", type=float, default=0.60)
    parser.add_argument("--ea-elite-count", type=int, default=6)
    parser.add_argument("--ea-seed", type=int, default=17)
    parser.add_argument("--ea-seed-count", type=int, default=8)
    parser.add_argument("--ea-proposal-multiplier", type=int, default=4)
    parser.add_argument("--ea-gene-key", default="hea_mask")
    parser.add_argument("--supernet-sidecar", default=None)
    parser.add_argument("--supernet-native-count", type=int, default=0)
    parser.add_argument("--supernet-native-layers", type=int, default=3)
    parser.add_argument("--supernet-native-supernet-num", type=int, default=2)
    parser.add_argument("--supernet-native-steps", type=int, default=20)
    parser.add_argument("--supernet-native-ranking-num", type=int, default=24)
    parser.add_argument("--supernet-native-finetune-steps", type=int, default=0)
    parser.add_argument("--supernet-native-seed", type=int, default=11)
    parser.add_argument("--supernet-native-device", default="cpu")
    args = parser.parse_args()

    sidecar = _load_supernet_sidecar(Path(args.supernet_sidecar) if args.supernet_sidecar else None)
    candidate_rows = [_attach_supernet_sidecar(row, sidecar) for row in _read_csv(Path(args.candidates))]
    benchmark_rows = _read_csv(Path(args.benchmark_table))
    completed = _completed_rows(benchmark_rows, args.protocol_version)
    task_context = _derive_task_context(benchmark_rows, args.protocol_version)
    labeled_ids = _labeled_ids(benchmark_rows, args.protocol_version)
    supernet_native_summary: dict[str, Any] = {"enabled": False, "generated_rows": 0}
    if int(args.supernet_native_count) > 0:
        hamiltonian_terms = _task_hamiltonian_terms(task_context)
        if hamiltonian_terms:
            from aicir.qas.vqe_loop.supernet_native import build_supernet_native_rows

            generated_rows, supernet_native_summary = build_supernet_native_rows(
                hamiltonian_terms=hamiltonian_terms,
                hamiltonian_id=task_context.get("hamiltonian_id", ""),
                hamiltonian_class=task_context.get("hamiltonian_class", ""),
                count=int(args.supernet_native_count),
                layers=int(args.supernet_native_layers),
                supernet_num=int(args.supernet_native_supernet_num),
                supernet_steps=int(args.supernet_native_steps),
                ranking_num=int(args.supernet_native_ranking_num),
                finetune_steps=int(args.supernet_native_finetune_steps),
                seed=int(args.supernet_native_seed),
                device=str(args.supernet_native_device),
                excluded_ids={row.get("architecture_id", "") for row in candidate_rows} | labeled_ids,
                params_dir=Path(args.output).parent,
            )
            candidate_rows.extend(generated_rows)
        else:
            supernet_native_summary = {
                "enabled": False,
                "generated_rows": 0,
                "reason": "missing_hamiltonian_terms",
            }
    candidate_by_id = {row["architecture_id"]: row for row in candidate_rows}
    candidate_records = [_row_to_candidate(row) for row in candidate_rows]
    scales = fit_distance_scales(candidate_records) if candidate_records else None
    completed_index = [
        (_row_to_candidate(row), float(row["fair_best_energy"]))
        for row in completed
    ]
    d_max = float(args.d_max) if args.d_max is not None else (_derive_d_max(completed_index, scales, args.k_min) if scales else 0.0)

    unlabeled_rows = [
        row for row in candidate_rows
        if row["architecture_id"] not in labeled_ids
        and row.get("zero_cost_status") != "hard_reject"
    ]
    unlabeled_records = [_row_to_candidate(row) for row in unlabeled_rows]
    row_by_id = {row["architecture_id"]: row for row in unlabeled_rows}

    local_scored: list[tuple[float, float, CandidateRecord]] = []
    ood_records: list[CandidateRecord] = []
    can_use_oracle = scales is not None and len(completed_index) >= int(args.k_min)
    ea_proposals: list[CandidateRecord] = []
    ea_backend = "not_run"
    track_a_mutations: list[dict[str, Any]] = []

    def oracle_eval(candidate: CandidateRecord) -> tuple[float | None, int, float, bool]:
        if not can_use_oracle or scales is None:
            return None, 0, float("inf"), False
        neighbors = _nearest(candidate, completed_index, scales)
        neighbor_count = sum(1 for distance, _candidate, _energy in neighbors if distance <= d_max)
        prediction = _predict(neighbors, int(args.k_min))
        nearest_distance = min_distance_to_set(candidate, [item[0] for item in completed_index], scales)
        in_trust_region = prediction is not None and neighbor_count >= int(args.k_min)
        return prediction, neighbor_count, nearest_distance, in_trust_region

    if can_use_oracle:
        seed_records = [
            candidate
            for candidate, _energy in sorted(completed_index, key=lambda item: (item[1], item[0].architecture_id))[: max(0, int(args.ea_seed_count))]
        ]

        def oracle_fitness(candidate: CandidateRecord) -> float:
            prediction, _neighbor_count, nearest_distance, in_trust_region = oracle_eval(candidate)
            if prediction is None or not in_trust_region:
                return -1.0e9 + min(float(nearest_distance), 10.0)
            # Lower predicted energy is better; a tiny diversity term keeps near-ties
            # from collapsing onto duplicate neighborhoods.
            return -float(prediction) + 1.0e-3 * float(nearest_distance)

        ea_limit = max(
            int(args.local) + int(args.boundary) + int(args.sparse),
            int(args.local) * max(1, int(args.ea_proposal_multiplier)),
        )
        ea_proposals, ea_backend = select_mog_ea_candidates(
            unlabeled_records,
            seeds=seed_records,
            excluded_ids=labeled_ids,
            fitness=oracle_fitness,
            population_size=int(args.ea_population),
            generations=int(args.ea_generations),
            mutation_rate=float(args.ea_mutation_rate),
            crossover_rate=float(args.ea_crossover_rate),
            elite_count=int(args.ea_elite_count),
            limit=ea_limit,
            random_seed=int(args.ea_seed),
            gene_key=str(args.ea_gene_key),
        )
        for candidate in ea_proposals:
            prediction, neighbor_count, nearest_distance, in_trust_region = oracle_eval(candidate)
            track_a_mutations.append(
                {
                    "architecture_id": candidate.architecture_id,
                    "canonical_arch_hash": candidate.canonical_arch_hash,
                    "prediction": prediction,
                    "neighbor_count": neighbor_count,
                    "nearest_distance": nearest_distance,
                    "abstain": not in_trust_region,
                }
            )
            if prediction is not None and in_trust_region:
                local_scored.append((float(prediction), float(nearest_distance), candidate))
            else:
                ood_records.append(candidate)
    else:
        ood_records = unlabeled_records

    track_a_abstain_rate = compute_abstain_rate(track_a_mutations)
    local_rows = [
        row_by_id[candidate.architecture_id]
        for _prediction, _distance, candidate in sorted(local_scored, key=lambda item: (item[0], -item[1], item[2].architecture_id))
    ]
    labeled_records = [candidate for candidate, _energy in completed_index]
    proposed_ids = {record.architecture_id for record in ea_proposals}
    ood_ids = {record.architecture_id for record in ood_records}
    ood_records.extend(
        record
        for record in unlabeled_records
        if record.architecture_id not in proposed_ids and record.architecture_id not in ood_ids
    )
    boundary_pool = [
        record
        for record in ood_records
        if record.metadata.get("zero_cost_status") == "soft_flag"
    ] or ood_records
    sparse_pool = [record for record in ood_records if record.architecture_id not in {item.architecture_id for item in boundary_pool}]
    if scales is None:
        boundary_records: list[CandidateRecord] = []
        sparse_records: list[CandidateRecord] = []
    else:
        boundary_limit = max(int(args.boundary), len(boundary_pool))
        boundary_ranked = select_farthest_first(boundary_pool, labeled_records, scales, count=boundary_limit)
        boundary_records = _rank_boundary_records(
            boundary_ranked,
            completed=completed_index,
            scales=scales,
            rows_by_id=row_by_id,
            count=int(args.boundary),
        )
        sparse_limit = max(int(args.sparse), len(sparse_pool or ood_records))
        sparse_ranked = select_farthest_first(sparse_pool or ood_records, labeled_records + boundary_records, scales, count=sparse_limit)
        sparse_records = sorted(
            sparse_ranked,
            key=lambda record: (_supernet_priority(row_by_id.get(record.architecture_id, {})), record.architecture_id),
        )[: int(args.sparse)]

    selected: list[tuple[dict[str, str], LabelSource]] = []
    _take_unique(selected, local_rows, LabelSource.TRACKA_LOCAL, int(args.local))
    _take_unique(selected, [row_by_id[item.architecture_id] for item in boundary_records], LabelSource.TRACKB_BOUNDARY, int(args.boundary))
    sparse_rows = [row_by_id[item.architecture_id] for item in sparse_records]
    supernet_sparse_rows = [row for row in sparse_rows if row.get("family") == "supernet_native"]
    regular_sparse_rows = [row for row in sparse_rows if row.get("family") != "supernet_native"]
    _take_unique(selected, supernet_sparse_rows, LabelSource.TRACKB_SUPERNET, int(args.sparse))
    used_sparse = len([1 for _row, source in selected if source == LabelSource.TRACKB_SUPERNET])
    _take_unique(selected, regular_sparse_rows, LabelSource.TRACKB_SPARSE, max(0, int(args.sparse) - used_sparse))
    remaining_for_control = sorted(
        [row for row in unlabeled_rows if row["architecture_id"] not in {selected_row["architecture_id"] for selected_row, _source in selected}],
        key=lambda row: (_supernet_priority(row), row.get("canonical_arch_hash", ""), row["architecture_id"]),
    )
    _take_unique(selected, remaining_for_control, LabelSource.CONTROL_RANDOM, int(args.control))

    # If trust-region labels are not available yet, reallocate unused local quota
    # to expansion/control instead of forcing unsupported oracle ranking.
    target_total = int(args.local) + int(args.boundary) + int(args.sparse) + int(args.control)
    if len(selected) < target_total:
        remaining = sorted(
            [row for row in unlabeled_rows if row["architecture_id"] not in {selected_row["architecture_id"] for selected_row, _source in selected}],
            key=lambda row: (_supernet_priority(row), row.get("canonical_arch_hash", ""), row["architecture_id"]),
        )
        _take_unique(selected, remaining, LabelSource.TRACKB_BOUNDARY, target_total)

    queue_rows = [
        _queue_row(
            row,
            source=source,
            protocol_version=args.protocol_version,
            batch_id=args.batch_id,
            task_context=task_context,
        )
        for row, source in selected[:target_total]
    ]
    _write_csv(Path(args.output), queue_rows)

    source_counts: dict[str, int] = {}
    for _row, source in selected[:target_total]:
        source_counts[source.value] = source_counts.get(source.value, 0) + 1
    summary = {
        "protocol_version": args.protocol_version,
        "batch_id": args.batch_id,
        "n_candidates": len(candidate_rows),
        "n_completed_labels": len(completed),
        "n_unlabeled_candidates": len(unlabeled_rows),
        "can_use_oracle": can_use_oracle,
        "k_min": int(args.k_min),
        "d_max": d_max,
        "ea": {
            "population": int(args.ea_population),
            "generations": int(args.ea_generations),
            "mutation_rate": float(args.ea_mutation_rate),
            "crossover_rate": float(args.ea_crossover_rate),
            "elite_count": int(args.ea_elite_count),
            "seed": int(args.ea_seed),
            "seed_count": int(args.ea_seed_count),
            "proposal_count": len(ea_proposals),
            "track_a_abstain_rate": track_a_abstain_rate,
            "gene_key": str(args.ea_gene_key),
            "backend": ea_backend,
        },
        "supernet_sidecar": {
            "path": str(args.supernet_sidecar or ""),
            "matched_records": sum(1 for row in candidate_rows if row.get("supernet_rank_score") not in {"", None}),
        },
        "supernet_native": supernet_native_summary,
        "requested_total": target_total,
        "planned_total": len(queue_rows),
        "source_counts": source_counts,
        "note": "Track A uses EA proposals scored by oracle only inside trust region; otherwise quota is reallocated to expansion/control.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

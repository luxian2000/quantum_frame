"""Plan one P1 mutation/oracle/fallback round.

This module writes fair-label queues but deliberately does not run fair VQE.
The generated queues can be handed to ``fair_labeling.py`` so P1 and baselines use
the same fair-call budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aicir.qas.vqe_loop.ansatz_family import summarize_ansatz_families
from aicir.qas.vqe_loop.benchmark_table import write_csv_rows
from aicir.qas.vqe_loop.oracle import predict_fair_energy
from aicir.qas.vqe_loop.p1_selection import (
    baseline_random_queue as _baseline_random_queue,
    baseline_selector_queue as _baseline_selector_queue,
    queue_row as _queue_row,
    rank_by_score as _rank_by_score,
    rank_fallback_rows as _rank_fallback_rows,
    score_rows as _score_rows,
    take_fill as _take_fill,
    task_context as _task_context,
)
from aicir.qas.vqe_loop.benchmark_table import (
    FairBudgetTracker,
    P1Quota,
    architecture_key,
    choose_quota,
    deduplicate_children,
    merge_quota_candidates,
    rank_with_zero_cost_soft_prefilter,
    resolve_p1_selector_fields,
)
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, LabelSource, LabelStatus, ZeroCostStatus
from aicir.qas.vqe_loop.training_free import annotate_training_free_rows
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float, is_empty as _is_empty
from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children, select_parent_rows


Evaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class P1RoundPlan:
    queue_rows: list[dict[str, Any]]
    child_rows: list[dict[str, Any]]
    oracle_rows: list[dict[str, Any]]
    fallback_rows: list[dict[str, Any]]
    baseline_queues: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]



def _source_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("p1_selection_source", ""))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _candidate_pool(
    child_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(child_rows) + list(control_rows):
        key = architecture_key(row)
        if key in seen:
            continue
        pool.append(dict(row))
        seen.add(key)
    return pool


def plan_p1_round(
    *,
    labeled_rows: Sequence[Mapping[str, Any]],
    known_unlabeled_rows: Sequence[Mapping[str, Any]] = (),
    control_rows: Sequence[Mapping[str, Any]] = (),
    parent_count: int,
    children_per_parent: int,
    fair_top_k: int,
    selector: str = "e2",
    cheap_eval_selector: str = "e2",
    evaluator_registry: Mapping[str, Evaluator] | None = None,
    k_min: int = 3,
    d_max: float = 0.1,
    batch_id: str = "p1_round1",
    protocol_version: str = "fair_vqe_protocol_v2",
    mutation_types: Sequence[str] = ("gate_mutation", "connectivity_mutation", "layer_mutation", "depth_mutation"),
    operator_pool: Sequence[str] | None = None,
    seed: int = 0,
    baseline_selector_fields: Sequence[str] = ("E2", "E5"),
    previous_oracle_trusted_fair_mean: float | None = None,
    previous_fallback_fair_mean: float | None = None,
    previous_oracle_hit_rate: float | None = None,
    enable_training_free_pruning: bool = False,
    training_free_soft_prefilter_multiplier: int = 2,
    fallback_audit_multiplier: int = 4,
    selection_policy: str = "no_regret",
    oracle_extra_top_k: int = 0,
    min_previous_oracle_hit_rate: float = 0.5,
    oracle_max_neighbor_std: float | None = None,
    trainability_soft_quantile: float = 0.10,
    expressibility_soft_quantile: float = 0.05,
    trainability_hard_floor: float = 0.01,
    expressibility_hard_floor: float = 0.01,
    entanglement_soft_floor: float = 0.0,
    max_params: int | None = None,
    max_two_q: int | None = None,
) -> P1RoundPlan:
    """Plan a P1 queue plus equal-budget baseline queues."""

    registry = evaluator_registry or {}
    k = max(0, int(fair_top_k))
    normalized_selection_policy = str(selection_policy).strip().lower()
    if normalized_selection_policy not in {"no_regret", "no_regret_lite", "quota"}:
        raise ValueError(f"unsupported P1 selection_policy: {selection_policy}")
    oracle_extra_k = max(0, int(oracle_extra_top_k)) if normalized_selection_policy == "no_regret_lite" else 0
    task = _task_context(list(labeled_rows) + list(known_unlabeled_rows))
    parents = select_parent_rows(labeled_rows, count=int(parent_count))
    child_rows = generate_mutation_children(
        parents,
        children_per_parent=int(children_per_parent),
        mutation_types=tuple(mutation_types),
        operator_pool=operator_pool,
        seed=int(seed),
    )
    dedup = deduplicate_children(
        child_rows,
        labeled_rows=labeled_rows,
        known_unlabeled_rows=known_unlabeled_rows,
    )
    annotated_children = (
        annotate_training_free_rows(
            dedup.new_children,
            trainability_soft_quantile=float(trainability_soft_quantile),
            expressibility_soft_quantile=float(expressibility_soft_quantile),
            trainability_hard_floor=float(trainability_hard_floor),
            expressibility_hard_floor=float(expressibility_hard_floor),
            entanglement_soft_floor=float(entanglement_soft_floor),
            max_params=max_params,
            max_two_q=max_two_q,
        )
        if enable_training_free_pruning
        else [dict(row) for row in dedup.new_children]
    )
    active_children = [
        dict(row)
        for row in annotated_children
        if str(row.get("zero_cost_status", "")).strip() != ZeroCostStatus.HARD_REJECT.value
    ]
    hard_reject_count = len(annotated_children) - len(active_children)
    soft_flag_count = sum(
        1
        for row in active_children
        if str(row.get("zero_cost_status", "")).strip() == ZeroCostStatus.SOFT_FLAG.value
    )

    oracle_rows: list[dict[str, Any]] = []
    abstain_rows: list[dict[str, Any]] = []
    previous_hit_gate_blocks = 0
    previous_hit_gate_active = (
        previous_oracle_hit_rate is not None
        and float(previous_oracle_hit_rate) < float(min_previous_oracle_hit_rate)
    )
    for child in active_children:
        prediction = predict_fair_energy(
            child,
            labeled_rows=labeled_rows,
            k_min=int(k_min),
            d_max=float(d_max),
            max_neighbor_std=oracle_max_neighbor_std,
        )
        enriched = dict(child)
        trusted = bool(prediction.trusted)
        reason = prediction.reason
        if trusted and prediction.reason != "exact_match" and previous_hit_gate_active:
            trusted = False
            reason = "previous_oracle_hit_rate_low"
            previous_hit_gate_blocks += 1
        enriched.update(
            {
                "predicted_fair_energy": "" if prediction.prediction is None else f"{float(prediction.prediction):.12f}",
                "oracle_reason": reason,
                "oracle_neighbor_count": int(prediction.neighbor_count),
                "oracle_nearest_distance": f"{float(prediction.nearest_distance):.12f}",
                "oracle_kth_distance": f"{float(prediction.kth_distance):.12f}",
                "oracle_neighbor_target_std": f"{float(prediction.neighbor_target_std):.12f}",
            }
        )
        if trusted:
            oracle_rows.append(enriched)
        else:
            abstain_rows.append(enriched)

    selector_fields = resolve_p1_selector_fields(selector, cheap_eval_selector=cheap_eval_selector)
    fallback_field = selector_fields[0]
    score_cache: dict[tuple[str, str], dict[str, Any]] = {}
    baseline_candidate_pool = _candidate_pool(active_children, control_rows)
    quota = choose_quota(
        k,
        oracle_trusted_count=len(oracle_rows),
        previous_oracle_trusted_fair_mean=previous_oracle_trusted_fair_mean,
        previous_fallback_fair_mean=previous_fallback_fair_mean,
    )
    fallback_source_rows = [dict(row) for row in abstain_rows]
    fallback_abstain_count = len(fallback_source_rows)
    fallback_audit_source_rows: list[dict[str, Any]] = []
    fallback_audit_target = 0
    fallback_target = k if normalized_selection_policy in {"no_regret", "no_regret_lite"} else int(quota.q1)
    fallback_shortfall = max(0, int(fallback_target) - fallback_abstain_count)
    if normalized_selection_policy == "no_regret" and oracle_rows:
        fallback_audit_target = len(oracle_rows)
        for row in _rank_by_score(oracle_rows, "predicted_fair_energy")[:fallback_audit_target]:
            audited = dict(row)
            audited["oracle_audit_for_fallback"] = "true"
            fallback_audit_source_rows.append(audited)
        fallback_source_rows.extend(fallback_audit_source_rows)
    elif normalized_selection_policy == "no_regret_lite" and fallback_shortfall > 0 and oracle_rows:
        fallback_audit_target = min(len(oracle_rows), fallback_shortfall)
        for row in _rank_by_score(oracle_rows, "predicted_fair_energy")[:fallback_audit_target]:
            audited = dict(row)
            audited["oracle_audit_for_fallback"] = "true"
            fallback_audit_source_rows.append(audited)
        fallback_source_rows.extend(fallback_audit_source_rows)
    elif fallback_shortfall > 0 and oracle_rows:
        fallback_audit_target = min(
            len(oracle_rows),
            max(int(quota.q0) + fallback_shortfall, int(fallback_audit_multiplier) * k),
        )
        for row in _rank_by_score(oracle_rows, "predicted_fair_energy")[:fallback_audit_target]:
            audited = dict(row)
            audited["oracle_audit_for_fallback"] = "true"
            fallback_audit_source_rows.append(audited)
        fallback_source_rows.extend(fallback_audit_source_rows)
    fallback_rows = _score_rows(fallback_source_rows, fallback_field, registry, score_cache)
    soft_prefilter_multiplier = (
        int(training_free_soft_prefilter_multiplier)
        if enable_training_free_pruning and int(training_free_soft_prefilter_multiplier) > 0
        else None
    )
    if normalized_selection_policy in {"no_regret", "no_regret_lite"}:
        selected: list[dict[str, Any]] = []
        fallback_ranked = _rank_fallback_rows(
            fallback_rows,
            fair_top_k=k,
            soft_prefilter_multiplier=soft_prefilter_multiplier,
        )
        _take_fill(selected, fallback_ranked, fair_top_k=k, selection_source="fallback_selector")
    else:
        selected = merge_quota_candidates(
            oracle_rows,
            fallback_rows,
            control_rows,
            quota=quota,
            fallback_score_field="fallback_score",
            fallback_soft_prefilter_multiplier=soft_prefilter_multiplier,
        )
    if len(selected) < k:
        fallback_fill_rows = _rank_fallback_rows(
            fallback_rows,
            fair_top_k=k,
            soft_prefilter_multiplier=soft_prefilter_multiplier,
        )
        _take_fill(selected, fallback_fill_rows, fair_top_k=k, selection_source="fallback_selector")
    if len(selected) < k:
        _take_fill(selected, _rank_by_score(oracle_rows, "predicted_fair_energy"), fair_top_k=k, selection_source="oracle_trusted")
    if len(selected) < k:
        _take_fill(selected, control_rows, fair_top_k=k, selection_source="control")
    fallback_fair_count = min(len(selected), k)
    if normalized_selection_policy == "no_regret_lite" and oracle_extra_k > 0:
        _take_fill(
            selected,
            _rank_by_score(oracle_rows, "predicted_fair_energy"),
            fair_top_k=k + oracle_extra_k,
            selection_source="oracle_trusted_extra",
        )
    oracle_extra_fair_calls = sum(1 for row in selected if row.get("p1_selection_source") == "oracle_trusted_extra")
    fair_queue_k = len(selected)

    queue_rows = [
        _queue_row(
            row,
            batch_id=batch_id,
            protocol_version=protocol_version,
            task_context=task,
            selection_source=str(row.get("p1_selection_source", "")),
        )
        for row in selected
    ]

    baseline_queues: dict[str, list[dict[str, Any]]] = {
        "random": _baseline_random_queue(
            baseline_candidate_pool,
            fair_top_k=fair_queue_k,
            seed=int(seed) + 1009,
            batch_id=f"{batch_id}_random",
            protocol_version=protocol_version,
            task_context=task,
        )
    }
    for field in baseline_selector_fields:
        baseline_queues[str(field)] = _baseline_selector_queue(
            baseline_candidate_pool,
            field=str(field),
            evaluator_registry=registry,
            score_cache=score_cache,
            fair_top_k=fair_queue_k,
            batch_id=f"{batch_id}_{str(field).lower()}_only",
            protocol_version=protocol_version,
            task_context=task,
            soft_prefilter_multiplier=soft_prefilter_multiplier,
        )

    budget = FairBudgetTracker(rounds=1, fair_top_k_per_round=fair_queue_k)
    budget.record_round(batch_id, len(queue_rows))
    budget_summary = budget.to_jsonable()
    budget_summary.update({
        "fallback_top_k_per_round": int(k),
        "fallback_fair_calls": int(fallback_fair_count),
        "oracle_extra_top_k": int(oracle_extra_k),
        "oracle_extra_fair_calls": int(oracle_extra_fair_calls),
    })
    cheap_eval_baseline_pool_size = len(baseline_candidate_pool)
    cheap_eval_p1_fallback_calls = len(fallback_rows)
    zero_cost_status_counts = {
        ZeroCostStatus.HARD_REJECT.value: hard_reject_count,
        ZeroCostStatus.SOFT_FLAG.value: soft_flag_count,
        ZeroCostStatus.PASS.value: len(active_children) - soft_flag_count,
    }
    summary = {
        "batch_id": batch_id,
        "parent_count": len(parents),
        "children_generated": len(child_rows),
        "new_children": len(dedup.new_children),
        "reused_labeled": len(dedup.reused_labeled),
        "skipped_duplicates": len(dedup.skipped_duplicate_architecture_ids),
        "training_free": {
            "enabled": bool(enable_training_free_pruning),
            "input_count": len(annotated_children),
            "hard_reject_count": hard_reject_count,
            "soft_flag_count": soft_flag_count,
            "active_count": len(active_children),
            "zero_cost_status_counts": dict(zero_cost_status_counts),
            "soft_prefilter_multiplier": int(training_free_soft_prefilter_multiplier),
        },
        "n_oracle_trusted": len(oracle_rows),
        "n_oracle_abstain": len(abstain_rows),
        "selector_fields": list(selector_fields),
        "selection_policy": normalized_selection_policy,
        "quota": quota.to_jsonable() if isinstance(quota, P1Quota) else dict(quota),
        "planned_total": len(queue_rows),
        "source_counts": _source_counts(queue_rows),
        "baseline_candidate_pool_size": cheap_eval_baseline_pool_size,
        "ansatz_families": summarize_ansatz_families(active_children),
        "budget": budget_summary,
        "cheap_eval": {
            "p1_fallback_field": fallback_field,
            "p1_fallback_calls": cheap_eval_p1_fallback_calls,
            "p1_fallback_abstain_calls": fallback_abstain_count,
            "p1_fallback_audit_calls": len(fallback_audit_source_rows),
            "p1_fallback_audit_target": fallback_audit_target,
            "p1_fallback_downgrade_calls": sum(
                1 for row in fallback_rows if not _is_empty(row.get("fallback_downgrade_reason"))
            ),
            "baseline_selector_calls_per_field": {
                str(field): cheap_eval_baseline_pool_size
                for field in baseline_selector_fields
            },
            "cheap_eval_calls_saved": max(0, cheap_eval_baseline_pool_size - cheap_eval_p1_fallback_calls),
        },
        "oracle_reliability": {
            "max_neighbor_std": oracle_max_neighbor_std,
            "previous_oracle_hit_rate": previous_oracle_hit_rate,
            "min_previous_oracle_hit_rate": float(min_previous_oracle_hit_rate),
            "previous_hit_gate_active": bool(previous_hit_gate_active),
            "previous_hit_gate_blocks": previous_hit_gate_blocks,
        },
        "baseline_queue_sizes": {name: len(rows) for name, rows in baseline_queues.items()},
        "note": "P1 round planner writes equal-size fair-label queues; fair_labeling.py supplies the fair COBYLA labels.",
    }
    return P1RoundPlan(
        queue_rows=queue_rows,
        child_rows=[dict(row) for row in annotated_children],
        oracle_rows=oracle_rows,
        fallback_rows=fallback_rows,
        baseline_queues=baseline_queues,
        summary=summary,
    )


def write_p1_round_outputs(plan: P1RoundPlan, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "queue": output / "p1_queue.csv",
        "children": output / "p1_children.csv",
        "oracle": output / "p1_oracle_predictions.csv",
        "fallback": output / "p1_fallback_rows.csv",
        "summary": output / "p1_round_summary.json",
    }
    write_csv_rows(paths["queue"], plan.queue_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    write_csv_rows(paths["children"], plan.child_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    write_csv_rows(paths["oracle"], plan.oracle_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    write_csv_rows(paths["fallback"], plan.fallback_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    for name, rows in plan.baseline_queues.items():
        key = f"baseline_{name}"
        paths[key] = output / f"queue_{str(name).lower()}_baseline.csv"
        write_csv_rows(paths[key], rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    paths["summary"].write_text(json.dumps(plan.summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


__all__ = [
    "Evaluator",
    "P1RoundPlan",
    "plan_p1_round",
    "write_p1_round_outputs",
]








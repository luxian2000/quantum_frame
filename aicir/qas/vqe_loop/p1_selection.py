"""Selection helpers for P1 mutation/oracle/fallback planning.

``p1_round.py`` remains the orchestration entry point.  This module owns row
scoring, fallback ranking, baseline queues, and queue-row conversion.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Mapping, Sequence

from aicir.qas.vqe_loop.benchmark_table import (
    architecture_key,
    rank_rows,
    rank_with_zero_cost_soft_prefilter,
    take_unique_with_source,
)
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, LabelSource, LabelStatus
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float, is_empty as _is_empty

Evaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]

def _row_gene_kind(row: Mapping[str, Any]) -> str:
    family = str(row.get("family", "") or "").strip().lower()
    if family:
        return family
    raw = row.get("ansatz_gene")
    if raw is None or str(raw).strip() == "":
        return ""
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if isinstance(payload, Mapping):
        return str(payload.get("kind", "") or "").strip().lower()
    return ""


def _should_downgrade_e5_to_e2(row: Mapping[str, Any], field: str, evaluator_registry: Mapping[str, Evaluator]) -> bool:
    if str(field).upper() != "E5" or "E2" not in evaluator_registry:
        return False
    kind = _row_gene_kind(row)
    return bool(kind and kind != "supernet_native")


def task_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("n_qubits", "hamiltonian_id", "hamiltonian_class", "hamiltonian_terms", "reference_energy")
    for row in rows:
        if row.get("hamiltonian_id") or row.get("hamiltonian_terms"):
            return {field: row.get(field, "") for field in fields}
    return {}


def _source_for_selection(selection_source: str) -> str:
    if selection_source in {"oracle_trusted", "oracle_trusted_extra"}:
        return LabelSource.P1_ORACLE.value
    if selection_source == "fallback_selector":
        return LabelSource.P1_FALLBACK.value
    if selection_source == "control":
        return LabelSource.P1_CONTROL.value
    if selection_source == "baseline_random":
        return LabelSource.BASELINE_RANDOM.value
    if selection_source == "baseline_e2":
        return LabelSource.BASELINE_E2.value
    if selection_source == "baseline_e5":
        return LabelSource.BASELINE_E5.value
    return LabelSource.BASELINE.value


def queue_row(
    row: Mapping[str, Any],
    *,
    batch_id: str,
    protocol_version: str,
    task_context: Mapping[str, Any],
    selection_source: str,
) -> dict[str, Any]:
    queued = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    queued.update({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})
    queued.update({field: value for field, value in task_context.items() if not _is_empty(value)})
    queued.update(
        {
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": _source_for_selection(selection_source),
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "failure_reason": "",
            "last_error_digest": "",
            "p1_selection_source": selection_source,
            "hamiltonian_id": queued.get("hamiltonian_id")
            or f"{queued.get('hamiltonian_class', 'tfim')}_{queued.get('n_qubits', '')}q",
        }
    )
    queued["hamiltonian_coverage_features"] = queued.get("hamiltonian_coverage_features") or queued.get("hamiltonian_coverage", "")
    return queued



def _score_row(
    row: Mapping[str, Any],
    field: str,
    evaluator_registry: Mapping[str, Evaluator],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    copied = dict(row)
    requested_field = str(field)
    active_field = "E2" if _should_downgrade_e5_to_e2(copied, requested_field, evaluator_registry) else requested_field
    if active_field != requested_field:
        copied["fallback_downgrade_reason"] = f"{requested_field}_not_applicable_for_{_row_gene_kind(copied)}"
    score = _as_float(copied.get(active_field))
    key = (architecture_key(copied), str(active_field))
    if score is None:
        if active_field not in evaluator_registry:
            raise KeyError(f"missing evaluator for fallback field: {active_field}")
        if key not in cache:
            result = dict(evaluator_registry[active_field](copied))
            cache[key] = result
        copied.update(cache[key])
        score = _as_float(copied.get(active_field))
    if score is None:
        raise ValueError(f"fallback evaluator did not produce numeric {active_field}")
    copied["fallback_selector"] = str(active_field)
    copied["fallback_score"] = f"{float(score):.12f}"
    return copied


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    evaluator_registry: Mapping[str, Evaluator],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    requested_field = str(field).upper()
    if requested_field != "ENSEMBLE":
        return [_score_row(row, field, evaluator_registry, cache) for row in rows]

    base_fields = ("E2", "VQE_TASK_PROXY", "GNN_PROXY")
    enriched = [dict(row) for row in rows]
    rank_sum = [0.0 for _row in enriched]
    rank_count = [0 for _row in enriched]

    for base_field in base_fields:
        scored: list[tuple[float, str, int]] = []
        for index, row in enumerate(enriched):
            if _as_float(row.get(base_field)) is None and base_field not in evaluator_registry:
                continue
            try:
                scored_row = _score_row(row, base_field, evaluator_registry, cache)
            except (KeyError, ValueError):
                continue
            enriched[index].update(scored_row)
            score = _as_float(enriched[index].get(base_field))
            if score is None:
                continue
            scored.append((score, str(enriched[index].get("architecture_id") or f"row:{index}"), index))

        for rank, (_score, _identifier, index) in enumerate(sorted(scored, key=lambda item: (item[0], item[1]))):
            rank_sum[index] += float(rank)
            rank_count[index] += 1

    output: list[dict[str, Any]] = []
    for index, row in enumerate(enriched):
        if rank_count[index] <= 0:
            continue
        copied = dict(row)
        ensemble_score = rank_sum[index] / float(rank_count[index])
        copied["ENSEMBLE"] = f"{ensemble_score:.12f}"
        copied["fallback_selector"] = "ENSEMBLE"
        copied["fallback_score"] = f"{ensemble_score:.12f}"
        output.append(copied)
    return output


def rank_by_score(rows: Sequence[Mapping[str, Any]], score_field: str) -> list[dict[str, Any]]:
    return rank_rows(rows, score_field, include_missing=False)


def rank_fallback_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fair_top_k: int,
    soft_prefilter_multiplier: int | None,
) -> list[dict[str, Any]]:
    return (
        rank_with_zero_cost_soft_prefilter(
            rows,
            "fallback_score",
            target_count=int(fair_top_k),
            window_multiplier=int(soft_prefilter_multiplier),
        )
        if soft_prefilter_multiplier is not None
        else rank_by_score(rows, "fallback_score")
    )


def take_fill(
    selected: list[dict[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    fair_top_k: int,
    selection_source: str,
) -> None:
    selected_keys = {architecture_key(row) for row in selected}
    selected.extend(
        take_unique_with_source(
            candidates,
            max(0, int(fair_top_k) - len(selected)),
            selection_source,
            selected_keys,
        )
    )


def baseline_random_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    fair_top_k: int,
    seed: int,
    batch_id: str,
    protocol_version: str,
    task_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    shuffled = [dict(row) for row in rows]
    random.Random(int(seed)).shuffle(shuffled)
    return [
        queue_row(
            row,
            batch_id=batch_id,
            protocol_version=protocol_version,
            task_context=task_context,
            selection_source="baseline_random",
        )
        for row in shuffled[: max(0, int(fair_top_k))]
    ]


def baseline_selector_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    evaluator_registry: Mapping[str, Evaluator],
    score_cache: dict[tuple[str, str], dict[str, Any]],
    fair_top_k: int,
    batch_id: str,
    protocol_version: str,
    task_context: Mapping[str, Any],
    soft_prefilter_multiplier: int | None = None,
) -> list[dict[str, Any]]:
    scored = score_rows(rows, field, evaluator_registry, score_cache)
    ranked = (
        rank_with_zero_cost_soft_prefilter(
            scored,
            "fallback_score",
            target_count=int(fair_top_k),
            window_multiplier=int(soft_prefilter_multiplier),
        )
        if soft_prefilter_multiplier is not None
        else rank_by_score(scored, "fallback_score")
    )
    selection_source = f"baseline_{field.lower()}"
    return [
        queue_row(
            row,
            batch_id=batch_id,
            protocol_version=protocol_version,
            task_context=task_context,
            selection_source=selection_source,
        )
        for row in ranked[: max(0, int(fair_top_k))]
    ]



__all__ = [
    "baseline_random_queue",
    "baseline_selector_queue",
    "queue_row",
    "rank_by_score",
    "rank_fallback_rows",
    "score_rows",
    "take_fill",
    "task_context",
]


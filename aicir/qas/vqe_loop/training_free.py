"""Training-free annotations for vqe_loop row dictionaries.

The P1 loop works with benchmark-table rows, not ``CandidateRecord`` objects.
This module provides the small row-level adapter needed by mutation children
while keeping the zero-cost score fields aligned with the existing CSV schema.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from aicir.metrics.circuit_structure import (
    entanglement_coverage_score,
    parameter_count,
    structural_expressibility_proxy_score,
)
from aicir.metrics.hardware import native_depth_twoq_efficiency
from aicir.metrics.trainability import structure_proxy
from aicir.qas.core.reward import RewardWeights
from aicir.qas.library.ansatz import SupernetAnsatzGene, architecture_from_supernet_gene
from aicir.qas.vqe_loop.benchmark_table import decoded_ansatz_gene_payload
from aicir.qas.vqe_loop.benchmark_table import ZeroCostStatus


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(max(0, min(len(ordered) - 1, round(float(fraction) * (len(ordered) - 1)))))
    return float(ordered[index])


def _gene_from_row(row: Mapping[str, Any]) -> SupernetAnsatzGene | None:
    try:
        payload = decoded_ansatz_gene_payload(row)
    except (TypeError, ValueError, KeyError):
        return None
    if payload is None or str(payload.get("kind", "")).lower() != "supernet_native":
        return None
    try:
        return SupernetAnsatzGene.from_jsonable(payload)
    except (TypeError, ValueError, KeyError):
        return None


def _two_q_count(circuit: Any) -> int:
    return int(
        len(
            [
                gate
                for gate in circuit.gates
                if gate.get("control_qubits") or gate.get("type") in {"rxx", "rzz"}
            ]
        )
    )


def training_free_scores_for_supernet_gene(gene: SupernetAnsatzGene) -> dict[str, float]:
    """Return structural training-free scores for a native supernet gene."""

    architecture = architecture_from_supernet_gene(gene)
    n_params = int(parameter_count(architecture.circuit))
    two_q_count = _two_q_count(architecture.circuit)
    entanglement = entanglement_coverage_score(
        two_q_count=float(two_q_count),
        n_qubits=int(gene.n_qubits),
        layers=int(gene.layers),
        topology="supernet_pairs",
    )
    expressibility = structural_expressibility_proxy_score(
        n_params=float(n_params),
        n_qubits=int(gene.n_qubits),
        layers=int(gene.layers),
        rotation_block="mixed_supernet",
        final_rotation="mixed_supernet",
        entanglement_score=float(entanglement),
    )
    trainability = structure_proxy(architecture.circuit)
    hardware = native_depth_twoq_efficiency(architecture.circuit)
    weights = RewardWeights()
    weighted = (
        weights.expressibility * float(expressibility)
        + weights.trainability * float(trainability)
        + weights.noise_robustness * float(entanglement)
        + weights.hardware_efficiency * float(hardware)
    )
    return {
        "n_params": float(n_params),
        "two_q_count": float(two_q_count),
        "expressibility_score": float(expressibility),
        "trainability_score": float(trainability),
        "entanglement_score": float(entanglement),
        "zero_cost_feature_score": float(weighted),
    }


def annotate_training_free_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    trainability_soft_quantile: float = 0.10,
    expressibility_soft_quantile: float = 0.05,
    trainability_hard_floor: float = 0.01,
    expressibility_hard_floor: float = 0.01,
    entanglement_soft_floor: float = 0.0,
    max_params: int | None = None,
    max_two_q: int | None = None,
) -> list[dict[str, Any]]:
    """Annotate rows with zero-cost scores and pass/soft/hard status."""

    scored: list[tuple[dict[str, Any], dict[str, float] | None]] = []
    for row in rows:
        copied = dict(row)
        gene = _gene_from_row(copied)
        scores = training_free_scores_for_supernet_gene(gene) if gene is not None else None
        scored.append((copied, scores))

    score_values = [scores for _row, scores in scored if scores is not None]
    expr_values = [float(scores["expressibility_score"]) for scores in score_values]
    train_values = [float(scores["trainability_score"]) for scores in score_values]
    expr_soft = _quantile(expr_values, expressibility_soft_quantile)
    train_soft = _quantile(train_values, trainability_soft_quantile)
    expr_has_signal = (max(expr_values) - min(expr_values)) > 1e-12 if expr_values else False
    train_has_signal = (max(train_values) - min(train_values)) > 1e-12 if train_values else False

    group_train: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, scores in scored:
        if scores is None:
            continue
        group_train[(str(row.get("family", "")), str(row.get("depth_group", "")))].append(
            float(scores["trainability_score"])
        )

    annotated: list[dict[str, Any]] = []
    for row, scores in scored:
        if scores is None:
            row.setdefault("zero_cost_status", ZeroCostStatus.PASS.value)
            row.setdefault("zero_cost_score_is_ranking_signal", "false")
            annotated.append(row)
            continue

        expressibility = float(scores["expressibility_score"])
        trainability = float(scores["trainability_score"])
        entanglement = float(scores["entanglement_score"])
        n_params = int(scores["n_params"])
        two_q_count = int(scores["two_q_count"])
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        if max_params is not None and n_params > int(max_params):
            hard_reasons.append("params_over_cap")
        if max_two_q is not None and two_q_count > int(max_two_q):
            hard_reasons.append("two_q_over_cap")
        if expressibility < float(expressibility_hard_floor):
            hard_reasons.append("expressibility_extreme_low")
        group_key = (str(row.get("family", "")), str(row.get("depth_group", "")))
        group_median_train = _quantile(group_train.get(group_key, []), 0.5)
        if trainability < float(trainability_hard_floor) and group_median_train < train_soft:
            hard_reasons.append("family_depth_trainability_collapse")
        if train_has_signal and trainability < train_soft:
            soft_reasons.append("trainability_low")
        if expr_has_signal and expressibility < expr_soft:
            soft_reasons.append("expressibility_low")
        if entanglement < float(entanglement_soft_floor):
            soft_reasons.append("entanglement_low")

        status = ZeroCostStatus.PASS
        reasons = soft_reasons
        if hard_reasons:
            status = ZeroCostStatus.HARD_REJECT
            reasons = hard_reasons + soft_reasons
        elif soft_reasons:
            status = ZeroCostStatus.SOFT_FLAG

        row.update(
            {
                "n_params": n_params,
                "two_q_count": two_q_count,
                "zero_cost_status": status.value,
                "zero_cost_reasons": ";".join(reasons),
                "expressibility_score": f"{expressibility:.12f}",
                "trainability_score": f"{trainability:.12f}",
                "entanglement_score": f"{entanglement:.12f}",
                "zero_cost_feature_score": f"{float(scores['zero_cost_feature_score']):.12f}",
                "zero_cost_score_is_ranking_signal": "false",
            }
        )
        annotated.append(row)
    return annotated



def _zero_cost_summary(rows: Sequence[Mapping[str, Any]], *, enabled: bool) -> dict[str, Any]:
    counts = {
        ZeroCostStatus.HARD_REJECT.value: 0,
        ZeroCostStatus.SOFT_FLAG.value: 0,
        ZeroCostStatus.PASS.value: 0,
    }
    for row in rows:
        status = str(row.get("zero_cost_status", ZeroCostStatus.PASS.value) or ZeroCostStatus.PASS.value).strip()
        if status in counts:
            counts[status] += 1
    return {
        "enabled": bool(enabled),
        "hard_reject_count": counts[ZeroCostStatus.HARD_REJECT.value],
        "soft_flag_count": counts[ZeroCostStatus.SOFT_FLAG.value],
        "pass_count": counts[ZeroCostStatus.PASS.value],
        "zero_cost_status_counts": dict(counts),
    }


def annotate_p0_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the shared zero-cost annotation boundary to P0 bootstrap rows."""

    if not enabled:
        copied = [dict(row) for row in rows]
        for row in copied:
            row.setdefault("zero_cost_status", ZeroCostStatus.PASS.value)
            row.setdefault("zero_cost_score_is_ranking_signal", "false")
        return copied, _zero_cost_summary(copied, enabled=False)

    annotated = annotate_training_free_rows(rows)
    active = [
        dict(row)
        for row in annotated
        if str(row.get("zero_cost_status", "")).strip() != ZeroCostStatus.HARD_REJECT.value
    ]
    return active, _zero_cost_summary(annotated, enabled=True)

__all__ = ["annotate_p0_bootstrap_rows", "annotate_training_free_rows", "training_free_scores_for_supernet_gene"]





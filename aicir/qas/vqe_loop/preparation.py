"""Prepare Stage-0/Stage-1.5 files for trust-region VQE-QAS.

This script does not run VQE.  It builds candidates, applies hard filters plus
zero-cost metric soft flags, writes the initial fair-label queue, and leaves
supernet screening to the optional sidecar scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.core.evaluator import evaluate_architectures
from aicir.metrics.circuit_structure import (
    entanglement_coverage_score,
    parameter_count,
    structural_expressibility_proxy_score,
)
from aicir.metrics.trainability import structure_proxy
from aicir.metrics.hardware import native_depth_twoq_efficiency
from aicir.ir import instruction_controls, instruction_name
from aicir.qas.core.reward import RewardWeights
from aicir.qas.library.ansatz import (
    HEAMask,
    LayerwiseAnsatzGene,
    architecture_from_hea_mask,
    architecture_from_layerwise_gene,
    enumerate_hea_masks,
    sample_layerwise_genes,
)
from aicir.qas.core.backend_utils import (
    resolve_qas_backend,
)
from aicir.qas.vqe_loop.protocol import (
    BENCHMARK_TABLE_FIELDS,
    DEFAULT_BATCH_QUOTAS,
    DEFAULT_RETRY_POLICY,
    DEFAULT_TRUST_REGION_RULES,
    LabelSource,
    LabelStatus,
    ZeroCostStatus,
)
from aicir.qas.vqe_loop.geometry import (
    CandidateRecord,
    fit_distance_scales,
    select_initial_label_batch,
    select_stage0_anchors,
)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _mask_family(mask: HEAMask) -> str:
    return f"{mask.rotation_block}_{mask.entangler}_{mask.entangle_pattern}_L{mask.layers}"


def _depth_group(mask: HEAMask) -> str:
    return f"L{mask.layers}"


def _hamiltonian_coverage(mask: HEAMask, hamiltonian_class: str) -> float:
    if hamiltonian_class.lower() == "tfim":
        score = 0.0
        if mask.entangler in {"rzz", "cz", "cx"}:
            score += {"rzz": 1.0, "cz": 0.8, "cx": 0.6}.get(mask.entangler, 0.0)
        if mask.rotation_block in {"rx_ry_rz", "ry_rz"}:
            score += {"rx_ry_rz": 1.0, "ry_rz": 0.7}.get(mask.rotation_block, 0.0)
        if mask.final_rotation in {"ry_rz", "ry"}:
            score += {"ry_rz": 0.5, "ry": 0.3}.get(mask.final_rotation, 0.0)
        return score
    return 0.0


def candidate_record_from_mask(mask: HEAMask, hamiltonian_class: str) -> CandidateRecord:
    architecture = architecture_from_hea_mask(mask)
    architecture_id = f"{mask.n_qubits}q_{architecture.name}"
    return CandidateRecord(
        architecture_id=architecture_id,
        canonical_arch_hash="|".join(str(item) for item in mask.key()),
        family=_mask_family(mask),
        entangler_type=mask.entangler,
        topology=mask.entangle_pattern,
        depth_group=_depth_group(mask),
        n_params=float(parameter_count(architecture.circuit)),
        two_q_count=float(
            len(
                [
                    instruction
                    for instruction in architecture.circuit.gates
                    if instruction_controls(instruction) or instruction_name(instruction) == "rzz"
                ]
            )
        ),
        hamiltonian_class=hamiltonian_class,
        hamiltonian_coverage=_hamiltonian_coverage(mask, hamiltonian_class),
        metadata={
            "n_qubits": mask.n_qubits,
            "layers": mask.layers,
            "rotation_block": mask.rotation_block,
            "final_rotation": mask.final_rotation,
            "hea_mask": list(mask.key()),
            "zero_cost_status": ZeroCostStatus.PASS.value,
            "zero_cost_reasons": "",
            "expressibility_score": "",
            "trainability_score": "",
            "entanglement_score": "",
            "zero_cost_feature_score": "",
            "zero_cost_score_is_ranking_signal": "false",
        },
    )


def candidate_record_from_layerwise_gene(gene: LayerwiseAnsatzGene, hamiltonian_class: str) -> CandidateRecord:
    architecture = architecture_from_layerwise_gene(gene)
    gene_payload = gene.to_jsonable()
    canonical = json.dumps(gene_payload, ensure_ascii=False, sort_keys=True)
    two_q_count = float(
        len(
            [
                instruction
                for instruction in architecture.circuit.gates
                if instruction_controls(instruction) or instruction_name(instruction) in {"rxx", "rzz"}
            ]
        )
    )
    return CandidateRecord(
        architecture_id=f"{gene.n_qubits}q_{architecture.name}",
        canonical_arch_hash=canonical,
        family="layerwise_gene",
        entangler_type="mixed_edge",
        topology=gene.entangle_pattern,
        depth_group=f"L{gene.layers}",
        n_params=float(parameter_count(architecture.circuit)),
        two_q_count=two_q_count,
        hamiltonian_class=hamiltonian_class,
        hamiltonian_coverage=1.0 if hamiltonian_class.lower() in {"h2", "molecular_h2"} else 0.5,
        metadata={
            "n_qubits": gene.n_qubits,
            "layers": gene.layers,
            "rotation_block": "mixed_layer",
            "final_rotation": gene.single_blocks[-1],
            "ansatz_gene": gene_payload,
            "zero_cost_status": ZeroCostStatus.PASS.value,
            "zero_cost_reasons": "",
            "expressibility_score": "",
            "trainability_score": "",
            "entanglement_score": "",
            "zero_cost_feature_score": "",
            "zero_cost_score_is_ranking_signal": "false",
        },
    )


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(max(0, min(len(ordered) - 1, round(float(fraction) * (len(ordered) - 1)))))
    return ordered[index]


def _score_distribution(values: Iterable[Any]) -> dict[str, float | int | None]:
    parsed = sorted(float(value) for value in values if value not in ("", None))
    if not parsed:
        return {"count": 0, "min": None, "p05": None, "median": None, "max": None}
    n_values = len(parsed)
    return {
        "count": n_values,
        "min": parsed[0],
        "p05": parsed[int(0.05 * (n_values - 1))],
        "median": parsed[int(0.5 * (n_values - 1))],
        "max": parsed[-1],
    }


def _entanglement_score(candidate: CandidateRecord) -> float:
    return entanglement_coverage_score(
        two_q_count=candidate.two_q_count,
        n_qubits=int(candidate.metadata.get("n_qubits", 2)),
        layers=int(candidate.metadata.get("layers", 1)),
        topology=candidate.topology,
    )


def _structural_expressibility_proxy_score(candidate: CandidateRecord) -> float:
    return structural_expressibility_proxy_score(
        n_params=candidate.n_params,
        n_qubits=int(candidate.metadata.get("n_qubits", 1)),
        layers=int(candidate.metadata.get("layers", 1)),
        rotation_block=str(candidate.metadata.get("rotation_block", "")),
        final_rotation=str(candidate.metadata.get("final_rotation", "")),
        entanglement_score=_entanglement_score(candidate),
    )


def _apply_zero_cost_stage1b(
    candidates: list[CandidateRecord],
    *,
    n_samples: int,
    trainability_soft_quantile: float,
    expressibility_soft_quantile: float,
    trainability_hard_floor: float,
    expressibility_hard_floor: float,
    entanglement_soft_floor: float,
    max_params: int | None,
    max_two_q: int | None,
    expressibility_metric: str,
    trainability_metric: str,
) -> list[CandidateRecord]:
    if n_samples <= 0:
        return candidates
    architectures = []
    for candidate in candidates:
        if candidate.metadata.get("ansatz_gene"):
            architectures.append(architecture_from_layerwise_gene(LayerwiseAnsatzGene.from_jsonable(candidate.metadata["ansatz_gene"])))
        else:
            architectures.append(architecture_from_hea_mask(HEAMask(*candidate.metadata["hea_mask"])))
    structural_mode = str(expressibility_metric).lower() in {"structural_proxy", "structure_proxy", "fast_structural"}
    score_rows: dict[str, dict[str, float]] = {}
    if structural_mode:
        weights = RewardWeights()
        for candidate, architecture in zip(candidates, architectures):
            expressibility = _structural_expressibility_proxy_score(candidate)
            trainability = structure_proxy(architecture.circuit)
            entanglement = _entanglement_score(candidate)
            hardware = native_depth_twoq_efficiency(architecture.circuit)
            weighted = (
                weights.expressibility * expressibility
                + weights.trainability * trainability
                + weights.noise_robustness * hardware
                + weights.hardware_efficiency * hardware
            )
            score_rows[candidate.canonical_arch_hash] = {
                "expressibility": expressibility,
                "trainability": trainability,
                "entanglement": entanglement,
                "weighted": float(weighted),
            }
    else:
        scores = evaluate_architectures(
            architectures,
            backend=resolve_qas_backend(kind="numpy", dtype="complex128"),
            n_samples=n_samples,
            active_metrics={
                "expressibility": expressibility_metric,
                "trainability": trainability_metric,
            },
        )
        for candidate, score in zip(candidates, scores):
            score_rows[candidate.canonical_arch_hash] = {
                "expressibility": float(score.expressibility.score),
                "trainability": float(score.trainability.score),
                "entanglement": _entanglement_score(candidate),
                "weighted": float(score.weighted_score),
            }
    expr_values = [row["expressibility"] for row in score_rows.values()]
    train_values = [row["trainability"] for row in score_rows.values()]
    expr_soft = _quantile(expr_values, expressibility_soft_quantile)
    train_soft = _quantile(train_values, trainability_soft_quantile)
    expr_has_signal = (max(expr_values) - min(expr_values)) > 1e-12 if expr_values else False
    train_has_signal = (max(train_values) - min(train_values)) > 1e-12 if train_values else False
    group_train: dict[tuple[str, str], list[float]] = {}
    for candidate in candidates:
        score = score_rows.get(candidate.canonical_arch_hash)
        if score is None:
            continue
        group_key = (candidate.family, candidate.depth_group)
        group_train.setdefault(group_key, []).append(float(score["trainability"]))

    annotated: list[CandidateRecord] = []
    for candidate in candidates:
        score = score_rows[candidate.canonical_arch_hash]
        expressibility = float(score["expressibility"])
        trainability = float(score["trainability"])
        entanglement = float(score["entanglement"])
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        if max_params is not None and candidate.n_params > int(max_params):
            hard_reasons.append("params_over_cap")
        if max_two_q is not None and candidate.two_q_count > int(max_two_q):
            hard_reasons.append("two_q_over_cap")
        if expressibility < float(expressibility_hard_floor):
            hard_reasons.append("expressibility_extreme_low")
        group_key = (candidate.family, candidate.depth_group)
        group_median_train = _quantile(group_train.get(group_key, []), 0.5)
        if trainability < float(trainability_hard_floor) and group_median_train < train_soft:
            hard_reasons.append("family_depth_trainability_collapse")
        if train_has_signal and trainability < train_soft:
            soft_reasons.append("trainability_low")
        if expr_has_signal and expressibility < expr_soft:
            soft_reasons.append("expressibility_low")
        if candidate.hamiltonian_class == "tfim" and entanglement < float(entanglement_soft_floor):
            soft_reasons.append("entanglement_low")

        status = ZeroCostStatus.PASS
        reasons = soft_reasons
        if hard_reasons:
            status = ZeroCostStatus.HARD_REJECT
            reasons = hard_reasons + soft_reasons
        elif soft_reasons:
            status = ZeroCostStatus.SOFT_FLAG
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "zero_cost_status": status.value,
                "zero_cost_reasons": ";".join(reasons),
                "expressibility_score": expressibility,
                "trainability_score": trainability,
                "entanglement_score": entanglement,
                "zero_cost_feature_score": float(score["weighted"]),
                "zero_cost_score_is_ranking_signal": "false",
            }
        )
        annotated.append(replace(candidate, metadata=metadata))
    return annotated


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_row(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "architecture_id": candidate.architecture_id,
        "canonical_arch_hash": candidate.canonical_arch_hash,
        "n_qubits": candidate.metadata.get("n_qubits", ""),
        "hamiltonian_class": candidate.hamiltonian_class,
        "family": candidate.family,
        "entangler_type": candidate.entangler_type,
        "topology": candidate.topology,
        "depth_group": candidate.depth_group,
        "n_params": int(candidate.n_params),
        "two_q_count": int(candidate.two_q_count),
        "hamiltonian_coverage": f"{candidate.hamiltonian_coverage:.6f}",
        "zero_cost_status": candidate.metadata.get("zero_cost_status", ""),
        "zero_cost_reasons": candidate.metadata.get("zero_cost_reasons", ""),
        "expressibility_score": candidate.metadata.get("expressibility_score", ""),
        "trainability_score": candidate.metadata.get("trainability_score", ""),
        "entanglement_score": candidate.metadata.get("entanglement_score", ""),
        "zero_cost_feature_score": candidate.metadata.get("zero_cost_feature_score", ""),
        "zero_cost_score_is_ranking_signal": candidate.metadata.get("zero_cost_score_is_ranking_signal", "false"),
        "hea_mask": json.dumps(candidate.metadata.get("hea_mask", []), ensure_ascii=False),
        "ansatz_gene": json.dumps(candidate.metadata.get("ansatz_gene", ""), ensure_ascii=False),
    }


def _label_queue_row(
    candidate: CandidateRecord,
    source: LabelSource,
    *,
    protocol_version: str,
    batch_id: str,
) -> dict[str, Any]:
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(_candidate_row(candidate))
    row.update(
        {
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": source.value,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "hamiltonian_id": f"{candidate.hamiltonian_class}_{candidate.metadata.get('n_qubits', '')}q",
            "hamiltonian_coverage_features": f"{candidate.hamiltonian_coverage:.6f}",
        }
    )
    return row


def _coverage_lines(candidates: list[CandidateRecord], anchors: list[CandidateRecord]) -> list[str]:
    lines = [
        "# VQE-QAS Stage-0 coverage report",
        "",
        f"n_candidates: {len(candidates)}",
        f"n_anchors: {len(anchors)}",
        "",
    ]
    for field in ["family", "entangler_type", "topology", "depth_group", "hamiltonian_class"]:
        counts = Counter(str(getattr(candidate, field)) for candidate in candidates)
        lines.append(f"## {field}")
        for key, count in sorted(counts.items()):
            lines.append(f"- {key}: {count}")
        lines.append("")
    zero_cost_counts = Counter(str(candidate.metadata.get("zero_cost_status", "")) for candidate in candidates)
    lines.append("## zero_cost_status")
    for key, count in sorted(zero_cost_counts.items()):
        lines.append(f"- {key}: {count}")
    zero_cost_reasons = Counter(
        reason
        for candidate in candidates
        for reason in str(candidate.metadata.get("zero_cost_reasons", "")).split(";")
        if reason
    )
    lines.append("")
    lines.append("## zero_cost_reasons")
    if zero_cost_reasons:
        for key, count in sorted(zero_cost_reasons.items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none: 0")
    lines.append("")
    for name in ["expressibility_score", "trainability_score", "entanglement_score", "zero_cost_feature_score"]:
        distribution = _score_distribution(candidate.metadata.get(name, "") for candidate in candidates)
        lines.extend(
            [
                f"## {name}",
                f"- count: {distribution['count']}",
                f"- min: {distribution['min']}",
                f"- p05: {distribution['p05']}",
                f"- median: {distribution['median']}",
                f"- max: {distribution['max']}",
                "",
            ]
        )
    numeric_fields = [
        ("n_params", [candidate.n_params for candidate in candidates]),
        ("two_q_count", [candidate.two_q_count for candidate in candidates]),
        ("hamiltonian_coverage", [candidate.hamiltonian_coverage for candidate in candidates]),
    ]
    for name, values in numeric_fields:
        ordered = sorted(values)
        lines.extend(
            [
                f"## {name}",
                f"- min: {ordered[0]:.6f}",
                f"- median: {ordered[len(ordered)//2]:.6f}",
                f"- max: {ordered[-1]:.6f}",
                "",
            ]
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare trust-region VQE-QAS oracle inputs")
    parser.add_argument("--scales", default="4,6,8")
    parser.add_argument("--hamiltonian-class", default="tfim")
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--oracle-protocol-version", default="vqe_qas_trust_region_oracle")
    parser.add_argument("--batch-id", default="initial")
    parser.add_argument("--initial-labels", type=int, default=96)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--k-min", type=int, default=int(DEFAULT_TRUST_REGION_RULES["k_min"]))
    parser.add_argument("--trust-d-max", type=float, default=None)
    parser.add_argument("--label-group-key", default="n_qubits")
    parser.add_argument("--zero-cost-samples", type=int, default=4)
    parser.add_argument("--zero-cost-expressibility-metric", default="structural_proxy")
    parser.add_argument("--zero-cost-trainability-metric", default="structure_proxy")
    parser.add_argument("--trainability-soft-quantile", type=float, default=0.10)
    parser.add_argument("--expressibility-soft-quantile", type=float, default=0.05)
    parser.add_argument("--trainability-hard-floor", type=float, default=0.01)
    parser.add_argument("--expressibility-hard-floor", type=float, default=0.01)
    parser.add_argument("--entanglement-soft-floor", type=float, default=0.25)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--max-two-q", type=int, default=None)
    parser.add_argument("--include-layerwise", action="store_true")
    parser.add_argument("--layerwise-count", type=int, default=0)
    parser.add_argument("--layerwise-layers", type=int, default=3)
    parser.add_argument("--layerwise-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/vqe_qas_oracle_prep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_candidates: list[CandidateRecord] = []
    for n_qubits in _parse_int_list(args.scales):
        for mask in enumerate_hea_masks(n_qubits):
            all_candidates.append(candidate_record_from_mask(mask, args.hamiltonian_class))
        if args.include_layerwise or int(args.layerwise_count) > 0:
            for gene in sample_layerwise_genes(
                n_qubits=n_qubits,
                layers=int(args.layerwise_layers),
                count=max(1, int(args.layerwise_count)),
                seed=int(args.layerwise_seed) + int(n_qubits),
            ):
                all_candidates.append(candidate_record_from_layerwise_gene(gene, args.hamiltonian_class))
    all_candidates = _apply_zero_cost_stage1b(
        all_candidates,
        n_samples=args.zero_cost_samples,
        trainability_soft_quantile=args.trainability_soft_quantile,
        expressibility_soft_quantile=args.expressibility_soft_quantile,
        trainability_hard_floor=args.trainability_hard_floor,
        expressibility_hard_floor=args.expressibility_hard_floor,
        entanglement_soft_floor=args.entanglement_soft_floor,
        max_params=args.max_params,
        max_two_q=args.max_two_q,
        expressibility_metric=args.zero_cost_expressibility_metric,
        trainability_metric=args.zero_cost_trainability_metric,
    )
    stage1_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.metadata.get("zero_cost_status") != ZeroCostStatus.HARD_REJECT.value
    ]

    anchors = select_stage0_anchors(stage1_candidates)
    scales = fit_distance_scales(stage1_candidates)
    labels = select_initial_label_batch(
        stage1_candidates,
        total_labels=args.initial_labels,
        holdout_fraction=args.holdout_fraction,
        group_key=args.label_group_key or None,
        trust_d_max=args.trust_d_max,
        k_min=args.k_min,
    )
    soft_label_ids = [
        candidate.architecture_id
        for candidate in stage1_candidates
        if candidate.architecture_id in labels
        and candidate.metadata.get("zero_cost_status") == ZeroCostStatus.SOFT_FLAG.value
        and labels[candidate.architecture_id] == LabelSource.INITIAL_TRAIN
    ]
    for architecture_id in soft_label_ids:
        labels[architecture_id] = LabelSource.HOLDOUT_BOUNDARY

    candidate_fields = [
        "architecture_id",
        "canonical_arch_hash",
        "n_qubits",
        "hamiltonian_class",
        "family",
        "entangler_type",
        "topology",
        "depth_group",
        "n_params",
        "two_q_count",
        "hamiltonian_coverage",
        "zero_cost_status",
        "zero_cost_reasons",
        "expressibility_score",
        "trainability_score",
        "entanglement_score",
        "zero_cost_feature_score",
        "zero_cost_score_is_ranking_signal",
        "hea_mask",
        "ansatz_gene",
    ]
    _write_csv(output_dir / "stage0_candidates.csv", (_candidate_row(candidate) for candidate in all_candidates), candidate_fields)
    _write_csv(output_dir / "stage0_anchors.csv", (_candidate_row(candidate) for candidate in anchors), candidate_fields)
    label_rows = [
        _label_queue_row(candidate, labels[candidate.architecture_id], protocol_version=args.protocol_version, batch_id=args.batch_id)
        for candidate in stage1_candidates
        if candidate.architecture_id in labels
    ]
    _write_csv(output_dir / "stage1_5_initial_label_queue.csv", label_rows, list(BENCHMARK_TABLE_FIELDS))
    _write_csv(output_dir / "benchmark_table_stub.csv", [], list(BENCHMARK_TABLE_FIELDS))

    coverage_path = output_dir / "stage0_coverage_report.md"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text("\n".join(_coverage_lines(all_candidates, anchors)) + "\n", encoding="utf-8")

    metadata = {
        "protocol_version": args.protocol_version,
        "oracle_protocol_version": args.oracle_protocol_version,
        "batch_id": args.batch_id,
        "n_candidates": len(all_candidates),
        "n_stage1_candidates": len(stage1_candidates),
        "n_anchors": len(anchors),
        "n_initial_labels": len(label_rows),
        "zero_cost_status_counts": dict(Counter(candidate.metadata.get("zero_cost_status", "") for candidate in all_candidates)),
        "zero_cost_reason_counts": dict(
            Counter(
                reason
                for candidate in all_candidates
                for reason in str(candidate.metadata.get("zero_cost_reasons", "")).split(";")
                if reason
            )
        ),
        "label_source_counts": dict(Counter(row["source"] for row in label_rows)),
        "label_source_by_n_qubits": dict(Counter(f"{row['n_qubits']}q:{row['source']}" for row in label_rows)),
        "distance_scales": scales.__dict__,
        "k_min": int(args.k_min),
        "trust_d_max": args.trust_d_max,
        "zero_cost_stage1b": {
            "n_samples": args.zero_cost_samples,
            "trainability_soft_quantile": args.trainability_soft_quantile,
            "expressibility_soft_quantile": args.expressibility_soft_quantile,
            "trainability_hard_floor": args.trainability_hard_floor,
            "expressibility_hard_floor": args.expressibility_hard_floor,
            "entanglement_soft_floor": args.entanglement_soft_floor,
            "max_params": args.max_params,
            "max_two_q": args.max_two_q,
            "expressibility_metric": args.zero_cost_expressibility_metric,
            "expressibility_validated": False,
            "trainability_metric": args.zero_cost_trainability_metric,
            "zero_cost_feature_score_is_ranking_signal": False,
            "score_distributions": {
                "expressibility_score": _score_distribution(candidate.metadata.get("expressibility_score", "") for candidate in all_candidates),
                "trainability_score": _score_distribution(candidate.metadata.get("trainability_score", "") for candidate in all_candidates),
                "entanglement_score": _score_distribution(candidate.metadata.get("entanglement_score", "") for candidate in all_candidates),
                "zero_cost_feature_score": _score_distribution(candidate.metadata.get("zero_cost_feature_score", "") for candidate in all_candidates),
            },
            "rule": "hard reject only extreme failures; soft flags are boundary candidates, not ranking weights",
        },
        "trust_region_defaults": DEFAULT_TRUST_REGION_RULES,
        "batch_defaults": DEFAULT_BATCH_QUOTAS,
        "retry_policy": DEFAULT_RETRY_POLICY,
    }
    (output_dir / "oracle_prep_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

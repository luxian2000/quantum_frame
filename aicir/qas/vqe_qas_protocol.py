"""Protocol helpers for trust-region VQE-QAS experiments.

This module intentionally contains no VQE execution code.  It freezes the
bookkeeping and selection rules needed before an offline oracle is trained:
label status values, source tags, conservative compatibility defaults,
Stage-0 anchor selection, initial holdout selection, Track-B expansion
selection, and the Track-A abstain-rate definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import ceil, isfinite
import random
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .MoG_VQE import (
    MOGVQEBlock,
    MOGVQECandidate,
    MOGVQEIndividual,
    nsga_ii_select,
)


class LabelStatus(str, Enum):
    """Allowed benchmark-table label states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_NONRETRYABLE = "failed_nonretryable"
    EXCLUDED_PROTOCOL_MISMATCH = "excluded_protocol_mismatch"
    EXCLUDED_INVALID_CANDIDATE = "excluded_invalid_candidate"
    EXCLUDED_DUPLICATE = "excluded_duplicate"


class LabelSource(str, Enum):
    """Allowed sources for fair-VQE labels and queued labels."""

    INITIAL_TRAIN = "initial_train"
    INITIAL_CALIBRATION = "initial_calibration"
    HOLDOUT_ID = "holdout_id"
    HOLDOUT_BOUNDARY = "holdout_boundary"
    HOLDOUT_SPARSE = "holdout_sparse"
    TRACKA_LOCAL = "trackA_local"
    TRACKB_BOUNDARY = "trackB_boundary"
    TRACKB_SPARSE = "trackB_sparse"
    CONTROL_RANDOM = "control_random"
    TARGET_FEWSHOT_TRAIN = "target_fewshot_train"
    TARGET_HOLDOUT_ID = "target_holdout_id"
    TARGET_HOLDOUT_BOUNDARY = "target_holdout_boundary"
    TARGET_HOLDOUT_SPARSE = "target_holdout_sparse"
    BASELINE = "baseline"


class ZeroCostStatus(str, Enum):
    """Stage-1b zero-cost feasibility status."""

    PASS = "pass"
    SOFT_FLAG = "soft_flag"
    HARD_REJECT = "hard_reject"


BENCHMARK_TABLE_FIELDS: tuple[str, ...] = (
    "architecture_id",
    "canonical_arch_hash",
    "protocol_version",
    "batch_id",
    "source",
    "label_status",
    "retry_count",
    "failure_reason",
    "last_error_digest",
    "n_qubits",
    "hamiltonian_id",
    "hamiltonian_class",
    "family",
    "depth_group",
    "entangler_type",
    "topology",
    "n_params",
    "two_q_count",
    "hamiltonian_coverage",
    "hamiltonian_coverage_features",
    "hamiltonian_terms",
    "zero_cost_status",
    "zero_cost_reasons",
    "expressibility_score",
    "trainability_score",
    "entanglement_score",
    "zero_cost_feature_score",
    "zero_cost_score_is_ranking_signal",
    "hea_mask",
    "ansatz_gene",
    "supernet_rank_score",
    "supernet_init_params_ref",
    "screening_energy",
    "screening_energy_is_final_label",
    "supernet_warm_start_status",
    "fair_best_energy",
    "fair_mean_energy",
    "fair_std_energy",
    "fair_success_rate",
    "delta_ref",
    "reference_energy",
    "optimizer",
    "n_seeds",
    "max_evals",
    "nfev",
    "walltime_s",
    "success_delta_ref",
    "best_trace",
    "dtype",
    "backend",
    "created_at",
)


def benchmark_row_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the task-local identity for benchmark row replacement."""

    architecture_id = str(row.get("architecture_id", ""))
    hamiltonian_id = str(row.get("hamiltonian_id", ""))
    if not hamiltonian_id:
        hamiltonian_class = str(row.get("hamiltonian_class", ""))
        n_qubits = str(row.get("n_qubits", ""))
        hamiltonian_id = f"{hamiltonian_class}_{n_qubits}q" if hamiltonian_class and n_qubits else ""
    protocol_version = str(row.get("protocol_version", ""))
    return architecture_id, hamiltonian_id, protocol_version


def append_benchmark_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    incoming_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Append label rows, replacing stale rows for the same architecture/task."""

    merged: list[dict[str, Any]] = [dict(row) for row in existing_rows]
    index_by_identity = {
        benchmark_row_identity(row): index
        for index, row in enumerate(merged)
        if benchmark_row_identity(row)[0]
    }
    for row in incoming_rows:
        copied = dict(row)
        identity = benchmark_row_identity(copied)
        if not identity[0]:
            merged.append(copied)
            continue
        if identity in index_by_identity:
            merged[index_by_identity[identity]] = copied
        else:
            index_by_identity[identity] = len(merged)
            merged.append(copied)
    return merged


DEFAULT_RETRY_POLICY: dict[str, float | int] = {
    "max_retry": 2,
    "running_timeout_multiplier": 1.5,
}


DEFAULT_TRUST_REGION_RULES: dict[str, float | int] = {
    "k_min": 5,
    "abstain_rate_max": 0.40,
    "target_sparse_abstain_rate": 0.80,
    "target_tr_coverage": 0.20,
    "target_tr_in_mae_ratio": 0.50,
}


DEFAULT_BATCH_QUOTAS: dict[str, int] = {
    "local": 12,
    "boundary": 10,
    "sparse": 6,
    "control": 4,
}


DEFAULT_SMALL_BATCH_QUOTAS: dict[str, int] = {
    "local": 6,
    "boundary": 5,
    "sparse": 3,
    "control": 2,
}


@dataclass(frozen=True)
class CandidateRecord:
    """Minimal architecture metadata used by protocol selection rules."""

    architecture_id: str
    canonical_arch_hash: str
    family: str
    entangler_type: str
    topology: str
    depth_group: str
    n_params: float
    two_q_count: float
    hamiltonian_id: str = ""
    hamiltonian_class: str = "tfim"
    hamiltonian_coverage: float = 0.0
    hamiltonian_features: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistanceScales:
    """Robust Stage-0 scales used for repeatable distance calculations."""

    n_params: float
    two_q_count: float
    hamiltonian_coverage: float


DEFAULT_COMPATIBILITY: dict[str, set[tuple[str, str]]] = {
    "entangler_type": {
        ("cx", "cz"),
        ("cz", "rzz"),
    },
    "topology": {
        ("linear", "ring"),
    },
    "family": {
        ("hea", "realamplitudes"),
        ("qaoa_like", "problem_inspired"),
    },
    "depth_group": set(),
}


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left).lower(), str(right).lower())))  # type: ignore[return-value]


def categorical_distance(
    left: str,
    right: str,
    *,
    feature: str,
    compatibility: Mapping[str, set[tuple[str, str]]] | None = None,
) -> float:
    """Return conservative categorical distance: same=0, compatible=0.5, else=1."""

    if str(left).lower() == str(right).lower():
        return 0.0
    compat = compatibility or DEFAULT_COMPATIBILITY
    pairs = compat.get(feature, set())
    return 0.5 if _ordered_pair(left, right) in pairs else 1.0


def _iqr(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    sorted_values = sorted(float(value) for value in values)
    low = sorted_values[int(0.25 * (len(sorted_values) - 1))]
    high = sorted_values[int(0.75 * (len(sorted_values) - 1))]
    spread = high - low
    if spread > 0:
        return spread
    value_range = sorted_values[-1] - sorted_values[0]
    return value_range if value_range > 0 else 1.0


def fit_distance_scales(candidates: Sequence[CandidateRecord]) -> DistanceScales:
    """Fit fixed Stage-0 robust scales for continuous distance features."""

    return DistanceScales(
        n_params=_iqr([candidate.n_params for candidate in candidates]),
        two_q_count=_iqr([candidate.two_q_count for candidate in candidates]),
        hamiltonian_coverage=_iqr([candidate.hamiltonian_coverage for candidate in candidates]),
    )


def normalized_abs_diff(left: float, right: float, scale: float) -> float:
    safe_scale = float(scale) if float(scale) > 0 else 1.0
    return min(abs(float(left) - float(right)) / safe_scale, 4.0)


def composite_distance(
    left: CandidateRecord,
    right: CandidateRecord,
    scales: DistanceScales,
    *,
    compatibility: Mapping[str, set[tuple[str, str]]] | None = None,
) -> float:
    """Distance used for Stage-0 holdout and Track-B farthest-first expansion."""

    categorical = [
        categorical_distance(left.family, right.family, feature="family", compatibility=compatibility),
        categorical_distance(
            left.entangler_type,
            right.entangler_type,
            feature="entangler_type",
            compatibility=compatibility,
        ),
        categorical_distance(left.topology, right.topology, feature="topology", compatibility=compatibility),
        categorical_distance(left.depth_group, right.depth_group, feature="depth_group", compatibility=compatibility),
        0.0 if left.hamiltonian_class == right.hamiltonian_class else 1.0,
    ]
    continuous = [
        normalized_abs_diff(left.n_params, right.n_params, scales.n_params),
        normalized_abs_diff(left.two_q_count, right.two_q_count, scales.two_q_count),
        normalized_abs_diff(
            left.hamiltonian_coverage,
            right.hamiltonian_coverage,
            scales.hamiltonian_coverage,
        ),
    ]
    return (sum(categorical) + sum(continuous)) / float(len(categorical) + len(continuous))


def select_stage0_anchors(candidates: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """Pick deterministic anchors from family x entangler_type x topology cells."""

    cells: dict[tuple[str, str, str], list[CandidateRecord]] = {}
    for candidate in candidates:
        key = (candidate.family, candidate.entangler_type, candidate.topology)
        cells.setdefault(key, []).append(candidate)

    anchors: list[CandidateRecord] = []
    for key in sorted(cells):
        cell = cells[key]
        median_params = median(candidate.n_params for candidate in cell)
        median_two_q = median(candidate.two_q_count for candidate in cell)
        selected = min(
            cell,
            key=lambda candidate: (
                abs(candidate.n_params - median_params) + abs(candidate.two_q_count - median_two_q),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )
        anchors.append(selected)
    return anchors


def min_distance_to_set(
    candidate: CandidateRecord,
    references: Sequence[CandidateRecord],
    scales: DistanceScales,
) -> float:
    if not references:
        return float("inf")
    return min(composite_distance(candidate, reference, scales) for reference in references)


def _sorted_reference_distances(
    candidate: CandidateRecord,
    references: Sequence[CandidateRecord],
    scales: DistanceScales,
) -> list[float]:
    return sorted(composite_distance(candidate, reference, scales) for reference in references)


def derive_trust_region_d_max(
    train: Sequence[CandidateRecord],
    scales: DistanceScales,
    *,
    k_min: int = 5,
    multiplier: float = 1.5,
) -> float:
    """Derive a local trust-region radius from train leave-one-out kth distances."""

    train = list(train)
    k = max(1, int(k_min))
    kth_distances: list[float] = []
    for index, candidate in enumerate(train):
        references = train[:index] + train[index + 1 :]
        distances = _sorted_reference_distances(candidate, references, scales)
        if len(distances) >= k:
            kth_distances.append(distances[k - 1])
    if not kth_distances:
        return 0.0
    return float(median(kth_distances) * float(multiplier))


def trust_region_geometry(
    candidate: CandidateRecord,
    train: Sequence[CandidateRecord],
    scales: DistanceScales,
    *,
    k_min: int = 5,
    d_max: float | None = None,
) -> dict[str, float | int | bool]:
    """Summarize a candidate's local geometry against the current train set."""

    distances = _sorted_reference_distances(candidate, train, scales)
    k = max(1, int(k_min))
    radius = float(d_max) if d_max is not None else derive_trust_region_d_max(train, scales, k_min=k)
    nearest = distances[0] if distances else float("inf")
    kth_distance = distances[k - 1] if len(distances) >= k else float("inf")
    neighbor_count = sum(1 for distance in distances if distance <= radius)
    return {
        "nearest_distance": float(nearest),
        "kth_distance": float(kth_distance),
        "neighbor_count": int(neighbor_count),
        "in_trust_region": bool(neighbor_count >= k),
    }


def assign_initial_holdout_sources(
    candidates: Sequence[CandidateRecord],
    anchors: Sequence[CandidateRecord],
    scales: DistanceScales,
    *,
    holdout_size: int,
) -> dict[str, LabelSource]:
    """Assign deterministic ID/boundary/sparse holdout tags from Stage-0 distances."""

    if holdout_size <= 0 or not candidates:
        return {}
    scored = [
        (candidate, min_distance_to_set(candidate, anchors, scales))
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (item[1], item[0].canonical_arch_hash, item[0].architecture_id))
    target_size = min(holdout_size, len(scored))
    id_count = max(1, int(round(target_size * 0.40)))
    boundary_count = max(1, int(round(target_size * 0.30))) if target_size >= 3 else max(0, target_size - id_count)
    sparse_count = max(0, target_size - id_count - boundary_count)

    n_scored = len(scored)
    id_region = scored[: max(1, int(0.60 * n_scored))]
    boundary_region = scored[int(0.60 * n_scored) : max(int(0.60 * n_scored) + 1, int(0.80 * n_scored))]
    sparse_region = scored[int(0.80 * n_scored) :]

    def pick_evenly(region: Sequence[tuple[CandidateRecord, float]], count: int) -> list[CandidateRecord]:
        if count <= 0 or not region:
            return []
        if count >= len(region):
            return [candidate for candidate, _distance in region]
        if count == 1:
            return [region[len(region) // 2][0]]
        step = (len(region) - 1) / float(count - 1)
        return [region[int(round(index * step))][0] for index in range(count)]

    sources: dict[str, LabelSource] = {}
    for candidate in pick_evenly(id_region, id_count):
        sources[candidate.architecture_id] = LabelSource.HOLDOUT_ID
    for candidate in pick_evenly(boundary_region, boundary_count):
        sources.setdefault(candidate.architecture_id, LabelSource.HOLDOUT_BOUNDARY)
    for candidate in pick_evenly(sparse_region, sparse_count):
        sources.setdefault(candidate.architecture_id, LabelSource.HOLDOUT_SPARSE)

    # Fill any collisions or empty percentile regions deterministically.
    for candidate, _distance in reversed(scored):
        if len(sources) >= target_size:
            break
        sources.setdefault(candidate.architecture_id, LabelSource.HOLDOUT_SPARSE)
    return sources


def assign_repaired_holdout_sources(
    candidates: Sequence[CandidateRecord],
    train: Sequence[CandidateRecord],
    scales: DistanceScales,
    *,
    holdout_size: int,
    k_min: int = 5,
    d_max: float | None = None,
) -> dict[str, LabelSource]:
    """Assign holdout tags using train-set trust-region geometry.

    The repaired semantics are:
    - ID: near the train distribution and inside the trust region.
    - boundary: kth-neighbor distance close to the trust-region radius.
    - sparse: fewer than ``k_min`` neighbors inside the trust-region radius.
    """

    candidates = list(candidates)
    train = list(train)
    if holdout_size <= 0 or not candidates:
        return {}
    if not train:
        anchors = select_stage0_anchors(candidates)
        return assign_initial_holdout_sources(candidates, anchors, scales, holdout_size=holdout_size)

    target_size = min(int(holdout_size), len(candidates))
    id_count = max(1, int(round(target_size * 0.40)))
    boundary_count = max(1, int(round(target_size * 0.30))) if target_size >= 3 else max(0, target_size - id_count)
    sparse_count = max(0, target_size - id_count - boundary_count)
    k = max(1, int(k_min))
    radius = float(d_max) if d_max is not None else derive_trust_region_d_max(train, scales, k_min=k)

    train_bucket_counts: dict[tuple[str, str, str], int] = {}
    for item in train:
        bucket = (item.entangler_type, item.topology, item.depth_group)
        train_bucket_counts[bucket] = train_bucket_counts.get(bucket, 0) + 1

    def bucket_weight(candidate: CandidateRecord) -> int:
        return train_bucket_counts.get((candidate.entangler_type, candidate.topology, candidate.depth_group), 0)

    geometry = {
        candidate.architecture_id: trust_region_geometry(candidate, train, scales, k_min=k, d_max=radius)
        for candidate in candidates
    }
    by_id = {candidate.architecture_id: candidate for candidate in candidates}
    selected: dict[str, LabelSource] = {}
    selected_counts = {
        LabelSource.HOLDOUT_ID: 0,
        LabelSource.HOLDOUT_BOUNDARY: 0,
        LabelSource.HOLDOUT_SPARSE: 0,
    }

    def take(source: LabelSource, ordered: Sequence[CandidateRecord], count: int) -> None:
        for candidate in ordered:
            if selected_counts[source] >= count or len(selected) >= target_size:
                return
            if candidate.architecture_id in selected:
                continue
            selected[candidate.architecture_id] = source
            selected_counts[source] += 1

    id_pool = [
        candidate
        for candidate in candidates
        if int(geometry[candidate.architecture_id]["neighbor_count"]) >= k
    ]
    id_pool.sort(
        key=lambda candidate: (
            -int(geometry[candidate.architecture_id]["neighbor_count"]),
            float(geometry[candidate.architecture_id]["kth_distance"]),
            -bucket_weight(candidate),
            candidate.canonical_arch_hash,
            candidate.architecture_id,
        )
    )

    sparse_pool = [
        candidate
        for candidate in candidates
        if int(geometry[candidate.architecture_id]["neighbor_count"]) < k
    ]
    sparse_pool.sort(
        key=lambda candidate: (
            int(geometry[candidate.architecture_id]["neighbor_count"]),
            -float(geometry[candidate.architecture_id]["nearest_distance"]),
            candidate.canonical_arch_hash,
            candidate.architecture_id,
        )
    )

    boundary_pool = [
        candidate
        for candidate in candidates
        if isfinite(float(geometry[candidate.architecture_id]["kth_distance"]))
    ]
    boundary_pool.sort(
        key=lambda candidate: (
            abs(float(geometry[candidate.architecture_id]["kth_distance"]) - radius),
            -bucket_weight(candidate),
            candidate.canonical_arch_hash,
            candidate.architecture_id,
        )
    )

    take(LabelSource.HOLDOUT_ID, id_pool, id_count)
    take(LabelSource.HOLDOUT_SPARSE, sparse_pool, sparse_count)
    take(LabelSource.HOLDOUT_BOUNDARY, boundary_pool, boundary_count)

    # Fill shortages deterministically while preserving the closest matching
    # source semantics available for each remaining candidate.
    remaining = [candidate for candidate in candidates if candidate.architecture_id not in selected]
    for candidate in sorted(
        remaining,
        key=lambda item: (
            abs(float(geometry[item.architecture_id]["kth_distance"]) - radius)
            if isfinite(float(geometry[item.architecture_id]["kth_distance"]))
            else float("inf"),
            item.canonical_arch_hash,
            item.architecture_id,
        ),
    ):
        if len(selected) >= target_size:
            break
        item_geometry = geometry[candidate.architecture_id]
        if int(item_geometry["neighbor_count"]) >= k and selected_counts[LabelSource.HOLDOUT_ID] < id_count:
            source = LabelSource.HOLDOUT_ID
        elif int(item_geometry["neighbor_count"]) < k and selected_counts[LabelSource.HOLDOUT_SPARSE] < sparse_count:
            source = LabelSource.HOLDOUT_SPARSE
        else:
            source = LabelSource.HOLDOUT_BOUNDARY
        selected[candidate.architecture_id] = source
        selected_counts[source] += 1

    # If all semantic pools were exhausted, keep the requested holdout size by
    # assigning the remaining farthest candidates as sparse expansion checks.
    if len(selected) < target_size:
        tail = sorted(
            (candidate for candidate in candidates if candidate.architecture_id not in selected),
            key=lambda candidate: (
                -float(geometry[candidate.architecture_id]["nearest_distance"]),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )
        for candidate in tail:
            if len(selected) >= target_size:
                break
            selected[candidate.architecture_id] = LabelSource.HOLDOUT_SPARSE

    return selected


def select_initial_label_batch(
    candidates: Sequence[CandidateRecord],
    *,
    total_labels: int = 96,
    holdout_fraction: float = 0.20,
    group_key: str | None = None,
    trust_d_max: float | None = None,
    k_min: int | None = None,
) -> dict[str, LabelSource]:
    """Select first fair-VQE label batch with priority coverage and holdouts."""

    if total_labels <= 0:
        return {}
    candidates = list(dict((candidate.architecture_id, candidate) for candidate in candidates).values())
    if not candidates:
        return {}
    if group_key:
        groups: dict[str, list[CandidateRecord]] = {}
        for candidate in candidates:
            if group_key == "n_qubits":
                key = str(candidate.metadata.get("n_qubits", ""))
            else:
                key = str(getattr(candidate, group_key))
            groups.setdefault(key, []).append(candidate)
        labels: dict[str, LabelSource] = {}
        remaining = min(total_labels, len(candidates))
        group_items = sorted(groups.items())
        for index, (_key, group_candidates) in enumerate(group_items):
            groups_left = len(group_items) - index
            group_budget = min(
                len(group_candidates),
                max(1, remaining // max(1, groups_left)),
            )
            labels.update(
                select_initial_label_batch(
                    group_candidates,
                    total_labels=group_budget,
                    holdout_fraction=holdout_fraction,
                    group_key=None,
                    trust_d_max=trust_d_max,
                    k_min=k_min,
                )
            )
            remaining -= group_budget
        return labels

    budget = min(total_labels, len(candidates))
    holdout_size = min(ceil(budget * holdout_fraction), max(0, budget - 1))
    scales = fit_distance_scales(candidates)
    anchors = select_stage0_anchors(candidates)
    train_size = budget - holdout_size
    effective_k_min = int(k_min if k_min is not None else DEFAULT_TRUST_REGION_RULES["k_min"])
    pool_geometry: dict[str, dict[str, float | int | bool]] = {}
    if trust_d_max is not None:
        for candidate in candidates:
            references = [item for item in candidates if item.architecture_id != candidate.architecture_id]
            pool_geometry[candidate.architecture_id] = trust_region_geometry(
                candidate,
                references,
                scales,
                k_min=effective_k_min,
                d_max=trust_d_max,
            )
    id_count = max(1, int(round(holdout_size * 0.40))) if holdout_size > 0 else 0
    boundary_count = max(1, int(round(holdout_size * 0.30))) if holdout_size >= 3 else max(0, holdout_size - id_count)
    sparse_count = max(0, holdout_size - id_count - boundary_count)
    sparse_reserve_count = min(sparse_count, max(0, len(candidates) - train_size))
    if pool_geometry:
        sparse_ordered = sorted(
            candidates,
            key=lambda candidate: (
                int(pool_geometry[candidate.architecture_id]["neighbor_count"]),
                -float(pool_geometry[candidate.architecture_id]["nearest_distance"]),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )
    else:
        sparse_ordered = [
            candidate
            for candidate, _distance in sorted(
                (
                    (candidate, min_distance_to_set(candidate, anchors, scales))
                    for candidate in candidates
                ),
                key=lambda item: (-item[1], item[0].canonical_arch_hash, item[0].architecture_id),
            )
        ]
    sparse_reserved_ids = {
        candidate.architecture_id
        for candidate in sparse_ordered[:sparse_reserve_count]
    }
    train_candidates = [
        candidate for candidate in candidates if candidate.architecture_id not in sparse_reserved_ids
    ]
    selected: dict[str, CandidateRecord] = {}

    def add_one_per_bucket(attribute: str, limit_per_bucket: int) -> None:
        buckets: dict[str, list[CandidateRecord]] = {}
        for candidate in train_candidates:
            buckets.setdefault(str(getattr(candidate, attribute)), []).append(candidate)
        for bucket in sorted(buckets):
            ordered = sorted(
                buckets[bucket],
                key=(
                    (
                        lambda candidate: (
                            -int(pool_geometry[candidate.architecture_id]["neighbor_count"]),
                            float(pool_geometry[candidate.architecture_id]["kth_distance"]),
                            candidate.canonical_arch_hash,
                            candidate.architecture_id,
                        )
                    )
                    if pool_geometry
                    else (lambda candidate: (candidate.canonical_arch_hash, candidate.architecture_id))
                ),
            )
            if pool_geometry:
                picked = ordered[:limit_per_bucket]
            elif limit_per_bucket >= len(ordered):
                picked = ordered
            elif limit_per_bucket <= 1:
                picked = [ordered[len(ordered) // 2]]
            else:
                step = (len(ordered) - 1) / float(limit_per_bucket - 1)
                picked = [ordered[int(round(index * step))] for index in range(limit_per_bucket)]
            for candidate in picked:
                if len(selected) >= train_size:
                    return
                selected.setdefault(candidate.architecture_id, candidate)

    coverage_order = (
        [
            ("family", 5),
            ("entangler_type", 5),
            ("depth_group", 3),
            ("topology", 3),
        ]
        if budget >= 96
        else [
            ("entangler_type", 2),
            ("depth_group", 2),
            ("topology", 2),
            ("family", 1),
        ]
    )
    for attribute, limit_per_bucket in coverage_order:
        add_one_per_bucket(attribute, limit_per_bucket)

    while len(selected) < train_size:
        remaining = [candidate for candidate in train_candidates if candidate.architecture_id not in selected]
        if not remaining:
            break
        references = list(selected.values()) or anchors
        next_candidate = max(
            remaining,
            key=lambda candidate: (
                min_distance_to_set(candidate, references, scales),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )
        selected[next_candidate.architecture_id] = next_candidate

    remaining_for_holdout = [
        candidate for candidate in candidates if candidate.architecture_id not in selected
    ]
    holdout_sources = assign_repaired_holdout_sources(
        remaining_for_holdout,
        list(selected.values()),
        scales,
        holdout_size=holdout_size,
        k_min=effective_k_min,
        d_max=trust_d_max,
    )

    labels = {architecture_id: LabelSource.INITIAL_TRAIN for architecture_id in selected}
    for architecture_id, source in holdout_sources.items():
        labels[architecture_id] = source
    return labels


def _term_coeff_and_pauli(term: Any) -> tuple[float, str]:
    if isinstance(term, Mapping):
        coeff = term.get("coefficient", term.get("coeff", term.get("weight", 0.0)))
        pauli = term.get("pauli", term.get("pauli_string", term.get("string", "")))
        return float(coeff), str(pauli).upper()
    if isinstance(term, (list, tuple)) and len(term) >= 2:
        return float(term[0]), str(term[1]).upper()
    raise ValueError(f"Unsupported Hamiltonian term format: {term!r}")


def parse_pauli_hamiltonian_terms(terms: Iterable[Any]) -> tuple[tuple[float, str], ...]:
    """Parse literal Pauli-sum terms into canonical ``(coeff, pauli)`` tuples."""

    return tuple(_term_coeff_and_pauli(term) for term in terms)


def _zero_pauli_hamiltonian_features() -> dict[str, float]:
    features = {
        "n_terms": 0.0,
        "n_qubits": 0.0,
        "coeff_l1": 0.0,
        "coeff_l2": 0.0,
        "coeff_max_abs": 0.0,
        "locality_mean": 0.0,
        "locality_max": 0.0,
        "one_body_fraction": 0.0,
        "two_body_fraction": 0.0,
        "many_body_fraction": 0.0,
        "x_fraction": 0.0,
        "y_fraction": 0.0,
        "z_fraction": 0.0,
        "mixed_pauli_fraction": 0.0,
        "active_coeff_l1": 0.0,
        "one_body_coeff_l1": 0.0,
        "two_body_coeff_l1": 0.0,
        "many_body_coeff_l1": 0.0,
        "one_body_coeff_fraction": 0.0,
        "two_body_coeff_fraction": 0.0,
        "many_body_coeff_fraction": 0.0,
        "two_body_mixed_coeff_l1": 0.0,
        "two_body_mixed_coeff_fraction": 0.0,
    }
    for symbol in ("x", "y", "z"):
        features[f"{symbol}_coeff_l1"] = 0.0
        features[f"{symbol}_coeff_fraction"] = 0.0
        features[f"one_body_{symbol}_coeff_l1"] = 0.0
        features[f"one_body_{symbol}_coeff_fraction"] = 0.0
    for pair in ("xx", "yy", "zz", "xy", "xz", "yz"):
        features[f"two_body_{pair}_coeff_l1"] = 0.0
        features[f"two_body_{pair}_coeff_fraction"] = 0.0
    return features


def extract_pauli_hamiltonian_features(terms: Iterable[Any]) -> dict[str, float]:
    """Extract compact task features from Pauli-sum Hamiltonian terms.

    Accepted term forms are ``(coefficient, pauli_string)`` tuples/lists or
    mappings with ``coeff``/``coefficient`` and ``pauli``/``pauli_string`` keys.
    The features are intentionally simple and deterministic so they can be
    serialized into benchmark tables and used by local-oracle distance rules.
    """

    parsed = list(parse_pauli_hamiltonian_terms(terms))
    if not parsed:
        return _zero_pauli_hamiltonian_features()

    n_terms = len(parsed)
    n_qubits = max(len(pauli) for _coeff, pauli in parsed)
    abs_coeffs = [abs(coeff) for coeff, _pauli in parsed]
    localities = [sum(1 for symbol in pauli if symbol != "I") for _coeff, pauli in parsed]
    active_symbols = [
        symbol
        for _coeff, pauli in parsed
        for symbol in pauli
        if symbol in {"X", "Y", "Z"}
    ]
    active_total = max(1, len(active_symbols))
    mixed_terms = 0
    features = _zero_pauli_hamiltonian_features()
    symbol_coeff_l1 = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    one_body_symbol_coeff_l1 = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    pair_coeff_l1 = {"XX": 0.0, "YY": 0.0, "ZZ": 0.0, "XY": 0.0, "XZ": 0.0, "YZ": 0.0}
    locality_coeff_l1 = {1: 0.0, 2: 0.0, "many": 0.0}
    for _coeff, pauli in parsed:
        coeff_abs = abs(_coeff)
        active = [symbol for symbol in pauli if symbol in {"X", "Y", "Z"}]
        kinds = set(active)
        if len(kinds) > 1:
            mixed_terms += 1
        for symbol in active:
            symbol_coeff_l1[symbol] += coeff_abs
        if len(active) == 1:
            locality_coeff_l1[1] += coeff_abs
            one_body_symbol_coeff_l1[active[0]] += coeff_abs
        elif len(active) == 2:
            locality_coeff_l1[2] += coeff_abs
            pair_key = "".join(sorted(active))
            if pair_key in pair_coeff_l1:
                pair_coeff_l1[pair_key] += coeff_abs
            else:
                features["two_body_mixed_coeff_l1"] += coeff_abs
        elif len(active) >= 3:
            locality_coeff_l1["many"] += coeff_abs

    active_coeff_l1 = sum(symbol_coeff_l1.values())
    coeff_l1 = float(sum(abs_coeffs))
    two_body_coeff_l1 = locality_coeff_l1[2]
    features.update({
        "n_terms": float(n_terms),
        "n_qubits": float(n_qubits),
        "coeff_l1": coeff_l1,
        "coeff_l2": float(sum(value * value for value in abs_coeffs) ** 0.5),
        "coeff_max_abs": float(max(abs_coeffs) if abs_coeffs else 0.0),
        "locality_mean": float(sum(localities) / n_terms),
        "locality_max": float(max(localities) if localities else 0.0),
        "one_body_fraction": float(sum(1 for value in localities if value == 1) / n_terms),
        "two_body_fraction": float(sum(1 for value in localities if value == 2) / n_terms),
        "many_body_fraction": float(sum(1 for value in localities if value >= 3) / n_terms),
        "x_fraction": float(sum(1 for symbol in active_symbols if symbol == "X") / active_total),
        "y_fraction": float(sum(1 for symbol in active_symbols if symbol == "Y") / active_total),
        "z_fraction": float(sum(1 for symbol in active_symbols if symbol == "Z") / active_total),
        "mixed_pauli_fraction": float(mixed_terms / n_terms),
        "active_coeff_l1": float(active_coeff_l1),
        "one_body_coeff_l1": float(locality_coeff_l1[1]),
        "two_body_coeff_l1": float(two_body_coeff_l1),
        "many_body_coeff_l1": float(locality_coeff_l1["many"]),
        "one_body_coeff_fraction": float(locality_coeff_l1[1] / coeff_l1) if coeff_l1 else 0.0,
        "two_body_coeff_fraction": float(two_body_coeff_l1 / coeff_l1) if coeff_l1 else 0.0,
        "many_body_coeff_fraction": float(locality_coeff_l1["many"] / coeff_l1) if coeff_l1 else 0.0,
        "two_body_mixed_coeff_fraction": (
            float(features["two_body_mixed_coeff_l1"] / two_body_coeff_l1) if two_body_coeff_l1 else 0.0
        ),
    })
    for symbol, key in (("X", "x"), ("Y", "y"), ("Z", "z")):
        features[f"{key}_coeff_l1"] = float(symbol_coeff_l1[symbol])
        features[f"{key}_coeff_fraction"] = (
            float(symbol_coeff_l1[symbol] / active_coeff_l1) if active_coeff_l1 else 0.0
        )
        features[f"one_body_{key}_coeff_l1"] = float(one_body_symbol_coeff_l1[symbol])
        features[f"one_body_{key}_coeff_fraction"] = (
            float(one_body_symbol_coeff_l1[symbol] / locality_coeff_l1[1]) if locality_coeff_l1[1] else 0.0
        )
    for pair_key, value in pair_coeff_l1.items():
        key = pair_key.lower()
        features[f"two_body_{key}_coeff_l1"] = float(value)
        features[f"two_body_{key}_coeff_fraction"] = float(value / two_body_coeff_l1) if two_body_coeff_l1 else 0.0
    return features


DEFAULT_HAMILTONIAN_DISTANCE_FEATURES: tuple[str, ...] = (
    "coeff_l1",
    "coeff_l2",
    "coeff_max_abs",
    "one_body_coeff_fraction",
    "two_body_coeff_fraction",
    "many_body_coeff_fraction",
    "x_coeff_fraction",
    "y_coeff_fraction",
    "z_coeff_fraction",
    "one_body_x_coeff_fraction",
    "one_body_y_coeff_fraction",
    "one_body_z_coeff_fraction",
    "two_body_xx_coeff_fraction",
    "two_body_yy_coeff_fraction",
    "two_body_zz_coeff_fraction",
    "two_body_xy_coeff_fraction",
    "two_body_xz_coeff_fraction",
    "two_body_yz_coeff_fraction",
    "two_body_mixed_coeff_fraction",
    "one_body_x_coeff_l1",
    "two_body_zz_coeff_l1",
)


def hamiltonian_feature_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    feature_names: Sequence[str] | None = None,
) -> float:
    """Return scale-free L1 distance between two Hamiltonian feature dicts."""

    keys = tuple(feature_names) if feature_names is not None else DEFAULT_HAMILTONIAN_DISTANCE_FEATURES
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        left_value = float(left.get(key, 0.0))
        right_value = float(right.get(key, 0.0))
        scale = max(abs(left_value), abs(right_value), 1.0)
        total += abs(left_value - right_value) / scale
    return total / float(len(keys))


def parse_hamiltonian_features(value: Any) -> dict[str, float]:
    """Parse serialized Hamiltonian features from benchmark-table cells."""

    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, float] = {}
    for key, item in raw.items():
        try:
            parsed[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return parsed


def task_aware_composite_distance(
    left: CandidateRecord,
    right: CandidateRecord,
    scales: DistanceScales,
    *,
    hamiltonian_weight: float = 0.25,
    compatibility: Mapping[str, set[tuple[str, str]]] | None = None,
) -> float:
    """Architecture distance plus a bounded Hamiltonian/task distance term."""

    arch_distance = composite_distance(left, right, scales, compatibility=compatibility)
    left_id = _candidate_hamiltonian_id(left)
    right_id = _candidate_hamiltonian_id(right)
    if not left_id and not right_id:
        id_distance = 0.0 if left.hamiltonian_class == right.hamiltonian_class else 1.0
    elif left_id and right_id and left_id == right_id:
        id_distance = 0.0
    elif left.hamiltonian_class == right.hamiltonian_class:
        id_distance = 0.5
    else:
        id_distance = 1.0
    feature_distance = hamiltonian_feature_distance(left.hamiltonian_features, right.hamiltonian_features)
    task_distance = 0.5 * id_distance + 0.5 * feature_distance
    weight = max(0.0, min(float(hamiltonian_weight), 1.0))
    return (1.0 - weight) * arch_distance + weight * task_distance


def _candidate_hamiltonian_id(candidate: CandidateRecord) -> str:
    return str(candidate.hamiltonian_id or candidate.metadata.get("hamiltonian_id", ""))


def select_target_fewshot_batch(
    candidates: Sequence[CandidateRecord],
    *,
    target_hamiltonian_id: str,
    total_labels: int,
    holdout_fraction: float = 0.25,
    group_key: str | None = None,
    trust_d_max: float | None = None,
    k_min: int | None = None,
) -> dict[str, LabelSource]:
    """Select target-task few-shot labels without mixing source labels."""

    target = [
        candidate
        for candidate in candidates
        if _candidate_hamiltonian_id(candidate) == str(target_hamiltonian_id)
    ]
    selected = select_initial_label_batch(
        target,
        total_labels=total_labels,
        holdout_fraction=holdout_fraction,
        group_key=group_key,
        trust_d_max=trust_d_max,
        k_min=k_min,
    )
    source_map = {
        LabelSource.INITIAL_TRAIN: LabelSource.TARGET_FEWSHOT_TRAIN,
        LabelSource.HOLDOUT_ID: LabelSource.TARGET_HOLDOUT_ID,
        LabelSource.HOLDOUT_BOUNDARY: LabelSource.TARGET_HOLDOUT_BOUNDARY,
        LabelSource.HOLDOUT_SPARSE: LabelSource.TARGET_HOLDOUT_SPARSE,
    }
    return {
        architecture_id: source_map.get(source, source)
        for architecture_id, source in selected.items()
    }


def select_farthest_first(
    candidates: Sequence[CandidateRecord],
    labeled: Sequence[CandidateRecord],
    scales: DistanceScales,
    *,
    count: int,
) -> list[CandidateRecord]:
    """Track-B expansion: choose candidates farthest from the labeled table."""

    selected: list[CandidateRecord] = []
    selected_ids: set[str] = set()
    references = list(labeled)
    pool = list(candidates)
    while len(selected) < count and pool:
        candidate = max(
            pool,
            key=lambda item: (
                min_distance_to_set(item, references, scales),
                item.canonical_arch_hash,
                item.architecture_id,
            ),
        )
        selected.append(candidate)
        selected_ids.add(candidate.architecture_id)
        references.append(candidate)
        pool = [item for item in pool if item.architecture_id not in selected_ids]
    return selected


def _candidate_gene(candidate: CandidateRecord, gene_key: str) -> tuple[str, ...] | None:
    value = candidate.metadata.get(gene_key)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return None
    return tuple(str(item) for item in value)


def _gene_options(genes: Sequence[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    if not genes:
        return ()
    width = len(genes[0])
    same_width = [gene for gene in genes if len(gene) == width]
    if len(same_width) != len(genes):
        return ()
    return tuple(
        tuple(sorted({gene[index] for gene in same_width}))
        for index in range(width)
    )


def _mutate_gene(
    gene: tuple[str, ...],
    options: Sequence[Sequence[str]],
    rng: random.Random,
    mutation_rate: float,
) -> tuple[str, ...]:
    values = list(gene)
    for index, choices in enumerate(options):
        if rng.random() >= float(mutation_rate):
            continue
        alternatives = [choice for choice in choices if choice != values[index]]
        if alternatives:
            values[index] = rng.choice(alternatives)
    return tuple(values)


def _crossover_gene(
    left: tuple[str, ...],
    right: tuple[str, ...],
    rng: random.Random,
    crossover_rate: float,
) -> tuple[str, ...]:
    if len(left) != len(right) or len(left) < 2 or rng.random() >= float(crossover_rate):
        return left
    cut = rng.randrange(1, len(left))
    return tuple(left[:cut] + right[cut:])


def select_ea_candidates(
    candidates: Sequence[CandidateRecord],
    *,
    seeds: Sequence[CandidateRecord] = (),
    excluded_ids: Iterable[str] = (),
    fitness: Callable[[CandidateRecord], float],
    population_size: int = 32,
    generations: int = 12,
    mutation_rate: float = 0.20,
    crossover_rate: float = 0.60,
    elite_count: int = 4,
    limit: int = 16,
    random_seed: int = 0,
    gene_key: str = "hea_mask",
) -> list[CandidateRecord]:
    """Run a deterministic, pool-bounded EA over candidate metadata genes.

    The helper never invents an architecture outside ``candidates``.  Mutation
    and crossover operate on the discrete gene stored in ``metadata[gene_key]``
    and a child is kept only when that gene maps back to a known, unlabeled
    candidate.  If the pool has no gene metadata, the function degrades to a
    deterministic fitness ranking so callers can still use it as a planner.
    """

    if int(limit) <= 0:
        return []
    excluded = {str(identifier) for identifier in excluded_ids}
    usable = [
        candidate
        for candidate in candidates
        if candidate.architecture_id not in excluded
    ]
    if not usable:
        return []

    def rank(items: Sequence[CandidateRecord]) -> list[CandidateRecord]:
        dedup = {candidate.architecture_id: candidate for candidate in items}
        return sorted(
            dedup.values(),
            key=lambda candidate: (
                -float(fitness(candidate)),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )

    genes_by_id = {
        candidate.architecture_id: _candidate_gene(candidate, gene_key)
        for candidate in usable
    }
    candidate_by_gene: dict[tuple[str, ...], CandidateRecord] = {}
    for candidate in sorted(usable, key=lambda item: (item.canonical_arch_hash, item.architecture_id)):
        gene = genes_by_id[candidate.architecture_id]
        if gene is not None:
            candidate_by_gene.setdefault(gene, candidate)

    if not candidate_by_gene:
        return rank(usable)[: int(limit)]

    options = _gene_options(list(candidate_by_gene))
    if not options:
        return rank(usable)[: int(limit)]

    rng = random.Random(int(random_seed))
    population_target = max(1, min(int(population_size), len(usable)))
    population: list[CandidateRecord] = []
    seen_ids: set[str] = set()

    def add(candidate: CandidateRecord) -> None:
        if candidate.architecture_id in excluded or candidate.architecture_id in seen_ids:
            return
        population.append(candidate)
        seen_ids.add(candidate.architecture_id)

    for seed in seeds:
        gene = _candidate_gene(seed, gene_key)
        if gene is not None and len(gene) == len(options):
            if gene in candidate_by_gene:
                add(candidate_by_gene[gene])
            for index, choices in enumerate(options):
                for choice in choices:
                    if choice == gene[index]:
                        continue
                    neighbor = list(gene)
                    neighbor[index] = choice
                    child = candidate_by_gene.get(tuple(neighbor))
                    if child is not None:
                        add(child)
                        if len(population) >= population_target:
                            break
                if len(population) >= population_target:
                    break
        if len(population) >= population_target:
            break
    for candidate in sorted(usable, key=lambda item: (item.canonical_arch_hash, item.architecture_id)):
        if len(population) >= population_target:
            break
        add(candidate)

    if not population:
        population = rank(usable)[:population_target]

    for _generation in range(max(0, int(generations))):
        ranked = rank(population)
        next_population = ranked[: max(1, min(int(elite_count), len(ranked)))]
        next_ids = {candidate.architecture_id for candidate in next_population}
        parent_pool = ranked[: max(2, min(len(ranked), len(ranked) // 2 or 1))]
        attempts = 0
        while len(next_population) < population_target and attempts < population_target * 20:
            attempts += 1
            left = rng.choice(parent_pool)
            right = rng.choice(parent_pool)
            left_gene = genes_by_id.get(left.architecture_id)
            right_gene = genes_by_id.get(right.architecture_id)
            if left_gene is None or right_gene is None:
                continue
            child_gene = _crossover_gene(left_gene, right_gene, rng, crossover_rate)
            child_gene = _mutate_gene(child_gene, options, rng, mutation_rate)
            child = candidate_by_gene.get(child_gene)
            if child is not None and child.architecture_id not in next_ids:
                next_population.append(child)
                next_ids.add(child.architecture_id)
        if len(next_population) < population_target:
            for candidate in rank(usable):
                if len(next_population) >= population_target:
                    break
                if candidate.architecture_id not in next_ids:
                    next_population.append(candidate)
                    next_ids.add(candidate.architecture_id)
        population = next_population

    return rank(population)[: int(limit)]


def _parse_int_from_depth_group(depth_group: str, default: int = 1) -> int:
    text = str(depth_group).strip().upper()
    if text.startswith("L"):
        text = text[1:]
    try:
        return max(0, int(text))
    except ValueError:
        return int(default)


def _candidate_n_qubits(candidate: CandidateRecord) -> int:
    for key in ("n_qubits", "num_qubits"):
        raw = candidate.metadata.get(key)
        if raw not in (None, ""):
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    mask = _candidate_gene(candidate, "hea_mask")
    if mask:
        try:
            return max(1, int(mask[0]))
        except (TypeError, ValueError):
            pass
    text = str(candidate.architecture_id)
    if text and text[0].isdigit() and "q" in text[:3]:
        try:
            return max(1, int(text.split("q", 1)[0]))
        except ValueError:
            pass
    return 4


def _topology_edges(n_qubits: int, topology: str) -> tuple[tuple[int, int], ...]:
    if n_qubits < 2:
        return ()
    normalized = str(topology).strip().lower()
    edges = [(index, index + 1) for index in range(n_qubits - 1)]
    if normalized == "ring" and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return tuple(edges)


def _candidate_mog_vqe_individual(candidate: CandidateRecord) -> MOGVQEIndividual:
    raw = candidate.metadata.get("mog_vqe_blocks")
    if raw is None:
        raw = candidate.metadata.get("mog_blocks")
    n_qubits = _candidate_n_qubits(candidate)
    if isinstance(raw, str):
        stripped = raw.strip()
        raw = json.loads(stripped) if stripped else None
    blocks: list[MOGVQEBlock] = []
    if raw:
        for item in raw:
            if isinstance(item, Mapping):
                blocks.append(
                    MOGVQEBlock(
                        int(item["control"]),
                        int(item["target"]),
                        str(item.get("block_type", "generalized_cnot")),
                    )
                )
            else:
                values = list(item)
                block_type = values[2] if len(values) > 2 else "generalized_cnot"
                blocks.append(MOGVQEBlock(int(values[0]), int(values[1]), str(block_type)))
    if not blocks:
        layers = _parse_int_from_depth_group(candidate.depth_group, default=1)
        for _layer in range(max(1, layers)):
            for control, target in _topology_edges(n_qubits, candidate.topology):
                blocks.append(MOGVQEBlock(control, target, "generalized_cnot"))
    return MOGVQEIndividual(
        n_qubits=n_qubits,
        blocks=tuple(blocks),
        metadata={
            "architecture_id": candidate.architecture_id,
            "canonical_arch_hash": candidate.canonical_arch_hash,
            "source": "qas_candidate_record",
        },
    )


def select_latest_ea_candidates(
    candidates: Sequence[CandidateRecord],
    *,
    seeds: Sequence[CandidateRecord] = (),
    excluded_ids: Iterable[str] = (),
    fitness: Callable[[CandidateRecord], float],
    population_size: int = 32,
    generations: int = 12,
    mutation_rate: float = 0.20,
    crossover_rate: float = 0.60,
    elite_count: int = 4,
    limit: int = 16,
    random_seed: int = 0,
    gene_key: str = "hea_mask",
) -> tuple[list[CandidateRecord], str]:
    """Select Stage-2 EA proposals with the latest MoG-VQE NSGA-II backend.

    Stage 2 already uses the local oracle as a cheap energy proxy, so this
    adapter does not call ``run_mog_vqe`` or run fresh VQE inside planning.
    It wraps each candidate as a ``MOGVQECandidate`` and uses MoG-VQE's
    NSGA-II selection on ``(oracle_energy_proxy, CNOT count)``.
    """

    if int(limit) <= 0:
        return [], "mog_vqe_nsga2_pool"
    excluded = {str(identifier) for identifier in excluded_ids}
    usable = [
        candidate
        for candidate in candidates
        if candidate.architecture_id not in excluded
    ]
    if not usable:
        return [], "mog_vqe_nsga2_pool"

    def rank(items: Sequence[CandidateRecord]) -> list[CandidateRecord]:
        dedup = {candidate.architecture_id: candidate for candidate in items}
        return sorted(
            dedup.values(),
            key=lambda candidate: (
                -float(fitness(candidate)),
                candidate.canonical_arch_hash,
                candidate.architecture_id,
            ),
        )

    population_target = max(1, min(int(population_size), len(usable)))
    by_id = {candidate.architecture_id: candidate for candidate in usable}
    mog_population: list[MOGVQECandidate] = []
    for candidate in usable:
        individual = _candidate_mog_vqe_individual(candidate)
        parameters = np.zeros(individual.parameter_count, dtype=float)
        mog_population.append(
            MOGVQECandidate(
                individual=individual,
                energy=-float(fitness(candidate)),
                cnot_count=max(0, int(round(candidate.two_q_count or individual.cnot_count))),
                parameters=parameters,
                circuit=individual.to_circuit(parameters),
                metadata={"architecture_id": candidate.architecture_id},
            )
        )

    selected = nsga_ii_select(mog_population, population_target)
    selected_ids = [
        str(item.metadata["architecture_id"])
        for item in sorted(
            selected,
            key=lambda item: (
                int(item.rank),
                -float(item.crowding_distance),
                float(item.energy),
                int(item.cnot_count),
                str(item.metadata["architecture_id"]),
            ),
        )
    ]
    selected_records = [by_id[architecture_id] for architecture_id in selected_ids if architecture_id in by_id]
    if len(selected_records) < int(limit):
        selected_ids_set = {record.architecture_id for record in selected_records}
        selected_records.extend(
            candidate
            for candidate in rank(usable)
            if candidate.architecture_id not in selected_ids_set
        )
    return selected_records[: int(limit)], "mog_vqe_nsga2_pool"


def compute_abstain_rate(track_a_mutations: Iterable[Mapping[str, Any]]) -> float:
    """Compute abstain rate over current Track-A mutated candidates after dedup."""

    dedup: dict[str, bool] = {}
    for row in track_a_mutations:
        identifier = str(row.get("architecture_id") or row.get("canonical_arch_hash") or len(dedup))
        dedup[identifier] = bool(row.get("abstain", False))
    if not dedup:
        return 0.0
    return sum(1 for abstain in dedup.values() if abstain) / float(len(dedup))


def next_label_status_after_failure(
    *,
    retry_count: int,
    max_retry: int = int(DEFAULT_RETRY_POLICY["max_retry"]),
) -> LabelStatus:
    """Return retryable/nonretryable state after a failed run."""

    return LabelStatus.FAILED_NONRETRYABLE if int(retry_count) >= int(max_retry) else LabelStatus.FAILED_RETRYABLE

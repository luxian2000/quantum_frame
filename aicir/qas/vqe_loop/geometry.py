"""Trust-region geometry, split sampling, and Hamiltonian features.

These helpers define what it means for two QAS candidates or Hamiltonian tasks
to be neighbors; they do not run VQE or choose the next online-search batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import ceil, isfinite
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol import DEFAULT_TRUST_REGION_RULES, LabelSource

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


def _candidate_hamiltonian_id(candidate: CandidateRecord) -> str:
    """Return the task identity used by Hamiltonian-aware distance rules."""

    hamiltonian_id = str(candidate.hamiltonian_id or "").strip()
    if hamiltonian_id:
        return hamiltonian_id
    n_qubits = candidate.metadata.get("n_qubits", "")
    if n_qubits in ("", None):
        return ""
    hamiltonian_class = str(candidate.hamiltonian_class or "").strip()
    return f"{hamiltonian_class}_{n_qubits}q" if hamiltonian_class else ""


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



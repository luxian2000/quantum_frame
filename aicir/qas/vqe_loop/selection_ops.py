"""Stage-2 batch-selection operators for VQE-QAS.

This is not a pluggable planner registry.  It contains the default selection
operators used by the Stage-2 loop: MoG-EA/NSGA-II local proposals,
trust-region abstain summaries, and Track-B farthest-first expansion choices.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..algorithms.mogvqe import MOGVQEBlock, MOGVQECandidate, MOGVQEIndividual, nsga_ii_select
from .geometry import CandidateRecord, DistanceScales, min_distance_to_set

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
    mask = candidate.metadata.get("hea_mask")
    if isinstance(mask, str):
        try:
            mask = json.loads(mask)
        except json.JSONDecodeError:
            mask = None
    if isinstance(mask, (list, tuple)) and mask:
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


def select_mog_ea_candidates(
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
    """Select Stage-2 proposals with MoG-VQE NSGA-II ranking.

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



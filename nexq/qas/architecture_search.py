"""Two-stage architecture-search orchestration for QAS."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..channel.backends.base import Backend
from ..channel.noise.model import NoiseModel
from ..metrics.hardware import HardwareProfile
from ._types import ArchitectureSpec, SearchConfig, SearchResult
from .candidates import build_common_architectures
from .evaluator import ArchitectureEvaluator
from .reward import RewardWeights
from .search_strategies import (
    candidate_from_supercircuit_mask,
    crossover_supercircuit_masks,
    generate_progressive_supercircuit_subcircuits,
    generate_supercircuit_subcircuits,
    is_valid_supercircuit_mask,
    mutate_supercircuit_mask,
    random_supercircuit_mask,
    sample_supercircuit_masks,
    supercircuit_blocks,
)


class ArchitectureSearch:
    """Generate candidates, then evaluate and rank them with orthogonal metrics."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        hardware_profile: Optional[HardwareProfile] = None,
        weights: Optional[RewardWeights] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
    ):
        self.backend = backend
        self.noise_model = noise_model
        self.hardware_profile = hardware_profile
        self.weights = weights or RewardWeights()
        self.evaluator = evaluator

    def generate_candidates(
        self,
        config: Optional[SearchConfig] = None,
        extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
    ) -> List[ArchitectureSpec]:
        cfg = config or SearchConfig()
        candidates: List[ArchitectureSpec] = []
        if cfg.include_common_candidates:
            candidates.extend(
                build_common_architectures(
                    n_qubits=cfg.n_qubits,
                    layers=cfg.candidate_layers,
                    backend=self.backend,
                )
            )
        if extra_candidates:
            candidates.extend(extra_candidates)
        if cfg.search_strategy == "supercircuit":
            candidates.extend(generate_supercircuit_subcircuits(cfg, backend=self.backend))
        elif cfg.search_strategy == "supercircuit_progressive":
            candidates.extend(
                generate_progressive_supercircuit_subcircuits(
                    cfg,
                    backend=self.backend,
                    hardware_profile=self.hardware_profile,
                )
            )
        elif cfg.search_strategy != "preset":
            raise ValueError(f"Unsupported search_strategy: {cfg.search_strategy!r}")
        candidates = self._filter_candidates(candidates, cfg)
        if cfg.candidate_budget is not None:
            candidates = candidates[: max(0, int(cfg.candidate_budget))]
        return candidates

    def _filter_candidates(self, candidates: Sequence[ArchitectureSpec], cfg: SearchConfig) -> List[ArchitectureSpec]:
        allowed_gates = None if cfg.allowed_gates is None else {str(gate) for gate in cfg.allowed_gates}
        topology = None if cfg.topology is None else {tuple(sorted(edge)) for edge in cfg.topology}
        filtered: List[ArchitectureSpec] = []
        for candidate in candidates:
            if cfg.max_depth is not None and candidate.n_gates > cfg.max_depth:
                continue
            if cfg.max_parameters is not None and candidate.parameter_count > cfg.max_parameters:
                continue
            if cfg.max_two_qubit_gates is not None and candidate.two_qubit_gate_count > cfg.max_two_qubit_gates:
                continue
            if allowed_gates is not None and any(gate.get("type") not in allowed_gates for gate in candidate.circuit.gates):
                continue
            if topology is not None and not self._matches_topology(candidate, topology):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _matches_topology(candidate: ArchitectureSpec, topology: set[tuple[int, int]]) -> bool:
        for gate in candidate.circuit.gates:
            edge = None
            if "control_qubits" in gate and "target_qubit" in gate:
                controls = list(gate.get("control_qubits", []))
                if len(controls) == 1:
                    edge = tuple(sorted((int(controls[0]), int(gate["target_qubit"]))))
            elif "qubit_1" in gate and "qubit_2" in gate:
                edge = tuple(sorted((int(gate["qubit_1"]), int(gate["qubit_2"]))))
            if edge is not None and edge not in topology:
                return False
        return True

    def evaluate_candidates(
        self,
        candidates: Sequence[ArchitectureSpec],
        config: Optional[SearchConfig] = None,
    ):
        cfg = config or SearchConfig()
        evaluator = self.evaluator or ArchitectureEvaluator(
            backend=self.backend,
            noise_model=self.noise_model,
            hardware_profile=self.hardware_profile,
            weights=self.weights,
            n_samples=cfg.n_samples,
            active_metrics=cfg.active_metrics,
        )
        return evaluator.evaluate_many(candidates)

    def run(
        self,
        config: Optional[SearchConfig] = None,
        extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
    ) -> SearchResult:
        cfg = config or SearchConfig()
        if cfg.search_strategy == "supercircuit_evolution":
            return self._run_supercircuit_evolution(cfg, extra_candidates=extra_candidates)
        candidates = self.generate_candidates(cfg, extra_candidates=extra_candidates)
        scores = self.evaluate_candidates(candidates, cfg)
        if cfg.top_k is not None:
            scores = scores[: max(0, int(cfg.top_k))]
        return SearchResult(
            candidates=candidates,
            scores=scores,
            metadata={
                "stage_1": "candidate_generation",
                "stage_2": "orthogonal_evaluation",
                "n_candidates": len(candidates),
                "top_k": cfg.top_k,
                "search_strategy": cfg.search_strategy,
            },
        )

    def _run_supercircuit_evolution(
        self,
        cfg: SearchConfig,
        extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
    ) -> SearchResult:
        blocks = supercircuit_blocks(cfg.candidate_layers)
        rng = np.random.default_rng(int(cfg.seed))
        population_size = max(1, int(cfg.population_size))
        generations = max(1, int(cfg.search_generations))
        elite_count = max(1, min(population_size, int(cfg.beam_width)))
        seen: set[tuple[int, ...]] = set()
        masks = sample_supercircuit_masks(cfg, sample_count=population_size)
        seen.update(masks)
        all_scores = []
        all_candidates_by_mask: Dict[tuple[int, ...], ArchitectureSpec] = {}

        for generation in range(generations):
            candidates = [
                candidate_from_supercircuit_mask(
                    cfg,
                    mask,
                    backend=self.backend,
                    generation=generation,
                    origin="initial" if generation == 0 else "evolved",
                )
                for mask in masks
            ]
            scores = self.evaluate_candidates(candidates, cfg)
            for rank, score in enumerate(scores, start=1):
                score.architecture.metadata["evolution_generation_rank"] = rank
                score.architecture.metadata["evolution_weighted_score"] = score.weighted_score
                mask = tuple(score.architecture.metadata["supercircuit_mask"])
                all_candidates_by_mask[mask] = score.architecture
            all_scores.extend(scores)
            elites = [tuple(score.architecture.metadata["supercircuit_mask"]) for score in scores[:elite_count]]
            if generation == generations - 1:
                break

            next_masks = list(elites)
            attempts = 0
            max_attempts = max(100, population_size * 30)
            while len(next_masks) < population_size and attempts < max_attempts:
                attempts += 1
                parent = elites[int(rng.integers(0, len(elites)))]
                if len(elites) > 1 and rng.random() < 0.5:
                    other = elites[int(rng.integers(0, len(elites)))]
                    child = crossover_supercircuit_masks(parent, other, rng)
                else:
                    child = parent
                child = mutate_supercircuit_mask(child, blocks, rng, mutation_rate=cfg.mutation_rate)
                if child in seen or not is_valid_supercircuit_mask(child, blocks):
                    continue
                seen.add(child)
                next_masks.append(child)
            while len(next_masks) < population_size:
                mask = random_supercircuit_mask(blocks, rng)
                if mask in seen:
                    mask = mutate_supercircuit_mask(mask, blocks, rng, mutation_rate=max(cfg.mutation_rate, 0.5))
                if mask in seen:
                    break
                seen.add(mask)
                next_masks.append(mask)
            masks = next_masks

        unique_scores = {}
        for score in all_scores:
            mask = tuple(score.architecture.metadata["supercircuit_mask"])
            previous = unique_scores.get(mask)
            if previous is None or score.weighted_score > previous.weighted_score:
                unique_scores[mask] = score
        scores = sorted(unique_scores.values(), key=lambda score: score.weighted_score, reverse=True)
        for rank, score in enumerate(scores, start=1):
            score.rank = rank
        if cfg.top_k is not None:
            scores = scores[: max(0, int(cfg.top_k))]
        candidates = [all_candidates_by_mask[mask] for mask in unique_scores]
        if extra_candidates:
            candidates.extend(extra_candidates)
        return SearchResult(
            candidates=candidates,
            scores=scores,
            metadata={
                "stage_1": "evolutionary_supercircuit_generation",
                "stage_2": "orthogonal_evaluation",
                "n_candidates": len(candidates),
                "top_k": cfg.top_k,
                "search_strategy": cfg.search_strategy,
                "search_generations": generations,
                "population_size": population_size,
                "elite_count": elite_count,
            },
        )

NoiseAdaptiveQAS = ArchitectureSearch

__all__ = ["ArchitectureSearch", "NoiseAdaptiveQAS"]


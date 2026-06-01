"""Two-stage architecture-search orchestration for QAS."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..channel.backends.base import Backend
from ..channel.noise.model import NoiseModel
from ._types import ArchitectureSpec, SearchConfig, SearchResult
from .candidates import build_common_architectures
from .evaluator import ArchitectureEvaluator
from .reward import RewardWeights


class ArchitectureSearch:
    """Generate candidates, then evaluate and rank them with orthogonal metrics."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        weights: Optional[RewardWeights] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
    ):
        self.backend = backend
        self.noise_model = noise_model
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
        return candidates

    def evaluate_candidates(
        self,
        candidates: Sequence[ArchitectureSpec],
        config: Optional[SearchConfig] = None,
    ):
        cfg = config or SearchConfig()
        evaluator = self.evaluator or ArchitectureEvaluator(
            backend=self.backend,
            noise_model=self.noise_model,
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
        candidates = self.generate_candidates(cfg, extra_candidates=extra_candidates)
        scores = self.evaluate_candidates(candidates, cfg)
        return SearchResult(
            candidates=candidates,
            scores=scores,
            metadata={
                "stage_1": "candidate_generation",
                "stage_2": "orthogonal_evaluation",
                "n_candidates": len(candidates),
            },
        )

NoiseAdaptiveQAS = ArchitectureSearch

__all__ = ["ArchitectureSearch", "NoiseAdaptiveQAS"]


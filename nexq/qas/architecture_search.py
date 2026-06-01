"""Two-stage architecture-search orchestration for QAS."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..channel.backends.base import Backend
from ..channel.noise.model import NoiseModel
from ..metrics.hardware import HardwareProfile
from ._types import ArchitectureSpec, SearchConfig, SearchResult
from .candidates import build_common_architectures
from .evaluator import ArchitectureEvaluator
from .reward import RewardWeights
from .search_strategies import generate_supercircuit_subcircuits


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

NoiseAdaptiveQAS = ArchitectureSearch

__all__ = ["ArchitectureSearch", "NoiseAdaptiveQAS"]


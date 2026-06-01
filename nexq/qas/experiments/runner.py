"""End-to-end QAS validation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ...channel.backends.numpy_backend import NumpyBackend
from ...channel.noise.model import NoiseModel
from .._types import ArchitectureScore, ArchitectureSpec, SearchConfig
from ..architecture_candidates import build_common_architectures
from ..architecture_search import ArchitectureSearch
from ..problems import ProblemInstance
from ..task_evaluation import OptimizerConfig, TaskEvaluationResult, optimize_task_parameters


@dataclass
class ValidationReport:
    """Task-level comparison report for baselines and QAS-selected circuits."""

    problem: ProblemInstance
    baseline_results: List[TaskEvaluationResult]
    qas_results: List[TaskEvaluationResult]
    prior_scores: List[ArchitectureScore] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_results(self) -> List[TaskEvaluationResult]:
        return list(self.baseline_results) + list(self.qas_results)

    @property
    def best_result(self) -> Optional[TaskEvaluationResult]:
        results = self.all_results
        if not results:
            return None
        key = lambda result: result.optimized_value
        return max(results, key=key) if self.problem.maximize else min(results, key=key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "baseline_results": [result.to_dict() for result in self.baseline_results],
            "qas_results": [result.to_dict() for result in self.qas_results],
            "prior_scores": [score.to_dict() for score in self.prior_scores],
            "metadata": dict(self.metadata),
            "best_result": None if self.best_result is None else self.best_result.to_dict(),
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"problem: {self.problem.name}",
            f"classical_optimum: {self.problem.classical_optimum:.6f}",
            "name | prior | optimized | ideal | noisy | ratio | gap",
        ]
        for result in self.all_results:
            prior = "-" if result.prior_score is None else f"{result.prior_score:.4f}"
            noisy = "-" if result.noisy_value is None else f"{result.noisy_value:.6f}"
            ratio = "-" if result.approximation_ratio is None else f"{result.approximation_ratio:.4f}"
            gap = "-" if result.normalized_gap is None else f"{result.normalized_gap:.4f}"
            lines.append(
                f"{result.architecture_name} | {prior} | {result.optimized_value:.6f} | "
                f"{result.ideal_value:.6f} | {noisy} | {ratio} | {gap}"
            )
        best = self.best_result
        if best is not None:
            lines.append(f"best: {best.architecture_name} ({best.optimized_value:.6f})")
        return lines


def _baseline_architectures(problem: ProblemInstance, layers: int, backend: NumpyBackend) -> List[ArchitectureSpec]:
    return build_common_architectures(
        n_qubits=problem.n_qubits,
        layers=layers,
        backend=backend,
        names=["qaoa_chain", "hea_linear", "real_amplitudes_linear"],
    )


def _optimize_many(
    architectures: Sequence[ArchitectureSpec],
    problem: ProblemInstance,
    optimizer_config: OptimizerConfig,
    backend: NumpyBackend,
    noise_model: Optional[NoiseModel],
    prior_by_name: Optional[Dict[str, float]] = None,
) -> List[TaskEvaluationResult]:
    prior_by_name = prior_by_name or {}
    return [
        optimize_task_parameters(
            architecture,
            problem,
            config=optimizer_config,
            backend=backend,
            noise_model=noise_model,
            prior_score=prior_by_name.get(architecture.name),
        )
        for architecture in architectures
    ]


def run_validation_experiment(
    problem: ProblemInstance,
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
) -> ValidationReport:
    """Rank candidates by QAS priors, then compare optimized task objectives."""
    backend = backend or NumpyBackend()
    search_cfg = search_config or SearchConfig(n_qubits=problem.n_qubits, candidate_layers=1, n_samples=16)
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=32)

    baselines = _baseline_architectures(problem, layers=search_cfg.candidate_layers, backend=backend)
    search = ArchitectureSearch(backend=backend, noise_model=noise_model)
    search_result = search.run(search_cfg, extra_candidates=extra_candidates)
    prior_scores = search_result.scores
    top_scores = prior_scores[: max(0, int(qas_top_k))]
    qas_architectures = [score.architecture for score in top_scores]
    prior_by_name = {score.architecture.name: score.weighted_score for score in top_scores}

    baseline_results = _optimize_many(
        baselines,
        problem,
        optimizer_cfg,
        backend=backend,
        noise_model=noise_model,
    )
    qas_results = _optimize_many(
        qas_architectures,
        problem,
        optimizer_cfg,
        backend=backend,
        noise_model=noise_model,
        prior_by_name=prior_by_name,
    )

    return ValidationReport(
        problem=problem,
        baseline_results=baseline_results,
        qas_results=qas_results,
        prior_scores=prior_scores,
        metadata={
            "qas_top_k": qas_top_k,
            "n_prior_candidates": len(prior_scores),
            "search_config": search_cfg.__dict__.copy(),
            "optimizer_config": optimizer_cfg.__dict__.copy(),
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
        },
    )


__all__ = ["ValidationReport", "run_validation_experiment"]

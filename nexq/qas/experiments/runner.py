"""End-to-end QAS validation runner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict, List, Optional, Sequence

from ...channel.backends.numpy_backend import NumpyBackend
from ...channel.noise.model import NoiseModel
from ...metrics.hardware import HardwareProfile
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


@dataclass
class MultiSeedValidationReport:
    """Aggregated validation report over repeated optimizer seeds."""

    reports: List[ValidationReport]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def problem(self) -> Optional[ProblemInstance]:
        return None if not self.reports else self.reports[0].problem

    @property
    def n_seeds(self) -> int:
        return len(self.reports)

    def architecture_summary(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple[str, str], List[TaskEvaluationResult]] = {}
        wins: Dict[str, int] = {}
        for report in self.reports:
            best = report.best_result
            if best is not None:
                best_group = str(best.metadata.get("result_group", "unknown"))
                best_key = f"{best_group}:{best.architecture_name}"
                wins[best_key] = wins.get(best_key, 0) + 1
            for result in report.all_results:
                result_group = str(result.metadata.get("result_group", "unknown"))
                grouped.setdefault((result_group, result.architecture_name), []).append(result)

        rows: List[Dict[str, Any]] = []
        for (result_group, name), results in grouped.items():
            values = [float(result.optimized_value) for result in results]
            ratios = [
                float(result.approximation_ratio)
                for result in results
                if result.approximation_ratio is not None
            ]
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            mean_ratio = None if not ratios else sum(ratios) / len(ratios)
            win_key = f"{result_group}:{name}"
            rows.append(
                {
                    "result_group": result_group,
                    "architecture_name": name,
                    "runs": len(results),
                    "mean_optimized": mean_value,
                    "std_optimized": math.sqrt(variance),
                    "best_optimized": max(values),
                    "worst_optimized": min(values),
                    "mean_ratio": mean_ratio,
                    "win_count": wins.get(win_key, 0),
                    "win_rate": wins.get(win_key, 0) / max(1, self.n_seeds),
                }
            )

        problem = self.problem
        reverse = True if problem is None else problem.maximize
        return sorted(rows, key=lambda row: row["mean_optimized"], reverse=reverse)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "problem": None if self.problem is None else self.problem.to_dict(),
            "n_seeds": self.n_seeds,
            "architecture_summary": self.architecture_summary(),
            "reports": [report.to_dict() for report in self.reports],
        }

    def summary_lines(self) -> List[str]:
        problem = self.problem
        lines = [
            f"problem: {None if problem is None else problem.name}",
            f"n_seeds: {self.n_seeds}",
            "group | name | runs | mean | std | best | worst | mean_ratio | win_rate",
        ]
        for row in self.architecture_summary():
            ratio = "-" if row["mean_ratio"] is None else f"{row['mean_ratio']:.4f}"
            lines.append(
                f"{row['result_group']} | {row['architecture_name']} | {row['runs']} | "
                f"{row['mean_optimized']:.6f} | {row['std_optimized']:.6f} | "
                f"{row['best_optimized']:.6f} | {row['worst_optimized']:.6f} | "
                f"{ratio} | {row['win_rate']:.2f}"
            )
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
    result_group: str = "candidate",
) -> List[TaskEvaluationResult]:
    prior_by_name = prior_by_name or {}
    results = []
    for architecture in architectures:
        result = optimize_task_parameters(
            architecture,
            problem,
            config=optimizer_config,
            backend=backend,
            noise_model=noise_model,
            prior_score=prior_by_name.get(architecture.name),
        )
        result.metadata["result_group"] = result_group
        results.append(result)
    return results


def run_validation_experiment(
    problem: ProblemInstance,
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
) -> ValidationReport:
    """Rank candidates by QAS priors, then compare optimized task objectives."""
    backend = backend or NumpyBackend()
    search_cfg = search_config or SearchConfig(n_qubits=problem.n_qubits, candidate_layers=1, n_samples=16)
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=32)

    baselines = _baseline_architectures(problem, layers=search_cfg.candidate_layers, backend=backend)
    search = ArchitectureSearch(backend=backend, noise_model=noise_model, hardware_profile=hardware_profile)
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
        result_group="baseline",
    )
    qas_results = _optimize_many(
        qas_architectures,
        problem,
        optimizer_cfg,
        backend=backend,
        noise_model=noise_model,
        prior_by_name=prior_by_name,
        result_group="qas",
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
            "hardware_profile": None if hardware_profile is None else hardware_profile.__dict__.copy(),
        },
    )


def run_multi_seed_validation_experiment(
    problem: ProblemInstance,
    seeds: Sequence[int],
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    hardware_profile: Optional[HardwareProfile] = None,
    extra_candidates: Optional[Sequence[ArchitectureSpec]] = None,
) -> MultiSeedValidationReport:
    """Run the same validation setup across optimizer seeds and aggregate it."""
    optimizer_cfg = optimizer_config or OptimizerConfig()
    seed_list = [int(seed) for seed in seeds]
    reports = [
        run_validation_experiment(
            problem,
            search_config=search_config,
            optimizer_config=replace(optimizer_cfg, seed=seed),
            qas_top_k=qas_top_k,
            backend=backend,
            noise_model=noise_model,
            hardware_profile=hardware_profile,
            extra_candidates=extra_candidates,
        )
        for seed in seed_list
    ]
    return MultiSeedValidationReport(
        reports=reports,
        metadata={
            "seeds": seed_list,
            "qas_top_k": qas_top_k,
            "search_config": None if search_config is None else search_config.__dict__.copy(),
            "optimizer_config": optimizer_cfg.__dict__.copy(),
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
            "hardware_profile": None if hardware_profile is None else hardware_profile.__dict__.copy(),
        },
    )


__all__ = ["MultiSeedValidationReport", "ValidationReport", "run_multi_seed_validation_experiment", "run_validation_experiment"]

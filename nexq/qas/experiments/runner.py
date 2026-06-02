"""End-to-end QAS validation runner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...channel.backends.numpy_backend import NumpyBackend
from ...channel.noise.model import NoiseModel
from ...metrics.hardware import HardwareProfile
from .._types import ArchitectureScore, ArchitectureSpec, SearchConfig
from ..architecture_candidates import build_common_architectures
from ..architecture_search import ArchitectureSearch
from ..problems import ProblemInstance
from ..search_strategies import (
    candidate_from_supercircuit_mask,
    crossover_supercircuit_masks,
    is_valid_supercircuit_mask,
    mutate_supercircuit_mask,
    random_supercircuit_mask,
    reflection_from_architecture_score,
    reflective_mutate_supercircuit_mask,
    sample_supercircuit_masks,
    supercircuit_blocks,
)
from ..task_evaluation import (
    OptimizerConfig,
    TaskEvaluationResult,
    evaluate_task_objective,
    optimize_task_parameters,
    parameter_count,
)


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


@dataclass
class StrategyComparisonReport:
    """Comparison of multiple QAS search strategies under one task budget."""

    problem: ProblemInstance
    reports: Dict[str, ValidationReport]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "metadata": dict(self.metadata),
            "reports": {name: report.to_dict() for name, report in self.reports.items()},
            "strategy_summary": self.strategy_summary(),
        }

    def strategy_summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for strategy, report in self.reports.items():
            baseline_best = report.best_result
            if report.baseline_results:
                baseline_best = _sort_task_results(self.problem, report.baseline_results)[0]
            qas_best = None if not report.qas_results else _sort_task_results(self.problem, report.qas_results)[0]
            overall_best = report.best_result
            rows.append(
                {
                    "strategy": strategy,
                    "baseline_best": None if baseline_best is None else baseline_best.optimized_value,
                    "qas_best": None if qas_best is None else qas_best.optimized_value,
                    "overall_best": None if overall_best is None else overall_best.optimized_value,
                    "qas_best_name": None if qas_best is None else qas_best.architecture_name,
                    "overall_best_name": None if overall_best is None else overall_best.architecture_name,
                    "qas_best_ratio": None if qas_best is None else qas_best.approximation_ratio,
                    "n_prior_candidates": report.metadata.get("n_prior_candidates"),
                }
            )
        return rows

    def summary_lines(self) -> List[str]:
        lines = [
            f"problem: {self.problem.name}",
            "strategy | baseline_best | qas_best | qas_ratio | overall_best | best_name | n_prior",
        ]
        for row in self.strategy_summary():
            baseline_best = "-" if row["baseline_best"] is None else f"{row['baseline_best']:.6f}"
            qas_best = "-" if row["qas_best"] is None else f"{row['qas_best']:.6f}"
            qas_ratio = "-" if row["qas_best_ratio"] is None else f"{row['qas_best_ratio']:.4f}"
            overall_best = "-" if row["overall_best"] is None else f"{row['overall_best']:.6f}"
            lines.append(
                f"{row['strategy']} | {baseline_best} | {qas_best} | {qas_ratio} | "
                f"{overall_best} | {row['overall_best_name']} | {row['n_prior_candidates']}"
            )
        return lines


@dataclass
class RandomProxyValidationRow:
    """Random-parameter task proxy statistics for one architecture."""

    architecture_name: str
    n_parameters: int
    n_random_samples: int
    random_mean: float
    random_std: float
    random_best: float
    random_worst: float
    random_p10: float
    random_p90: float
    short_optimized: float
    short_evaluations: int
    approximation_ratio: Optional[float]
    normalized_gap: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "n_parameters": self.n_parameters,
            "n_random_samples": self.n_random_samples,
            "random_mean": self.random_mean,
            "random_std": self.random_std,
            "random_best": self.random_best,
            "random_worst": self.random_worst,
            "random_p10": self.random_p10,
            "random_p90": self.random_p90,
            "short_optimized": self.short_optimized,
            "short_evaluations": self.short_evaluations,
            "approximation_ratio": self.approximation_ratio,
            "normalized_gap": self.normalized_gap,
        }


@dataclass
class RandomProxyValidationReport:
    """Report for checking whether random-parameter objective is a useful proxy."""

    problem: ProblemInstance
    rows: List[RandomProxyValidationRow]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "metadata": dict(self.metadata),
            "correlations": self.correlations(),
        }

    def correlations(self) -> Dict[str, Optional[float]]:
        if len(self.rows) < 2:
            return {"pearson_best": None, "spearman_best": None, "pearson_mean": None}
        sign = 1.0 if self.problem.maximize else -1.0
        optimized = [sign * row.short_optimized for row in self.rows]
        best = [sign * row.random_best for row in self.rows]
        mean = [sign * row.random_mean for row in self.rows]
        return {
            "pearson_best": _pearson(best, optimized),
            "spearman_best": _spearman(best, optimized),
            "pearson_mean": _pearson(mean, optimized),
        }

    def summary_lines(self) -> List[str]:
        corr = self.correlations()
        lines = [
            f"problem: {self.problem.name}",
            f"n_architectures: {len(self.rows)}",
            (
                "correlation | pearson_random_best_vs_short | "
                "spearman_random_best_vs_short | pearson_random_mean_vs_short"
            ),
            (
                f"values | {_format_optional(corr['pearson_best'])} | "
                f"{_format_optional(corr['spearman_best'])} | {_format_optional(corr['pearson_mean'])}"
            ),
            "name | n_params | random_best | random_mean | random_std | short_optimized | ratio | gap",
        ]
        for row in self.rows:
            ratio = "-" if row.approximation_ratio is None else f"{row.approximation_ratio:.4f}"
            gap = "-" if row.normalized_gap is None else f"{row.normalized_gap:.4f}"
            lines.append(
                f"{row.architecture_name} | {row.n_parameters} | {row.random_best:.6f} | "
                f"{row.random_mean:.6f} | {row.random_std:.6f} | {row.short_optimized:.6f} | {ratio} | {gap}"
            )
        return lines


def _format_optional(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size != y.size or x.size < 2:
        return None
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and abs(indexed[end][1] - indexed[start][1]) <= 1e-12:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in range(start, end):
            ranks[indexed[index][0]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    return _pearson(_rankdata(left), _rankdata(right))


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


def run_random_proxy_validation_experiment(
    problem: ProblemInstance,
    architectures: Optional[Sequence[ArchitectureSpec]] = None,
    n_random_samples: int = 100,
    optimizer_config: Optional[OptimizerConfig] = None,
    random_seed: int = 1234,
    parameter_scale: float = 2.0 * np.pi,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
) -> RandomProxyValidationReport:
    """Check whether random-parameter task objective separates candidate architectures."""
    backend = backend or NumpyBackend()
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=32, seed=random_seed)
    candidates = list(architectures) if architectures is not None else build_common_architectures(
        n_qubits=problem.n_qubits,
        layers=2,
        backend=backend,
        names=[
            "hea_linear",
            "real_amplitudes_linear",
            "efficient_su2_ring",
            "qaoa_chain",
            "brickwork_cx",
            "mera_like",
        ],
    )
    rng = np.random.default_rng(int(random_seed))
    rows: List[RandomProxyValidationRow] = []
    for architecture in candidates:
        n_params = parameter_count(architecture.circuit)
        values: List[float] = []
        if n_params == 0:
            values.append(
                evaluate_task_objective(
                    architecture,
                    problem,
                    [],
                    backend=backend,
                    noise_model=noise_model,
                )
            )
        else:
            for _ in range(max(1, int(n_random_samples))):
                params = rng.uniform(-parameter_scale, parameter_scale, size=n_params)
                values.append(
                    evaluate_task_objective(
                        architecture,
                        problem,
                        params,
                        backend=backend,
                        noise_model=noise_model,
                    )
                )
        array = np.asarray(values, dtype=float)
        random_best = float(array.max() if problem.maximize else array.min())
        random_worst = float(array.min() if problem.maximize else array.max())
        short_result = optimize_task_parameters(
            architecture,
            problem,
            config=optimizer_cfg,
            backend=backend,
            noise_model=noise_model,
        )
        rows.append(
            RandomProxyValidationRow(
                architecture_name=architecture.name,
                n_parameters=n_params,
                n_random_samples=len(values),
                random_mean=float(array.mean()),
                random_std=float(array.std()),
                random_best=random_best,
                random_worst=random_worst,
                random_p10=float(np.percentile(array, 10)),
                random_p90=float(np.percentile(array, 90)),
                short_optimized=float(short_result.optimized_value),
                short_evaluations=int(short_result.evaluations),
                approximation_ratio=short_result.approximation_ratio,
                normalized_gap=short_result.normalized_gap,
            )
        )
    rows = sorted(rows, key=lambda row: row.short_optimized, reverse=problem.maximize)
    return RandomProxyValidationReport(
        problem=problem,
        rows=rows,
        metadata={
            "n_random_samples": n_random_samples,
            "random_seed": random_seed,
            "parameter_scale": parameter_scale,
            "optimizer_config": optimizer_cfg.__dict__.copy(),
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
        },
    )


def _sort_task_results(problem: ProblemInstance, results: Sequence[TaskEvaluationResult]) -> List[TaskEvaluationResult]:
    return sorted(results, key=lambda result: result.optimized_value, reverse=problem.maximize)


def _unique_masks(masks: Sequence[Sequence[int]]) -> List[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    unique: List[tuple[int, ...]] = []
    for mask in masks:
        mask_tuple = tuple(int(value) for value in mask)
        if mask_tuple in seen:
            continue
        seen.add(mask_tuple)
        unique.append(mask_tuple)
    return unique


def _zero_cost_evolve_from_masks(
    seed_masks: Sequence[Sequence[int]],
    search_cfg: SearchConfig,
    backend: NumpyBackend,
    noise_model: Optional[NoiseModel],
    hardware_profile: Optional[HardwareProfile],
) -> List[ArchitectureScore]:
    """Run zero-cost evolutionary refinement from explicit seed masks."""
    blocks = supercircuit_blocks(search_cfg.candidate_layers)
    rng = np.random.default_rng(int(search_cfg.seed) + 2718)
    population_size = max(1, int(search_cfg.population_size))
    generations = max(1, int(search_cfg.search_generations))
    elite_count = max(1, min(population_size, int(search_cfg.beam_width)))
    search = ArchitectureSearch(backend=backend, noise_model=noise_model, hardware_profile=hardware_profile)
    reflective = search_cfg.search_strategy == "supercircuit_reflective" or bool(search_cfg.reflective_mutation)
    current_masks = _unique_masks(seed_masks)[:population_size]
    seen = set(current_masks)
    reflection_by_mask: Dict[tuple[int, ...], dict] = {}
    while len(current_masks) < population_size:
        mask = random_supercircuit_mask(blocks, rng)
        if mask in seen:
            continue
        seen.add(mask)
        current_masks.append(mask)

    all_scores: Dict[tuple[int, ...], ArchitectureScore] = {}
    for generation in range(generations):
        candidates = [
            candidate_from_supercircuit_mask(
                search_cfg,
                mask,
                backend=backend,
                generation=generation,
                origin="hybrid_seed" if generation == 0 else "hybrid_evolved",
            )
            for mask in current_masks
        ]
        for candidate in candidates:
            mask = tuple(candidate.metadata["supercircuit_mask"])
            if mask in reflection_by_mask:
                candidate.metadata["reflection"] = reflection_by_mask[mask]
        scores = search.evaluate_candidates(candidates, search_cfg)
        for rank, score in enumerate(scores, start=1):
            score.architecture.metadata["hybrid_evolution_generation_rank"] = rank
            score.architecture.metadata["hybrid_evolution_score"] = score.weighted_score
            mask = tuple(score.architecture.metadata["supercircuit_mask"])
            previous = all_scores.get(mask)
            if previous is None or score.weighted_score > previous.weighted_score:
                all_scores[mask] = score
        elite_scores = scores[:elite_count]
        elites = [tuple(score.architecture.metadata["supercircuit_mask"]) for score in elite_scores]
        if generation == generations - 1:
            break

        next_masks = list(elites)
        attempts = 0
        while len(next_masks) < population_size and attempts < max(100, population_size * 30):
            attempts += 1
            parent_index = int(rng.integers(0, len(elites)))
            parent = elites[parent_index]
            if len(elites) > 1 and rng.random() < 0.5:
                other = elites[int(rng.integers(0, len(elites)))]
                child = crossover_supercircuit_masks(parent, other, rng)
            else:
                child = parent
            if reflective:
                reflection = reflection_from_architecture_score(elite_scores[parent_index])
                child = reflective_mutate_supercircuit_mask(
                    child,
                    blocks,
                    rng,
                    reflection,
                    mutation_rate=search_cfg.mutation_rate,
                    strength=search_cfg.reflection_strength,
                )
            else:
                reflection = None
                child = mutate_supercircuit_mask(child, blocks, rng, mutation_rate=search_cfg.mutation_rate)
            if child in seen or not is_valid_supercircuit_mask(child, blocks):
                continue
            seen.add(child)
            if reflection is not None:
                reflection_by_mask[child] = reflection
            next_masks.append(child)
        current_masks = next_masks

    scores = sorted(all_scores.values(), key=lambda score: score.weighted_score, reverse=True)
    for rank, score in enumerate(scores, start=1):
        score.rank = rank
    return scores


def _task_feedback_from_masks(
    seed_masks: Sequence[Sequence[int]],
    problem: ProblemInstance,
    search_cfg: SearchConfig,
    optimizer_cfg: OptimizerConfig,
    qas_top_k: int,
    feedback_generations: int,
    feedback_population_size: Optional[int],
    feedback_elite_count: Optional[int],
    backend: NumpyBackend,
    noise_model: Optional[NoiseModel],
) -> tuple[List[TaskEvaluationResult], int]:
    """Run task-feedback mutation from explicit SuperCircuit masks."""
    population_size = max(1, int(feedback_population_size or search_cfg.population_size))
    elite_count = max(1, min(population_size, int(feedback_elite_count or search_cfg.beam_width or qas_top_k)))
    generations = max(1, int(feedback_generations))
    blocks = supercircuit_blocks(search_cfg.candidate_layers)
    rng = np.random.default_rng(int(search_cfg.seed) + 9173)
    current_masks = _unique_masks(seed_masks)[:population_size]
    seen = set(current_masks)
    while len(current_masks) < population_size:
        mask = random_supercircuit_mask(blocks, rng)
        if mask in seen:
            continue
        seen.add(mask)
        current_masks.append(mask)

    task_results: List[TaskEvaluationResult] = []
    result_by_mask: Dict[tuple[int, ...], TaskEvaluationResult] = {}
    for generation in range(generations):
        candidates = [
            candidate_from_supercircuit_mask(
                search_cfg,
                mask,
                backend=backend,
                generation=generation,
                origin="hybrid_task_feedback_seed" if generation == 0 else "hybrid_task_feedback_mutation",
            )
            for mask in current_masks
        ]
        generation_results = _optimize_many(
            candidates,
            problem,
            optimizer_cfg,
            backend=backend,
            noise_model=noise_model,
            result_group="qas_hybrid",
        )
        for candidate, result in zip(candidates, generation_results):
            mask = tuple(candidate.metadata["supercircuit_mask"])
            result.metadata.update(
                {
                    "feedback_generation": generation,
                    "supercircuit_mask": mask,
                    "search_origin": candidate.metadata.get("search_origin"),
                }
            )
            previous = result_by_mask.get(mask)
            is_better = previous is None or (
                result.optimized_value > previous.optimized_value
                if problem.maximize
                else result.optimized_value < previous.optimized_value
            )
            if is_better:
                result_by_mask[mask] = result
        task_results.extend(generation_results)
        if generation == generations - 1:
            break
        elites = [
            tuple(result.metadata["supercircuit_mask"])
            for result in _sort_task_results(problem, generation_results)[:elite_count]
        ]
        next_masks = list(elites)
        attempts = 0
        while len(next_masks) < population_size and attempts < max(100, population_size * 30):
            attempts += 1
            parent = elites[int(rng.integers(0, len(elites)))]
            child = mutate_supercircuit_mask(parent, blocks, rng, mutation_rate=search_cfg.mutation_rate)
            if child in seen:
                continue
            seen.add(child)
            next_masks.append(child)
        current_masks = next_masks
    return _sort_task_results(problem, list(result_by_mask.values()))[: max(0, int(qas_top_k))], len(task_results)


def run_task_feedback_validation_experiment(
    problem: ProblemInstance,
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    feedback_generations: int = 2,
    feedback_population_size: Optional[int] = None,
    feedback_elite_count: Optional[int] = None,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    hardware_profile: Optional[HardwareProfile] = None,
) -> ValidationReport:
    """Task-feedback SuperCircuit search after zero-cost candidate filtering.

    This is the P1.3 route: unlike zero-cost QAS, it deliberately uses a small,
    fixed task-optimization budget as feedback for mutating SubCircuit masks.
    """
    backend = backend or NumpyBackend()
    search_cfg = search_config or SearchConfig(
        n_qubits=problem.n_qubits,
        candidate_layers=1,
        n_samples=16,
        search_strategy="supercircuit_evolution",
    )
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=16)
    population_size = max(1, int(feedback_population_size or search_cfg.population_size))
    elite_count = max(1, min(population_size, int(feedback_elite_count or search_cfg.beam_width or qas_top_k)))
    generations = max(1, int(feedback_generations))
    blocks = supercircuit_blocks(search_cfg.candidate_layers)
    rng = np.random.default_rng(int(search_cfg.seed) + 9173)

    baselines = _baseline_architectures(problem, layers=search_cfg.candidate_layers, backend=backend)
    search = ArchitectureSearch(backend=backend, noise_model=noise_model, hardware_profile=hardware_profile)
    zero_cost_result = search.run(search_cfg)
    prior_scores = zero_cost_result.scores
    seed_masks = [
        tuple(score.architecture.metadata["supercircuit_mask"])
        for score in prior_scores
        if "supercircuit_mask" in score.architecture.metadata
    ]
    if not seed_masks:
        seed_masks = sample_supercircuit_masks(search_cfg, sample_count=population_size)
    seed_masks = seed_masks[:population_size]

    seen = set(seed_masks)
    current_masks = list(seed_masks)
    task_results: List[TaskEvaluationResult] = []
    result_by_mask: Dict[tuple[int, ...], TaskEvaluationResult] = {}

    for generation in range(generations):
        candidates = [
            candidate_from_supercircuit_mask(
                search_cfg,
                mask,
                backend=backend,
                generation=generation,
                origin="task_feedback_seed" if generation == 0 else "task_feedback_mutation",
            )
            for mask in current_masks
        ]
        generation_results = _optimize_many(
            candidates,
            problem,
            optimizer_cfg,
            backend=backend,
            noise_model=noise_model,
            result_group="qas_task_feedback",
        )
        for candidate, result in zip(candidates, generation_results):
            mask = tuple(candidate.metadata["supercircuit_mask"])
            result.metadata.update(
                {
                    "feedback_generation": generation,
                    "supercircuit_mask": mask,
                    "search_origin": candidate.metadata.get("search_origin"),
                }
            )
            previous = result_by_mask.get(mask)
            is_better = previous is None or (
                result.optimized_value > previous.optimized_value
                if problem.maximize
                else result.optimized_value < previous.optimized_value
            )
            if is_better:
                result_by_mask[mask] = result
        task_results.extend(generation_results)

        if generation == generations - 1:
            break
        elites = [
            tuple(result.metadata["supercircuit_mask"])
            for result in _sort_task_results(problem, generation_results)[:elite_count]
        ]
        next_masks = list(elites)
        attempts = 0
        max_attempts = max(100, population_size * 30)
        while len(next_masks) < population_size and attempts < max_attempts:
            attempts += 1
            parent = elites[int(rng.integers(0, len(elites)))]
            child = mutate_supercircuit_mask(parent, blocks, rng, mutation_rate=search_cfg.mutation_rate)
            if child in seen:
                continue
            seen.add(child)
            next_masks.append(child)
        current_masks = next_masks

    baseline_results = _optimize_many(
        baselines,
        problem,
        optimizer_cfg,
        backend=backend,
        noise_model=noise_model,
        result_group="baseline",
    )
    qas_results = _sort_task_results(problem, list(result_by_mask.values()))[: max(0, int(qas_top_k))]
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
            "feedback_generations": generations,
            "feedback_population_size": population_size,
            "feedback_elite_count": elite_count,
            "task_feedback_evaluated": len(task_results),
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
            "hardware_profile": None if hardware_profile is None else hardware_profile.__dict__.copy(),
        },
    )


def run_hybrid_qas_validation_experiment(
    problem: ProblemInstance,
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    progressive_keep: Optional[int] = None,
    feedback_generations: int = 2,
    feedback_population_size: Optional[int] = None,
    feedback_elite_count: Optional[int] = None,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    hardware_profile: Optional[HardwareProfile] = None,
) -> ValidationReport:
    """Run progressive prefilter -> zero-cost evolution -> task-feedback refinement."""
    backend = backend or NumpyBackend()
    base_cfg = search_config or SearchConfig(n_qubits=problem.n_qubits, candidate_layers=1, n_samples=16)
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=16)
    search = ArchitectureSearch(backend=backend, noise_model=noise_model, hardware_profile=hardware_profile)

    progressive_cfg = replace(
        base_cfg,
        search_strategy="supercircuit_progressive",
        include_common_candidates=False,
        progressive_keep=progressive_keep or base_cfg.progressive_keep or base_cfg.population_size,
    )
    progressive_result = search.run(progressive_cfg)
    progressive_masks = [
        tuple(score.architecture.metadata["supercircuit_mask"])
        for score in progressive_result.scores
        if "supercircuit_mask" in score.architecture.metadata
    ]

    evolution_cfg = replace(
        base_cfg,
        search_strategy="supercircuit_reflective",
        include_common_candidates=False,
        reflective_mutation=True,
    )
    evolved_scores = _zero_cost_evolve_from_masks(
        progressive_masks,
        evolution_cfg,
        backend=backend,
        noise_model=noise_model,
        hardware_profile=hardware_profile,
    )
    evolved_masks = [
        tuple(score.architecture.metadata["supercircuit_mask"])
        for score in evolved_scores[: max(1, int(feedback_population_size or base_cfg.population_size))]
    ]
    qas_results, task_evaluated = _task_feedback_from_masks(
        evolved_masks,
        problem,
        evolution_cfg,
        optimizer_cfg,
        qas_top_k,
        feedback_generations,
        feedback_population_size,
        feedback_elite_count,
        backend,
        noise_model,
    )
    baselines = _optimize_many(
        _baseline_architectures(problem, layers=base_cfg.candidate_layers, backend=backend),
        problem,
        optimizer_cfg,
        backend=backend,
        noise_model=noise_model,
        result_group="baseline",
    )
    return ValidationReport(
        problem=problem,
        baseline_results=baselines,
        qas_results=qas_results,
        prior_scores=evolved_scores,
        metadata={
            "qas_top_k": qas_top_k,
            "n_prior_candidates": len(evolved_scores),
            "search_config": base_cfg.__dict__.copy(),
            "optimizer_config": optimizer_cfg.__dict__.copy(),
            "hybrid_pipeline": "progressive->reflective_evolution->task_feedback",
            "progressive_candidates": len(progressive_masks),
            "feedback_generations": feedback_generations,
            "feedback_population_size": feedback_population_size,
            "feedback_elite_count": feedback_elite_count,
            "task_feedback_evaluated": task_evaluated,
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
            "hardware_profile": None if hardware_profile is None else hardware_profile.__dict__.copy(),
        },
    )


def run_search_strategy_comparison(
    problem: ProblemInstance,
    search_config: Optional[SearchConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    qas_top_k: int = 3,
    strategies: Sequence[str] = (
        "supercircuit_progressive",
        "supercircuit_evolution",
        "supercircuit_reflective",
        "task_feedback",
        "hybrid",
    ),
    feedback_generations: int = 2,
    feedback_population_size: Optional[int] = None,
    feedback_elite_count: Optional[int] = None,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    hardware_profile: Optional[HardwareProfile] = None,
) -> StrategyComparisonReport:
    """Run several search strategies with the same task and optimizer budget."""
    backend = backend or NumpyBackend()
    base_cfg = search_config or SearchConfig(n_qubits=problem.n_qubits, candidate_layers=1, n_samples=8)
    optimizer_cfg = optimizer_config or OptimizerConfig(max_evaluations=16)
    reports: Dict[str, ValidationReport] = {}

    for strategy in strategies:
        if strategy == "hybrid":
            reports[strategy] = run_hybrid_qas_validation_experiment(
                problem,
                search_config=base_cfg,
                optimizer_config=optimizer_cfg,
                qas_top_k=qas_top_k,
                feedback_generations=feedback_generations,
                feedback_population_size=feedback_population_size,
                feedback_elite_count=feedback_elite_count,
                backend=backend,
                noise_model=noise_model,
                hardware_profile=hardware_profile,
            )
        elif strategy == "task_feedback":
            task_cfg = replace(base_cfg, search_strategy="supercircuit_evolution", include_common_candidates=False)
            reports[strategy] = run_task_feedback_validation_experiment(
                problem,
                search_config=task_cfg,
                optimizer_config=optimizer_cfg,
                qas_top_k=qas_top_k,
                feedback_generations=feedback_generations,
                feedback_population_size=feedback_population_size,
                feedback_elite_count=feedback_elite_count,
                backend=backend,
                noise_model=noise_model,
                hardware_profile=hardware_profile,
            )
        else:
            reports[strategy] = run_validation_experiment(
                problem,
                search_config=replace(base_cfg, search_strategy=strategy),
                optimizer_config=optimizer_cfg,
                qas_top_k=qas_top_k,
                backend=backend,
                noise_model=noise_model,
                hardware_profile=hardware_profile,
            )

    return StrategyComparisonReport(
        problem=problem,
        reports=reports,
        metadata={
            "strategies": [str(strategy) for strategy in strategies],
            "qas_top_k": qas_top_k,
            "search_config": base_cfg.__dict__.copy(),
            "optimizer_config": optimizer_cfg.__dict__.copy(),
            "feedback_generations": feedback_generations,
            "feedback_population_size": feedback_population_size,
            "feedback_elite_count": feedback_elite_count,
            "noise_model": type(noise_model).__name__ if noise_model is not None else None,
            "hardware_profile": None if hardware_profile is None else hardware_profile.__dict__.copy(),
        },
    )


__all__ = [
    "MultiSeedValidationReport",
    "RandomProxyValidationReport",
    "RandomProxyValidationRow",
    "StrategyComparisonReport",
    "ValidationReport",
    "run_hybrid_qas_validation_experiment",
    "run_multi_seed_validation_experiment",
    "run_random_proxy_validation_experiment",
    "run_search_strategy_comparison",
    "run_task_feedback_validation_experiment",
    "run_validation_experiment",
]

"""Task-level QAS validation utilities.

Architecture priors are useful for filtering, but final validation should compare
optimized task objectives under the same parameter budget.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..channel.backends.numpy_backend import NumpyBackend
from ..channel.noise.model import NoiseModel
from ..core.circuit import Circuit
from ..measure.measure import Measure
from ._types import ArchitectureSpec
from .problems import ProblemInstance


@dataclass
class OptimizerConfig:
    """Shared optimizer budget for fair task-level comparison."""

    max_evaluations: int = 64
    seed: int = 1234
    parameter_scale: float = 2.0 * np.pi
    include_zero_initialization: bool = True


@dataclass
class TaskEvaluationResult:
    """Optimized task-level score for one architecture/problem pair."""

    architecture_name: str
    problem_name: str
    optimized_value: float
    best_parameters: List[float]
    evaluations: int
    approximation_ratio: Optional[float]
    normalized_gap: Optional[float]
    ideal_value: float
    noisy_value: Optional[float] = None
    prior_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "problem_name": self.problem_name,
            "optimized_value": self.optimized_value,
            "best_parameters": list(self.best_parameters),
            "evaluations": self.evaluations,
            "approximation_ratio": self.approximation_ratio,
            "normalized_gap": self.normalized_gap,
            "ideal_value": self.ideal_value,
            "noisy_value": self.noisy_value,
            "prior_score": self.prior_score,
            "metadata": dict(self.metadata),
        }


def parameter_count(circuit: Circuit) -> int:
    count = 0
    for gate in circuit.gates:
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        param = gate["parameter"]
        if isinstance(param, (list, tuple, np.ndarray)):
            count += int(np.asarray(param).size)
        else:
            count += 1
    return count


def bind_parameters(circuit: Circuit, parameters: Sequence[float]) -> Circuit:
    """Return a copy of ``circuit`` with parameterized gates filled in order."""
    values = [float(value) for value in parameters]
    gates = deepcopy(circuit.gates)
    cursor = 0
    for gate in gates:
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        param = gate["parameter"]
        if isinstance(param, (list, tuple, np.ndarray)):
            shape = np.asarray(param).shape
            size = int(np.asarray(param).size)
            gate["parameter"] = np.asarray(values[cursor : cursor + size], dtype=float).reshape(shape).tolist()
            cursor += size
        else:
            gate["parameter"] = values[cursor]
            cursor += 1
    if cursor != len(values):
        raise ValueError(f"Expected {cursor} parameters, got {len(values)}")
    return Circuit(*gates, n_qubits=circuit.n_qubits, backend=circuit.backend)


def evaluate_task_objective(
    architecture: ArchitectureSpec,
    problem: ProblemInstance,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
) -> float:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and problem must use the same number of qubits")
    backend = backend or NumpyBackend()
    circuit = architecture.circuit
    n_params = parameter_count(circuit)
    if parameters is None:
        parameters = [0.0] * n_params
    if len(parameters) != n_params:
        raise ValueError(f"Expected {n_params} parameters, got {len(parameters)}")
    bound = bind_parameters(circuit, parameters) if n_params else circuit
    bound.bind_backend(backend)
    measure = Measure(backend)
    if noise_model is None:
        result = measure.run(bound, return_state=False)
    else:
        result = measure.run_density_matrix(bound, noise_model=noise_model, return_state=False)
    return problem.expected_objective(result.probabilities)


def optimize_task_parameters(
    architecture: ArchitectureSpec,
    problem: ProblemInstance,
    config: Optional[OptimizerConfig] = None,
    backend: Optional[NumpyBackend] = None,
    noise_model: Optional[NoiseModel] = None,
    prior_score: Optional[float] = None,
) -> TaskEvaluationResult:
    """Budgeted random-search optimizer for small validation smoke tests."""
    cfg = config or OptimizerConfig()
    backend = backend or NumpyBackend()
    n_params = parameter_count(architecture.circuit)
    rng = np.random.default_rng(cfg.seed)
    best_params = np.zeros(n_params, dtype=float)
    evaluations = 0

    def score(params: np.ndarray) -> float:
        return evaluate_task_objective(architecture, problem, params, backend=backend, noise_model=noise_model)

    if n_params == 0:
        best_value = score(best_params)
        evaluations = 1
    else:
        candidates: List[np.ndarray] = []
        if cfg.include_zero_initialization:
            candidates.append(np.zeros(n_params, dtype=float))
        while len(candidates) < cfg.max_evaluations:
            candidates.append(rng.uniform(-cfg.parameter_scale, cfg.parameter_scale, size=n_params))
        best_value = -np.inf if problem.maximize else np.inf
        for params in candidates[: cfg.max_evaluations]:
            value = score(params)
            evaluations += 1
            is_better = value > best_value if problem.maximize else value < best_value
            if is_better:
                best_value = value
                best_params = params.copy()

    ideal_value = evaluate_task_objective(architecture, problem, best_params, backend=backend)
    noisy_value = None
    if noise_model is not None:
        noisy_value = evaluate_task_objective(architecture, problem, best_params, backend=backend, noise_model=noise_model)

    optimized_value = noisy_value if noisy_value is not None else ideal_value
    return TaskEvaluationResult(
        architecture_name=architecture.name,
        problem_name=problem.name,
        optimized_value=float(optimized_value),
        best_parameters=[float(value) for value in best_params],
        evaluations=evaluations,
        approximation_ratio=problem.approximation_ratio(float(optimized_value)),
        normalized_gap=problem.normalized_gap(float(optimized_value)),
        ideal_value=float(ideal_value),
        noisy_value=None if noisy_value is None else float(noisy_value),
        prior_score=prior_score,
        metadata={
            "n_parameters": n_params,
            "optimizer": "budgeted_random_search",
            "seed": cfg.seed,
            "max_evaluations": cfg.max_evaluations,
        },
    )


__all__ = [
    "OptimizerConfig",
    "TaskEvaluationResult",
    "bind_parameters",
    "evaluate_task_objective",
    "optimize_task_parameters",
    "parameter_count",
]

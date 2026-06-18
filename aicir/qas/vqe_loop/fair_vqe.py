"""Fair VQE execution helpers for QAS labels.

The actual VQE engine is ``aicir.vqc.BasicVQE``; this wrapper adds QAS-specific
fair-label policy: fixed budgets, multi-start initialization, warm-start input,
best-parameter tracing, and benchmark-table-ready metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...backends.base import Backend
from ...core.circuit import Circuit
from ...metrics.circuit_structure import parameter_count
from ...optimizer import COBYLA
from ...vqc import BasicVQE
from ..core._types import ArchitectureSpec
from ..primitives.backend_utils import resolve_qas_backend
from ..problems.hamiltonians import VQEProblem, h2_demo_problem, hamiltonian_matrix


THETA_INIT_RANDOM_UNIFORM_PI = "random_uniform_pi"
THETA_INIT_ZERO_DIAGNOSTIC = "zero"
COBYLA_RHOBEG = 1.0
COBYLA_TOL = 1e-6


@dataclass
class VQEOptimizationResult:
    architecture: ArchitectureSpec
    energy: float
    best_parameters: List[float]
    evaluations: int
    n_starts: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _bind_parameters(circuit: Circuit, parameters: Sequence[float]) -> Circuit:
    """Return a bound copy of ``circuit`` using QAS gate-list order."""

    import copy

    values = [float(value) for value in parameters]
    gates = copy.deepcopy(circuit.gates)
    cursor = 0
    for gate in gates:
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        parameter = gate["parameter"]
        if isinstance(parameter, (list, tuple, np.ndarray)):
            shape = np.asarray(parameter).shape
            size = int(np.asarray(parameter).size)
            gate["parameter"] = np.asarray(values[cursor : cursor + size], dtype=float).reshape(shape).tolist()
            cursor += size
        else:
            gate["parameter"] = values[cursor]
            cursor += 1
    if cursor != len(values):
        raise ValueError(f"Expected {cursor} parameters, got {len(values)}")
    return Circuit(*gates, n_qubits=circuit.n_qubits, backend=circuit.backend)


def fair_vqe_screening_maxfev(n_params: int) -> int:
    """Low-fidelity proxy budget, counted as selection cost."""

    return max(500, 80 * max(1, int(n_params)))


def fair_vqe_final_maxfev(n_params: int) -> int:
    """Final-label COBYLA budget for fair VQE."""

    return max(1000, 200 * max(1, int(n_params)))


def fair_vqe_top_k(candidate_count: int) -> int:
    """Default deployment top-K rule."""

    return min(max(0, int(candidate_count)), max(10, int(np.ceil(0.1 * max(0, int(candidate_count))))))


def evaluate_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEProblem,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[Backend] = None,
) -> float:
    """Evaluate an architecture energy through the shared BasicVQE exact path."""

    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    params = [0.0] * n_params if parameters is None else list(parameters)
    if len(params) != n_params:
        raise ValueError(f"Expected {n_params} parameters, got {len(params)}")

    def ansatz(theta: np.ndarray):
        return _bind_parameters(architecture.circuit, theta.reshape(-1)) if n_params else architecture.circuit

    solver = BasicVQE(
        hamiltonian_matrix(problem.hamiltonian),
        n_qubits=problem.n_qubits,
        ansatz=ansatz,
        n_params=n_params,
        backend=backend,
        shots=None,
        energy_estimator="exact",
    )
    return float(solver.energy(np.asarray(params, dtype=float)))


def evaluate_h2_energy(
    architecture: ArchitectureSpec,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[Backend] = None,
) -> float:
    return evaluate_vqe_energy(architecture, h2_demo_problem(), parameters=parameters, backend=backend)


def _evaluation_budget(architecture: ArchitectureSpec, evals_per_param: int, max_evaluations: int) -> int:
    n_params = max(1, parameter_count(architecture.circuit))
    return min(int(max_evaluations), max(1, int(evals_per_param) * n_params))


def adaptive_fair_n_starts(architecture: ArchitectureSpec, min_starts: int = 3, params_per_start: int = 15) -> int:
    n_params = max(1, parameter_count(architecture.circuit))
    return max(int(min_starts), int(np.ceil(n_params / max(1, int(params_per_start)))))


def is_b1_improvement_valid(b1_energy: float, floor_energy: float = -3.0, tolerance: float = 0.01) -> bool:
    return abs(float(b1_energy) - float(floor_energy)) > float(tolerance)


def _initial_starts(
    *,
    n_params: int,
    n_starts: int,
    rng: np.random.Generator,
    init_mode: str,
    init_scale: float,
    initial_parameters: Optional[Sequence[float]],
) -> tuple[list[np.ndarray], str]:
    mode = str(init_mode).strip().lower()
    start_count = max(1, int(n_starts))
    if initial_parameters is not None:
        first_start = np.asarray(initial_parameters, dtype=float)
        if first_start.shape != (n_params,):
            raise ValueError(f"initial_parameters must have length {n_params}")
        starts = [first_start]
        for _ in range(start_count - 1):
            starts.append(rng.uniform(-float(init_scale), float(init_scale), size=n_params))
        return starts, "warm_start_then_random" if start_count > 1 else "warm_start"
    if mode in {THETA_INIT_ZERO_DIAGNOSTIC, "zeros"}:
        return [np.zeros(n_params, dtype=float) for _ in range(start_count)], mode
    if mode in {THETA_INIT_RANDOM_UNIFORM_PI, "random", "uniform"}:
        return [rng.uniform(-float(init_scale), float(init_scale), size=n_params) for _ in range(start_count)], mode
    if mode in {"zero_then_random", "mixed"}:
        starts = [np.zeros(n_params, dtype=float)]
        for _ in range(start_count - 1):
            starts.append(rng.uniform(-float(init_scale), float(init_scale), size=n_params))
        return starts, mode
    raise ValueError("init_mode must be one of: random_uniform_pi, zero, zero_then_random")


def optimize_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEProblem,
    seed: int = 1234,
    n_starts: int = 1,
    evals_per_param: int = 10,
    max_evaluations: int = 80,
    budget_override: Optional[int] = None,
    backend: Optional[Backend] = None,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
    initial_parameters: Optional[Sequence[float]] = None,
) -> VQEOptimizationResult:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    rng = np.random.default_rng(int(seed))
    budget = int(budget_override) if budget_override is not None else _evaluation_budget(architecture, evals_per_param, max_evaluations)
    budget = max(1, budget)
    starts, mode = _initial_starts(
        n_params=n_params,
        n_starts=n_starts,
        rng=rng,
        init_mode=init_mode,
        init_scale=init_scale,
        initial_parameters=initial_parameters,
    )
    hamiltonian = hamiltonian_matrix(problem.hamiltonian)

    def ansatz(theta: np.ndarray):
        return _bind_parameters(architecture.circuit, theta.reshape(-1)) if n_params else architecture.circuit

    best_energy = float("inf")
    best_params = np.zeros(n_params, dtype=float)
    total_evals = 0
    per_start: List[Dict[str, Any]] = []

    for start_index, start in enumerate(starts):
        start_time = perf_counter()
        if n_params == 0:
            energy = evaluate_vqe_energy(architecture, problem, start, backend=backend)
            nfev = 1
            params = start
        else:
            optimizer = COBYLA(options={"maxiter": budget, "rhobeg": COBYLA_RHOBEG, "tol": COBYLA_TOL})
            solver = BasicVQE(
                hamiltonian,
                n_qubits=problem.n_qubits,
                ansatz=ansatz,
                n_params=n_params,
                backend=backend,
                optimizer=optimizer,
                shots=None,
                energy_estimator="exact",
            )
            try:
                result = solver.run(init_params=start, optimizer=optimizer)
                energy = float(result.energy)
                params = np.asarray(result.parameters, dtype=float).reshape(-1)
                raw_optimizer = result.optimizer_result
                nfev = int(getattr(raw_optimizer, "nfev", budget))
            except ModuleNotFoundError:
                values = [start]
                while len(values) < budget:
                    values.append(rng.uniform(-np.pi, np.pi, size=n_params))
                scored = [(float(solver.energy(params)), params) for params in values]
                energy, params = min(scored, key=lambda item: item[0])
                nfev = len(scored)
        total_evals += nfev
        per_start.append(
            {
                "start_index": start_index,
                "seed": int(seed),
                "energy": float(energy),
                "nfev": int(nfev),
                "walltime_s": float(perf_counter() - start_time),
                "init_mode": mode,
                "init_l2": float(np.linalg.norm(start)) if n_params else 0.0,
            }
        )
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

    return VQEOptimizationResult(
        architecture=architecture,
        energy=float(best_energy),
        best_parameters=[float(value) for value in best_params],
        evaluations=total_evals,
        n_starts=len(starts),
        metadata={
            "optimizer": "COBYLA",
            "vqe_engine": "BasicVQE",
            "budget_per_start": budget,
            "nfev": total_evals,
            "per_start": per_start,
            "theta_init_mode": mode,
            "cobyla_rhobeg": COBYLA_RHOBEG,
            "cobyla_tol": COBYLA_TOL,
        },
    )


def optimize_h2_energy(
    architecture: ArchitectureSpec,
    seed: int = 1234,
    n_starts: int = 1,
    evals_per_param: int = 10,
    max_evaluations: int = 80,
    backend: Optional[Backend] = None,
) -> VQEOptimizationResult:
    return optimize_vqe_energy(
        architecture,
        h2_demo_problem(),
        seed=seed,
        n_starts=n_starts,
        evals_per_param=evals_per_param,
        max_evaluations=max_evaluations,
        backend=backend,
    )


__all__ = [
    "THETA_INIT_RANDOM_UNIFORM_PI",
    "THETA_INIT_ZERO_DIAGNOSTIC",
    "COBYLA_RHOBEG",
    "COBYLA_TOL",
    "VQEOptimizationResult",
    "adaptive_fair_n_starts",
    "evaluate_h2_energy",
    "evaluate_vqe_energy",
    "is_b1_improvement_valid",
    "optimize_h2_energy",
    "optimize_vqe_energy",
    "fair_vqe_final_maxfev",
    "fair_vqe_screening_maxfev",
    "fair_vqe_top_k",
]

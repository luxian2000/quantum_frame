"""Fair VQE execution helpers for QAS labels.

The actual VQE objective is evaluated through QAS's shared fair-label policy:
fixed budgets, multi-start initialization, warm-start input, best-parameter
tracing, and benchmark-table-ready metadata.  Large Pauli Hamiltonians are kept
in Pauli-term form; fair labeling must not expand them into dense 2^n x 2^n
matrices because 18q chemistry Hamiltonians would exceed device memory before
optimization even starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...backends.base import Backend
from ...core.circuit import Circuit
from ...core.gates import apply_gate_to_state
from ...core.state import State
from ...ir import circuit_instructions, instruction_name, instruction_parameter, instruction_with_parameter
from ...metrics.circuit_structure import parameter_count
from ...optimizer import COBYLA
from ..core._types import ArchitectureSpec
from ..core.backend_utils import resolve_qas_backend
from ..problems.hamiltonians import VQEProblem, h2_demo_problem, hamiltonian_matrix

try:  # pragma: no cover - torch is optional in NumPy-only environments.
    import torch
except Exception:  # pragma: no cover
    torch = None


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

    values = [float(value) for value in parameters]
    instructions = []
    cursor = 0
    for instruction in circuit_instructions(circuit):
        parameter = instruction_parameter(instruction)
        if parameter is None:
            instructions.append(instruction)
            continue
        if isinstance(parameter, (list, tuple, np.ndarray)):
            shape = np.asarray(parameter).shape
            size = int(np.asarray(parameter).size)
            instruction = instruction_with_parameter(
                instruction,
                np.asarray(values[cursor : cursor + size], dtype=float).reshape(shape).tolist(),
            )
            cursor += size
        else:
            instruction = instruction_with_parameter(instruction, values[cursor])
            cursor += 1
        instructions.append(instruction)
    if cursor != len(values):
        raise ValueError(f"Expected {cursor} parameters, got {len(values)}")
    return Circuit(*instructions, n_qubits=circuit.n_qubits, backend=circuit.backend)


def fair_vqe_screening_maxfev(n_params: int) -> int:
    """Low-fidelity proxy budget, counted as selection cost."""

    return max(500, 80 * max(1, int(n_params)))


def fair_vqe_final_maxfev(n_params: int) -> int:
    """Final-label COBYLA budget for fair VQE."""

    return max(1000, 200 * max(1, int(n_params)))


def fair_vqe_top_k(candidate_count: int) -> int:
    """Default deployment top-K rule."""

    return min(max(0, int(candidate_count)), max(10, int(np.ceil(0.1 * max(0, int(candidate_count))))))


def _pauli_term_cache(hamiltonian: Sequence[tuple[float, str]], *, n_qubits: int) -> list[tuple[int, int, int, float, float]]:
    cached_terms: list[tuple[int, int, int, float, float]] = []
    for coefficient, pauli in hamiltonian:
        labels = str(pauli).strip().upper()
        if len(labels) != int(n_qubits):
            raise ValueError("Hamiltonian Pauli strings must match problem.n_qubits")
        invalid = sorted(set(labels) - {"I", "X", "Y", "Z"})
        if invalid:
            raise ValueError(f"unsupported Pauli character(s): {', '.join(invalid)}")
        flip_mask = 0
        sign_mask = 0
        y_count = 0
        for qubit, label in enumerate(labels):
            bit_mask = 1 << (int(n_qubits) - qubit - 1)
            if label in {"X", "Y"}:
                flip_mask ^= bit_mask
            if label in {"Y", "Z"}:
                sign_mask ^= bit_mask
            if label == "Y":
                y_count += 1
        coeff = complex(coefficient)
        cached_terms.append((flip_mask, sign_mask, y_count % 4, float(coeff.real), float(coeff.imag)))
    if not cached_terms:
        raise ValueError("Hamiltonian must contain at least one Pauli term")
    return cached_terms


def _simulate_statevector(circuit: Circuit, *, backend: Backend):
    state = State.zero_state(int(circuit.n_qubits), backend).data
    for instruction in circuit_instructions(circuit):
        name = str(instruction_name(instruction)).strip().lower()
        if name in {"measure", "measurement", "reset", "if", "while"}:
            raise ValueError(f"fair VQE exact statevector path does not support nonunitary/control-flow gate {name!r}")
        updated = apply_gate_to_state(instruction, state, int(circuit.n_qubits), backend)
        if updated is None:
            raise ValueError(f"fair VQE exact statevector path cannot apply gate {name!r} without dense expansion")
        state = updated
    return state.reshape(-1)


def _torch_pauli_signs(basis_indices, sign_mask: int):
    if sign_mask == 0:
        return None
    parity = torch.zeros_like(basis_indices, dtype=torch.bool)
    bit = 0
    mask = int(sign_mask)
    while mask:
        if mask & 1:
            parity = torch.logical_xor(parity, ((basis_indices >> bit) & 1).to(torch.bool))
        mask >>= 1
        bit += 1
    ones = torch.ones_like(basis_indices, dtype=torch.float32)
    return torch.where(parity, -ones, ones)


def _torch_pauli_expectation(state, pauli_cache, *, backend: Backend):
    flat = state.reshape(-1)
    basis_indices = torch.arange(flat.numel(), dtype=torch.long, device=flat.device)
    if hasattr(backend, "hamiltonian_expectation_pauli"):
        return backend.hamiltonian_expectation_pauli(flat, basis_indices, pauli_cache)
    state_real = torch.real(flat)
    state_imag = torch.imag(flat)
    energy = torch.zeros((), dtype=torch.float32, device=flat.device)
    for flip_mask, sign_mask, y_phase, coefficient_real, coefficient_imag in pauli_cache:
        if flip_mask:
            mapped_indices = torch.bitwise_xor(basis_indices, int(flip_mask))
            mapped_real = state_real.index_select(0, mapped_indices)
            mapped_imag = state_imag.index_select(0, mapped_indices)
        else:
            mapped_real = state_real
            mapped_imag = state_imag
        overlap_real = mapped_real * state_real + mapped_imag * state_imag
        overlap_imag = mapped_real * state_imag - mapped_imag * state_real
        signs = _torch_pauli_signs(basis_indices, sign_mask)
        if signs is not None:
            overlap_real = overlap_real * signs
            overlap_imag = overlap_imag * signs
        if y_phase == 0:
            term_real = overlap_real.sum()
            term_imag = overlap_imag.sum()
        elif y_phase == 1:
            term_real = -overlap_imag.sum()
            term_imag = overlap_real.sum()
        elif y_phase == 2:
            term_real = -overlap_real.sum()
            term_imag = -overlap_imag.sum()
        else:
            term_real = overlap_imag.sum()
            term_imag = -overlap_real.sum()
        energy = energy + coefficient_real * term_real - coefficient_imag * term_imag
    return energy


def _numpy_pauli_expectation(state: np.ndarray, pauli_cache) -> float:
    flat = np.asarray(state, dtype=np.complex128).reshape(-1)
    basis_indices = np.arange(flat.size, dtype=np.int64)
    state_conj = np.conjugate(flat)
    energy = 0.0 + 0.0j
    for flip_mask, sign_mask, y_phase, coefficient_real, coefficient_imag in pauli_cache:
        mapped = flat[np.bitwise_xor(basis_indices, np.int64(flip_mask))] if flip_mask else flat
        term = state_conj * mapped
        if sign_mask:
            parity = np.zeros(flat.size, dtype=bool)
            bit = 0
            mask = int(sign_mask)
            while mask:
                if mask & 1:
                    parity ^= ((basis_indices >> bit) & 1).astype(bool)
                mask >>= 1
                bit += 1
            term = np.where(parity, -term, term)
        if y_phase:
            term = term * (1j ** int(y_phase))
        energy += complex(coefficient_real, coefficient_imag) * np.sum(term)
    return float(np.real(energy))


def _evaluate_pauli_state_energy(
    architecture: ArchitectureSpec,
    problem: VQEProblem,
    parameters: Sequence[float],
    *,
    backend: Backend,
) -> float:
    n_params = parameter_count(architecture.circuit)
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    bound = _bind_parameters(architecture.circuit, theta) if n_params else architecture.circuit
    bound = Circuit(*list(circuit_instructions(bound)), n_qubits=bound.n_qubits, backend=backend)
    state = _simulate_statevector(bound, backend=backend)
    pauli_cache = _pauli_term_cache(problem.hamiltonian, n_qubits=problem.n_qubits)
    if torch is not None and isinstance(state, torch.Tensor):
        value = _torch_pauli_expectation(state, pauli_cache, backend=backend)
        return float(np.asarray(backend.to_numpy(value)).reshape(()))
    return _numpy_pauli_expectation(state, pauli_cache)


def evaluate_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEProblem,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[Backend] = None,
) -> float:
    """Evaluate an architecture energy through the shared fair Pauli-term path."""

    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    params = [0.0] * n_params if parameters is None else list(parameters)
    if len(params) != n_params:
        raise ValueError(f"Expected {n_params} parameters, got {len(params)}")
    return _evaluate_pauli_state_energy(architecture, problem, params, backend=backend)


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

            def objective(theta: np.ndarray) -> float:
                return _evaluate_pauli_state_energy(architecture, problem, theta.reshape(-1), backend=backend)

            try:
                result = optimizer.minimize(objective, start)
                energy = float(getattr(result, "best_fun", getattr(result, "fun")))
                params = np.asarray(getattr(result, "best_x", getattr(result, "x")), dtype=float).reshape(-1)
                nfev = int(getattr(result, "nfev", budget))
            except ModuleNotFoundError:
                values = [start]
                while len(values) < budget:
                    values.append(rng.uniform(-np.pi, np.pi, size=n_params))
                scored = [(float(objective(params)), params) for params in values]
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
            "vqe_engine": "statevector_pauli_terms",
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

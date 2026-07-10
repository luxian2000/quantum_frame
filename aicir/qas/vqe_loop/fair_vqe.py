"""Fair VQE execution helpers for QAS labels.

The actual VQE objective is evaluated through QAS's shared fair-label policy:
fixed budgets, multi-start initialization, warm-start input, best-parameter
tracing, and benchmark-table-ready metadata.  Large Pauli Hamiltonians are kept
in Pauli-term form; fair labeling must not expand them into dense 2^n x 2^n
matrices because 18q chemistry Hamiltonians would exceed device memory before
optimization even starts.
"""

from __future__ import annotations

from contextlib import nullcontext
import os
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
CPU_PAULI_EXPECTATION_MIN_TERMS = 1024


def _fair_trace_enabled() -> bool:
    return str(os.environ.get("AICIR_QAS_FAIR_TRACE", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _describe_value(value: Any) -> str:
    parts = []
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None:
        try:
            parts.append(f"shape={tuple(shape)}")
        except TypeError:
            parts.append(f"shape={shape}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if device is not None:
        parts.append(f"device={device}")
    return " ".join(parts) if parts else f"type={type(value).__name__}"


def _instruction_qubits_text(instruction: Any) -> str:
    for key in ("qubits", "wires", "targets"):
        if isinstance(instruction, dict) and key in instruction:
            return str(instruction.get(key))
        value = getattr(instruction, key, None)
        if value is not None:
            return str(value)
    if isinstance(instruction, dict):
        values = []
        for key in ("qubit", "qubit_1", "qubit_2", "qubit_3", "qubit_4"):
            if key in instruction:
                values.append(instruction[key])
        if values:
            return str(values)
    return "unknown"


def _fair_trace(message: str) -> None:
    if _fair_trace_enabled():
        print(f"[fair_vqe_trace] {message}", flush=True)

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
    _fair_trace(f"stage=simulate_start n_qubits={int(circuit.n_qubits)} state={_describe_value(state)}")
    for gate_index, instruction in enumerate(circuit_instructions(circuit)):
        name = str(instruction_name(instruction)).strip().lower()
        qubits = _instruction_qubits_text(instruction)
        _fair_trace(
            f"stage=simulate_gate_begin gate_index={gate_index} gate_type={name} qubits={qubits} state={_describe_value(state)}"
        )
        if name in {"measure", "measurement", "reset", "if", "while"}:
            _fair_trace(f"stage=simulate_gate_error gate_index={gate_index} gate_type={name} error=nonunitary_control_flow")
            raise ValueError(f"fair VQE exact statevector path does not support nonunitary/control-flow gate {name!r}")
        try:
            updated = apply_gate_to_state(instruction, state, int(circuit.n_qubits), backend)
        except BaseException as exc:
            _fair_trace(f"stage=simulate_gate_error gate_index={gate_index} gate_type={name} qubits={qubits} error={exc!r}")
            raise
        if updated is None:
            _fair_trace(f"stage=simulate_gate_error gate_index={gate_index} gate_type={name} qubits={qubits} error=dense_expansion_required")
            raise ValueError(f"fair VQE exact statevector path cannot apply gate {name!r} without dense expansion")
        state = updated
        _fair_trace(
            f"stage=simulate_gate_end gate_index={gate_index} gate_type={name} qubits={qubits} state={_describe_value(state)}"
        )
    result = state.reshape(-1)
    _fair_trace(f"stage=simulate_end state={_describe_value(result)}")
    return result


def _is_backend_tensor_like(value: Any) -> bool:
    if torch is not None:
        try:
            if torch.is_tensor(value):
                return True
        except Exception:
            pass
    return (
        not isinstance(value, np.ndarray)
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "device")
    )


def _torch_inference_context():
    if torch is None:
        return nullcontext()
    context = getattr(torch, "inference_mode", None)
    if callable(context):
        return context()
    return torch.no_grad()


def _device_type(value: Any) -> str:
    if value is None:
        return ""
    device_type = getattr(value, "type", None)
    if device_type:
        return str(device_type).lower()
    text = str(value).strip().lower()
    if ":" in text:
        text = text.split(":", 1)[0]
    return text


def _prefer_cpu_pauli_expectation(state: Any, pauli_cache, *, backend: Backend) -> bool:
    if not _is_backend_tensor_like(state):
        return False
    device_type = _device_type(getattr(state, "device", None)) or _device_type(getattr(backend, "_device", None))
    return device_type == "npu" and len(pauli_cache) >= CPU_PAULI_EXPECTATION_MIN_TERMS


def _backend_state_to_numpy_vector(state: Any, *, backend: Backend) -> np.ndarray:
    if not hasattr(backend, "to_numpy"):
        raise TypeError("Backend returned a tensor-like state but does not expose to_numpy for explicit CPU transfer")
    return np.asarray(backend.to_numpy(state), dtype=np.complex128).reshape(-1)


def _evaluate_state_pauli_energy(state: Any, pauli_cache, *, backend: Backend) -> float:
    _fair_trace(f"stage=pauli_expectation_begin terms={len(pauli_cache)} state={_describe_value(state)}")
    try:
        if _is_backend_tensor_like(state):
            if _prefer_cpu_pauli_expectation(state, pauli_cache, backend=backend):
                _fair_trace("stage=pauli_expectation_cpu_transfer_begin reason=large_npu_pauli_cache")
                state_np = _backend_state_to_numpy_vector(state, backend=backend)
                _fair_trace(f"stage=pauli_expectation_cpu_transfer_end state={_describe_value(state_np)}")
                value = _numpy_pauli_expectation(state_np, pauli_cache)
                _fair_trace(f"stage=pauli_expectation_end path=cpu_numpy energy={value}")
                return value
            if torch is None:
                raise TypeError("Backend returned a tensor-like state, but torch is not importable")
            value = _torch_pauli_expectation(state, pauli_cache, backend=backend)
            result = float(np.asarray(backend.to_numpy(value)).reshape(()))
            _fair_trace(f"stage=pauli_expectation_end path=torch_backend energy={result}")
            return result
        result = _numpy_pauli_expectation(state, pauli_cache)
        _fair_trace(f"stage=pauli_expectation_end path=numpy energy={result}")
        return result
    except BaseException as exc:
        _fair_trace(f"stage=pauli_expectation_error error={exc!r}")
        raise


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
    if _is_backend_tensor_like(state):
        raise TypeError("Refusing to evaluate a backend tensor through NumPy Pauli expectation; use the torch/NPU path")
    flat = np.asarray(state, dtype=np.complex128).reshape(-1)
    basis_indices = np.arange(flat.size, dtype=np.int64)
    state_conj = np.conjugate(flat)
    energy = 0.0 + 0.0j
    total_terms = len(pauli_cache)
    progress_interval = max(1, int(os.environ.get("AICIR_QAS_FAIR_TRACE_PAULI_INTERVAL", "512")))
    _fair_trace(
        f"stage=numpy_pauli_expectation_begin terms={total_terms} state_size={flat.size} progress_interval={progress_interval}"
    )
    for term_index, (flip_mask, sign_mask, y_phase, coefficient_real, coefficient_imag) in enumerate(pauli_cache, start=1):
        try:
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
            term_energy = complex(coefficient_real, coefficient_imag) * np.sum(term)
            energy += term_energy
        except BaseException as exc:
            _fair_trace(
                "stage=numpy_pauli_expectation_term_error "
                f"term_index={term_index} terms={total_terms} flip_mask={flip_mask} sign_mask={sign_mask} "
                f"y_phase={y_phase} coeff=({coefficient_real},{coefficient_imag}) error={exc!r}"
            )
            raise
        if term_index == total_terms or term_index % progress_interval == 0:
            _fair_trace(
                "stage=numpy_pauli_expectation_progress "
                f"term_index={term_index} terms={total_terms} partial_energy={float(np.real(energy))}"
            )
    result = float(np.real(energy))
    _fair_trace(f"stage=numpy_pauli_expectation_end terms={total_terms} energy={result}")
    return result


def _evaluate_pauli_state_energy(
    architecture: ArchitectureSpec,
    problem: VQEProblem,
    parameters: Sequence[float],
    *,
    backend: Backend,
) -> float:
    n_params = parameter_count(architecture.circuit)
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    _fair_trace(
        f"stage=evaluate_begin architecture_id={getattr(architecture, 'architecture_id', getattr(architecture, 'name', 'unknown'))} n_qubits={architecture.n_qubits} n_params={n_params} problem={problem.name}"
    )
    bound = _bind_parameters(architecture.circuit, theta) if n_params else architecture.circuit
    bound = Circuit(*list(circuit_instructions(bound)), n_qubits=bound.n_qubits, backend=backend)
    _fair_trace(f"stage=bind_end gates={len(list(circuit_instructions(bound)))}")
    pauli_cache = _pauli_term_cache(problem.hamiltonian, n_qubits=problem.n_qubits)
    _fair_trace(f"stage=pauli_cache_end terms={len(pauli_cache)}")
    try:
        with _torch_inference_context():
            state = _simulate_statevector(bound, backend=backend)
            return _evaluate_state_pauli_energy(state, pauli_cache, backend=backend)
    except BaseException as exc:
        _fair_trace(f"stage=evaluate_error error={exc!r}")
        raise


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
    zero_candidate: Dict[str, Any] = {"included": False}

    if n_params > 0:
        zero_start = np.zeros(n_params, dtype=float)
        zero_time = perf_counter()
        zero_energy = evaluate_vqe_energy(architecture, problem, zero_start, backend=backend)
        total_evals += 1
        zero_candidate = {
            "included": True,
            "energy": float(zero_energy),
            "nfev": 1,
            "walltime_s": float(perf_counter() - zero_time),
            "init_mode": "zero_parameter_candidate",
        }
        if float(zero_energy) < best_energy:
            best_energy = float(zero_energy)
            best_params = zero_start.copy()

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
            "zero_parameter_candidate": zero_candidate,
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

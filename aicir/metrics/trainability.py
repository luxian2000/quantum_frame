"""Trainability metrics for quantum circuits and ansatz templates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..channel.backends.numpy_backend import NumpyBackend
from ..core.circuit import Circuit
from ..core.gates import apply_gate_to_state
from ._utils import count_gate_families, depth_proxy


def structure_proxy(circuit: Circuit) -> float:
    """Low-cost trainability proxy based on depth, entangling density, and parameter density."""
    n_qubits = int(circuit.n_qubits)
    n_gates = len(circuit.gates)
    single_qubit_ops, two_qubit_ops = count_gate_families(circuit)
    circuit_depth_proxy = depth_proxy(circuit) if n_qubits > 0 else 1.0
    two_qubit_ratio = two_qubit_ops / n_gates if n_gates > 0 else 0.0
    params_per_qubit = single_qubit_ops / n_qubits if n_qubits > 0 else 0.0

    depth_score = np.exp(-circuit_depth_proxy / 10.0)
    entanglement_score = np.exp(-2.0 * two_qubit_ratio)
    parameter_score = np.exp(-params_per_qubit / 5.0)
    return float(np.clip(0.4 * depth_score + 0.4 * entanglement_score + 0.2 * parameter_score, 0.0, 1.0))


def structure_proxy_details(circuit: Circuit) -> Dict[str, Any]:
    single_qubit_ops, two_qubit_ops = count_gate_families(circuit)
    return {
        "structure_proxy_score": structure_proxy(circuit),
        "single_qubit_gate_count": single_qubit_ops,
        "two_qubit_gate_count": two_qubit_ops,
        "depth_proxy": depth_proxy(circuit),
    }


def _parameter_slices(circuit: Circuit) -> List[tuple[int, Optional[tuple[int, ...]], int]]:
    slices: List[tuple[int, Optional[tuple[int, ...]], int]] = []
    for gate_index, gate in enumerate(circuit.gates):
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        parameter = gate["parameter"]
        if isinstance(parameter, (list, tuple, np.ndarray)):
            array = np.asarray(parameter, dtype=float)
            for flat_index in range(array.size):
                slices.append((gate_index, np.unravel_index(flat_index, array.shape), flat_index))
        else:
            slices.append((gate_index, None, 0))
    return slices


def _bind_parameter_values(circuit: Circuit, values: Sequence[float]) -> Circuit:
    gates = deepcopy(circuit.gates)
    cursor = 0
    for gate in gates:
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        parameter = gate["parameter"]
        if isinstance(parameter, (list, tuple, np.ndarray)):
            template = np.asarray(parameter, dtype=float)
            size = int(template.size)
            gate["parameter"] = np.asarray(values[cursor : cursor + size], dtype=float).reshape(template.shape).tolist()
            cursor += size
        else:
            gate["parameter"] = float(values[cursor])
            cursor += 1
    if cursor != len(values):
        raise ValueError(f"Expected {cursor} parameters, got {len(values)}")
    return Circuit(*gates, n_qubits=circuit.n_qubits, backend=circuit.backend)


def _z_expectations_from_probabilities(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    probs = probs / total
    values = np.zeros(n_qubits, dtype=float)
    for state_index, probability in enumerate(probs):
        for qubit in range(n_qubits):
            bit = (state_index >> (n_qubits - qubit - 1)) & 1
            values[qubit] += probability * (1.0 if bit == 0 else -1.0)
    return values


def _simulate_probabilities(circuit: Circuit, backend: Optional[NumpyBackend] = None) -> np.ndarray:
    bk = backend or circuit.backend or NumpyBackend()
    state = bk.zeros_state(int(circuit.n_qubits))
    for gate in circuit.gates:
        state = apply_gate_to_state(gate, state, int(circuit.n_qubits), bk)
    return bk.to_numpy(bk.measure_probs(state)).reshape(-1)


def local_probe_objective(
    circuit: Circuit,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[NumpyBackend] = None,
    probe: str = "mean_z",
) -> float:
    """Task-agnostic local probe objective used for zero-cost trainability."""
    bound = _bind_parameter_values(circuit, parameters) if parameters is not None else circuit
    probabilities = _simulate_probabilities(bound, backend=backend)
    z_values = _z_expectations_from_probabilities(probabilities, int(bound.n_qubits))
    if probe == "mean_z":
        return float(np.mean(z_values))
    if probe == "mean_abs_z":
        return float(np.mean(np.abs(z_values)))
    raise ValueError(f"Unsupported trainability probe: {probe!r}")


def local_probe_gradient_statistics(
    circuit: Circuit,
    samples: int = 8,
    seed: int = 1234,
    shift: float = np.pi / 2.0,
    parameter_scale: float = 2.0 * np.pi,
    backend: Optional[NumpyBackend] = None,
    probe: str = "mean_z",
    zero_threshold: float = 1e-8,
) -> Dict[str, Any]:
    """Estimate zero-cost trainability from local-probe parameter-shift gradients."""
    parameter_positions = _parameter_slices(circuit)
    n_parameters = len(parameter_positions)
    if n_parameters == 0:
        return {
            "n_parameters": 0,
            "n_gradient_samples": 0,
            "probe": probe,
            "mean_gradient_norm": 0.0,
            "gradient_variance": 0.0,
            "mean_abs_gradient": 0.0,
            "zero_gradient_fraction": 1.0,
        }

    rng = np.random.default_rng(seed)
    gradients = np.zeros((max(1, int(samples)), n_parameters), dtype=float)
    for sample_index in range(gradients.shape[0]):
        theta = rng.uniform(-parameter_scale, parameter_scale, size=n_parameters)
        for parameter_index in range(n_parameters):
            plus = theta.copy()
            minus = theta.copy()
            plus[parameter_index] += shift
            minus[parameter_index] -= shift
            value_plus = local_probe_objective(circuit, plus, backend=backend, probe=probe)
            value_minus = local_probe_objective(circuit, minus, backend=backend, probe=probe)
            gradients[sample_index, parameter_index] = 0.5 * (value_plus - value_minus)

    norms = np.linalg.norm(gradients, axis=1)
    gradient_variance = float(np.mean(np.var(gradients, axis=0))) if gradients.shape[0] > 1 else 0.0
    return {
        "n_parameters": n_parameters,
        "n_gradient_samples": int(gradients.shape[0]),
        "probe": probe,
        "mean_gradient_norm": float(np.mean(norms)),
        "gradient_variance": gradient_variance,
        "mean_abs_gradient": float(np.mean(np.abs(gradients))),
        "zero_gradient_fraction": float(np.mean(np.abs(gradients) < zero_threshold)),
    }


def gradient_norm_score(circuit: Circuit, **kwargs: Any) -> float:
    stats = local_probe_gradient_statistics(circuit, **kwargs)
    scale = 1.0 / max(1, int(circuit.n_qubits))
    return float(np.clip(1.0 - np.exp(-stats["mean_gradient_norm"] / max(scale, 1e-12)), 0.0, 1.0))


def gradient_variance_score(circuit: Circuit, **kwargs: Any) -> float:
    stats = local_probe_gradient_statistics(circuit, **kwargs)
    scale = 1.0 / max(1, int(circuit.n_qubits)) ** 2
    variance_score = 1.0 - np.exp(-stats["gradient_variance"] / max(scale, 1e-12))
    norm_score = 1.0 - np.exp(-stats["mean_gradient_norm"] / max(np.sqrt(scale), 1e-12))
    return float(np.clip(0.65 * variance_score + 0.35 * norm_score, 0.0, 1.0))


__all__ = [
    "gradient_norm_score",
    "gradient_variance_score",
    "local_probe_gradient_statistics",
    "local_probe_objective",
    "structure_proxy",
    "structure_proxy_details",
]

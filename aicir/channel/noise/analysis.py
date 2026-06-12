"""Shared noise-analysis utilities for circuits and NoiseModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..backends.base import Backend
from .model import NoiseModel
from ...core.circuit import Circuit
from ...core.gates import gate_to_matrix
from ...core.state import State
from ...ir import circuit_instruction_count, circuit_instructions, instruction_name


@dataclass
class NoiseSensitivityResult:
    """Result of noise sensitivity analysis for a quantum circuit."""

    circuit: Circuit
    n_qubits: int
    noise_model: Optional[NoiseModel]
    ideal_avg_fidelity: float
    noisy_avg_fidelity: float
    avg_fidelity_loss: float
    gate_type_sensitivity: Dict[str, float]
    n_gates_total: int
    n_gates_by_type: Dict[str, int]
    noise_strength: float

    def summary(self) -> str:
        lines = [
            "=== Noise Sensitivity Analysis ===",
            f"Circuit gates: {self.n_gates_total}",
            f"Ideal avg fidelity: {self.ideal_avg_fidelity:.4f}",
            f"Noisy avg fidelity: {self.noisy_avg_fidelity:.4f}",
            f"Fidelity loss: {self.avg_fidelity_loss:.4f}",
            f"Noise strength: {self.noise_strength:.4f}",
            "Gate-type sensitivity:",
        ]
        for gate_type, sensitivity in sorted(self.gate_type_sensitivity.items()):
            lines.append(f"  {gate_type}: {sensitivity:.4f}")
        return "\n".join(lines)


def default_plus_state(n_qubits: int, backend: Backend) -> State:
    dim = 1 << n_qubits
    plus_data = np.ones(dim, dtype=np.complex64) / np.sqrt(dim)
    return State.from_array(plus_data, n_qubits=n_qubits, backend=backend)


def evolve_density_gatewise(
    circuit: Circuit,
    backend: Backend,
    initial_state: State,
    noise_model: Optional[NoiseModel] = None,
) -> State:
    rho = initial_state.to_density_matrix()
    for gate in circuit_instructions(circuit):
        gate_unitary = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        rho = rho.evolve(gate_unitary)
        if noise_model is not None:
            rho = State(
                noise_model.apply(
                    rho.data,
                    n_qubits=circuit.n_qubits,
                    backend=backend,
                    gate_type=instruction_name(gate),
                    gate=gate,
                ),
                circuit.n_qubits,
                backend,
            )
    return rho


def estimate_noise_strength(noise_model: NoiseModel) -> float:
    total_strength = 0.0
    n_channels = 0
    for rule in noise_model.rules:
        channel = rule.channel
        p = getattr(channel, "p", None)
        if p is not None:
            total_strength += p
            n_channels += 1
        gamma = getattr(channel, "gamma", None)
        if gamma is not None:
            total_strength += gamma
            n_channels += 1
    return total_strength / n_channels if n_channels > 0 else 0.01


def noise_sensitivity(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    noise_model: Optional[NoiseModel] = None,
    n_samples: int = 200,
    initial_state: Optional[State] = None,
) -> NoiseSensitivityResult:
    """Compare ideal and noisy execution and summarize gate-type sensitivity."""
    if backend is None:
        from ..backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    n_qubits = circuit.n_qubits
    if initial_state is None:
        initial_state = default_plus_state(n_qubits, backend)

    rho_ideal = evolve_density_gatewise(circuit, backend, initial_state, noise_model=None)
    rho_noisy = evolve_density_gatewise(circuit, backend, initial_state, noise_model=noise_model)
    probs_ideal = rho_ideal.probabilities()
    probs_noisy = rho_noisy.probabilities()
    fidelity_noisy = float(np.square(np.sum(np.sqrt(probs_ideal) * np.sqrt(probs_noisy))))
    avg_fidelity_loss = 1.0 - fidelity_noisy

    n_gates_by_type: Dict[str, int] = {}
    for gate in circuit_instructions(circuit):
        gate_type = instruction_name(gate)
        n_gates_by_type[gate_type] = n_gates_by_type.get(gate_type, 0) + 1

    return NoiseSensitivityResult(
        circuit=circuit,
        n_qubits=n_qubits,
        noise_model=noise_model,
        ideal_avg_fidelity=1.0,
        noisy_avg_fidelity=fidelity_noisy,
        avg_fidelity_loss=avg_fidelity_loss,
        gate_type_sensitivity=analyze_gate_type_sensitivity(circuit, noise_model),
        n_gates_total=circuit_instruction_count(circuit),
        n_gates_by_type=n_gates_by_type,
        noise_strength=estimate_noise_strength(noise_model) if noise_model else 0.0,
    )


def analyze_gate_type_sensitivity(circuit: Circuit, noise_model: Optional[NoiseModel]) -> Dict[str, float]:
    if noise_model is None:
        return {}
    sensitivity: Dict[str, float] = {}
    names = [instruction_name(gate) for gate in circuit_instructions(circuit)]
    for gate_type in set(names):
        count = sum(1 for name in names if name == gate_type)
        if gate_type in {"cx", "cnot", "cy", "cz", "rzz", "rxx", "swap", "crx", "cry", "crz"}:
            sensitivity[gate_type] = 0.02 * count
        elif gate_type in {"rx", "ry", "rz", "u2", "u3"}:
            sensitivity[gate_type] = 0.01 * count
        elif gate_type in {"hadamard", "s_gate", "t_gate"}:
            sensitivity[gate_type] = 0.005 * count
        else:
            sensitivity[gate_type] = 0.01 * count
    return sensitivity


__all__ = [
    "NoiseSensitivityResult",
    "analyze_gate_type_sensitivity",
    "default_plus_state",
    "estimate_noise_strength",
    "evolve_density_gatewise",
    "noise_sensitivity",
]

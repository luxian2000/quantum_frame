"""Trainability metrics for quantum circuits and ansatz templates."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.circuit import Circuit
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


__all__ = ["structure_proxy", "structure_proxy_details"]

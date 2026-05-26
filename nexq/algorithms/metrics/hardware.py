"""Hardware-efficiency metrics for quantum circuits."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ...core.circuit import Circuit
from ._utils import count_two_qubit_gates, depth_proxy


DEFAULT_NATIVE_GATES = ("hadamard", "rx", "ry", "rz", "cx", "cnot")


def native_depth_twoq_efficiency(
    circuit: Circuit,
    native_gates: Optional[Sequence[str]] = None,
    max_depth: int = 100,
) -> float:
    """Native-gate, depth, and two-qubit-density hardware efficiency proxy."""
    native_set = set(native_gates or DEFAULT_NATIVE_GATES)
    n_gates = len(circuit.gates)
    native_count = sum(1 for gate in circuit.gates if gate.get("type", "") in native_set)
    native_ratio = native_count / n_gates if n_gates > 0 else 1.0
    circuit_depth_proxy = depth_proxy(circuit)
    depth_score = min(1.0, float(max_depth) / max(1.0, circuit_depth_proxy * 10.0))
    two_qubit_ratio = count_two_qubit_gates(circuit) / max(1, int(circuit.n_qubits))
    twoq_efficiency = np.exp(-two_qubit_ratio / 3.0)
    return float(np.clip(0.4 * native_ratio + 0.3 * depth_score + 0.3 * twoq_efficiency, 0.0, 1.0))


def native_depth_twoq_efficiency_details(
    circuit: Circuit,
    native_gates: Optional[Sequence[str]] = None,
    max_depth: int = 100,
) -> Dict[str, Any]:
    native_set = set(native_gates or DEFAULT_NATIVE_GATES)
    n_gates = len(circuit.gates)
    native_count = sum(1 for gate in circuit.gates if gate.get("type", "") in native_set)
    return {
        "native_depth_twoq_efficiency_score": native_depth_twoq_efficiency(circuit, native_gates, max_depth),
        "native_gate_count": native_count,
        "native_gate_ratio": native_count / n_gates if n_gates > 0 else 1.0,
        "two_qubit_gate_count": count_two_qubit_gates(circuit),
        "depth_proxy": depth_proxy(circuit),
    }


__all__ = [
    "DEFAULT_NATIVE_GATES",
    "native_depth_twoq_efficiency",
    "native_depth_twoq_efficiency_details",
]

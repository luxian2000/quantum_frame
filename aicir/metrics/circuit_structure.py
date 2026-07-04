"""Circuit-level metrics shared across QAS and VQE utilities.

This module owns measurements of circuit structure.  QAS callers should import
these helpers instead of defining local copies, so filtering, oracle features,
and fair-label metadata count circuits the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.circuit import Circuit
from ..ir import circuit_instructions, instruction_parameter


def parameter_count(circuit: Circuit) -> int:
    """Count scalar trainable parameters stored on circuit gate records."""

    count = 0
    for instruction in circuit_instructions(circuit):
        parameter: Any = instruction_parameter(instruction)
        if parameter is None:
            continue
        if isinstance(parameter, (list, tuple, np.ndarray)):
            count += int(np.asarray(parameter).size)
        else:
            count += 1
    return count


def entanglement_coverage_score(two_q_count: float, n_qubits: int, layers: int, topology: str) -> float:
    """Estimate how fully an ansatz covers the available two-qubit edges."""

    n_qubits = max(2, int(n_qubits))
    layers = max(1, int(layers))
    max_edges = (n_qubits if str(topology) == "ring" else n_qubits - 1) * layers
    return min(1.0, float(two_q_count) / max(1.0, float(max_edges)))


def structural_expressibility_proxy_score(
    *,
    n_params: float,
    n_qubits: int,
    layers: int,
    rotation_block: str,
    final_rotation: str,
    entanglement_score: float,
) -> float:
    """Fast Stage-1 proxy; not a sampled Haar/KL/MMD expressibility metric."""

    n_qubits = max(1, int(n_qubits))
    layers = max(1, int(layers))
    rotation_richness = {
        "ry": 1.0 / 3.0,
        "ry_rz": 2.0 / 3.0,
        "rx_ry_rz": 1.0,
    }.get(str(rotation_block), 0.0)
    final_bonus = 0.15 if str(final_rotation) == "ry_rz" else 0.05
    param_density = min(1.0, float(n_params) / max(1.0, float(3 * n_qubits * (layers + 1))))
    score = 0.45 * rotation_richness + 0.35 * param_density + 0.20 * float(entanglement_score) + final_bonus
    return float(max(0.0, min(1.0, score)))


__all__ = ["entanglement_coverage_score", "parameter_count", "structural_expressibility_proxy_score"]

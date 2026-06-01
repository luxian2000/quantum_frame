"""Small MaxCut problem instances for QAS validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .base import ProblemInstance

WeightedEdge = Tuple[int, int, float]


def _bit(index: int, qubit: int, n_qubits: int) -> int:
    return (index >> (n_qubits - qubit - 1)) & 1


def _maxcut_values(n_qubits: int, edges: Sequence[WeightedEdge]) -> np.ndarray:
    values = np.zeros(1 << n_qubits, dtype=float)
    for state_index in range(1 << n_qubits):
        total = 0.0
        for source, target, weight in edges:
            if _bit(state_index, source, n_qubits) != _bit(state_index, target, n_qubits):
                total += float(weight)
        values[state_index] = total
    return values


@dataclass
class MaxCutInstance(ProblemInstance):
    """Weighted MaxCut instance with brute-force optimum."""

    edges: List[WeightedEdge] = None

    @classmethod
    def from_edges(cls, name: str, n_qubits: int, edges: Iterable[Tuple[int, int, float]]) -> "MaxCutInstance":
        edge_list = [(int(i), int(j), float(w)) for i, j, w in edges]
        for source, target, _ in edge_list:
            if source == target or source < 0 or target < 0 or source >= n_qubits or target >= n_qubits:
                raise ValueError(f"Invalid MaxCut edge ({source}, {target}) for {n_qubits} qubits")
        values = _maxcut_values(n_qubits, edge_list)
        return cls(
            name=name,
            n_qubits=n_qubits,
            objective_values=values,
            classical_optimum=float(values.max()),
            maximize=True,
            metadata={"family": "maxcut", "edges": edge_list},
            edges=edge_list,
        )


def maxcut_line(n_qubits: int = 4, weight: float = 1.0) -> MaxCutInstance:
    edges = [(qubit, qubit + 1, weight) for qubit in range(n_qubits - 1)]
    return MaxCutInstance.from_edges(f"maxcut_line_{n_qubits}", n_qubits, edges)


def maxcut_ring(n_qubits: int = 4, weight: float = 1.0) -> MaxCutInstance:
    edges = [(qubit, (qubit + 1) % n_qubits, weight) for qubit in range(n_qubits)]
    return MaxCutInstance.from_edges(f"maxcut_ring_{n_qubits}", n_qubits, edges)

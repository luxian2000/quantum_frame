"""Shared problem abstractions for QAS validation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ProblemInstance:
    """Classical objective used to judge optimized candidate circuits."""

    name: str
    n_qubits: int
    objective_values: np.ndarray
    classical_optimum: float
    maximize: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.objective_values, dtype=float).reshape(-1)
        expected = 1 << int(self.n_qubits)
        if values.shape[0] != expected:
            raise ValueError(f"objective_values must have length {expected}, got {values.shape[0]}")
        self.objective_values = values
        self.classical_optimum = float(self.classical_optimum)

    def evaluate_bitstring(self, bitstring: str) -> float:
        """Evaluate a computational-basis bitstring such as ``'0101'``."""
        if len(bitstring) != self.n_qubits or any(bit not in "01" for bit in bitstring):
            raise ValueError(f"bitstring must contain {self.n_qubits} binary characters")
        return float(self.objective_values[int(bitstring, 2)])

    def expected_objective(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.shape != self.objective_values.shape:
            raise ValueError("probabilities length must match objective_values length")
        total = float(probs.sum())
        if total <= 0.0:
            raise ValueError("probabilities must have positive total mass")
        probs = probs / total
        return float(np.dot(probs, self.objective_values))

    def approximation_ratio(self, value: float) -> Optional[float]:
        if abs(self.classical_optimum) < 1e-12:
            return None
        return float(value / self.classical_optimum)

    def normalized_gap(self, value: float) -> Optional[float]:
        if abs(self.classical_optimum) < 1e-12:
            return None
        if self.maximize:
            return float((self.classical_optimum - value) / abs(self.classical_optimum))
        return float((value - self.classical_optimum) / abs(self.classical_optimum))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "classical_optimum": self.classical_optimum,
            "maximize": self.maximize,
            "metadata": dict(self.metadata),
        }

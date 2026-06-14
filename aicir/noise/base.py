"""Abstract interfaces for noise channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class NoiseChannel(ABC):
    """Kraus channel abstraction: E(rho) = sum_k K_k rho K_k^dagger."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name."""

    @abstractmethod
    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        """Return Kraus operators embedded in the full n-qubit system."""

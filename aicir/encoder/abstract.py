"""Abstract base definitions for quantum data encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class BaseEncoder(ABC):
    """Abstract interface that all encoder implementations must satisfy."""

    def __init__(self, n_qubits: Optional[int] = None):
        """Initialize the encoder with optional qubit count."""
        self.n_qubits = n_qubits

    @abstractmethod
    def encode(self, data, *, cir: str = "dict", backend=None) -> Tuple[Any, Any]:
        """Encode classical data into a quantum representation.

        Returns:
            Tuple[circuit_repr, state]
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, quantum_state):
        """Decode quantum state back to classical data."""
        raise NotImplementedError

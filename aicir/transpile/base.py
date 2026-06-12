"""Base classes for circuit transpilation passes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.circuit import Circuit


class TransformationPass(ABC):
    """Base class for passes that return a transformed circuit."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def run(self, circuit: Circuit) -> Circuit:
        """Return a transformed copy of ``circuit``."""

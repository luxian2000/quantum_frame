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

    def _local_kraus(self, n_qubits: int, backend):
        """Return ``(local_matrix, logical_targets)`` pairs.

        Distributed density-matrix execution uses this protected hook to avoid
        constructing full-system Kraus matrices. Third-party channels must
        opt in explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未提供分布式执行所需的局部 Kraus 表示"
        )

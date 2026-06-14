"""Common single-qubit noise channels embedded through Kraus operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import NoiseChannel

_I2 = np.array([[1, 0], [0, 1]], dtype=np.complex64)
_X2 = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def _embed_single_qubit(op_2x2: np.ndarray, target_qubit: int, n_qubits: int, backend):
    mats = []
    for q in range(n_qubits):
        mats.append(backend.cast(op_2x2 if q == target_qubit else _I2))
    return backend.tensor_product(*mats)


@dataclass
class _SingleQubitChannel(NoiseChannel):
    target_qubit: int

    def _validate_target(self, n_qubits: int) -> None:
        if self.target_qubit < 0 or self.target_qubit >= n_qubits:
            raise ValueError(f"target_qubit={self.target_qubit} out of range [0, {n_qubits})")


@dataclass
class DepolarizingChannel(_SingleQubitChannel):
    """Single-qubit depolarizing channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "depolarizing"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("depolarizing p must be in [0, 1]")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p / 3.0) * _X2
        k2 = np.sqrt(p / 3.0) * _Y2
        k3 = np.sqrt(p / 3.0) * _Z2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k3, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class BitFlipChannel(_SingleQubitChannel):
    """Single-qubit bit-flip channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "bit_flip"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("bit flip p must be in [0, 1]")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p) * _X2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class PhaseFlipChannel(_SingleQubitChannel):
    """Single-qubit phase-flip channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "phase_flip"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("phase flip p must be in [0, 1]")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p) * _Z2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class AmplitudeDampingChannel(_SingleQubitChannel):
    """Single-qubit amplitude-damping channel with gamma in [0, 1]."""

    gamma: float

    @property
    def name(self) -> str:
        return "amplitude_damping"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("amplitude damping gamma must be in [0, 1]")

        g = float(self.gamma)
        k0 = np.array([[1, 0], [0, np.sqrt(1.0 - g)]], dtype=np.complex64)
        k1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex64)
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]

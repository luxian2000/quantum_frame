"""Quantum Fourier transform circuit builders."""

from __future__ import annotations

import math
import operator
from typing import List

from ..core.circuit import Circuit, crz, hadamard, swap


def _validate_qubit_range(n_qubits: int, start_qubit: int) -> tuple[int, int]:
    try:
        n = operator.index(n_qubits)
        start = operator.index(start_qubit)
    except TypeError as exc:
        raise TypeError("n_qubits and start_qubit must be integers") from exc
    if n <= 0:
        raise ValueError("n_qubits must be a positive integer")
    if start < 0:
        raise ValueError("start_qubit must be a non-negative integer")
    return n, start


def qft(n_qubits: int, start_qubit: int = 0) -> List[dict]:
    """Build a QFT gate sequence over consecutive qubits.

    Args:
        n_qubits: Number of consecutive qubits included in the QFT.
        start_qubit: Index of the first qubit. The QFT acts on
            ``start_qubit`` through ``start_qubit + n_qubits - 1``.

    Returns:
        A list of nexq core gate dictionaries.
    """
    n, start = _validate_qubit_range(n_qubits, start_qubit)
    stop = start + n
    gates: List[dict] = []

    for target in range(start, stop):
        gates.append(hadamard(target))
        for control in range(target + 1, stop):
            angle = math.pi / (2 ** (control - target))
            gates.append(crz(angle, target, [control]))

    for offset in range(n // 2):
        gates.append(swap(start + offset, stop - 1 - offset))

    return gates


def qft_circuit(n_qubits: int, start_qubit: int = 0) -> Circuit:
    """Build a QFT ``Circuit`` over consecutive qubits."""
    n, start = _validate_qubit_range(n_qubits, start_qubit)
    gates = qft(n, start)
    return Circuit(*gates, n_qubits=start + n)


__all__ = ["qft", "qft_circuit"]

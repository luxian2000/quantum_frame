"""Pauli-Hamiltonian problem helpers for QAS/VQE loops.

Closed-loop QAS accepts literal Pauli terms from queue rows; these presets are
small convenience constructors for smoke tests and demos, not separate search
pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


H2_HAMILTONIAN = (
    (-1.0523732, "II"),
    (0.3979374, "ZI"),
    (-0.3979374, "IZ"),
    (-0.0112801, "ZZ"),
    (0.1809312, "XX"),
)
H2_REFERENCE_ENERGY = -1.8572
ISING4_HAMILTONIAN = (
    (-1.0, "ZZII"),
    (-1.0, "IZZI"),
    (-1.0, "IIZZ"),
    (-0.5, "XIII"),
    (-0.5, "IXII"),
    (-0.5, "IIXI"),
    (-0.5, "IIIX"),
)


@dataclass(frozen=True)
class VQEProblem:
    """Pauli-Hamiltonian VQE task consumed by QAS fair-labeling."""

    name: str
    n_qubits: int
    hamiltonian: Sequence[tuple[float, str]]
    reference_energy: float


VQEDemoProblem = VQEProblem


def _pauli_matrix(label: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    result = np.array([[1]], dtype=np.complex128)
    for char in label:
        if char not in matrices:
            raise ValueError(f"unsupported Pauli character: {char!r}")
        result = np.kron(result, matrices[char])
    return result


def hamiltonian_matrix(hamiltonian: Sequence[tuple[float, str]]) -> np.ndarray:
    if not hamiltonian:
        raise ValueError("Hamiltonian must contain at least one Pauli term")
    n_qubits = len(hamiltonian[0][1])
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for coeff, pauli in hamiltonian:
        if len(pauli) != n_qubits:
            raise ValueError("All Pauli labels must have the same length")
        matrix += float(coeff) * _pauli_matrix(pauli)
    return matrix


def exact_ground_energy(hamiltonian: Sequence[tuple[float, str]]) -> float:
    return float(np.min(np.linalg.eigvalsh(hamiltonian_matrix(hamiltonian))))


def tfim_chain_hamiltonian(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> tuple[tuple[float, str], ...]:
    """Return H=-J sum ZZ - h sum X for an open or periodic chain."""

    n_qubits = int(n_qubits)
    if n_qubits < 2:
        raise ValueError("TFIM chain requires at least 2 qubits")

    terms: list[tuple[float, str]] = []
    edge_count = n_qubits if periodic else n_qubits - 1
    for index in range(edge_count):
        left = index
        right = (index + 1) % n_qubits
        pauli = ["I"] * n_qubits
        pauli[left] = "Z"
        pauli[right] = "Z"
        terms.append((-float(J), "".join(pauli)))
    for index in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[index] = "X"
        terms.append((-float(h), "".join(pauli)))
    return tuple(terms)


def h2_demo_problem() -> VQEProblem:
    return VQEProblem(
        name="h2_toy_2q",
        n_qubits=2,
        hamiltonian=H2_HAMILTONIAN,
        reference_energy=H2_REFERENCE_ENERGY,
    )


def ising4_demo_problem() -> VQEProblem:
    return VQEProblem(
        name="tfim_chain_4q_J1_h0.5",
        n_qubits=4,
        hamiltonian=ISING4_HAMILTONIAN,
        reference_energy=exact_ground_energy(ISING4_HAMILTONIAN),
    )


def tfim_chain_demo_problem(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> VQEProblem:
    hamiltonian = tfim_chain_hamiltonian(n_qubits=n_qubits, J=J, h=h, periodic=periodic)
    boundary = "ring" if periodic else "chain"
    return VQEProblem(
        name=f"tfim_{boundary}_{int(n_qubits)}q_J{J:g}_h{h:g}",
        n_qubits=int(n_qubits),
        hamiltonian=hamiltonian,
        reference_energy=exact_ground_energy(hamiltonian),
    )


def h2_hamiltonian_matrix() -> np.ndarray:
    return hamiltonian_matrix(H2_HAMILTONIAN)


__all__ = [
    "H2_HAMILTONIAN",
    "H2_REFERENCE_ENERGY",
    "ISING4_HAMILTONIAN",
    "VQEDemoProblem",
    "VQEProblem",
    "exact_ground_energy",
    "h2_demo_problem",
    "h2_hamiltonian_matrix",
    "hamiltonian_matrix",
    "ising4_demo_problem",
    "tfim_chain_demo_problem",
    "tfim_chain_hamiltonian",
]

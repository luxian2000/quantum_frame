"""Preset molecular qubit Hamiltonians.

The presets in this module are small benchmark Hamiltonians with fixed
geometry, basis, and qubit mapping. They are intended for VQE examples and
tests, not as a replacement for an electronic-structure pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..channel.backends.numpy_backend import NumpyBackend
from ..channel.operators import Hamiltonian


PauliTerm = tuple[complex, str]


def _normalize_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("molecule name cannot be empty")
    return key


@dataclass(frozen=True)
class MoleculeHamiltonian:
    """A fixed molecular qubit Hamiltonian preset."""

    name: str
    formula: str
    n_qubits: int
    terms: tuple[PauliTerm, ...]
    basis: str
    mapping: str
    geometry: str
    source: str
    description: str = ""

    def to_hamiltonian(self) -> Hamiltonian:
        """Build a fresh :class:`Hamiltonian` instance from the preset terms."""

        terms: list[tuple[str, complex]] = []
        for coefficient, pauli in self.terms:
            if len(pauli) != self.n_qubits:
                raise ValueError(
                    f"Preset {self.name!r} has Pauli string {pauli!r} "
                    f"with length {len(pauli)} for n_qubits={self.n_qubits}"
                )
            terms.append((pauli, coefficient))
        return Hamiltonian(n_qubits=self.n_qubits, terms=terms)

    def to_matrix(self, backend=None) -> np.ndarray:
        """Return the dense_matrix representation using ``backend``."""

        backend = backend if backend is not None else NumpyBackend()
        return np.asarray(backend.to_numpy(self.to_hamiltonian().to_matrix(backend)))


_QISKIT_NATURE_MAPPER_TUTORIAL = (
    "https://qiskit-community.github.io/qiskit-nature/tutorials/06_qubit_mappers.html"
)


# H2, STO-3G, PySCFDriver default geometry, parity mapper with two-qubit
# reduction.
H2_STO3G_PARITY_2Q = MoleculeHamiltonian(
    name="h2",
    formula="H2",
    n_qubits=2,
    basis="STO-3G",
    mapping="ParityMapper(two_qubit_reduction)",
    geometry="PySCFDriver default H2 geometry",
    source=_QISKIT_NATURE_MAPPER_TUTORIAL,
    description="Compact two-qubit H2 Hamiltonian commonly used in VQE examples.",
    terms=(
        (-1.05237325, "II"),
        (0.39793742, "IZ"),
        (-0.39793742, "ZI"),
        (-0.01128010, "ZZ"),
        (0.18093120, "XX"),
    ),
)


# H2, STO-3G, Jordan-Wigner mapper before symmetry reduction.
H2_STO3G_JW_4Q = MoleculeHamiltonian(
    name="h2_jw",
    formula="H2",
    n_qubits=4,
    basis="STO-3G",
    mapping="JordanWignerMapper",
    geometry="PySCFDriver default H2 geometry",
    source=_QISKIT_NATURE_MAPPER_TUTORIAL,
    description="Four-qubit Jordan-Wigner H2 Hamiltonian.",
    terms=(
        (-0.81054798, "IIII"),
        (0.17218393, "IIIZ"),
        (-0.22575349, "IIZI"),
        (0.12091263, "IIZZ"),
        (0.17218393, "IZII"),
        (0.16892754, "IZIZ"),
        (-0.22575349, "ZIII"),
        (0.16614543, "ZIIZ"),
        (0.04523280, "YYYY"),
        (0.04523280, "XXYY"),
        (0.04523280, "YYXX"),
        (0.04523280, "XXXX"),
        (0.16614543, "IZZI"),
        (0.17464343, "ZIZI"),
        (0.12091263, "ZZII"),
    ),
)


# H2, STO-3G, one-qubit tapered Hamiltonian from the same mapper tutorial.
H2_STO3G_TAPERED_1Q = MoleculeHamiltonian(
    name="h2_tapered",
    formula="H2",
    n_qubits=1,
    basis="STO-3G",
    mapping="TaperedQubitMapper",
    geometry="PySCFDriver default H2 geometry",
    source=_QISKIT_NATURE_MAPPER_TUTORIAL,
    description="One-qubit tapered H2 Hamiltonian.",
    terms=(
        (-1.04109314, "I"),
        (-0.79587485, "Z"),
        (-0.18093120, "X"),
    ),
)


MOLECULES: dict[str, MoleculeHamiltonian] = {
    H2_STO3G_PARITY_2Q.name: H2_STO3G_PARITY_2Q,
    H2_STO3G_JW_4Q.name: H2_STO3G_JW_4Q,
    H2_STO3G_TAPERED_1Q.name: H2_STO3G_TAPERED_1Q,
}


def available_molecules() -> tuple[str, ...]:
    """Return available canonical preset names."""

    return tuple(sorted(MOLECULES))


def get_molecule(name: str) -> MoleculeHamiltonian:
    """Return a molecular Hamiltonian preset by canonical name."""

    key = _normalize_name(name)
    try:
        return MOLECULES[key]
    except KeyError as exc:
        available = ", ".join(available_molecules())
        raise KeyError(f"Unknown molecule Hamiltonian {name!r}. Available: {available}") from exc


def molecule_hamiltonian(name: str) -> Hamiltonian:
    """Return a fresh :class:`Hamiltonian` for ``name``."""

    return get_molecule(name).to_hamiltonian()


def molecule_matrix(name: str, backend=None) -> np.ndarray:
    """Return a dense_matrix Hamiltonian for ``name``."""

    return get_molecule(name).to_matrix(backend=backend)


def iter_molecules(names: Iterable[str] | None = None) -> tuple[MoleculeHamiltonian, ...]:
    """Return presets for ``names`` or all canonical presets."""

    if names is None:
        return tuple(MOLECULES[name] for name in available_molecules())
    return tuple(get_molecule(name) for name in names)


__all__ = [
    "MoleculeHamiltonian",
    "H2_STO3G_PARITY_2Q",
    "H2_STO3G_JW_4Q",
    "H2_STO3G_TAPERED_1Q",
    "MOLECULES",
    "available_molecules",
    "get_molecule",
    "iter_molecules",
    "molecule_hamiltonian",
    "molecule_matrix",
]

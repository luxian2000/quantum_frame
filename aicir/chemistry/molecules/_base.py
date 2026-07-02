"""Molecule preset infrastructure: dataclass, registry, and accessors.

Each verified molecule lives in its own ``molecules/<name>.py`` module and calls
:func:`register_molecule` at import time. This module holds only the shared
machinery; it registers no molecules itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ...backends.numpy_backend import NumpyBackend
from ...core.operators import Hamiltonian

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
    n_electrons: int | None = None
    hf_occupation: tuple[int, ...] | None = None
    excitations: tuple[tuple[str, tuple[int, ...]], ...] | None = None

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


MOLECULES: dict[str, MoleculeHamiltonian] = {}


def register_molecule(molecule: MoleculeHamiltonian) -> MoleculeHamiltonian:
    """Register ``molecule`` into :data:`MOLECULES` (keyed by canonical name)."""

    key = _normalize_name(molecule.name)
    if key in MOLECULES:
        raise ValueError(f"molecule {molecule.name!r} already registered")
    MOLECULES[key] = molecule
    return molecule


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

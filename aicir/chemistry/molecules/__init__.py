"""Verified molecular qubit Hamiltonian presets.

Each molecule lives in its own module named by formula (``H2.py``, ``LiH.py``,
``H2O.py``, ``NH3.py``, ``N2.py``, ``BeH2.py``) and registers itself into
:data:`MOLECULES` on import. Presets are fixed benchmark Hamiltonians with
coefficients from PySCF / Qiskit Nature; this package is not an
electronic-structure pipeline.

Small presets (H2/LiH/H2O/NH3) are verified by dense-matrix ground energy in the
tests; the 14–16 qubit N2/BeH2 are too large for dense diagonalization and carry
structural guards only.

Public API is unchanged from the previous single-module layout: import
``MoleculeHamiltonian``/``get_molecule``/``molecule_hamiltonian`` etc. from
``aicir.chemistry``.
"""

from __future__ import annotations

from ._base import (
    MOLECULES,
    MoleculeHamiltonian,
    available_molecules,
    get_molecule,
    iter_molecules,
    molecule_hamiltonian,
    molecule_matrix,
    register_molecule,
)

# Import side effect: each module registers its preset(s) into MOLECULES.
from .BeH2 import BEH2_321G_JW_16Q
from .H2 import H2_STO3G_JW_4Q, H2_STO3G_PARITY_2Q, H2_STO3G_TAPERED_1Q
from .H2O import H2O_STO3G_JW_6Q
from .LiH import LIH_STO3G_JW_4Q
from .N2 import N2_STO3G_JW_14Q
from .NH3 import NH3_STO3G_JW_12Q

__all__ = [
    "MoleculeHamiltonian",
    "H2_STO3G_PARITY_2Q",
    "H2_STO3G_JW_4Q",
    "H2_STO3G_TAPERED_1Q",
    "LIH_STO3G_JW_4Q",
    "H2O_STO3G_JW_6Q",
    "NH3_STO3G_JW_12Q",
    "N2_STO3G_JW_14Q",
    "BEH2_321G_JW_16Q",
    "MOLECULES",
    "register_molecule",
    "available_molecules",
    "get_molecule",
    "iter_molecules",
    "molecule_hamiltonian",
    "molecule_matrix",
]

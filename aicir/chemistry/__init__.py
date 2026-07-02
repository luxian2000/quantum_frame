"""aicir.chemistry

量子化学工具模块（电子结构、哈密顿映射、分子算符构建）。
"""

from .molecules import (
    BEH2_321G_JW_16Q,
    H2_STO3G_JW_4Q,
    H2_STO3G_PARITY_2Q,
    H2_STO3G_TAPERED_1Q,
    H2O_STO3G_JW_6Q,
    LIH_STO3G_JW_4Q,
    MOLECULES,
    MoleculeHamiltonian,
    N2_STO3G_JW_14Q,
    NH3_STO3G_JW_12Q,
    available_molecules,
    get_molecule,
    iter_molecules,
    molecule_hamiltonian,
    molecule_matrix,
    register_molecule,
)
from .pipeline import build_molecule

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
    "build_molecule",
]

"""aicir.chemistry

量子化学工具模块（电子结构、哈密顿映射、分子算符构建）。
"""

from .molecule import (
    H2_STO3G_JW_4Q,
    H2_STO3G_JW_R0735_4Q,
    H2_STO3G_PARITY_2Q,
    H2_STO3G_TAPERED_1Q,
    MOLECULES,
    MoleculeHamiltonian,
    available_molecules,
    get_molecule,
    iter_molecules,
    molecule_hamiltonian,
    molecule_matrix,
)
from .spec import (
    GeneratedHamiltonian,
    MolecularSpec,
    PauliTermsSpec,
    PresetSpec,
    generate_hamiltonian,
    load_hamiltonian_input,
    spec_from_mapping,
)

__all__ = [
    "MoleculeHamiltonian",
    "H2_STO3G_PARITY_2Q",
    "H2_STO3G_JW_4Q",
    "H2_STO3G_JW_R0735_4Q",
    "H2_STO3G_TAPERED_1Q",
    "MOLECULES",
    "available_molecules",
    "get_molecule",
    "iter_molecules",
    "molecule_hamiltonian",
    "molecule_matrix",
    "GeneratedHamiltonian",
    "MolecularSpec",
    "PauliTermsSpec",
    "PresetSpec",
    "generate_hamiltonian",
    "load_hamiltonian_input",
    "spec_from_mapping",
]

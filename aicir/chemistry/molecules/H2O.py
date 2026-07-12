"""H2O active-space qubit Hamiltonian preset.

PySCF/Qiskit Nature: sto3g basis, 4-electron/3-orbital active space, JordanWignerMapper.
Verified against the dense-matrix ground energy in tests/chemistry/test_molecules.py.
"""

from __future__ import annotations

from ._base import MoleculeHamiltonian, register_molecule


# n_electrons/hf_occupation/excitations 静态补齐：build_molecule(同几何/基组，
# active_electrons=4, active_orbitals=3, mapping="jordan_wigner") 现算对照，terms
# 逐项吻合（误差 <3e-15）。
H2O_STO3G_JW_6Q = register_molecule(
    MoleculeHamiltonian(
        name="h2o",
        formula="H2O",
        n_qubits=6,
        basis="sto3g",
        mapping="JordanWignerMapper",
        geometry="O 0.000000 0.000000 0.000000; H 0.757160 0.586260 0.000000; H -0.757160 0.586260 0.000000",
        source="PySCF/Qiskit Nature (ActiveSpaceTransformer + JordanWignerMapper)",
        description="H2O 4e/3o active-space Hamiltonian (6 qubits).",
        n_electrons=4,
        hf_occupation=(0, 1, 1, 0, 1, 1),
        excitations=(
            ("single", (5, 3)),
            ("single", (4, 3)),
            ("single", (2, 0)),
            ("single", (1, 0)),
            ("double", (5, 2, 3, 0)),
            ("double", (5, 1, 3, 0)),
            ("double", (4, 2, 3, 0)),
            ("double", (4, 1, 3, 0)),
        ),
        terms=(
            (-4.5241061234101245, "IIIIII"),
            (0.5153159797373158, "IIIIIZ"),
            (-0.0777577585041152, "IIIYZY"),
            (-0.0777577585041152, "IIIXZX"),
            (0.4813205218163351, "IIIIZI"),
            (0.0902426376429984, "IIIZII"),
            (0.5153159797373158, "IIZIII"),
            (-0.0777577585041153, "YZYIII"),
            (-0.0777577585041153, "XZXIII"),
            (0.4813205218163352, "IZIIII"),
            (0.0902426376429983, "ZIIIII"),
            (0.1682539303981457, "IIIIZZ"),
            (0.1200923081829575, "IIIZIZ"),
            (0.19568956513667, "IIZIIZ"),
            (-0.0303380764625682, "YZYIIZ"),
            (-0.0303380764625682, "XZXIIZ"),
            (0.1822330598869542, "IZIIIZ"),
            (0.1372661146010317, "ZIIIIZ"),
            (-0.0295021027117906, "IIIYIY"),
            (-0.0295021027117906, "IIIXIX"),
            (0.0139791294888085, "IYYIYY"),
            (0.0139791294888085, "IXXIYY"),
            (0.0139791294888085, "IYYIXX"),
            (0.0139791294888085, "IXXIXX"),
            (-0.0004375325735518, "YYIIYY"),
            (-0.0004375325735518, "XXIIYY"),
            (-0.0004375325735518, "YYIIXX"),
            (-0.0004375325735518, "XXIIXX"),
            (-0.0303380764625682, "IIZYZY"),
            (-0.0303380764625682, "IIZXZX"),
            (0.0171738064180742, "YZYYZY"),
            (0.0171738064180742, "XZXYZY"),
            (0.0171738064180742, "YZYXZX"),
            (0.0171738064180742, "XZXXZX"),
            (-0.0290645701382388, "IZIYZY"),
            (-0.0290645701382388, "IZIXZX"),
            (-0.0111466920449807, "ZIIYZY"),
            (-0.0111466920449807, "ZIIXZX"),
            (0.1375870149850004, "IIIZZI"),
            (0.1822330598869542, "IIZIZI"),
            (-0.0290645701382388, "YZYIZI"),
            (-0.0290645701382388, "XZXIZI"),
            (0.2200397733437616, "IZIIZI"),
            (0.1472359927649237, "ZIIIZI"),
            (-0.0004375325735518, "IYYYYI"),
            (-0.0004375325735518, "IXXYYI"),
            (-0.0004375325735518, "IYYXXI"),
            (-0.0004375325735518, "IXXXXI"),
            (0.0096489777799233, "YYIYYI"),
            (0.0096489777799233, "XXIYYI"),
            (0.0096489777799233, "YYIXXI"),
            (0.0096489777799233, "XXIXXI"),
            (0.1372661146010317, "IIZZII"),
            (-0.0111466920449807, "YZYZII"),
            (-0.0111466920449807, "XZXZII"),
            (0.1472359927649237, "IZIZII"),
            (0.1492816648983666, "ZIIZII"),
            (0.1682539303981457, "IZZIII"),
            (0.1200923081829575, "ZIZIII"),
            (-0.0295021027117906, "YIYIII"),
            (-0.0295021027117906, "XIXIII"),
            (0.1375870149850004, "ZZIIII"),
        ),
    )
)

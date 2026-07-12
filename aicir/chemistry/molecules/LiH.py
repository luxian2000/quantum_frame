"""LiH active-space qubit Hamiltonian preset.

PySCF/Qiskit Nature: sto3g basis, 2-electron/2-orbital active space, JordanWignerMapper.
Verified against the dense-matrix ground energy in tests/chemistry/test_molecules.py.
"""

from __future__ import annotations

from ._base import MoleculeHamiltonian, register_molecule


# n_electrons/hf_occupation/excitations 静态补齐：build_molecule(同几何/基组，
# active_electrons=2, active_orbitals=2, mapping="jordan_wigner") 现算对照，terms
# 逐项吻合（误差 <4e-16）。
LIH_STO3G_JW_4Q = register_molecule(
    MoleculeHamiltonian(
        name="lih",
        formula="LiH",
        n_qubits=4,
        basis="sto3g",
        mapping="JordanWignerMapper",
        geometry="Li 0.000000 0.000000 0.000000; H 0.000000 0.000000 1.595000",
        source="PySCF/Qiskit Nature (ActiveSpaceTransformer + JordanWignerMapper)",
        description="LiH 2e/2o active-space Hamiltonian (4 qubits).",
        n_electrons=2,
        hf_occupation=(0, 1, 0, 1),
        excitations=(
            ("single", (3, 2)),
            ("single", (1, 0)),
            ("double", (3, 1, 2, 0)),
        ),
        terms=(
            (-0.705940988128576, "IIII"),
            (0.156139531200633, "IIIZ"),
            (-0.0149911864321253, "IIZI"),
            (0.0526847769863971, "IIZZ"),
            (0.156139531200633, "IZII"),
            (0.121914456537852, "IZIZ"),
            (0.0139782944733026, "YYII"),
            (0.0121237381452783, "YYIZ"),
            (0.0139782944733026, "XXII"),
            (0.0121237381452783, "XXIZ"),
            (-0.0149911864321253, "ZIII"),
            (0.0559382693417738, "ZIIZ"),
            (0.0139782944733026, "IIYY"),
            (0.0121237381452783, "IZYY"),
            (0.0139782944733026, "IIXX"),
            (0.0121237381452783, "IZXX"),
            (0.0032534923553767, "YYYY"),
            (0.0032534923553767, "XXYY"),
            (0.0032534923553767, "YYXX"),
            (0.0032534923553767, "XXXX"),
            (-0.0018545501857935, "ZIYY"),
            (-0.0018545501857935, "ZIXX"),
            (0.0559382693417738, "IZZI"),
            (-0.0018545501857935, "YYZI"),
            (-0.0018545501857935, "XXZI"),
            (0.0844837493973667, "ZIZI"),
            (0.0526847769863971, "ZZII"),
        ),
    )
)

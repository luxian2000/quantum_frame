"""H2 qubit Hamiltonian presets (STO-3G).

Coefficients from the Qiskit Nature qubit-mapper tutorial. Verified against the
dense-matrix ground energy in ``tests/chemistry/test_molecules.py``.
"""

from __future__ import annotations

from ._base import MoleculeHamiltonian, register_molecule

_QISKIT_NATURE_MAPPER_TUTORIAL = (
    "https://qiskit-community.github.io/qiskit-nature/tutorials/06_qubit_mappers.html"
)


# H2, STO-3G, PySCFDriver default geometry, parity mapper with two-qubit reduction.
#
# n_electrons/hf_occupation/excitations 静态补齐：用 build_molecule(与本 preset 同一
# 几何/基组/mapping) 现算对照，terms 逐项吻合（误差 <5e-9，来自 preset 系数四舍五入到
# 8 位小数），再抄录 build_molecule 的元数据输出。Parity mapper 的 hf_occupation/
# excitations 是变换后 bitstring/结构索引，不声明 mapper-correct 化学 UCCSD 激发
# （build_molecule docstring 已注明）。
H2_STO3G_PARITY_2Q = register_molecule(
    MoleculeHamiltonian(
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
        n_electrons=2,
        hf_occupation=(0, 1),
        excitations=(("single", (1, 0)),),
    )
)


# H2, STO-3G, Jordan-Wigner mapper before symmetry reduction.
#
# n_electrons/hf_occupation/excitations 同上：build_molecule(同几何/基组，
# mapping="jordan_wigner") 现算对照，terms 逐项吻合（误差 <3e-9）。
H2_STO3G_JW_4Q = register_molecule(
    MoleculeHamiltonian(
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
        n_electrons=2,
        hf_occupation=(0, 1, 0, 1),
        excitations=(
            ("single", (3, 2)),
            ("single", (1, 0)),
            ("double", (3, 1, 2, 0)),
        ),
    )
)


# H2, STO-3G, one-qubit tapered Hamiltonian from the same mapper tutorial.
#
# 未做 n_electrons/hf_occupation/excitations 补齐：TaperedQubitMapper 不在
# pipeline.build_molecule 支持的三种 mapper（jordan_wigner/parity/bravyi_kitaev）
# 之列，没有可比对的现算路径；且 tapering 后的单比特表示不再与任一自旋轨道占据数
# 一一对应，无法静态推导有意义的 HF bitstring/结构激发，故保持 None，跳过补齐。
H2_STO3G_TAPERED_1Q = register_molecule(
    MoleculeHamiltonian(
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
)

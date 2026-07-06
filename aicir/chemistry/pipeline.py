"""电子结构流水线：给定分子几何/基组/映射，现算 qubit Hamiltonian。

底层调 Qiskit Nature（内部包 PySCF）。属可选能力，需 ``pip install -e ".[chem]"``。
与固定预置并列——预置是快速、零依赖的常用分子；本流水线支持任意分子。
"""

from __future__ import annotations

from .molecules._base import MoleculeHamiltonian

_CHEM_INSTALL_HINT = (
    "电子结构流水线需要 qiskit-nature 与 pyscf；请安装可选依赖："
    'pip install -e ".[chem]"'
)


def _qiskit_nature_available() -> bool:
    try:
        import qiskit_nature  # noqa: F401
    except ImportError:
        return False
    return True


def _require_qiskit_nature():
    if not _qiskit_nature_available():
        raise ImportError(_CHEM_INSTALL_HINT)


def build_molecule(
    geometry,
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    mapping: str = "jordan_wigner",
    active_electrons=None,
    active_orbitals=None,
    two_qubit_reduction: bool = False,
    name: str = "custom",
) -> MoleculeHamiltonian:
    """给定分子几何/基组/映射，现算 qubit Hamiltonian。

    仅 ``mapping="jordan_wigner"`` 填充 ``n_electrons``/``hf_occupation``/
    ``excitations``；``parity``/``bravyi_kitaev`` 仍返回可用 Hamiltonian，但元数据为 None。
    """

    _require_qiskit_nature()
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    driver = PySCFDriver(atom=geometry, basis=basis, charge=charge, spin=spin)
    problem = driver.run()
    if active_electrons is not None and active_orbitals is not None:
        problem = ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)

    second_q_op = problem.hamiltonian.second_q_op()

    mapping_key = mapping.lower()
    if mapping_key in ("jordan_wigner", "jw"):
        mapper = JordanWignerMapper()
    elif mapping_key == "parity":
        mapper = (
            ParityMapper(num_particles=problem.num_particles)
            if two_qubit_reduction
            else ParityMapper()
        )
    elif mapping_key in ("bravyi_kitaev", "bk"):
        mapper = BravyiKitaevMapper()
    else:
        raise ValueError(f"未知 mapping: {mapping!r}")

    qubit_op = mapper.map(second_q_op)
    terms = _sparse_pauli_to_terms(qubit_op)  # 转 aicir PauliTerm，对齐比特序
    n_qubits = qubit_op.num_qubits

    n_electrons = hf_occupation = excitations = None
    if mapping_key in ("jordan_wigner", "jw"):
        n_electrons = sum(problem.num_particles)
        hf_occupation = _jw_hf_occupation(problem, n_qubits)
        excitations = _jw_excitations(problem, n_qubits)

    return MoleculeHamiltonian(
        name=name,
        formula=name.upper(),
        n_qubits=n_qubits,
        terms=terms,
        basis=basis.upper(),
        mapping=mapping,
        geometry=str(geometry),
        source="aicir.chemistry.build_molecule (Qiskit Nature/PySCF)",
        n_electrons=n_electrons,
        hf_occupation=hf_occupation,
        excitations=excitations,
    )


def _sparse_pauli_to_terms(qubit_op):
    """Qiskit SparsePauliOp → aicir PauliTerm 元组。

    经验证：现有 aicir 分子预置（如 ``h2_jw``）直接采用 Qiskit
    ``SparsePauliOp.to_list()`` 的标签顺序（qubit 0 在字符串最右），并未翻转；
    为与已提交预置的 ``terms`` 逐项吻合（免费 oracle），此处保持标签原样，不做翻转。
    """

    out = []
    for label, coeff in sorted(qubit_op.to_list(), key=lambda t: t[0]):
        out.append((complex(coeff), label))
    return tuple(out)


def _jw_hf_occupation(problem, n_qubits):
    """JW 下 HF 占据 bitstring，元组下标即 aicir 比特序号（与 ``_sparse_pauli_to_terms`` 同一比特序）。"""

    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, JordanWignerMapper())
    # HartreeFock._bitstr[i] 是 Qiskit 自旋轨道序（qubit i，label 最右字符）下的占据情况；
    # 而 aicir 的比特序是 leftmost=qubit0=MSB，terms 里的 Pauli label 仍保持 Qiskit
    # 原序未翻转（为复现预置），因此元数据须显式按 n-1-k 反转，才能与 terms 的比特
    # 对齐——否则用 hf_occupation 摆 HF 态、算 ⟨HF|H|HF⟩ 会对错 qubit 施加 X 门。
    qiskit_bits = [int(bit) for bit in hf._bitstr]
    return tuple(qiskit_bits[n_qubits - 1 - i] for i in range(n_qubits))


def _jw_excitations(problem, n_qubits):
    """singles+doubles 费米子激发 → aicir qubit 索引元组。"""

    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    out = []
    for order, kind in ((1, "single"), (2, "double")):
        raw = generate_fermionic_excitations(
            order, problem.num_spatial_orbitals, problem.num_particles
        )
        for occ, vir in raw:
            # 同 _jw_hf_occupation：Qiskit 自旋轨道索引 k 需按 n_qubits-1-k 映射为
            # aicir 比特序号，才能与（未翻转的）terms 对齐；元组内顺序无关紧要，
            # uccsd 会自行排序。
            idx = tuple(n_qubits - 1 - int(i) for i in (*occ, *vir))
            out.append((kind, idx))
    return tuple(out)

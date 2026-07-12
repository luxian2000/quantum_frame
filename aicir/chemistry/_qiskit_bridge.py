"""Qiskit Nature 交互的共享私有模块。

``pipeline.build_molecule`` 与 ``spec.generate_hamiltonian`` 的 ``MolecularSpec``
分支都需要同一套 Qiskit Nature 操作：driver 构造、mapper 选择、active-space
变换、``SparsePauliOp -> aicir PauliTerm`` 转换。此前两条路径各自维护一份，
且比特序互相镜像（``spec.py`` 对 label 做了 ``[::-1]`` 翻转）——本模块把共享部分
收敛到一处，两条路径改为委托这里，确保比特序自动保持一致。

``qiskit_nature``/``pyscf`` 均为可选依赖（``chem`` extra），本模块所有函数都在
函数体内惰性 import，绝不在模块顶层触碰这两个包。

比特序约定
----------
``QUBIT_ORDER = "qiskit_label"``：canonical 比特序就是 Qiskit
``SparsePauliOp.to_list()`` 返回的 label 原样字符串，不做任何翻转。这与已冻结的
``aicir/chemistry/molecules/*.py`` 预置完全一致（预置的 terms 就是这样生成、
未经翻转地抄录下来的），是 aicir 全仓关于分子 Hamiltonian 的 canonical 比特序。
若某个调用方确实需要旧的镜像序，请显式调用 :func:`reverse_pauli_labels`——默认
生成路径都不会调用它。
"""

from __future__ import annotations

from typing import Any, Mapping

QUBIT_ORDER = "qiskit_label"


def qiskit_nature_available() -> bool:
    """惰性探测 ``qiskit_nature`` 是否可导入。"""

    try:
        import qiskit_nature  # noqa: F401
    except ImportError:
        return False
    return True


def qiskit_distance_unit(unit: str):
    """字符串距离单位 -> ``qiskit_nature.units.DistanceUnit``。"""

    from qiskit_nature.units import DistanceUnit

    key = str(unit).strip().lower()
    if key in {"angstrom", "ang", "a"}:
        return DistanceUnit.ANGSTROM
    if key in {"bohr", "b"}:
        return DistanceUnit.BOHR
    raise ValueError(f"Unsupported molecular distance unit {unit!r}")


def build_driver(
    geometry: str,
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    unit=None,
):
    """构造 ``PySCFDriver``。``unit`` 为 None 时不传（沿用 PySCFDriver 默认）。"""

    from qiskit_nature.second_q.drivers import PySCFDriver

    kwargs: dict[str, Any] = dict(atom=geometry, basis=basis, charge=charge, spin=spin)
    if unit is not None:
        kwargs["unit"] = unit
    return PySCFDriver(**kwargs)


def select_mapper(mapping: str, *, num_particles=None, two_qubit_reduction: bool = False):
    """按名字选 mapper（``jordan_wigner``/``parity``/``bravyi_kitaev``，含别名）。"""

    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    key = str(mapping).strip().lower().replace("-", "_")
    if key in ("jordan_wigner", "jw"):
        return JordanWignerMapper()
    if key == "parity":
        if two_qubit_reduction:
            return ParityMapper(num_particles=num_particles)
        return ParityMapper()
    if key in ("bravyi_kitaev", "bk"):
        return BravyiKitaevMapper()
    raise ValueError(f"未知 mapping: {mapping!r}")


def apply_active_space(problem, *, active_electrons=None, active_orbitals=None):
    """按 (electrons, orbitals) 整数对做 active-space 变换（``pipeline`` 风格）。"""

    if active_electrons is None or active_orbitals is None:
        return problem
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    return ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)


def apply_active_space_mapping(problem, active_space: Mapping[str, Any] | None):
    """按 mapping 形式的 active-space 描述做变换（``spec.MolecularSpec`` 风格）。"""

    if not active_space:
        return problem
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    kwargs: dict[str, Any] = {}
    if "num_electrons" in active_space:
        value = active_space["num_electrons"]
        kwargs["num_electrons"] = tuple(value) if isinstance(value, list) else value
    if "num_spatial_orbitals" in active_space:
        kwargs["num_spatial_orbitals"] = int(active_space["num_spatial_orbitals"])
    if "active_orbitals" in active_space:
        kwargs["active_orbitals"] = tuple(int(item) for item in active_space["active_orbitals"])
    transformer = ActiveSpaceTransformer(**kwargs)
    return transformer.transform(problem)


def sparse_pauli_to_terms(qubit_op, *, sort: bool = True) -> tuple[tuple[complex, str], ...]:
    """``SparsePauliOp`` -> ``(complex 系数, label)`` 元组，canonical 比特序，不翻转。

    已验证：现有 aicir 分子预置（如 ``h2_jw``）直接采用 Qiskit
    ``SparsePauliOp.to_list()`` 的标签顺序，并未翻转；为与已提交预置的 ``terms``
    逐项吻合（免费 oracle），这里保持标签原样。
    """

    items = qubit_op.to_list()
    if sort:
        items = sorted(items, key=lambda item: item[0])
    return tuple((complex(coeff), label) for label, coeff in items)


def reverse_pauli_labels(
    terms: tuple[tuple[complex, str], ...]
) -> tuple[tuple[complex, str], ...]:
    """显式翻转每个 Pauli label 的比特序（``label[::-1]``）。

    仅供明确需要旧"镜像"比特序的调用方使用；``pipeline.build_molecule`` 与
    ``spec.generate_hamiltonian`` 的默认路径都**不**调用此函数——canonical
    比特序是 :data:`QUBIT_ORDER`（``qiskit_label``，即不翻转），与已冻结预置一致。
    """

    return tuple((coeff, str(label)[::-1]) for coeff, label in terms)


def hf_occupation_from_mapper(problem, mapper, n_qubits: int) -> tuple[int, ...]:
    """mapper 派生 HF 占据 bitstring，下标即 aicir（=canonical，未翻转）比特序号。"""

    from qiskit_nature.second_q.circuit.library import HartreeFock

    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    # HartreeFock._bitstr[i] 是 Qiskit 自旋轨道序（qubit i，label 最右字符）下的占据情况；
    # 而 aicir 的比特序是 leftmost=qubit0=MSB，terms 里的 Pauli label 仍保持 Qiskit
    # 原序未翻转（为复现预置），因此元数据须显式按 n-1-k 反转，才能与 terms 的比特
    # 对齐——否则用 hf_occupation 摆 HF 态、算 ⟨HF|H|HF⟩ 会对错 qubit 施加 X 门。
    qiskit_bits = [int(bit) for bit in hf._bitstr]
    if len(qiskit_bits) != n_qubits:
        qiskit_bits = qiskit_bits[:n_qubits]
    return tuple(qiskit_bits[n_qubits - 1 - i] for i in range(n_qubits))


def structural_excitations(problem, n_qubits: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """singles+doubles 费米子激发 -> aicir 结构索引元组（比特序同上，不翻转对齐）。"""

    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    out = []
    for order, kind in ((1, "single"), (2, "double")):
        raw = generate_fermionic_excitations(
            order, problem.num_spatial_orbitals, problem.num_particles
        )
        for occ, vir in raw:
            # 同 hf_occupation_from_mapper：Qiskit 自旋轨道索引 k 需按 n_qubits-1-k
            # 映射为 aicir 比特序号，才能与（未翻转的）terms 对齐；元组内顺序无关紧要，
            # uccsd 会自行排序。
            idx = tuple(n_qubits - 1 - int(i) for i in (*occ, *vir))
            if all(0 <= i < n_qubits for i in idx):
                out.append((kind, idx))
    return tuple(out)


__all__ = [
    "QUBIT_ORDER",
    "apply_active_space",
    "apply_active_space_mapping",
    "build_driver",
    "hf_occupation_from_mapper",
    "qiskit_distance_unit",
    "qiskit_nature_available",
    "reverse_pauli_labels",
    "select_mapper",
    "sparse_pauli_to_terms",
    "structural_excitations",
]

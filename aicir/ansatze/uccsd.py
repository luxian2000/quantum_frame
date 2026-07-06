"""UCCSD ansatz 模板。

吃纯数据（n_qubits + HF 占据 + 激发列表），产出参数化 ``Circuit``，与 chemistry
子包解耦（镜像 ``hea`` 只吃结构参数的风格）。非相邻激发的精确电路见 ``_excitation``。
调用方通常从 ``MoleculeHamiltonian`` 的 JW 元数据桥接：
``uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.circuit import Circuit, Parameter, pauli_x
from ._excitation import double_excitation_ops, single_excitation_ops


def _flatten(parameters: Sequence[Any] | None) -> list[Any] | None:
    if parameters is None:
        return None
    if isinstance(parameters, (str, bytes)):
        raise TypeError("parameters 必须是非字符串序列")
    if hasattr(parameters, "reshape"):
        flat = parameters.reshape(-1)
        return [flat[i] for i in range(len(flat))]
    return list(parameters)


def uccsd_parameter_count(excitations, *, reps: int = 1) -> int:
    """UCCSD 参数个数 = 激发数 × reps。"""

    if reps < 1:
        raise ValueError(f"reps 必须 ≥1，得到 {reps}")
    return len(tuple(excitations)) * int(reps)


def _validate(n_qubits: int, hf_occupation, excitations) -> None:
    if hf_occupation is None:
        raise ValueError("hf_occupation 为 None；UCCSD 需 Jordan-Wigner 映射的分子元数据")
    if excitations is None:
        raise ValueError("excitations 为 None；UCCSD 需 Jordan-Wigner 映射的分子元数据")
    if len(hf_occupation) != n_qubits:
        raise ValueError(
            f"hf_occupation 长度 {len(hf_occupation)} 与 n_qubits={n_qubits} 不符"
        )
    if any(bit not in (0, 1) for bit in hf_occupation):
        raise ValueError("hf_occupation 元素必须为 0/1")
    for kind, idx in excitations:
        arity = {"single": 2, "double": 4}.get(kind)
        if arity is None:
            raise ValueError(f"未知激发类型 {kind!r}")
        if len(idx) != arity:
            raise ValueError(f"{kind} 激发需 {arity} 个索引，得到 {idx!r}")
        if any(not (0 <= i < n_qubits) for i in idx):
            raise ValueError(f"激发索引越界（out of range）：{idx!r}, n_qubits={n_qubits}")


def uccsd(
    n_qubits: int,
    hf_occupation,
    excitations,
    *,
    reps: int = 1,
    parameter_prefix: str = "theta",
    parameters: Sequence[Any] | None = None,
    backend: Any = None,
) -> Circuit:
    """构建 UCCSD ansatz 电路。

    Args:
        n_qubits: 量子比特数。
        hf_occupation: 长度 == n_qubits 的 0/1 序列，HF 参考态占据。
        excitations: ``("single",(i,a))`` / ``("double",(i,j,a,b))`` 的序列。
        reps: 激发层重复次数（每次重复用独立参数集）。
        parameter_prefix: 生成符号参数的前缀。
        parameters: 可选的扁平参数值序列（缺省则生成符号 ``Parameter``）；
            顺序为 **先 reps 外层、后激发内层**：``[rep0_exc0, rep0_exc1, ..., rep1_exc0, ...]``。
        backend: 可选，绑定到返回 ``Circuit`` 的后端。

    Returns:
        参数化 ``Circuit``（未提供 parameters 时含符号参数）。
    """

    n_qubits = int(n_qubits)
    if reps < 1:
        raise ValueError(f"reps 必须 ≥1，得到 {reps}")
    _validate(n_qubits, hf_occupation, excitations)
    excitations = tuple(excitations)

    values = _flatten(parameters)
    total = uccsd_parameter_count(excitations, reps=reps)
    if values is not None and len(values) != total:
        raise ValueError(f"需要 {total} 个参数值，得到 {len(values)}")

    gates: list[Any] = []
    # HF 参考态：占据位施 pauli_x
    for qubit, bit in enumerate(hf_occupation):
        if bit == 1:
            gates.append(pauli_x(qubit))

    index = 0
    for rep in range(reps):
        for kind, idx in excitations:
            param = values[index] if values is not None else Parameter(f"{parameter_prefix}_{index}")
            index += 1
            if kind == "single":
                # 单激发是两个 orbital 间对称的占据/未占据翻转，数值顺序不影响物理，
                # 排序只是让 single_excitation_ops 的 p<q 前置条件恒满足。
                p, q = sorted(idx)
                gates.extend(single_excitation_ops(param, p, q))
            else:
                # 双激发的角色（创生对 vs 湮灭对）由 idx 的位置顺序决定，*不能*
                # 按数值排序——占据/未占据在比特序上可能交错（见
                # double_excitation_ops 文档字符串），排序会拆错创生/湮灭配对。
                p, q, r, s = idx
                gates.extend(double_excitation_ops(param, p, q, r, s))

    return Circuit(*gates, n_qubits=n_qubits, backend=backend)

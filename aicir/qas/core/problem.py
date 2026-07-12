"""QAS 问题输入归一化。

各 QAS 方法目前各自接受不同形态的“问题”输入：crlqas/qdrats/dqas 接受
``np.ndarray | Hamiltonian`` 哈密顿量矩阵，pporb/pprdql 接受目标密度矩阵，
supernet 通过 objective 回调间接消费。``aicir/qas/problems/hamiltonians.py``
里的预置项使用 ``(coeff, label)`` 顺序（如 ``(-J, "ZZ")``），而
``aicir.core.operators.Hamiltonian`` 的规范构造顺序是 ``(label, coeff)``
（如 ``("ZZ", -J)``）——两处顺序相反，容易在跨模块传递 Pauli 项时踩坑。

本模块只新增归一化工具（:func:`normalize_problem`/:func:`normalize_terms` 等），
3a 阶段不改变任何现有方法的输入签名或行为，接入留给 3b。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Sequence

import numpy as np

from ...core.operators import Hamiltonian
from ...core.state import State

# ──────────────────────────────────────────────────────────────────────────────
# Pauli 项顺序转换
# ──────────────────────────────────────────────────────────────────────────────


def terms_label_first(terms: Sequence[tuple[Any, Any]]) -> list[tuple[str, complex]]:
    """将 ``[(coeff, label), ...]``（如 ``problems/hamiltonians.py`` 预置项）转换为
    ``[(label, coeff), ...]``（``Hamiltonian`` 规范顺序）。

    纯转换，不做类型猜测——调用方需确保输入确实是 coeff-first 顺序。
    """

    return [(label, coeff) for coeff, label in terms]


def terms_coeff_first(terms: Sequence[tuple[Any, Any]]) -> list[tuple[complex, str]]:
    """将 ``[(label, coeff), ...]``（``Hamiltonian`` 规范顺序）转换为
    ``[(coeff, label), ...]``（``problems/hamiltonians.py`` 预置项顺序）。

    纯转换，不做类型猜测——调用方需确保输入确实是 label-first 顺序。
    """

    return [(coeff, label) for label, coeff in terms]


def _is_coefficient(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def normalize_terms(terms: Sequence[tuple[Any, Any]]) -> list[tuple[str, complex]]:
    """接受 ``(label, coeff)`` 或 ``(coeff, label)`` 任一顺序，统一为 ``(label, coeff)``。

    按元素类型消歧：二元组中恰好一个是字符串（Pauli 标签）、另一个是数值（系数，
    ``bool`` 不计入数值）时才能判定顺序；否则（两者都是字符串/都是数值/长度不为 2）
    视为歧义，抛出 ``ValueError``。
    """

    normalized: list[tuple[str, complex]] = []
    for index, term in enumerate(terms):
        if not isinstance(term, tuple) or len(term) != 2:
            raise ValueError(f"第 {index} 项必须是长度为 2 的元组，得到 {term!r}")
        first, second = term
        first_is_label, second_is_label = isinstance(first, str), isinstance(second, str)
        first_is_coeff, second_is_coeff = _is_coefficient(first), _is_coefficient(second)
        if first_is_label and second_is_coeff:
            normalized.append((first, second))
        elif first_is_coeff and second_is_label:
            normalized.append((second, first))
        else:
            raise ValueError(
                f"第 {index} 项 {term!r} 无法消歧顺序：需要恰好一个字符串（Pauli 标签）"
                "和一个数值（系数）"
            )
    return normalized


def _looks_like_terms(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    return all(isinstance(item, tuple) and len(item) == 2 for item in obj)


# ──────────────────────────────────────────────────────────────────────────────
# QASProblem
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QASProblem:
    """QAS 方法的统一问题描述。

    字段:
        kind: 问题种类，``"hamiltonian"``/``"state"``/``"density_matrix"`` 之一。
        hamiltonian: 已构造好的 :class:`~aicir.core.operators.Hamiltonian`；
            仅当输入本身是 ``Hamiltonian``/Pauli 项列表/可 ``to_hamiltonian()``
            的对象时填充，矩阵形式的哈密顿量走 ``matrix`` 字段。
        matrix: 稠密矩阵；``kind="hamiltonian"`` 时是哈密顿量矩阵，
            ``kind="density_matrix"`` 时是密度矩阵。
        state: :class:`~aicir.core.state.State` 实例（``kind="state"``）。
        n_qubits: 比特数。
        metadata: 附加信息（来源类型、是否由矩阵直接构造等），默认为空字典。
    """

    kind: str
    hamiltonian: Hamiltonian | None = None
    matrix: np.ndarray | None = None
    state: State | None = None
    n_qubits: int = 0
    metadata: dict = field(default_factory=dict)


def _infer_qubits_from_dim(dim: int) -> int:
    if dim <= 0:
        raise ValueError(f"维数 {dim} 必须为正整数")
    n_qubits = dim.bit_length() - 1
    if (1 << n_qubits) != dim:
        raise ValueError(f"维数 {dim} 不是 2 的幂，无法推断比特数")
    return n_qubits


def _check_explicit_n_qubits(n_qubits: int | None, inferred: int) -> int:
    if n_qubits is not None and int(n_qubits) != inferred:
        raise ValueError(f"给定的 n_qubits={n_qubits} 与输入实际比特数 {inferred} 不一致")
    return inferred


def _classify_square_matrix(matrix: np.ndarray) -> str:
    """默认哈密顿量矩阵，仅当迹≈1 且半正定（PSD）时归类为密度矩阵。"""

    trace = complex(np.trace(matrix))
    if abs(trace.imag) > 1e-6 or abs(trace.real - 1.0) > 1e-6:
        return "hamiltonian"
    hermitian_part = (matrix + matrix.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(hermitian_part)
    if np.all(eigenvalues >= -1e-6):
        return "density_matrix"
    return "hamiltonian"


def _normalize_array(array: np.ndarray, *, n_qubits: int | None, kind: str | None) -> QASProblem:
    array = np.asarray(array)

    if array.ndim == 2 and array.shape[0] == array.shape[1] and array.shape[0] > 1:
        dim = int(array.shape[0])
        width = _check_explicit_n_qubits(n_qubits, _infer_qubits_from_dim(dim))
        resolved_kind = kind if kind is not None else _classify_square_matrix(array)
        if resolved_kind not in {"hamiltonian", "density_matrix"}:
            raise ValueError(f"kind={resolved_kind!r} 对方阵输入无效，只能是 'hamiltonian' 或 'density_matrix'")
        if resolved_kind == "density_matrix":
            return QASProblem(kind="density_matrix", matrix=array, n_qubits=width)
        return QASProblem(kind="hamiltonian", matrix=array, n_qubits=width, metadata={"from_matrix": True})

    if (array.ndim == 2 and array.shape[1] == 1) or array.ndim == 1:
        flat = array.reshape(-1)
        width = _check_explicit_n_qubits(n_qubits, _infer_qubits_from_dim(int(flat.shape[0])))
        state = State.from_array(flat, n_qubits=width)
        return QASProblem(kind="state", state=state, n_qubits=width)

    raise ValueError(f"无法识别的 ndarray 输入形状 {array.shape!r}")


def normalize_problem(obj: Any, *, n_qubits: int | None = None, kind: str | None = None) -> QASProblem:
    """将多种输入形态归一化为统一的 :class:`QASProblem`。

    支持的输入:
        - :class:`aicir.core.operators.Hamiltonian` 实例
        - :class:`aicir.core.state.State` 实例
        - 二维方阵 ``ndarray``：默认视为哈密顿量矩阵；若迹≈1 且半正定（PSD）
          则视为密度矩阵；可用 ``kind=`` 强制指定 ``"hamiltonian"``/``"density_matrix"``
          覆盖启发式判定
        - 列向量/一维态向量 ``ndarray``（shape ``(2**n, 1)`` 或 ``(2**n,)``）
        - 具备 ``to_hamiltonian()`` 方法的对象（duck-type，覆盖
          ``aicir.chemistry`` 的 ``MoleculeHamiltonian``/``GeneratedHamiltonian``，
          不做硬导入以避免引入 chemistry 依赖）
        - Pauli 项列表，元素为 ``(label, coeff)`` 或 ``(coeff, label)`` 二元组
          （经 :func:`normalize_terms` 消歧）

    ``n_qubits``：对 ndarray/态向量输入用于交叉校验（与实际形状不一致会报错）；
    对 Pauli 项列表输入用于指定比特数（默认从标签长度推断）。
    ``kind``：仅对方阵 ndarray 输入生效，用于覆盖迹/PSD 启发式；其余输入类型忽略。

    无法识别的输入类型抛出 ``ValueError``。
    """

    if isinstance(obj, Hamiltonian):
        return QASProblem(kind="hamiltonian", hamiltonian=obj, n_qubits=obj.n_qubits)

    if isinstance(obj, State):
        return QASProblem(
            kind="state",
            state=obj,
            n_qubits=obj.n_qubits,
            metadata={"is_density": obj.is_density},
        )

    if isinstance(obj, np.ndarray):
        return _normalize_array(obj, n_qubits=n_qubits, kind=kind)

    to_hamiltonian = getattr(obj, "to_hamiltonian", None)
    if callable(to_hamiltonian):
        hamiltonian = to_hamiltonian()
        return QASProblem(
            kind="hamiltonian",
            hamiltonian=hamiltonian,
            n_qubits=hamiltonian.n_qubits,
            metadata={"source_type": type(obj).__name__},
        )

    if _looks_like_terms(obj):
        label_first = normalize_terms(obj)
        width = n_qubits if n_qubits is not None else max(len(label) for label, _ in label_first)
        hamiltonian = Hamiltonian(n_qubits=width, terms=label_first)
        return QASProblem(kind="hamiltonian", hamiltonian=hamiltonian, n_qubits=hamiltonian.n_qubits)

    raise ValueError(
        f"无法识别的 QAS problem 输入类型 {type(obj)!r}；支持 Hamiltonian、State、"
        "方阵/态向量 ndarray、带 to_hamiltonian() 的对象，或 (label,coeff)/(coeff,label) 项列表。"
    )


__all__ = [
    "QASProblem",
    "normalize_problem",
    "normalize_terms",
    "terms_coeff_first",
    "terms_label_first",
]

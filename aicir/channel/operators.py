"""
aicir/channel/operators.py

量子算符层：PauliOp、PauliString、Hamiltonian。

提供从物理描述到矩阵表示的转换，与具体后端解耦。

示例::

    from aicir.channel.backends import TorchBackend
    from aicir.channel.operators import Hamiltonian

    bk = TorchBackend()

    # H = -0.5 * Z₀Z₁  +  0.3 * X₀X₁
    H = Hamiltonian([
        ("ZZ", -0.5),
        ("XX", 0.3),
    ])
    mat = H.to_matrix(bk)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from .backends.base import Backend
    from ..core.state import StateVector
    from ..core.density import DensityMatrix

# ── 单比特泡利矩阵（NumPy 常量，用于构造复杂算符）──────────────────────
_I = np.array([[1, 0], [0, 1]], dtype=np.complex64)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

PAULI_MAP: Dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


# ──────────────────────────────────────────────────────────────────────────────
# PauliOp
# ──────────────────────────────────────────────────────────────────────────────

class PauliOp:
    """
    作用在单个量子比特上的泡利算符。

    示例::

        Z0 = PauliOp('Z', qubit=0)
        mat = Z0.to_matrix(n_qubits=2, backend=bk)  # 4×4 矩阵
    """

    def __init__(self, pauli: str, qubit: int = 0):
        """
        参数:
            pauli: 泡利标签 'I'/'X'/'Y'/'Z'（不区分大小写）
            qubit: 作用的量子比特索引（从 0 开始）
        """
        label = pauli.upper()
        if label not in PAULI_MAP:
            raise ValueError(f"未知泡利算符 '{pauli}'，只支持 I/X/Y/Z")
        self.pauli = label
        self.qubit = qubit

    def to_matrix(self, n_qubits: int, backend: "Backend"):
        """
        在 n_qubits 个量子比特的希尔伯特空间中构造完整矩阵。

        其他比特位填充单位矩阵 I。

        参数:
            n_qubits: 总量子比特数
            backend:  计算后端
        返回:
            shape (2^n, 2^n) 后端原生张量
        """
        if self.qubit >= n_qubits:
            raise IndexError(
                f"qubit={self.qubit} 超出 n_qubits={n_qubits} 的范围"
            )
        matrices = [
            backend.cast(PAULI_MAP[self.pauli if i == self.qubit else "I"])
            for i in range(n_qubits)
        ]
        return backend.tensor_product(*matrices)

    def __repr__(self) -> str:
        return f"PauliOp({self.pauli}, qubit={self.qubit})"


# ──────────────────────────────────────────────────────────────────────────────
# PauliString
# ──────────────────────────────────────────────────────────────────────────────

class PauliString:
    """
    多比特泡利串：coefficient × ∏_i σ_i。

    推荐用字符串描述完整或局部泡利串；也支持用 ``qubits``
    将局部泡利串放到指定比特上；还兼容
    {pauli_label: [qubit_indices]} 字典描述非单位项，其余默认为 I。

    示例::

        ps = PauliString("ZX", coefficient=0.5)
        ps_sparse = PauliString("ZZ", coefficient=-1.0, n_qubits=4, qubits=[0, 3])
    """

    def __init__(
        self,
        paulistring: str | Mapping[str, Sequence[int]] | None = None,
        coefficient: complex = 1.0,
        n_qubits: int | None = None,
        *,
        terms: str | Mapping[str, Sequence[int]] | None = None,
        qubits: Sequence[int] | None = None,
    ):
        """
        参数:
            paulistring: 完整或局部泡利串，例如 "ZIX"；或 {pauli_label: [qubit_indices]} 字典
            coefficient: 系数（复数，默认 1.0）
            n_qubits:    总量子比特数。若为 None，则从 paulistring 中自动推导
            qubits:       可选；指定字符串中每个泡利算符作用的比特下标
        """
        if paulistring is None:
            paulistring = terms
        elif terms is not None:
            raise ValueError("请只传入 paulistring 或 terms 之一")

        if paulistring is None:
            raise ValueError("paulistring 不能为空")

        if isinstance(paulistring, str):
            labels = [label.upper() for label in paulistring.strip()]
            if not labels:
                raise ValueError("Pauli 字符串不能为空")
            invalid = sorted({label for label in labels if label not in PAULI_MAP})
            if invalid:
                raise ValueError(f"未知泡利算符 '{invalid[0]}'，只支持 I/X/Y/Z")
            qubit_indices = (
                [int(qubit) for qubit in qubits]
                if qubits is not None
                else list(range(len(labels)))
            )
            if len(labels) != len(qubit_indices):
                raise ValueError("Pauli 字符串长度必须与 qubits 长度一致")
            if len(set(qubit_indices)) != len(qubit_indices):
                raise ValueError("qubits 不能包含重复下标")
            for qubit in qubit_indices:
                if qubit < 0:
                    raise IndexError(f"量子比特索引 {qubit} 不能为负数")
            if n_qubits is None:
                n_qubits = max(qubit_indices) + 1 if qubit_indices else 0
            self.n_qubits = int(n_qubits)
            if self.n_qubits <= 0:
                raise ValueError("n_qubits 必须为正整数")
            self.coefficient = complex(coefficient)
            self._qubit_labels = ["I"] * self.n_qubits
            for label, qubit in zip(labels, qubit_indices):
                if qubit >= self.n_qubits:
                    raise IndexError(
                        f"量子比特索引 {qubit} 超出范围 [0, {self.n_qubits})"
                    )
                self._qubit_labels[qubit] = label
            return

        if qubits is not None:
            raise ValueError("qubits 只能与字符串形式的 Pauli 串一起使用")

        if n_qubits is None:
            max_qubit = -1
            for qubits in paulistring.values():
                for q in qubits:
                    if q < 0:
                        raise IndexError(f"量子比特索引 {q} 不能为负数")
                    if q > max_qubit:
                        max_qubit = q
            n_qubits = max_qubit + 1 if max_qubit >= 0 else 0

        self.n_qubits = n_qubits
        self.coefficient = complex(coefficient)

        # 每个比特位的泡利标签，默认 'I'
        self._qubit_labels: List[str] = ["I"] * n_qubits
        for label, qubits in paulistring.items():
            label = label.upper()
            if label not in PAULI_MAP:
                raise ValueError(f"未知泡利算符 '{label}'，只支持 I/X/Y/Z")
            for q in qubits:
                if q < 0 or q >= n_qubits:
                    raise IndexError(
                        f"量子比特索引 {q} 超出范围 [0, {n_qubits})"
                    )
                self._qubit_labels[q] = label

    def to_matrix(self, backend: "Backend"):
        """
        构造完整的 2^n × 2^n 矩阵（系数已乘入）。

        参数:
            backend: 计算后端
        返回:
            shape (2^n, 2^n) 后端原生张量
        """
        matrices = [backend.cast(PAULI_MAP[lbl]) for lbl in self._qubit_labels]
        mat_np = backend.to_numpy(backend.tensor_product(*matrices))
        return backend.cast(self.coefficient * mat_np)

    @property
    def qubit_labels(self) -> List[str]:
        """每个比特位对应的泡利标签列表。"""
        return list(self._qubit_labels)

    def __repr__(self) -> str:
        s = "⊗".join(self._qubit_labels)
        return f"PauliString({self.coefficient:.3g} × {s})"


def _is_qubit_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


_HamiltonianTerm = (
    PauliString
    | str
    | tuple[str | Mapping[str, Sequence[int]]]
    | tuple[str | Mapping[str, Sequence[int]], complex | Sequence[int]]
    | tuple[str, complex, Sequence[int]]
    | tuple[str, Sequence[int], complex]
)


# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

class Hamiltonian:
    """
    哈密顿量：加权 PauliString 的线性组合  H = Σ_i cᵢ Pᵢ。

    推荐使用 Pauli 在前的构造方式：
    ``Hamiltonian([("ZI", 0.3), ("XX", 0.5)])``。若只想写局部 Pauli
    串，可以额外传入比特下标：``Hamiltonian(n_qubits=4, terms=[("ZZ", [0, 3], -1.0)])``。

    示例::

        bk = TorchBackend()

        # H = -Z₀Z₁  +  0.5 X₀X₁  +  0.3 Z₀
        H = Hamiltonian([
            ("ZZ", -1.0),
            ("XX", 0.5),
            ("ZI", 0.3),
        ])
        H03 = Hamiltonian(n_qubits=4, terms=[("ZZ", [0, 3], -1.0)])

        mat = H.to_matrix(bk)
        print(H.expectation(sv, bk))   # sv 是 StateVector
    """

    def __init__(
        self,
        n_qubits: int | Iterable[_HamiltonianTerm] | None = None,
        terms: Iterable[_HamiltonianTerm] | None = None,
    ):
        if terms is None and n_qubits is not None and not isinstance(n_qubits, (int, np.integer)):
            terms = n_qubits
            n_qubits = None
        if n_qubits is not None and not isinstance(n_qubits, (int, np.integer)):
            raise TypeError("n_qubits 必须是整数；若要传入 Pauli 项，请使用 terms= 或省略 n_qubits")
        width = int(n_qubits) if n_qubits is not None else None

        raw_terms = list(terms or ())
        if width is None and raw_terms:
            width = max(self._infer_term_width(term) for term in raw_terms)

        parsed_terms = [self._parse_term(term, width) for term in raw_terms]
        if width is None:
            if not parsed_terms:
                raise ValueError("n_qubits 不能为 None，除非 terms 中至少包含一个 Pauli 字符串")
            width = parsed_terms[0].n_qubits
        if width <= 0:
            raise ValueError("n_qubits 必须为正整数")
        for term in parsed_terms:
            if term.n_qubits != width:
                raise ValueError(
                    f"Pauli 项宽度 {term.n_qubits} 与 n_qubits={width} 不一致"
                )
        self.n_qubits = width
        self._terms: List[PauliString] = parsed_terms

    @staticmethod
    def _infer_term_width(term: _HamiltonianTerm) -> int:
        if isinstance(term, PauliString):
            return term.n_qubits
        if isinstance(term, str):
            return len(term.strip())
        if not isinstance(term, tuple):
            raise TypeError("Hamiltonian terms 必须是 PauliString 或 Pauli 项元组")
        if len(term) not in {1, 2, 3}:
            raise TypeError("Hamiltonian terms 必须是 (pauli,), (pauli, coefficient), (pauli, qubits) 或 (pauli, coefficient, qubits)")

        pauli = term[0]
        if len(term) == 2 and isinstance(pauli, str) and _is_qubit_sequence(term[1]):
            qubit_indices = [int(qubit) for qubit in term[1]]
            return max(qubit_indices) + 1 if qubit_indices else 0
        elif len(term) == 3:
            qubits = term[1] if _is_qubit_sequence(term[1]) else term[2]
            qubit_indices = [int(qubit) for qubit in qubits]
            return max(qubit_indices) + 1 if qubit_indices else 0
        if isinstance(pauli, str):
            return len(pauli.strip())
        if isinstance(pauli, Mapping):
            max_qubit = -1
            for qubits in pauli.values():
                for qubit in qubits:
                    max_qubit = max(max_qubit, int(qubit))
            return max_qubit + 1 if max_qubit >= 0 else 0
        raise TypeError("Hamiltonian 项的第一个元素必须是 Pauli 字符串或 Pauli 字典")

    @staticmethod
    def _parse_term(
        term: _HamiltonianTerm,
        n_qubits: int | None,
    ) -> PauliString:
        if isinstance(term, PauliString):
            return term
        if isinstance(term, str):
            return PauliString(term, coefficient=1.0, n_qubits=n_qubits)
        if not isinstance(term, tuple):
            raise TypeError("Hamiltonian terms 必须是 PauliString 或 Pauli 项元组")
        if len(term) == 1:
            pauli = term[0]
            return PauliString(pauli, coefficient=1.0, n_qubits=n_qubits)
        if len(term) == 2:
            pauli, value = term
            if isinstance(pauli, str) and _is_qubit_sequence(value):
                return PauliString(pauli, coefficient=1.0, n_qubits=n_qubits, qubits=value)
            coefficient = value
            return PauliString(pauli, coefficient=coefficient, n_qubits=n_qubits)
        if len(term) == 3:
            pauli = term[0]
            if not isinstance(pauli, str):
                raise ValueError("带 qubits 的 Hamiltonian 项必须使用字符串形式的 Pauli 串")
            if _is_qubit_sequence(term[1]):
                qubits = term[1]
                coefficient = term[2]
            else:
                coefficient = term[1]
                qubits = term[2]
            return PauliString(pauli, coefficient=coefficient, n_qubits=n_qubits, qubits=qubits)
        raise TypeError("Hamiltonian terms 必须是 (pauli,), (pauli, coefficient), (pauli, qubits) 或 (pauli, coefficient, qubits)")

    @classmethod
    def from_list(
        cls,
        terms: Iterable[
            str
            | tuple[str | Mapping[str, Sequence[int]]]
            | tuple[str | Mapping[str, Sequence[int]], complex | Sequence[int]]
            | tuple[str, complex, Sequence[int]]
            | tuple[str, Sequence[int], complex]
        ],
        n_qubits: int | None = None,
    ) -> "Hamiltonian":
        """用 ``[(pauli, coefficient), ...]`` 或带 qubits 的三元组构造哈密顿量。"""

        return cls(n_qubits=n_qubits, terms=terms)

    def to_matrix(self, backend: "Backend"):
        """
        构造完整哈密顿量矩阵，shape (2^n, 2^n)。

        参数:
            backend: 计算后端
        返回:
            后端原生张量（complex）
        """
        dim = 1 << self.n_qubits
        result_np = np.zeros((dim, dim), dtype=np.complex64)
        for term in self._terms:
            mat_np = backend.to_numpy(term.to_matrix(backend))
            result_np = result_np + mat_np
        return backend.cast(result_np)

    def expectation(
        self,
        state: Union["StateVector", "DensityMatrix"],
        backend: "Backend",
    ) -> float:
        """
        计算量子态对哈密顿量的期望值。

        参数:
            state:   StateVector 或 DensityMatrix 实例
            backend: 计算后端
        返回:
            实数期望值
        """
        from ..core.state import StateVector
        from ..core.density import DensityMatrix

        H_mat = self.to_matrix(backend)
        if isinstance(state, StateVector):
            return state.expectation(H_mat)
        elif isinstance(state, DensityMatrix):
            return state.expectation(H_mat)
        else:
            # 兼容原始后端张量（态向量）
            return backend.expectation_sv(state, H_mat)

    @property
    def terms(self) -> List[PauliString]:
        """返回所有 PauliString 项（只读副本）。"""
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return f"Hamiltonian(n_qubits={self.n_qubits}, terms={len(self._terms)})"

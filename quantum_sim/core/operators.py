"""
quantum_sim/core/operators.py

量子算符层：PauliOp、PauliString、Hamiltonian。

提供从物理描述到矩阵表示的转换，与具体后端解耦。

示例::

    from quantum_sim.core.backends import TorchBackend
    from quantum_sim.core.operators import Hamiltonian

    bk = TorchBackend()

    # H = -0.5 * Z₀Z₁  +  0.3 * X₀X₁
    H = (Hamiltonian(n_qubits=2)
         .add_term(-0.5, {'Z': [0, 1]})
         .add_term( 0.3, {'X': [0, 1]}))
    mat = H.to_matrix(bk)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from .backends.base import Backend
    from .states.state_vector import StateVector
    from .states.density_matrix import DensityMatrix

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

    用 {pauli_label: [qubit_indices]} 字典描述非单位项，其余默认为 I。

    示例::

        # 0.5 × Z₀ ⊗ X₁
        ps = PauliString({'Z': [0], 'X': [1]}, n_qubits=2, coefficient=0.5)
    """

    def __init__(
        self,
        terms: Dict[str, List[int]],
        n_qubits: int,
        coefficient: complex = 1.0,
    ):
        """
        参数:
            terms:       {pauli_label: [qubit_indices]}，例如 {'Z': [0, 1], 'X': [2]}
            n_qubits:    总量子比特数
            coefficient: 系数（复数，默认 1.0）
        """
        self.n_qubits = n_qubits
        self.coefficient = complex(coefficient)

        # 每个比特位的泡利标签，默认 'I'
        self._qubit_labels: List[str] = ["I"] * n_qubits
        for label, qubits in terms.items():
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


# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

class Hamiltonian:
    """
    哈密顿量：加权 PauliString 的线性组合  H = Σ_i cᵢ Pᵢ。

    示例::

        bk = TorchBackend()

        # H = -Z₀Z₁  +  0.5 X₀X₁  +  0.3 Z₀
        H = (Hamiltonian(n_qubits=2)
             .add_term(-1.0,  {'Z': [0, 1]})
             .add_term( 0.5,  {'X': [0, 1]})
             .add_term( 0.3,  {'Z': [0]}))

        mat = H.to_matrix(bk)
        print(H.expectation(sv, bk))   # sv 是 StateVector
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self._terms: List[PauliString] = []

    def add_term(
        self,
        coefficient: complex,
        pauli_dict: Dict[str, List[int]],
    ) -> "Hamiltonian":
        """
        添加一项 coefficient × PauliString(pauli_dict)，支持链式调用。

        参数:
            coefficient: 系数（实数或复数）
            pauli_dict:  {pauli_label: [qubit_indices]}，例如 {'Z': [0, 1]}
        返回:
            self
        """
        term = PauliString(pauli_dict, self.n_qubits, coefficient)
        self._terms.append(term)
        return self

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
        from .states.state_vector import StateVector
        from .states.density_matrix import DensityMatrix

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

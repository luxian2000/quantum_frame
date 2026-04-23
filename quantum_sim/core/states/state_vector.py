"""
quantum_sim/core/states/state_vector.py

纯量子态 |ψ⟩ 的面向对象封装。

设计原则：
- 内部数据始终保持列向量形式，shape (2^n, 1)
- 所有数值运算委托给注入的 Backend 实例
- 返回新对象而非原地修改（不可变风格），便于函数式组合
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from ..backends.base import Backend
    from .density_matrix import DensityMatrix


class StateVector:
    """
    纯量子态 |ψ⟩ 的面向对象封装。

    形状约定：内部数据 shape (2^n, 1)，复数类型。

    示例::

        from quantum_sim.core.backends import TorchBackend
        from quantum_sim.core.states import StateVector

        bk = TorchBackend()
        sv = StateVector.zero_state(2, bk)          # |00⟩
        U  = ...                                    # 某个 4×4 酉矩阵
        sv2 = sv.evolve(U)                          # |ψ'⟩ = U|ψ⟩
        print(sv2.probabilities())
        print(sv2.measure(shots=1024))
    """

    def __init__(self, data, n_qubits: int, backend: "Backend"):
        """
        参数:
            data:     后端张量，shape (2^n, 1) 或 (2^n,)
            n_qubits: 量子比特数
            backend:  计算后端实例
        """
        self._backend = backend
        self._n_qubits = n_qubits

        # 统一存储为列向量 (2^n, 1)
        np_data = backend.to_numpy(data)
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
        self._data = backend.cast(np_data)

        # 维度检查
        expected = 1 << n_qubits
        if self._data.shape[0] != expected:
            raise ValueError(
                f"数据长度 {self._data.shape[0]} 与 n_qubits={n_qubits} 不符（期望 {expected}）"
            )

    # ──────────────────────────── 工厂方法 ──────────────────────────

    @classmethod
    def zero_state(cls, n_qubits: int, backend: "Backend") -> "StateVector":
        """创建 |0⊗n⟩ 计算基基态。"""
        data = backend.zeros_state(n_qubits)
        return cls(data, n_qubits, backend)

    @classmethod
    def from_array(cls, array, n_qubits: int, backend: "Backend") -> "StateVector":
        """
        从 numpy array / list 构造态向量。

        参数:
            array:    长度为 2^n 的一维或 (2^n,1) 的复数序列
            n_qubits: 量子比特数
            backend:  计算后端
        """
        data = backend.cast(np.asarray(array, dtype=np.complex64))
        return cls(data, n_qubits, backend)

    # ──────────────────────────── 属性 ──────────────────────────────

    @property
    def data(self):
        """后端原生张量，shape (2^n, 1)。"""
        return self._data

    @property
    def n_qubits(self) -> int:
        """量子比特数。"""
        return self._n_qubits

    @property
    def dim(self) -> int:
        """希尔伯特空间维数 2^n。"""
        return 1 << self._n_qubits

    @property
    def backend(self) -> "Backend":
        """当前使用的计算后端。"""
        return self._backend

    # ──────────────────────────── 量子操作 ──────────────────────────

    def evolve(self, unitary) -> "StateVector":
        """
        将酉矩阵作用于当前态，返回新的 StateVector（不修改原对象）。

        参数:
            unitary: (2^n, 2^n) 酉矩阵（后端原生张量）
        返回:
            StateVector — 演化后的新态
        """
        new_data = self._backend.apply_unitary(self._data, unitary)
        return StateVector(new_data, self._n_qubits, self._backend)

    def probabilities(self):
        """
        计算计算基上的测量概率分布。

        返回:
            shape (2^n,) 实数概率向量（后端原生张量），总和为 1
        """
        return self._backend.measure_probs(self._data)

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """
        模拟 shots 次投影测量，返回各基态出现次数。

        参数:
            shots: 测量次数（正整数）
        返回:
            {"|00⟩": count, "|11⟩": count, ...} 字典（仅含非零项）
        """
        probs = self.probabilities()
        counts_arr = self._backend.sample(probs, shots)
        counts_np = self._backend.to_numpy(counts_arr).astype(int).reshape(-1)
        return {
            f"|{idx:0{self._n_qubits}b}>": int(c)
            for idx, c in enumerate(counts_np)
            if c > 0
        }

    def expectation(self, operator) -> float:
        """
        计算期望值 ⟨ψ|O|ψ⟩，返回实数。

        参数:
            operator: (2^n, 2^n) Hermitian 算符（后端原生张量）
        """
        return self._backend.expectation_sv(self._data, operator)

    def inner_product(self, other: "StateVector"):
        """
        计算内积 ⟨self|other⟩。

        返回:
            复数标量张量
        """
        if self._n_qubits != other._n_qubits:
            raise ValueError("内积要求两态具有相同 n_qubits")
        return self._backend.inner_product(self._data, other._data)

    def norm(self) -> float:
        """
        计算态向量的范数（归一化时应约等于 1.0）。
        """
        probs_np = self._backend.to_numpy(self.probabilities()).real
        return float(probs_np.sum()) ** 0.5

    def to_density_matrix(self) -> "DensityMatrix":
        """
        将纯态转换为密度矩阵 ρ = |ψ⟩⟨ψ|。
        """
        from .density_matrix import DensityMatrix

        bk = self._backend
        rho = bk.matmul(self._data, bk.dagger(self._data))
        return DensityMatrix(rho, self._n_qubits, bk)

    def to_numpy(self) -> np.ndarray:
        """
        导出为 numpy 一维复数数组，shape (2^n,)。
        """
        return self._backend.to_numpy(self._data).reshape(-1)

    # ──────────────────────────── 运算符重载 ────────────────────────

    def __len__(self) -> int:
        return self.dim

    def __repr__(self) -> str:
        return (
            f"StateVector(n_qubits={self._n_qubits}, "
            f"backend={self._backend.name})"
        )

"""
quantum_sim/core/states/density_matrix.py

量子密度矩阵 ρ 的面向对象封装，同时支持纯态与混合态。

形状约定：(2^n, 2^n)，复数类型。

设计原则与 StateVector 相同：
- 数值运算委托给后端
- 返回新对象，不原地修改
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from ..backends.base import Backend
    from .state_vector import StateVector


class DensityMatrix:
    """
    量子密度矩阵 ρ 的面向对象封装。

    示例::

        from quantum_sim.core.backends import TorchBackend
        from quantum_sim.core.states import StateVector, DensityMatrix

        bk = TorchBackend()
        sv  = StateVector.zero_state(2, bk)
        rho = sv.to_density_matrix()
        print("purity:", rho.purity())          # 纯态 → 1.0
        red = rho.partial_trace(keep=[0])       # 保留第 0 比特
        print("entropy:", red.von_neumann_entropy())
    """

    def __init__(self, data, n_qubits: int, backend: "Backend"):
        """
        参数:
            data:     后端张量，shape (2^n, 2^n)
            n_qubits: 量子比特数
            backend:  计算后端实例
        """
        self._backend = backend
        self._n_qubits = n_qubits

        np_data = backend.to_numpy(data)
        dim = 1 << n_qubits
        if np_data.shape != (dim, dim):
            raise ValueError(
                f"密度矩阵形状 {np_data.shape} 与 n_qubits={n_qubits} 不符（期望 ({dim},{dim})）"
            )
        self._data = backend.cast(np_data)

    # ──────────────────────────── 工厂方法 ──────────────────────────

    @classmethod
    def zero_state(cls, n_qubits: int, backend: "Backend") -> "DensityMatrix":
        """创建 |0⊗n⟩⟨0⊗n| 密度矩阵。"""
        dim = 1 << n_qubits
        rho_np = np.zeros((dim, dim), dtype=np.complex64)
        rho_np[0, 0] = 1.0 + 0j
        return cls(backend.cast(rho_np), n_qubits, backend)

    @classmethod
    def from_state_vector(cls, sv: "StateVector") -> "DensityMatrix":
        """从 StateVector 构造纯态密度矩阵 ρ = |ψ⟩⟨ψ|。"""
        return sv.to_density_matrix()

    @classmethod
    def from_array(cls, array, n_qubits: int, backend: "Backend") -> "DensityMatrix":
        """从 numpy array / list 构造密度矩阵。"""
        data = backend.cast(np.asarray(array, dtype=np.complex64))
        return cls(data, n_qubits, backend)

    @classmethod
    def maximally_mixed(cls, n_qubits: int, backend: "Backend") -> "DensityMatrix":
        """创建最大混合态 ρ = I / 2^n。"""
        dim = 1 << n_qubits
        rho_np = np.eye(dim, dtype=np.complex64) / dim
        return cls(backend.cast(rho_np), n_qubits, backend)

    # ──────────────────────────── 属性 ──────────────────────────────

    @property
    def data(self):
        """后端原生张量，shape (2^n, 2^n)。"""
        return self._data

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def dim(self) -> int:
        return 1 << self._n_qubits

    @property
    def backend(self) -> "Backend":
        return self._backend

    # ──────────────────────────── 量子操作 ──────────────────────────

    def evolve(self, unitary) -> "DensityMatrix":
        """
        酉演化 ρ' = U ρ U†。

        参数:
            unitary: (2^n, 2^n) 酉矩阵（后端原生张量）
        返回:
            DensityMatrix — 演化后的新密度矩阵
        """
        bk = self._backend
        new_data = bk.matmul(bk.matmul(unitary, self._data), bk.dagger(unitary))
        return DensityMatrix(new_data, self._n_qubits, bk)

    def partial_trace(self, keep: List[int]) -> "DensityMatrix":
        """
        对指定子系统求偏迹，返回约化密度矩阵。

        参数:
            keep: 要保留的量子比特索引列表（从 0 开始）
        返回:
            DensityMatrix — 形状 (2^k, 2^k)，k = len(keep)
        """
        bk = self._backend
        rho_red = bk.partial_trace(self._data, keep, self._n_qubits)
        return DensityMatrix(rho_red, len(keep), bk)

    def probabilities(self) -> np.ndarray:
        """
        计算基测量概率（密度矩阵对角元的实部），shape (2^n,)，返回 numpy array。
        """
        rho_np = self._backend.to_numpy(self._data)
        diag = np.real(rho_np.diagonal())
        diag = np.clip(diag, 0, None)
        total = diag.sum()
        return diag / total if total > 0 else diag

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """
        模拟 shots 次测量，返回各基态出现次数。

        返回:
            {"|00>": count, ...} 字典（仅含非零项）
        """
        probs = self.probabilities()
        indices = np.random.choice(len(probs), size=shots, p=probs)
        counts_arr = np.bincount(indices, minlength=len(probs))
        return {
            f"|{idx:0{self._n_qubits}b}>": int(c)
            for idx, c in enumerate(counts_arr)
            if c > 0
        }

    def expectation(self, operator) -> float:
        """
        计算期望值 Tr(ρO)，返回实数。

        参数:
            operator: (2^n, 2^n) Hermitian 算符（后端原生张量）
        """
        return self._backend.expectation_dm(self._data, operator)

    # ──────────────────────────── 物理量 ────────────────────────────

    def purity(self) -> float:
        """
        纯度 Tr(ρ²)。

        - 纯态：purity = 1.0
        - 完全混合态：purity = 1 / 2^n
        """
        bk = self._backend
        val = bk.trace(bk.matmul(self._data, self._data))
        return float(np.real(bk.to_numpy(val)))

    def eigenvalues(self) -> np.ndarray:
        """
        密度矩阵特征值（即各本征态的占据概率），升序排列，numpy array。
        """
        rho_np = self._backend.to_numpy(self._data)
        return np.linalg.eigvalsh(rho_np)

    def von_neumann_entropy(self) -> float:
        """
        冯·诺依曼熵 S(ρ) = -Tr(ρ log ρ)（以自然对数为底）。

        - 纯态：S = 0
        - 完全混合态（n 比特）：S = n ln 2
        """
        eigs = self.eigenvalues()
        eigs = eigs[eigs > 1e-15]
        return float(-np.sum(eigs * np.log(eigs)))

    def is_pure(self, tol: float = 1e-5) -> bool:
        """判断是否为纯态（purity ≈ 1）。"""
        return abs(self.purity() - 1.0) < tol

    def to_numpy(self) -> np.ndarray:
        """导出为 numpy 二维复数数组，shape (2^n, 2^n)。"""
        return self._backend.to_numpy(self._data)

    # ──────────────────────────── 魔法方法 ──────────────────────────

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(n_qubits={self._n_qubits}, "
            f"backend={self._backend.name})"
        )

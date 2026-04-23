"""
quantum_sim/core/backends/base.py

量子模拟器后端抽象基类。

约定
----
- 态向量   shape: (2^n, 1)，complex dtype
- 密度矩阵 shape: (2^n, 2^n)，complex dtype
- 所有返回张量的方法返回后端原生张量类型
- `to_numpy` 提供统一的转换出口，用于输出/调试

实现新后端时，继承 Backend 并实现全部抽象方法即可；
上层代码（StateVector、Circuit、Engine …）只依赖此接口，不感知底层框架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Backend(ABC):
    """量子模拟器计算后端抽象基类。"""

    # ──────────────────────────── 元信息 ────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """后端的唯一名称标识符（用于日志/repr）。"""

    # ──────────────────────────── 张量工厂 ──────────────────────────

    @abstractmethod
    def zeros(self, shape: tuple, dtype=None):
        """
        创建全零张量。

        参数:
            shape: 张量形状
            dtype: 数据类型（None 时使用后端默认 complex dtype）
        返回:
            后端原生张量
        """

    @abstractmethod
    def eye(self, dim: int):
        """
        创建 dim × dim 复数单位矩阵。

        返回:
            shape (dim, dim) 后端原生张量
        """

    @abstractmethod
    def cast(self, array, dtype=None):
        """
        将 numpy array 或嵌套 list/Python 标量转换为后端张量。

        参数:
            array: 输入数据（numpy array、list、Python 数值或后端张量）
            dtype: 目标数据类型（None 时使用后端默认）
        返回:
            后端原生张量
        """

    @abstractmethod
    def to_numpy(self, tensor) -> np.ndarray:
        """
        将后端张量转换为 numpy array（用于输出、调试、后端间交互）。

        注意：Torch 张量需先 detach().cpu()，此方法负责处理。
        """

    # ──────────────────────────── 量子态初始化 ──────────────────────

    @abstractmethod
    def zeros_state(self, n_qubits: int):
        """
        创建 |0⊗n⟩ 基态列向量，shape (2^n, 1)，复数类型。

        参数:
            n_qubits: 量子比特数
        返回:
            shape (2^n, 1) 后端原生张量
        """

    # ──────────────────────────── 线性代数 ──────────────────────────

    @abstractmethod
    def matmul(self, a, b):
        """矩阵乘法 a @ b。"""

    @abstractmethod
    def kron(self, a, b):
        """Kronecker（张量）积 a ⊗ b。"""

    @abstractmethod
    def dagger(self, matrix):
        """共轭转置（Hermitian adjoint）：matrix†。"""

    @abstractmethod
    def trace(self, matrix):
        """矩阵的迹，返回标量张量。"""

    @abstractmethod
    def real(self, tensor):
        """逐元素取实部。"""

    @abstractmethod
    def abs_sq(self, tensor):
        """逐元素取模平方 |x|²，返回实数张量。"""

    # ──────────────────────────── 量子操作 ──────────────────────────

    @abstractmethod
    def apply_unitary(self, state, unitary):
        """
        将酉矩阵作用于态向量：|ψ'⟩ = U|ψ⟩。

        参数:
            state:   (2^n, 1) 态向量
            unitary: (2^n, 2^n) 酉矩阵
        返回:
            (2^n, 1) 新态向量
        """

    @abstractmethod
    def inner_product(self, bra, ket):
        """
        内积 ⟨bra|ket⟩（bra 自动取共轭）。

        参数:
            bra: (2^n, 1) 或 (2^n,) 张量
            ket: (2^n, 1) 或 (2^n,) 张量
        返回:
            复数标量张量
        """

    @abstractmethod
    def measure_probs(self, state):
        """
        由态向量计算计算基上的测量概率分布。

        参数:
            state: (2^n, 1) 或 (2^n,) 归一化态向量
        返回:
            (2^n,) 实数概率向量，总和为 1
        """

    @abstractmethod
    def partial_trace(self, rho, keep: List[int], n_qubits: int):
        """
        对密度矩阵执行偏迹，保留 keep 中列出的量子比特子系统。

        参数:
            rho:      (2^n, 2^n) 密度矩阵
            keep:     要保留的量子比特索引列表（从 0 开始）
            n_qubits: 总量子比特数
        返回:
            (2^k, 2^k) 约化密度矩阵，k = len(keep)
        """

    @abstractmethod
    def sample(self, probs, shots: int):
        """
        按概率分布进行 shots 次测量采样。

        参数:
            probs: (2^n,) 概率向量（实数，总和为 1）
            shots: 采样次数（正整数）
        返回:
            (2^n,) int 计数向量，总和为 shots
        """

    @abstractmethod
    def expectation_sv(self, state, operator):
        """
        纯态期望值 ⟨ψ|O|ψ⟩。

        参数:
            state:    (2^n, 1) 态向量
            operator: (2^n, 2^n) Hermitian 算符
        返回:
            实数标量张量
        """

    @abstractmethod
    def expectation_dm(self, rho, operator):
        """
        混合态期望值 Tr(ρO)。

        参数:
            rho:      (2^n, 2^n) 密度矩阵
            operator: (2^n, 2^n) Hermitian 算符
        返回:
            实数标量张量
        """

    # ──────────────────────────── 便利方法（非抽象）────────────────────

    def tensor_product(self, *matrices):
        """
        多个矩阵的 Kronecker 积（从左到右）。

        等价于 kron(kron(m0, m1), m2) ...
        """
        if not matrices:
            raise ValueError("tensor_product 至少需要一个矩阵")
        result = matrices[0]
        for m in matrices[1:]:
            result = self.kron(result, m)
        return result

    def matrix_product(self, *matrices):
        """
        多个矩阵的乘积（从左到右）：m0 @ m1 @ m2 ...
        """
        if not matrices:
            raise ValueError("matrix_product 至少需要一个矩阵")
        result = matrices[0]
        for m in matrices[1:]:
            result = self.matmul(result, m)
        return result

    def __repr__(self) -> str:
        return self.name

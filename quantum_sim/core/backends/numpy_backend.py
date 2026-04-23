"""
quantum_sim/core/backends/numpy_backend.py

基于 NumPy 的 CPU 计算后端（参考实现）。

特点：
- 无外部深度学习依赖，环境要求最低
- 不支持自动微分（参数优化需用 TorchBackend）
- 适合小规模验证和算法原型
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .base import Backend

_CDTYPE = np.complex64


class NumpyBackend(Backend):
    """基于 NumPy 的 CPU 计算后端（无自动微分支持）。"""

    def __init__(self, dtype=None):
        """
        参数:
            dtype: 复数数据类型，默认 np.complex64
        """
        self._dtype = dtype or _CDTYPE

    # ──────────────────────── 元信息 ────────────────────────────

    @property
    def name(self) -> str:
        return f"NumpyBackend(dtype={self._dtype.__name__ if hasattr(self._dtype, '__name__') else self._dtype})"

    # ──────────────────────── 张量工厂 ──────────────────────────

    def zeros(self, shape: tuple, dtype=None):
        return np.zeros(shape, dtype=dtype or self._dtype)

    def eye(self, dim: int):
        return np.eye(dim, dtype=self._dtype)

    def cast(self, array, dtype=None):
        if isinstance(array, np.ndarray):
            return array.astype(dtype or self._dtype)
        return np.array(array, dtype=dtype or self._dtype)

    def to_numpy(self, tensor) -> np.ndarray:
        return np.asarray(tensor)

    # ──────────────────────── 量子态初始化 ──────────────────────

    def zeros_state(self, n_qubits: int):
        dim = 1 << n_qubits
        state = np.zeros((dim, 1), dtype=self._dtype)
        state[0, 0] = 1.0 + 0j
        return state

    # ──────────────────────── 线性代数 ──────────────────────────

    def matmul(self, a, b):
        return a @ b

    def kron(self, a, b):
        return np.kron(a, b)

    def dagger(self, matrix):
        return np.conj(matrix).T

    def trace(self, matrix):
        return np.trace(matrix)

    def real(self, tensor):
        return np.real(tensor)

    def abs_sq(self, tensor):
        return np.abs(tensor) ** 2

    # ──────────────────────── 量子操作 ──────────────────────────

    def apply_unitary(self, state, unitary):
        return unitary @ state

    def inner_product(self, bra, ket):
        b = np.asarray(bra).reshape(-1)
        k = np.asarray(ket).reshape(-1)
        return np.conj(b) @ k

    def measure_probs(self, state):
        probs = (np.abs(np.asarray(state).reshape(-1)) ** 2).real
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def partial_trace(self, rho, keep: List[int], n_qubits: int):
        rho = np.asarray(rho)
        if n_qubits is None:
            n_qubits = int(math.log2(rho.shape[0]))

        keep = sorted(set(int(k) for k in keep))
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if not trace_out:
            return rho.copy()

        reshaped = rho.reshape([2] * n_qubits + [2] * n_qubits)
        perm = (keep + trace_out
                + [k + n_qubits for k in keep]
                + [t + n_qubits for t in trace_out])
        permuted = reshaped.transpose(perm)

        d_keep = 1 << len(keep)
        d_trace = 1 << len(trace_out)
        permuted = permuted.reshape(d_keep, d_trace, d_keep, d_trace)
        return np.einsum("abcb->ac", permuted)

    def sample(self, probs, shots: int):
        probs_real = np.real(np.asarray(probs)).astype(np.float64)
        probs_real = np.clip(probs_real, 0, None)
        total = probs_real.sum()
        if total == 0:
            raise ValueError("概率全为零，无法采样")
        probs_real = probs_real / total
        indices = np.random.choice(len(probs_real), size=shots, p=probs_real)
        return np.bincount(indices, minlength=len(probs_real))

    def expectation_sv(self, state, operator):
        s = np.asarray(state).reshape(-1, 1)
        val = (np.conj(s).T @ np.asarray(operator) @ s)[0, 0]
        return float(np.real(val))

    def expectation_dm(self, rho, operator):
        val = np.trace(np.asarray(rho) @ np.asarray(operator))
        return float(np.real(val))

"""
quantum_sim/core/backends/torch_backend.py

基于 PyTorch 的计算后端。

特点：
- 支持 GPU（CUDA）加速
- 支持自动微分（参数化门、VQE 等变分算法需要）
- 与深度学习工作流深度集成
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

from .base import Backend

_CDTYPE = torch.complex64


class TorchBackend(Backend):
    """基于 PyTorch 的计算后端，支持 GPU 加速与自动微分。"""

    def __init__(self, dtype=None, device=None):
        """
        参数:
            dtype:  torch 复数数据类型，默认 torch.complex64
            device: 计算设备，默认自动选择（cuda > cpu）
        """
        self._dtype = dtype or _CDTYPE
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

    # ──────────────────────── 元信息 ────────────────────────────

    @property
    def name(self) -> str:
        return f"TorchBackend(dtype={self._dtype}, device={self._device})"

    # ──────────────────────── 张量工厂 ──────────────────────────

    def zeros(self, shape: tuple, dtype=None):
        return torch.zeros(shape, dtype=dtype or self._dtype, device=self._device)

    def eye(self, dim: int):
        return torch.eye(dim, dtype=self._dtype, device=self._device)

    def cast(self, array, dtype=None):
        target_dtype = dtype or self._dtype
        if isinstance(array, torch.Tensor):
            return array.to(dtype=target_dtype, device=self._device)
        return torch.tensor(np.asarray(array), dtype=target_dtype, device=self._device)

    def to_numpy(self, tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    # ──────────────────────── 量子态初始化 ──────────────────────

    def zeros_state(self, n_qubits: int):
        dim = 1 << n_qubits
        state = torch.zeros((dim, 1), dtype=self._dtype, device=self._device)
        state[0, 0] = 1.0 + 0j
        return state

    # ──────────────────────── 线性代数 ──────────────────────────

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def kron(self, a, b):
        return torch.kron(a, b)

    def dagger(self, matrix):
        return torch.conj(torch.transpose(matrix, -2, -1))

    def trace(self, matrix):
        return torch.trace(matrix)

    def real(self, tensor):
        return torch.real(tensor)

    def abs_sq(self, tensor):
        return torch.abs(tensor) ** 2

    # ──────────────────────── 量子操作 ──────────────────────────

    def apply_unitary(self, state, unitary):
        return torch.matmul(unitary, state)

    def inner_product(self, bra, ket):
        b = bra.reshape(-1)
        k = ket.reshape(-1)
        return torch.dot(torch.conj(b), k)

    def measure_probs(self, state):
        probs = (torch.abs(state.reshape(-1)) ** 2).real
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def partial_trace(self, rho, keep: List[int], n_qubits: int):
        if n_qubits is None:
            n_qubits = int(math.log2(rho.shape[0]))

        keep = sorted(set(int(k) for k in keep))
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if not trace_out:
            return rho.clone()

        reshaped = rho.reshape([2] * n_qubits + [2] * n_qubits)
        perm = (keep + trace_out
                + [k + n_qubits for k in keep]
                + [t + n_qubits for t in trace_out])
        permuted = reshaped.permute(perm)

        d_keep = 1 << len(keep)
        d_trace = 1 << len(trace_out)
        permuted = permuted.reshape(d_keep, d_trace, d_keep, d_trace)
        return torch.einsum("abcb->ac", permuted)

    def sample(self, probs, shots: int):
        if probs.is_complex():
            probs_real = torch.real(probs).to(torch.float32)
        else:
            probs_real = probs.float()
        probs_real = torch.clamp(probs_real, min=0)
        total = probs_real.sum()
        if total == 0:
            raise ValueError("概率全为零，无法采样")
        probs_real = probs_real / total
        indices = torch.multinomial(probs_real, num_samples=shots, replacement=True)
        return torch.bincount(indices, minlength=int(probs_real.numel()))

    def expectation_sv(self, state, operator):
        s = state.reshape(-1, 1)
        val = (self.dagger(s) @ operator @ s)[0, 0]
        return torch.real(val)

    def expectation_dm(self, rho, operator):
        val = torch.trace(torch.matmul(rho, operator))
        return torch.real(val)

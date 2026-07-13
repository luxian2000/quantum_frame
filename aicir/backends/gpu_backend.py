"""
aicir/backends/gpu_backend.py

基于 PyTorch 的 CPU/GPU 计算后端。

特点：
- 支持 GPU（CUDA）加速
- 支持自动微分（参数化门、VQE 等变分算法需要）
- 与深度学习工作流深度集成

注：Ascend NPU 使用 ``npu_backend.NPUBackend``（继承自本类，覆写缺少
complex64 内核的算子）。``TorchBackend`` 为本类的过时别名，保留向后兼容。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

from .base import Backend

_CDTYPE = torch.complex64


class GPUBackend(Backend):
    """基于 PyTorch 的 CPU/GPU 计算后端，支持 GPU 加速与自动微分。"""

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
        return f"GPUBackend(dtype={self._dtype}, device={self._device})"

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

    def tensordot(self, a, b, axes):
        return torch.tensordot(a, b, dims=(list(axes[0]), list(axes[1])))

    def transpose(self, a, axes):
        return a.permute(*[int(x) for x in axes])

    def reshape(self, a, shape):
        return a.reshape(tuple(int(s) for s in shape))

    def conj(self, a):
        return torch.conj(a)

    def svd(self, matrix):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u, s, vh

    def take(self, a, axis, index):
        return torch.select(a, int(axis), int(index))

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def abs_sq(self, tensor):
        return torch.abs(tensor) ** 2

    # ──────────────────────── 量子操作 ──────────────────────────

    def apply_unitary(self, state, unitary):
        return torch.matmul(unitary, state)

    def inner_product(self, bra, ket):
        b = bra.reshape(-1)
        k = ket.reshape(-1)
        if torch.is_complex(b) or torch.is_complex(k):
            # 实虚部分解为 4 个实数 dot：⟨b|k⟩ = Σ conj(b)·k。
            # 规避 torch.dot 在部分构建（实测 torch 2.4 / aarch64 CPU）上对
            # torch.conj 惰性共轭视图返回全零的内核缺陷；.real/.imag 为可微视图，
            # autograd 语义与原实现一致。
            dtype = torch.promote_types(b.dtype, k.dtype)
            b = b.to(dtype)
            k = k.to(dtype)
            real = torch.dot(b.real, k.real) + torch.dot(b.imag, k.imag)
            imag = torch.dot(b.real, k.imag) - torch.dot(b.imag, k.real)
            return torch.complex(real, imag)
        return torch.dot(b, k)

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

        # 逐比特求迹，每步整形为 (L, 2, R, L, 2, R) 的秩-6 张量并对该比特的行/列
        # 同一指标求和，避免一次性整形为 [2]*n + [2]*n（秩 2n）。CUDA 允许至多 64
        # 维，故仅 n>32 时才会触发原限制；此处与 NPU 路径保持一致，工作张量的秩恒为
        # 6。按降序求迹使保留比特维持原有（升序）次序。
        remaining = list(range(n_qubits))
        cur = rho
        for qubit in sorted(trace_out, reverse=True):
            pos = remaining.index(qubit)
            m = len(remaining)
            left = 1 << pos
            right = 1 << (m - pos - 1)
            block = cur.reshape(left, 2, right, left, 2, right)
            cur = block[:, 0, :, :, 0, :] + block[:, 1, :, :, 1, :]
            cur = cur.reshape(left * right, left * right)
            remaining.pop(pos)
        return cur

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


# Deprecated alias kept for backward compatibility. ``GPUBackend`` is the
# canonical name; ``TorchBackend`` will be removed in a future release.
TorchBackend = GPUBackend

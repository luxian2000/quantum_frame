"""
aicir/core/batch.py

批量 (batched) 态矢量模拟路径。

设计动机
--------
aicir 的主路径 (``State`` / ``apply_gate_to_state``) 一次只演化单个
``(2^n, 1)`` 态矢量, 且门参数为标量。深度学习场景 (例如把变分量子线路当作
神经网络的一层) 需要:

- 一次模拟一批 (batch) 态矢量;
- 旋转门角度可以逐样本 (per-sample) 不同 (例如数据编码角度依赖输入);
- Ascend NPU 安全 —— NPU 缺少 complex64 的 ``aclnnAdd`` / ``aclnnMul`` 内核,
  因此全程以实部/虚部两个实张量表示, 只用实数乘加, 自动求导不会触发复数累加;
- 端到端可微 (autograd)。

本模块即为该批量路径。门矩阵与 aicir 单态路径采用同一套定义 (复用
``_single_qubit_base_for_gate`` 提供常量门/标量旋转门的基矩阵), 仅在逐样本张量
角度时按相同公式构造批量基矩阵, 保持单一事实来源。

约定
----
- 量子比特端序与 aicir 主路径一致: **qubit 0 为最高位**, qubit ``q`` 的比特权重
  为 ``2^(n-1-q)``。
- 内部态以 ``real`` / ``imag`` 两个形状 ``(batch, 2^n)`` 的实张量表示。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..gates import canonical_gate_name
from ..ir import as_instruction, instruction_control_states, instruction_controls, instruction_name, instruction_parameter, instruction_qubits
from .gates import _single_qubit_base_for_gate

__all__ = ["BatchSV"]


# (n_qubits, device, dtype) -> (idx_tensor, zsign_tensor)
_BIT_CACHE: dict = {}


def _bit_tensors(n_qubits: int, device, real_dtype):
    key = (int(n_qubits), str(device), str(real_dtype))
    cached = _BIT_CACHE.get(key)
    if cached is not None:
        return cached
    dim = 1 << n_qubits
    idx = torch.arange(dim, device=device)
    # zsign[q, basis] = 1 - 2*bit_q, aicir 端序 (qubit 0 = MSB)。
    zsign = torch.stack(
        [1.0 - 2.0 * ((idx >> (n_qubits - 1 - q)) & 1).to(real_dtype) for q in range(n_qubits)],
        dim=0,
    )  # (n, D)
    cached = (idx, zsign)
    _BIT_CACHE[key] = cached
    return cached


def _real_dtype_for(backend) -> torch.dtype:
    cdtype = getattr(backend, "_dtype", torch.complex64)
    return torch.float64 if cdtype == torch.complex128 else torch.float32


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


class BatchSV:
    """一次演化一批态矢量, 支持逐样本门参数, NPU 安全且可微。

    Args:
        n_qubits: 量子比特数。
        batch_size: 批大小。
        backend: aicir 后端 (用于获取目标 device / dtype, 推荐 ``NPUBackend``)。
        device: 覆盖后端 device (可选)。
        real_dtype: 覆盖实张量 dtype (可选, 默认由后端复数 dtype 推断)。
    """

    def __init__(self, n_qubits: int, batch_size: int, backend, *,
                 device=None, real_dtype: Optional[torch.dtype] = None):
        if n_qubits < 1:
            raise ValueError("n_qubits 必须 >= 1")
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        self.n_qubits = int(n_qubits)
        self.batch_size = int(batch_size)
        self.backend = backend
        self.dim = 1 << self.n_qubits
        self.device = device if device is not None else getattr(backend, "_device", torch.device("cpu"))
        self.real_dtype = real_dtype if real_dtype is not None else _real_dtype_for(backend)

        # |0...0> 基态: real[:, 0] = 1, 其余为 0。
        real = torch.zeros((self.batch_size, self.dim), dtype=self.real_dtype, device=self.device)
        real[:, 0] = 1.0
        self.real = real
        self.imag = torch.zeros_like(real)

    # ──────────────────────────── 门基矩阵 ────────────────────────────

    def _base_2x2(self, gate):
        """返回单比特基矩阵的 (real, imag), 每个形状 (2,2) 或 (batch,2,2)。

        标量/常量门复用 aicir 单态路径的 ``_single_qubit_base_for_gate``;
        逐样本张量角度时按相同公式构造批量基矩阵。
        """
        gate_type = canonical_gate_name(instruction_name(gate))
        parameter = instruction_parameter(gate)

        if _is_tensor(parameter):
            # 张量角度: 0 维标量(保留梯度)或逐样本 (batch,) —— 构造 (k, 2, 2)
            # 实/虚基矩阵, k=1 时按 batch 广播, k=batch 时逐样本。
            t = parameter.to(device=self.device, dtype=self.real_dtype).reshape(-1)
            half = t * 0.5
            cos = torch.cos(half)
            sin = torch.sin(half)
            zero = torch.zeros_like(cos)
            if gate_type in ("ry", "cry"):
                real = torch.stack([torch.stack([cos, -sin], -1),
                                    torch.stack([sin, cos], -1)], -2)
                imag = torch.zeros_like(real)
                return real, imag
            if gate_type in ("rx", "crx"):
                real = torch.stack([torch.stack([cos, zero], -1),
                                    torch.stack([zero, cos], -1)], -2)
                imag = torch.stack([torch.stack([zero, -sin], -1),
                                    torch.stack([-sin, zero], -1)], -2)
                return real, imag
            if gate_type in ("rz", "crz"):
                real = torch.stack([torch.stack([cos, zero], -1),
                                    torch.stack([zero, cos], -1)], -2)
                imag = torch.stack([torch.stack([-sin, zero], -1),
                                    torch.stack([zero, sin], -1)], -2)
                return real, imag
            raise ValueError(
                f"批量逐样本参数暂不支持门类型: {gate_type} (仅 rx/ry/rz 及其受控形式)"
            )

        # 标量参数 / 常量门: 复用 aicir 单态路径的基矩阵定义 (单一事实来源)。
        base = _single_qubit_base_for_gate(gate)
        if base is None:
            raise ValueError(f"批量路径暂不支持门类型: {gate_type}")
        base = np.asarray(base)
        real = torch.tensor(np.real(base), dtype=self.real_dtype, device=self.device)
        imag = torch.tensor(np.imag(base), dtype=self.real_dtype, device=self.device)
        return real, imag

    def _active_mask(self, controls, control_states):
        """受控门: 返回布尔张量 (dim,), 标记控制比特满足要求的基态。"""
        idx, _ = _bit_tensors(self.n_qubits, self.device, self.real_dtype)
        mask = torch.ones(self.dim, dtype=torch.bool, device=self.device)
        for c, st in zip(controls, control_states):
            bit = (idx >> (self.n_qubits - 1 - int(c))) & 1
            mask = mask & (bit == int(st))
        return mask

    # ──────────────────────────── 门作用 ──────────────────────────────

    def apply_gate(self, gate) -> "BatchSV":
        """就地作用一个门 (dict 或 Operation), 返回自身以便链式调用。"""
        gate = as_instruction(gate)
        gate_type = canonical_gate_name(instruction_name(gate))

        if gate_type == "identity":
            return self

        if gate_type in ("rzz", "rxx"):
            qs = [int(q) for q in instruction_qubits(gate)]
            if len(qs) != 2 or qs[0] == qs[1]:
                raise ValueError(f"{gate_type} 需要两个不同的量子比特, 得到 {qs}")
            theta = self._angle_tensor(instruction_parameter(gate))
            if gate_type == "rzz":
                self._apply_rzz(qs[0], qs[1], theta)
            else:
                self._apply_rxx(qs[0], qs[1], theta)
            return self

        target = int(instruction_qubits(gate)[0])
        controls = [int(c) for c in instruction_controls(gate)]
        control_states = [int(s) for s in instruction_control_states(gate)]
        if target in controls:
            raise ValueError("控制比特与目标比特不能相同")

        real, imag = self._base_2x2(gate)
        active = self._active_mask(controls, control_states) if controls else None
        self._apply_single_qubit(target, real, imag, active)
        return self

    def _apply_single_qubit(self, q, ur, ui, active):
        n = self.n_qubits
        B = self.batch_size
        hi = 1 << q
        lo = 1 << (n - 1 - q)

        vr = self.real.reshape(B, hi, 2, lo)
        vi = self.imag.reshape(B, hi, 2, lo)
        a0r, a0i = vr[:, :, 0, :], vi[:, :, 0, :]
        a1r, a1i = vr[:, :, 1, :], vi[:, :, 1, :]

        def el(mat, i, j):
            v = mat[..., i, j]
            if _is_tensor(v) and v.dim() == 1:  # (k,): k=1 广播, k=batch 逐样本
                return v.view(-1, 1, 1)
            return v  # 0 维标量, 自动广播

        u00r, u01r, u10r, u11r = el(ur, 0, 0), el(ur, 0, 1), el(ur, 1, 0), el(ur, 1, 1)
        u00i, u01i, u10i, u11i = el(ui, 0, 0), el(ui, 0, 1), el(ui, 1, 0), el(ui, 1, 1)

        # 复数乘加 (全程实张量)。
        out0r = u00r * a0r - u00i * a0i + u01r * a1r - u01i * a1i
        out0i = u00r * a0i + u00i * a0r + u01r * a1i + u01i * a1r
        out1r = u10r * a0r - u10i * a0i + u11r * a1r - u11i * a1i
        out1i = u10r * a0i + u10i * a0r + u11r * a1i + u11i * a1r

        if active is not None:
            # 控制比特与目标无关, 同一对 (target=0/1) 共享 active。
            m = active.reshape(hi, 2, lo)[:, 0, :].unsqueeze(0)  # (1, hi, lo)
            out0r = torch.where(m, out0r, a0r)
            out0i = torch.where(m, out0i, a0i)
            out1r = torch.where(m, out1r, a1r)
            out1i = torch.where(m, out1i, a1i)

        new_real = torch.stack([out0r, out1r], dim=2).reshape(B, self.dim)
        new_imag = torch.stack([out0i, out1i], dim=2).reshape(B, self.dim)
        self.real = new_real
        self.imag = new_imag

    # ──────────────────────────── 双比特门 ────────────────────────────

    def _angle_tensor(self, parameter) -> torch.Tensor:
        """把门角度归一为 (k,) 实张量, k=1(广播) 或 batch(逐样本)。"""
        if _is_tensor(parameter):
            t = parameter.to(device=self.device, dtype=self.real_dtype).reshape(-1)
        else:
            t = torch.tensor([float(parameter)], dtype=self.real_dtype, device=self.device)
        if t.numel() not in (1, self.batch_size):
            raise ValueError(f"角度长度 {t.numel()} 须为 1 或 batch={self.batch_size}")
        return t

    def _apply_rzz(self, q1: int, q2: int, theta: torch.Tensor) -> None:
        """rzz(θ) = diag(e^{∓iθ/2})：对每个基态施相位 -θ/2·zz, 全程实张量。"""
        _, zsign = _bit_tensors(self.n_qubits, self.device, self.real_dtype)
        zz = zsign[q1] * zsign[q2]                    # (D,), ±1
        phi = (-0.5 * theta).reshape(-1, 1) * zz      # (k, D) 广播
        c, s = torch.cos(phi), torch.sin(phi)
        new_real = self.real * c - self.imag * s
        new_imag = self.real * s + self.imag * c
        self.real, self.imag = new_real, new_imag

    def _apply_rxx(self, q1: int, q2: int, theta: torch.Tensor) -> None:
        """rxx(θ) = cos(θ/2)·I - i·sin(θ/2)·X⊗X：X⊗X 即双比特同时翻转（索引异或）。"""
        idx, _ = _bit_tensors(self.n_qubits, self.device, self.real_dtype)
        mask = (1 << (self.n_qubits - 1 - q1)) | (1 << (self.n_qubits - 1 - q2))
        perm = idx ^ mask                             # (D,)
        re_f = self.real.index_select(1, perm)
        im_f = self.imag.index_select(1, perm)
        half = (0.5 * theta).reshape(-1, 1)
        c, s = torch.cos(half), torch.sin(half)
        # -i·sin·(a+bi) = sin·b - i·sin·a
        new_real = c * self.real + s * im_f
        new_imag = c * self.imag - s * re_f
        self.real, self.imag = new_real, new_imag

    # ──────────────────────────── 读出 ────────────────────────────────

    def probabilities(self) -> torch.Tensor:
        """计算基测量概率, 返回 (batch, 2^n) 实张量。"""
        return self.real * self.real + self.imag * self.imag

    def z_expectations(self) -> torch.Tensor:
        """逐比特泡利 Z 期望 <Z_q>, 返回 (batch, n_qubits) 实张量。"""
        _, zsign = _bit_tensors(self.n_qubits, self.device, self.real_dtype)
        probs = self.probabilities()
        return probs @ zsign.transpose(0, 1)

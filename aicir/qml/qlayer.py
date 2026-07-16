"""把 :class:`~aicir.qml.qfun.QFun` 封装成 PyTorch ``nn.Module`` 量子层。

设计：不重写梯度，而是把 ``QFun`` 桥接进 torch autograd —— 前向调用
``qfun(params)`` 取期望值，反向调用 ``qfun.grad(params)`` 取参数移位 Jacobian。
因此该层与 ``QFun`` 的后端解耦，``device="numpy"/"gpu"/"npu"`` 皆可用，梯度方法
仍走 ``aicir.qml.diff`` 注册表这一单一真源。

经典输入与可训练权重经 ``torch.cat`` 拼成单个参数向量喂给 ``qfun``（``cat`` 可微，
故梯度同时回流到前置经典层的输入与本层权重），可一行嵌入经典 PyTorch 网络：

```python
import torch
from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun, QLayer

@qfun(observable=Hamiltonian([("Z", 1.0)]))
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta[0], 0))
    return c

model = torch.nn.Sequential(torch.nn.Linear(4, 1), QLayer(cost, n_weights=0))
```
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from ..core.batch import BatchSV
from ..core.circuit import Parameter
from ..gates import canonical_gate_name
from ..ir import instruction_name
from .qfun import QFun

# BatchSV 支持张量角度的参数门（rx/ry/rz 及受控形式来自 _base_2x2 张量分支，
# rzz/rxx 来自双比特实张量路径）
_BATCH_PARAM_GATES = {"rx", "ry", "rz", "crx", "cry", "crz", "rzz", "rxx"}


class _QFunApply(torch.autograd.Function):
    """把 ``QFun`` 的前向/参数移位反向接入 torch autograd 的桥。

    ``params`` 恒为一维张量；``qfun(params_np)`` 返回标量（单观测量）或 ``(n_obs,)``
    数组（多观测量），``qfun.grad`` 相应返回 ``(P,)`` 或 ``(n_obs, P)`` Jacobian。
    """

    @staticmethod
    def forward(ctx, params: torch.Tensor, qfun: QFun) -> torch.Tensor:
        ctx.qfun = qfun
        ctx.save_for_backward(params)
        p = params.detach().cpu().numpy().astype(float)
        out = np.asarray(qfun(p), dtype=float)
        return torch.as_tensor(out, dtype=params.dtype, device=params.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (params,) = ctx.saved_tensors
        p = params.detach().cpu().numpy().astype(float)
        jac = np.asarray(ctx.qfun.grad(p), dtype=float)
        g = grad_output.detach().cpu().numpy().astype(float)
        if jac.ndim == 1:
            # 单观测量：grad_output 标量，jac 形如 (P,)
            grad_p = g * jac
        else:
            # 多观测量：g 形如 (n_obs,)，jac 形如 (n_obs, P)
            grad_p = g.reshape(-1) @ jac
        grad_params = torch.as_tensor(grad_p, dtype=params.dtype, device=params.device)
        return grad_params, None


class QLayer(torch.nn.Module):
    """把 ``QFun`` 封装成可嵌入 PyTorch 的量子层。

    Args:
        qfun: 已声明设备/观测量/梯度方法的 :class:`QFun`（须为 ``expval`` 返回）。
        n_weights: 本层可训练权重个数（拼在输入之后传给 ``qfun``）。为 0 时纯粹
            由外部输入驱动（如做数据编码层）。
        init: 权重初值（数组/张量，长度须为 ``n_weights``）；缺省在 ``[0, 2π)``
            均匀采样。
        dtype: 权重与输出张量的浮点 dtype，默认 ``torch.float32``（对齐 torch 生态）。

    ``forward(inputs=None)``：
        - ``inputs is None`` → 仅用权重，返回标量/``(n_obs,)``。
        - ``inputs`` 一维 → 拼 ``[inputs, weights]``，返回标量/``(n_obs,)``。
        - ``inputs`` 二维 ``(batch, features)`` → 逐行求值堆叠，返回 ``(batch,)``
          或 ``(batch, n_obs)``。
    """

    def __init__(
        self,
        qfun: QFun,
        n_weights: int,
        *,
        init: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not isinstance(qfun, QFun):
            raise TypeError("QLayer 需要一个 aicir.qml.QFun 实例")
        if int(n_weights) < 0:
            raise ValueError("n_weights 不能为负")
        self.qfun = qfun
        self.n_weights = int(n_weights)
        if init is None:
            data = torch.empty(self.n_weights, dtype=dtype).uniform_(0.0, 2.0 * np.pi)
        else:
            data = torch.as_tensor(np.asarray(init, dtype=float), dtype=dtype).reshape(-1)
            if data.numel() != self.n_weights:
                raise ValueError(f"init 长度 {data.numel()} 与 n_weights={self.n_weights} 不符")
        self.weights = torch.nn.Parameter(data)

    def _run(self, params: torch.Tensor) -> torch.Tensor:
        return _QFunApply.apply(params, self.qfun)

    def forward(self, inputs: Any = None) -> torch.Tensor:
        w = self.weights
        if inputs is None:
            if self.n_weights == 0:
                raise ValueError("n_weights=0 的 QLayer 需要提供 inputs")
            return self._run(w)

        x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
        x = x.to(dtype=w.dtype, device=w.device)
        if x.dim() <= 1:
            return self._run(torch.cat([x.reshape(-1), w]))
        rows = [self._run(torch.cat([row.reshape(-1), w])) for row in x]
        return torch.stack(rows, dim=0)


class BatchLayer(torch.nn.Module):
    """固定模板线路的批量量子层：BatchSV 整批前向 + 原生 autograd 反向。

    与 :class:`QLayer` 的分工：QLayer 包任意 ``QFun``、逐行求值、参数移位反向，
    通用但每样本一次演化；BatchLayer 要求**固定模板线路**（参数门限
    ``rx/ry/rz/crx/cry/crz/rzz/rxx``），换来整批一次演化与端到端 torch
    autograd —— 全程实部/虚部实张量，NPU 安全（无复数内核）。

    参数约定（与 QLayer 的 ``cat([inputs, weights])`` 同序）：模板
    ``circuit.parameters()`` 首用序的前 ``n_inputs`` 个为数据编码参数
    （逐样本取 ``inputs`` 对应列），其余为本层可训练权重。

    读出为逐比特 ``<Z_q>``：``forward(inputs (batch, n_inputs))`` 返回
    ``(batch, n_qubits)``；一维输入返回 ``(n_qubits,)``。

    Args:
        circuit: 含符号 :class:`~aicir.core.circuit.Parameter` 的模板线路。
        n_inputs: 数据编码参数个数（模板参数首用序的前缀）。
        backend: torch 系后端（``GPUBackend``/``NPUBackend``），决定 device/dtype。
        init: 权重初值（长度 = 模板参数数 - n_inputs）；缺省 ``[0, 2π)`` 均匀采样。
        dtype: 权重 dtype，默认 ``torch.float32``。
    """

    def __init__(
        self,
        circuit,
        n_inputs: int,
        *,
        backend,
        init: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        params = list(circuit.parameters)
        n_inputs = int(n_inputs)
        if not 0 <= n_inputs <= len(params):
            raise ValueError(f"n_inputs={n_inputs} 越界（模板共 {len(params)} 个参数）")
        index_of = {id(p): i for i, p in enumerate(params)}

        specs = []  # (gate, param_index or None)
        for gate in circuit.gates:
            raw = tuple(getattr(gate, "params", ()) or ())
            symbolic = [v for v in raw if isinstance(v, Parameter)]
            if symbolic:
                name = canonical_gate_name(instruction_name(gate))
                if name not in _BATCH_PARAM_GATES or len(raw) != 1:
                    raise ValueError(
                        f"BatchLayer 参数门仅支持单参数的 {sorted(_BATCH_PARAM_GATES)}，得到 {name}")
                specs.append((gate, index_of[id(symbolic[0])]))
            else:
                specs.append((gate, None))

        self.n_qubits = int(circuit.n_qubits)
        self.n_inputs = n_inputs
        self.n_weights = len(params) - n_inputs
        self._specs = specs
        self._backend = backend
        if init is None:
            data = torch.empty(self.n_weights, dtype=dtype).uniform_(0.0, 2.0 * np.pi)
        else:
            data = torch.as_tensor(np.asarray(init, dtype=float), dtype=dtype).reshape(-1)
            if data.numel() != self.n_weights:
                raise ValueError(f"init 长度 {data.numel()} 与权重数 {self.n_weights} 不符")
        self.weights = torch.nn.Parameter(data)

    def forward(self, inputs: Any = None) -> torch.Tensor:
        if inputs is None:
            if self.n_inputs != 0:
                raise ValueError(f"模板含 {self.n_inputs} 个数据编码参数，需提供 inputs")
            x = None
            batch = 1
            single = True
        else:
            x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
            single = x.dim() <= 1
            x = x.reshape(1, -1) if single else x
            if x.shape[1] != self.n_inputs:
                raise ValueError(f"inputs 特征数 {x.shape[1]} 与 n_inputs={self.n_inputs} 不符")
            batch = x.shape[0]

        sv = BatchSV(self.n_qubits, batch, self._backend)
        x = None if x is None else x.to(dtype=sv.real_dtype)
        for gate, idx in self._specs:
            if idx is None:
                sv.apply_gate(gate)
            else:
                angle = x[:, idx] if idx < self.n_inputs else self.weights[idx - self.n_inputs]
                sv.apply_gate(dataclasses.replace(gate, params=(angle,)))
        out = sv.z_expectations()  # (batch, n_qubits)，位于后端设备
        # 读出移回模块参数所在设备（默认 CPU），使下游经典 nn.Module 无需设备对齐；
        # 演化仍在后端设备（NPU/GPU），仅 (batch, n_qubits) 小实张量跨设备（autograd 安全）。
        out = out.to(self.weights.device)
        return out.reshape(-1) if single else out

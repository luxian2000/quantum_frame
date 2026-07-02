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
from aicir.qml import qfun, TorchLayer

@qfun(observable=Hamiltonian([("Z", 1.0)]))
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta[0], 0))
    return c

model = torch.nn.Sequential(torch.nn.Linear(4, 1), TorchLayer(cost, n_weights=0))
```
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .qfun import QFun


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


class TorchLayer(torch.nn.Module):
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
            raise TypeError("TorchLayer 需要一个 aicir.qml.QFun 实例")
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
                raise ValueError("n_weights=0 的 TorchLayer 需要提供 inputs")
            return self._run(w)

        x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
        x = x.to(dtype=w.dtype, device=w.device)
        if x.dim() <= 1:
            return self._run(torch.cat([x.reshape(-1), w]))
        rows = [self._run(torch.cat([row.reshape(-1), w])) for row in x]
        return torch.stack(rows, dim=0)

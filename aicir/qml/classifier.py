"""端到端量子分类器工厂（qml 成熟化 #6）。

把「角度编码 → 硬件高效纠缠层 → 逐比特 <Z_q> 读出 → 线性头」组合成一个
标准 ``torch.nn.Module``，可直接用 torch 优化器/损失训练，可嵌入经典网络。
量子部分走 :class:`~aicir.qml.qlayer.BatchLayer`（整批一次演化、实/虚分离、
NPU 安全、原生 autograd），故大 batch 训练在 NPU 上高效。

模板构造：``n_features`` 个 ``rx`` 数据编码（逐样本），随后 ``layers`` 层
``ry`` 局部旋转 + ``rzz``/``cx`` 环形纠缠（权重参数）。这套门集正是 BatchSV
批量路径支持的子集（见 ``aicir.core.batch``）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch 可选
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..core.circuit import Circuit, Parameter, cx, rx, ry, rzz
from .qlayer import BatchLayer


def classifier_template(n_features: int, n_qubits: int, layers: int) -> Circuit:
    """构造分类器模板线路：前 ``n_features`` 个参数为数据编码，其余为权重。

    参数首用序满足 :class:`BatchLayer` 约定（前 ``n_features`` 个逐样本编码）。
    """
    if n_features < 1 or n_qubits < 1 or layers < 1:
        raise ValueError("n_features/n_qubits/layers 均须 ≥1")
    xs = [Parameter(f"x{i}") for i in range(n_features)]
    gates = [rx(xs[i], i % n_qubits) for i in range(n_features)]  # 数据编码（逐样本）
    w = 0
    for _ in range(layers):
        for q in range(n_qubits):
            gates.append(ry(Parameter(f"w{w}"), q)); w += 1
        if n_qubits > 1:
            for q in range(n_qubits):
                gates.append(rzz(Parameter(f"w{w}"), q, (q + 1) % n_qubits)); w += 1
                gates.append(cx((q + 1) % n_qubits, [q]))
    return Circuit(*gates, n_qubits=n_qubits)


def build_classifier(
    *,
    n_features: int,
    n_classes: int,
    backend,
    n_qubits: int | None = None,
    layers: int = 2,
    seed: int | None = None,
) -> "torch.nn.Module":
    """构造量子分类器 ``nn.Module``：BatchLayer 读出 → 线性头 → logits。

    Args:
        n_features: 输入特征数（= 数据编码参数数）。
        n_classes: 类别数（线性头输出维度）。
        backend: torch 系后端（``GPUBackend``/``NPUBackend``）。
        n_qubits: 量子比特数，缺省 = ``n_features``。
        layers: 纠缠层数。
        seed: 权重初始化随机种子（可复现）。

    Returns:
        ``nn.Module``：``forward(x (batch, n_features)) -> logits (batch, n_classes)``。
    """
    if torch is None:  # pragma: no cover
        raise ImportError("build_classifier 需要 torch")
    n_qubits = int(n_features if n_qubits is None else n_qubits)
    if seed is not None:
        torch.manual_seed(int(seed))
    template = classifier_template(n_features, n_qubits, layers)
    qlayer = BatchLayer(template, n_inputs=n_features, backend=backend)
    head = torch.nn.Linear(n_qubits, int(n_classes))
    return torch.nn.Sequential(qlayer, head)

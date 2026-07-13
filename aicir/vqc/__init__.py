"""Variational quantum computing algorithms.

``VQE``/``VQD``/``SSVQE``/``QAOA`` 四个子模块的 numpy 路径均不依赖 torch
（``..qml.deriv`` 的 torch 探测/设备驻留分支是函数体内惰性 import，参见
``aicir/qml/deriv/_coerce.py``/``fn_gradient.py``），因此本包无需再对
``BasicVQE``/``BasicVQD``/``BasicSSVQE`` 做 torch 可选降级：无 torch 环境下
``from aicir.vqc import BasicVQE`` 恒可成功，numpy 后端 VQE 端到端可运行。
"""

from __future__ import annotations

from .QAOA import BasicQAOA, QAOAResult, run_qaoa
from .SSVQE import BasicSSVQE, SSVQEResult, run_ssvqe
from .VQD import BasicVQD, VQDResult, run_vqd
from .VQE import BasicVQE, VQEResult, run_vqe

__all__ = [
    "BasicQAOA",
    "QAOAResult",
    "run_qaoa",
    "BasicVQE",
    "VQEResult",
    "run_vqe",
    "BasicVQD",
    "VQDResult",
    "run_vqd",
    "BasicSSVQE",
    "SSVQEResult",
    "run_ssvqe",
]

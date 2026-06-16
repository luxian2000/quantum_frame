"""Sampler / Estimator primitives：算法层的统一执行入口（NEXT.md 第 4 节第一片）。

- :class:`ShotSampler`：有限 shots 采样，包装 ``Measure``。
- :class:`StatevectorEstimator`：精确期望（态向量路径）。
- :class:`ShotEstimator`：有限 shots 能量估计，包装 ``PauliEstimator``。

约定见 :mod:`aicir.primitives.base`：接收已绑定参数的电路；单个入参返回
单个结果，序列入参返回列表；单个可观测量可广播。``Noisy*``/``Backend*``
变体留待需要时新增。
"""

from .base import BaseEstimator, BaseSampler
from .estimator import ShotEstimator, StatevectorEstimator
from .results import EstimateResult, SampleResult
from .sampler import ShotSampler

__all__ = [
    "BaseEstimator",
    "BaseSampler",
    "EstimateResult",
    "SampleResult",
    "ShotEstimator",
    "ShotSampler",
    "StatevectorEstimator",
]

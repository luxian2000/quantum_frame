"""Sampler / Estimator primitives：算法层的统一执行入口（NEXT.md 第 4 节）。

采样：

- :class:`StatevectorSampler`：精确解析概率（无散粒噪声）。
- :class:`ShotSampler`：有限 shots 采样，包装 ``Measure``。
- :class:`NoisySampler`：带噪声采样（密度矩阵路径）。

估计：

- :class:`StatevectorEstimator`：精确期望（态向量路径）。
- :class:`ShotEstimator`：有限 shots 能量估计，包装 ``PauliEstimator``。
- :class:`NoisyEstimator`：带噪声期望（密度矩阵路径）。

扩展点：

- :class:`BackendSampler` / :class:`BackendEstimator`：包装用户注入的 ``runner``，
  面向真实硬件或远端服务。

约定见 :mod:`aicir.primitives.base`：接收电路（可经 ``parameter_values=`` 延迟
绑定模板）；单个入参返回单个结果，序列入参返回列表；单个可观测量可广播。
"""

from .backend import BackendEstimator, BackendSampler
from .base import BaseEstimator, BaseSampler
from .estimator import NoisyEstimator, ShotEstimator, StatevectorEstimator
from .results import EstimateResult, SampleResult
from .sampler import NoisySampler, ShotSampler, StatevectorSampler

__all__ = [
    "BackendEstimator",
    "BackendSampler",
    "BaseEstimator",
    "BaseSampler",
    "EstimateResult",
    "NoisyEstimator",
    "NoisySampler",
    "SampleResult",
    "ShotEstimator",
    "ShotSampler",
    "StatevectorEstimator",
    "StatevectorSampler",
]

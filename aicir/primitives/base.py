"""Sampler/Estimator 抽象基类与输入归一化约定。

第一片约定（NEXT.md 第 4 节）：

- 接收**已绑定参数**的电路（``parameter_values=`` 延迟绑定留待后续）。
- 单个电路入参返回单个结果；序列入参返回结果列表。
- Estimator 支持单个可观测量广播到多个电路；电路与可观测量均为序列时
  按位置配对（长度必须一致）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def normalize_run_inputs(circuits: Any) -> tuple[list[Any], bool]:
    """把电路入参归一为列表，并返回是否为单个入参。"""

    if isinstance(circuits, (list, tuple)):
        return list(circuits), False
    return [circuits], True


def pair_observables(circuits: list[Any], observables: Any) -> list[Any]:
    """把可观测量入参配对到电路列表（单个可观测量广播）。"""

    if isinstance(observables, (list, tuple)):
        if len(observables) != len(circuits):
            raise ValueError(
                f"observables length {len(observables)} does not match "
                f"circuits length {len(circuits)}"
            )
        return list(observables)
    return [observables] * len(circuits)


class BaseSampler(ABC):
    """采样 primitive 统一接口。"""

    @abstractmethod
    def run(self, circuits, *, shots: int | None = None):
        """对电路采样，返回 :class:`SampleResult`（或其列表）。"""


class BaseEstimator(ABC):
    """期望值估计 primitive 统一接口。"""

    @abstractmethod
    def run(self, circuits, observables, *, shots: int | None = None):
        """估计期望值，返回 :class:`EstimateResult`（或其列表）。"""

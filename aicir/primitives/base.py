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

import numpy as np


def _bind_one(circuit: Any, values: Any) -> Any:
    """按 ``parameter_values`` 绑定单个电路；``values is None`` 时原样返回。"""

    if values is None:
        return circuit
    return circuit.bind_parameters(np.asarray(values, dtype=float).reshape(-1))


def normalize_run_inputs(
    circuits: Any, parameter_values: Any = None
) -> tuple[list[Any], bool]:
    """把电路入参归一为列表并返回是否为单个入参。

    若提供 ``parameter_values``，按与电路相同的单/列表形状对齐后绑定模板电路
    （单电路 → 一维数组；电路序列 → 各电路对应的一维数组序列）。
    """

    if isinstance(circuits, (list, tuple)):
        items = list(circuits)
        single = False
        pvs = list(parameter_values) if parameter_values is not None else [None] * len(items)
    else:
        items = [circuits]
        single = True
        pvs = [parameter_values]

    if len(pvs) != len(items):
        raise ValueError(
            f"parameter_values length {len(pvs)} does not match circuits length {len(items)}"
        )
    items = [_bind_one(circuit, values) for circuit, values in zip(items, pvs)]
    return items, single


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
    def run(self, circuits, *, shots: int | None = None, parameter_values=None):
        """对电路采样，返回 :class:`SampleResult`（或其列表）。"""


class BaseEstimator(ABC):
    """期望值估计 primitive 统一接口。"""

    @abstractmethod
    def run(self, circuits, observables, *, shots: int | None = None, parameter_values=None):
        """估计期望值，返回 :class:`EstimateResult`（或其列表）。"""

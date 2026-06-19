"""QAS 搜索策略协议（模块化第一片）。

每种架构搜索算法（supernet / RL / 进化 …）实现统一的 :class:`SearchStrategy`
契约，从而可经 :mod:`aicir.qas.core.registry` 按名解析、按需替换，取代
``runner.py`` 中按方法名硬编码的 ``if`` 分发链。

约定：``run(request)`` 接收方法无关的请求对象（``QASRunConfig`` 或等价体），
由各策略自行提取所需字段并调用其底层算法，返回该算法的原生结果。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SearchStrategy(ABC):
    """单个架构搜索策略的统一执行接口。"""

    name: str

    @abstractmethod
    def run(self, request: Any) -> Any:
        """执行搜索，返回底层算法的结果。"""

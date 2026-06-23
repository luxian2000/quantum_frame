"""把内置 QAS 算法适配为 :class:`SearchStrategy` 并注册（模块化第一片）。

导入本模块即完成内置策略的注册。当前仅迁移 ``supernet``；其余算法仍走
``runner.py`` 的旧分支，后续逐个适配。
"""

from __future__ import annotations

from typing import Any

from .registry import StrategySpec, register_strategy
from .strategy import SearchStrategy


class SupernetStrategy(SearchStrategy):
    """权重共享 supernet 架构搜索。"""

    name = "supernet"

    def run(self, request: Any) -> Any:
        from ..algorithms.supernet import train_supernet

        return train_supernet(
            objective=request.objective,
            config=request.config,
            dataset=request.dataset,
            hamiltonian=request.hamiltonian,
        )


register_strategy(StrategySpec("supernet", SupernetStrategy(), requires_torch=True))

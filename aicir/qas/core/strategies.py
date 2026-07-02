"""内置 QAS 策略适配与注册（QAS README §2.1）。

import 本模块即把内置策略注册进 ``registry``（import 副作用）。**新增同类算法
时只改本文件**：写一个 ``SearchStrategy`` 子类、再 ``register_strategy(...)`` 即可，
框架（``registry.py``）无需改动。同时从本模块再导出注册表 API，故
``from .strategies import get_strategy`` 等既能用又会触发注册。

当前 ``supernet`` 和 ``dqas`` 已迁移为 ``SearchStrategy``；``ppo_rb``/``ppr_dql``/
``crlqas``/``qdrats``/``supernet_classification``/``supernet_h2`` 仍走 ``runner`` 的 ``_Spec``
分发表，行为不变，后续逐个适配。
"""

from __future__ import annotations

from typing import Any

from .registry import (
    SearchStrategy,
    StrategySpec,
    get_spec,
    get_strategy,
    register_strategy,
    registered_strategies,
    unregister_strategy,
)


class SupernetStrategy(SearchStrategy):
    """``run("supernet", ...)`` 的适配器：分发到 ``train_supernet``。"""

    _PARAMS = ("objective", "config", "dataset", "hamiltonian")

    def run(self, request: Any) -> Any:
        # 懒导入：supernet 依赖 torch，避免在无 torch 环境 import 本模块即失败。
        from ..algorithms.supernet import train_supernet

        return train_supernet(**{name: getattr(request, name, None) for name in self._PARAMS})


class DQASStrategy(SearchStrategy):
    """``run("dqas", ...)`` 的适配器：分发到 ``train_dqas``。"""

    _PARAMS = ("hamiltonian", "config")

    def run(self, request: Any) -> Any:
        from ..algorithms.dqas import train_dqas

        if getattr(request, "hamiltonian", None) is None:
            raise ValueError("dqas requires hamiltonian.")
        return train_dqas(**{name: getattr(request, name, None) for name in self._PARAMS})


register_strategy(StrategySpec("supernet", SupernetStrategy(), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("dqas", DQASStrategy(), aliases=("differentiable_qas",), requires_torch=True), overwrite=True)


__all__ = [
    "SearchStrategy",
    "DQASStrategy",
    "StrategySpec",
    "SupernetStrategy",
    "get_spec",
    "get_strategy",
    "register_strategy",
    "registered_strategies",
    "unregister_strategy",
]

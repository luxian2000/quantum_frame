"""内置 QAS 策略注册（QAS README §2.3）。

import 本模块即把内置策略注册进 ``registry``（import 副作用，由 ``runner``
在模块加载时触发）。**新增同类算法时改两处**：在 ``adapters.py`` 写一个
``SearchStrategy`` 子类，再在本文件底部 ``register_strategy(...)`` 一行；
``registry.py`` 的框架部分不用改。同时从本模块再导出注册表 API，故
``from .strategies import get_strategy`` 等既能用又会触发注册。

3b 起 ``core.config._FACTORIES`` 的全部 10 个方法
（``supernet``/``supernet_classification``/``supernet_h2``/``pporb``/
``pprdql``/``crlqas``/``qdrats``/``dqas``/``vqe_loop``/``mogvqe``）都已迁移为
``SearchStrategy``；``runner`` 的旧 ``_Spec`` 分发表（``_TABLE``）已删除，
``run()`` 现在只查本注册表。除 ``vqe_loop``/``mogvqe`` 外均 ``requires_torch=True``
（对应底层算法模块在 import 期硬依赖 torch）。
"""

from __future__ import annotations

from .adapters import (
    CRLQASStrategy,
    DQASStrategy,
    MOGVQEStrategy,
    PPORBStrategy,
    PPRDQLStrategy,
    QDRATSStrategy,
    SupernetClassificationStrategy,
    SupernetH2Strategy,
    SupernetStrategy,
    VqeLoopStrategy,
)
from .registry import (
    SearchStrategy,
    StrategySpec,
    get_spec,
    get_strategy,
    register_strategy,
    registered_strategies,
    unregister_strategy,
)

register_strategy(StrategySpec("supernet", SupernetStrategy(), requires_torch=True), overwrite=True)
register_strategy(
    StrategySpec("supernet_classification", SupernetClassificationStrategy(), requires_torch=True), overwrite=True
)
register_strategy(StrategySpec("supernet_h2", SupernetH2Strategy(), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("pporb", PPORBStrategy(), aliases=("ppo", "ppo_rb"), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("pprdql", PPRDQLStrategy(), aliases=("ppr", "ppr_dql"), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("crlqas", CRLQASStrategy(), aliases=("crl",), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("qdrats", QDRATSStrategy(), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("dqas", DQASStrategy(), aliases=("differentiable_qas",), requires_torch=True), overwrite=True)
register_strategy(StrategySpec("vqe_loop", VqeLoopStrategy(), aliases=("vqe_qas", "vqe_closed_loop"), requires_torch=False), overwrite=True)
register_strategy(StrategySpec("mogvqe", MOGVQEStrategy(), requires_torch=False), overwrite=True)


__all__ = [
    "SearchStrategy",
    "CRLQASStrategy",
    "DQASStrategy",
    "MOGVQEStrategy",
    "PPORBStrategy",
    "PPRDQLStrategy",
    "QDRATSStrategy",
    "StrategySpec",
    "SupernetClassificationStrategy",
    "SupernetH2Strategy",
    "SupernetStrategy",
    "VqeLoopStrategy",
    "get_spec",
    "get_strategy",
    "register_strategy",
    "registered_strategies",
    "unregister_strategy",
]

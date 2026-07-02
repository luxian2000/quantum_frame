"""QAS ``SearchStrategy`` 协议 + 策略注册表（QAS README §2.1）。

框架部分（接口 + 注册表）：与 ``aicir.gates``（``GateSpec``）、``aicir.qml.diff``
（``DiffMethod``）同一习惯。具体策略实现与注册放在 ``strategies.py``——新增同类
算法时只改那一个文件，本模块通常不动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class SearchStrategy(ABC):
    """QAS 搜索算法的统一执行协议。

    ``request`` 是 ``aicir.qas.core.runner.QASRunConfig`` 风格的请求对象，
    按字段读取 ``objective``/``config``/``hamiltonian``/``target_state`` 等。
    """

    @abstractmethod
    def run(self, request: Any) -> Any:
        """按 ``request`` 字段执行底层搜索，返回该方法的结果对象。"""


@dataclass(frozen=True)
class StrategySpec:
    """单个搜索策略的注册规格。

    - ``name``：规范方法名（如 ``"supernet"``）。
    - ``strategy``：``SearchStrategy`` 实例。
    - ``aliases``：可选别名。
    - ``requires_torch``：该策略是否依赖 ``torch``（仅作元信息暴露）。
    """

    name: str
    strategy: SearchStrategy
    aliases: tuple[str, ...] = ()
    requires_torch: bool = False


_REGISTRY: dict[str, StrategySpec] = {}
_ALIASES: dict[str, str] = {}


def _canonical(name: str) -> str:
    return _ALIASES.get(name, name)


def register_strategy(spec: StrategySpec, *, overwrite: bool = False) -> StrategySpec:
    """注册一个 ``StrategySpec``；``overwrite=False`` 时重名抛 ``ValueError``。"""
    if not isinstance(spec, StrategySpec):
        raise TypeError("register_strategy expects a StrategySpec")
    if not overwrite and spec.name in _REGISTRY:
        raise ValueError(f"strategy {spec.name!r} already registered")
    unregister_strategy(spec.name)
    _REGISTRY[spec.name] = spec
    _ALIASES[spec.name] = spec.name
    for alias in spec.aliases:
        _ALIASES[alias] = spec.name
    return spec


def unregister_strategy(name: str) -> None:
    """注销策略（含其别名）；未注册时静默返回。"""
    canon = _canonical(name)
    if _REGISTRY.pop(canon, None) is None:
        return
    for alias in [a for a, target in _ALIASES.items() if target == canon]:
        del _ALIASES[alias]


def get_spec(name: str) -> StrategySpec | None:
    """按名（含别名）返回 ``StrategySpec``；未注册返回 ``None``。"""
    return _REGISTRY.get(_canonical(name))


def get_strategy(name: str) -> SearchStrategy | None:
    """按名（含别名）返回 ``SearchStrategy``；未注册返回 ``None``。"""
    spec = get_spec(name)
    return spec.strategy if spec is not None else None


def registered_strategies() -> tuple[str, ...]:
    """返回已注册的规范策略名（排序）。"""
    return tuple(sorted(_REGISTRY))


__all__ = [
    "SearchStrategy",
    "StrategySpec",
    "get_spec",
    "get_strategy",
    "register_strategy",
    "registered_strategies",
    "unregister_strategy",
]

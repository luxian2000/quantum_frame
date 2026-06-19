"""QAS 搜索策略注册表（模块化第一片）。

按规范名或别名登记 :class:`~aicir.qas.core.strategy.SearchStrategy`，供
``runner.run`` 按名解析分发。镜像 ``aicir.qml.diff`` / ``aicir.gates`` 的
frozen dataclass + 模块级注册表习惯。
"""

from __future__ import annotations

from dataclasses import dataclass

from .strategy import SearchStrategy


@dataclass(frozen=True)
class StrategySpec:
    """单个策略的元信息。

    - ``name``：规范方法名，如 ``"supernet"``。
    - ``strategy``：对应的 :class:`SearchStrategy` 实例。
    - ``aliases``：等价写法。
    - ``requires_torch``：是否需要 Torch（RL 方法为真）。仅元信息，调用方自行守卫。
    """

    name: str
    strategy: SearchStrategy
    aliases: tuple[str, ...] = ()
    requires_torch: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("StrategySpec name cannot be empty")
        if not isinstance(self.strategy, SearchStrategy):
            raise TypeError("StrategySpec strategy must be a SearchStrategy")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", tuple(str(a) for a in self.aliases))


_REGISTRY: dict[str, StrategySpec] = {}
_LOOKUP: dict[str, StrategySpec] = {}


def register_strategy(spec: StrategySpec, *, overwrite: bool = False) -> StrategySpec:
    """注册一个策略；``overwrite=False`` 时名称/别名冲突会报错。"""

    if not isinstance(spec, StrategySpec):
        raise TypeError("register_strategy expects a StrategySpec")

    names = (spec.name, *spec.aliases)
    if not overwrite:
        for name in names:
            existing = _LOOKUP.get(name)
            if existing is not None and existing.name != spec.name:
                raise ValueError(f"strategy name or alias already registered: {name!r}")
        if spec.name in _REGISTRY:
            raise ValueError(f"strategy already registered: {spec.name!r}")
    else:
        unregister_strategy(spec.name)

    _REGISTRY[spec.name] = spec
    for name in names:
        _LOOKUP[name] = spec
    return spec


def unregister_strategy(name: str) -> None:
    """移除一个已注册策略（含别名）；未注册时静默返回。"""

    spec = _REGISTRY.pop(str(name), None)
    if spec is None:
        return
    for key in (spec.name, *spec.aliases):
        if _LOOKUP.get(key) is spec:
            del _LOOKUP[key]


def get_spec(name: str) -> StrategySpec | None:
    """按规范名或别名查 spec；未注册返回 ``None``。"""

    return _LOOKUP.get(str(name))


def get_strategy(name: str) -> SearchStrategy | None:
    """按规范名或别名查策略实例；未注册返回 ``None``。"""

    spec = _LOOKUP.get(str(name))
    return spec.strategy if spec is not None else None


def registered_strategies() -> tuple[str, ...]:
    """返回全部已注册策略的规范名。"""

    return tuple(_REGISTRY)

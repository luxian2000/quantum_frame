"""DiffMethod 注册表：按规范名或别名查询梯度方法、解析 fn。

NEXT.md §6 第一片。内置 fn-based 全梯度方法在导入时注册一次：
``psr``/``fd``/``auto``/``spsa``/``spsr``。``mpsr`` 因返回标量混合偏导、
不满足 ``(fn, params) -> 梯度向量`` 契约，刻意排除（仍可用 ``qml.mpsr``）。
"""

from __future__ import annotations

from typing import Any, Callable

from ..deriv import auto, fd, psr, spsa, spsr
from .spec import DiffMethod

_REGISTRY: dict[str, DiffMethod] = {}
_LOOKUP: dict[str, DiffMethod] = {}


def register_diff_method(spec: DiffMethod, *, overwrite: bool = False) -> DiffMethod:
    """注册一个梯度方法；``overwrite=False`` 时名称/别名冲突会报错。"""

    if not isinstance(spec, DiffMethod):
        raise TypeError("register_diff_method expects a DiffMethod")

    names = (spec.name, *spec.aliases)
    if not overwrite:
        for name in names:
            existing = _LOOKUP.get(name)
            if existing is not None and existing.name != spec.name:
                raise ValueError(f"diff method name or alias already registered: {name!r}")
        if spec.name in _REGISTRY:
            raise ValueError(f"diff method already registered: {spec.name!r}")
    else:
        unregister_diff_method(spec.name)

    _REGISTRY[spec.name] = spec
    for name in names:
        _LOOKUP[name] = spec
    return spec


def unregister_diff_method(name: str) -> None:
    """移除一个已注册方法（含其全部别名）；未注册时静默返回。"""

    spec = _REGISTRY.pop(str(name), None)
    if spec is None:
        return
    for key in (spec.name, *spec.aliases):
        if _LOOKUP.get(key) is spec:
            del _LOOKUP[key]


def get_diff_method(name: str) -> DiffMethod | None:
    """按规范名或别名查询；未注册返回 ``None``。"""

    return _LOOKUP.get(str(name))


def registered_diff_methods() -> tuple[str, ...]:
    """返回全部已注册方法的规范名。"""

    return tuple(_REGISTRY)


def canonical_diff_name(name: str) -> str:
    """把方法名（含别名）解析为规范名；未注册的名称原样返回。"""

    spec = _LOOKUP.get(str(name))
    return spec.name if spec is not None else str(name)


def resolve_diff_method(name: str) -> Callable[..., Any]:
    """返回方法对应的可调用 ``fn``；未注册名抛 ``ValueError`` 并列出已注册方法。"""

    spec = _LOOKUP.get(str(name))
    if spec is None:
        available = ", ".join(sorted(registered_diff_methods()))
        raise ValueError(f"unknown diff method {name!r}; registered methods: {available}")
    return spec.fn


# ---------------------------------------------------------------------------
# 内置 fn-based 全梯度方法（与 deriv.py 中函数一一对应）
# ---------------------------------------------------------------------------

_STANDARD_METHODS = (
    DiffMethod("psr", psr, exact=True),
    DiffMethod("fd", fd),
    DiffMethod(
        "auto", auto, exact=True, requires_torch=True,
        supports_shots=False, supports_noise=False,
    ),
    DiffMethod("spsa", spsa, stochastic=True),
    DiffMethod("spsr", spsr, stochastic=True),
)

for _spec in _STANDARD_METHODS:
    register_diff_method(_spec)
del _spec

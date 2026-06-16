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


def register_diff(spec: DiffMethod, *, overwrite: bool = False) -> DiffMethod:
    """注册一个梯度方法；``overwrite=False`` 时名称/别名冲突会报错。"""

    if not isinstance(spec, DiffMethod):
        raise TypeError("register_diff expects a DiffMethod")

    names = (spec.name, *spec.aliases)
    if not overwrite:
        for name in names:
            existing = _LOOKUP.get(name)
            if existing is not None and existing.name != spec.name:
                raise ValueError(f"diff method name or alias already registered: {name!r}")
        if spec.name in _REGISTRY:
            raise ValueError(f"diff method already registered: {spec.name!r}")
    else:
        unregister_diff(spec.name)

    _REGISTRY[spec.name] = spec
    for name in names:
        _LOOKUP[name] = spec
    return spec


def unregister_diff(name: str) -> None:
    """移除一个已注册方法（含其全部别名）；未注册时静默返回。"""

    spec = _REGISTRY.pop(str(name), None)
    if spec is None:
        return
    for key in (spec.name, *spec.aliases):
        if _LOOKUP.get(key) is spec:
            del _LOOKUP[key]


def get_diff(name: str) -> DiffMethod | None:
    """按规范名或别名查询；未注册返回 ``None``。"""

    return _LOOKUP.get(str(name))


def registered_diffs() -> tuple[str, ...]:
    """返回全部已注册方法的规范名。"""

    return tuple(_REGISTRY)


def canonical_diff(name: str) -> str:
    """把方法名（含别名）解析为规范名；未注册的名称原样返回。"""

    spec = _LOOKUP.get(str(name))
    return spec.name if spec is not None else str(name)


def resolve_diff(name: str) -> Callable[..., Any]:
    """返回方法对应的可调用 ``fn``；未注册名抛 ``ValueError`` 并列出已注册方法。"""

    spec = _LOOKUP.get(str(name))
    if spec is None:
        available = ", ".join(sorted(registered_diffs()))
        raise ValueError(f"unknown diff method {name!r}; registered methods: {available}")
    return spec.fn


def _is_torch_family_backend(backend: Any) -> bool:
    """是否为支持自动微分的 Torch 系后端（GPU/NPU）。仅按类名/设备判定，不导入 torch。"""

    if backend is None:
        return False
    if type(backend).__name__ in {"GPUBackend", "TorchBackend", "NPUBackend"}:
        return True
    device = getattr(backend, "_device", None)
    return getattr(device, "type", None) in {"cuda", "npu"}


_SELECT_PREFERENCE = ("auto", "psr", "fd")


def select_diff(*, backend: Any = None, shots: Any = None, noisy: bool = False) -> str:
    """按 NEXT.md §6 策略选择梯度方法名（纯函数）。

    过滤：``requires_torch`` 仅在 Torch 系后端保留；有 shots 丢弃
    ``supports_shots=False``；``noisy`` 丢弃 ``supports_noise=False``。
    偏好顺序：``auto -> psr -> fd``（``spsa``/``spsr`` 不参与自动优选）。
    """

    has_shots = shots is not None and int(shots) > 0
    torch_family = _is_torch_family_backend(backend)

    def compatible(spec: DiffMethod) -> bool:
        if spec.requires_torch and not torch_family:
            return False
        if has_shots and not spec.supports_shots:
            return False
        if noisy and not spec.supports_noise:
            return False
        return True

    for name in _SELECT_PREFERENCE:
        spec = _REGISTRY.get(name)
        if spec is not None and compatible(spec):
            return name
    return "fd"


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
    register_diff(_spec)
del _spec

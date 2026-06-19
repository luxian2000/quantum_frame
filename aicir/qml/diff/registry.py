"""DiffMethod 注册表：按规范名或别名查询微分方法、解析 fn。

NEXT.md §6。内置方法在导入时注册一次，按 ``category`` 分三类：

- ``fn_gradient``：``psr``/``fd``/``auto``/``spsa``/``spsr``；
- ``circuit_gradient``：``ad``（伴随微分）；
- ``preconditioner``：``qng``/``bdqng``/``kqng``/``dqng``。

``mpsr`` 因返回标量混合偏导、不满足任何统一契约，刻意排除（仍可用 ``qml.mpsr``）。
``resolve_diff``/``select_diff`` 只对 ``fn_gradient`` 生效，保证经典优化器分发安全；
``ad``/``qng`` 族仅供 ``get_diff``/``registered_diffs(category=...)`` 检索发现。
"""

from __future__ import annotations

from typing import Any, Callable

from ..deriv import ad, auto, bdqng, dqng, fd, kqng, psr, qng, spsa, spsr
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


def registered_diffs(category: str | None = None) -> tuple[str, ...]:
    """返回已注册方法的规范名；``category`` 非空时按类别过滤。"""

    if category is None:
        return tuple(_REGISTRY)
    return tuple(name for name, spec in _REGISTRY.items() if spec.category == category)


def canonical_diff(name: str) -> str:
    """把方法名（含别名）解析为规范名；未注册的名称原样返回。"""

    spec = _LOOKUP.get(str(name))
    return spec.name if spec is not None else str(name)


def resolve_diff(name: str) -> Callable[..., Any]:
    """返回 ``fn_gradient`` 方法对应的可调用 ``fn``。

    仅解析 ``(fn, params) -> 梯度向量`` 契约的方法，供经典优化器统一分发；
    未注册名、或 ``circuit_gradient``/``preconditioner`` 类别（如 ``ad``/``qng``）
    均抛 ``ValueError``。
    """

    spec = _LOOKUP.get(str(name))
    if spec is None or spec.category != "fn_gradient":
        available = ", ".join(sorted(registered_diffs(category="fn_gradient")))
        raise ValueError(f"unknown fn-gradient method {name!r}; registered methods: {available}")
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
# 内置微分方法（与 deriv.py 中函数一一对应）。capability 字段只为 fn_gradient
# 的 select_diff 服务；circuit_gradient/preconditioner 均从态向量求值，故标注
# supports_shots/noise=False。
# ---------------------------------------------------------------------------

_STANDARD_METHODS = (
    # fn_gradient: (fn, params) -> 梯度向量
    DiffMethod("psr", psr, exact=True),
    DiffMethod("fd", fd),
    DiffMethod(
        "auto", auto, exact=True, requires_torch=True,
        supports_shots=False, supports_noise=False,
    ),
    DiffMethod("spsa", spsa, stochastic=True),
    DiffMethod("spsr", spsr, stochastic=True),
    # circuit_gradient: (circuit, observable) -> 梯度（伴随微分，精确，需态向量）
    DiffMethod(
        "ad", ad, category="circuit_gradient", exact=True,
        supports_shots=False, supports_noise=False,
    ),
    # preconditioner: (fn, state_fn, params) -> 方向/度规（量子自然梯度族，需态向量）
    DiffMethod("qng", qng, category="preconditioner", supports_shots=False, supports_noise=False),
    DiffMethod("bdqng", bdqng, category="preconditioner", supports_shots=False, supports_noise=False),
    DiffMethod("kqng", kqng, category="preconditioner", supports_shots=False, supports_noise=False),
    DiffMethod("dqng", dqng, category="preconditioner", supports_shots=False, supports_noise=False),
)

for _spec in _STANDARD_METHODS:
    register_diff(_spec)
del _spec

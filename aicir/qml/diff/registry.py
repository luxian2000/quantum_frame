"""DiffMethod 注册表：按规范名或别名查询微分方法、解析 fn。

NEXT.md §6。内置方法在导入时注册一次，按 ``category`` 分三类：

- ``fn_gradient``：``psr``/``psr4``/``fd``/``auto``/``spsa``/``spsr``；
- ``circuit_gradient``：``ad``（伴随微分）；
- ``preconditioner``：``qng``/``bdqng``/``kqng``/``dqng``。

``mpsr`` 因返回标量混合偏导、不满足任何统一契约，刻意排除（仍可用 ``qml.mpsr``）。
``resolve_diff``/``select_diff`` 只对 ``fn_gradient`` 生效，保证经典优化器分发安全；
``ad``/``qng`` 族仅供 ``get_diff``/``registered_diffs(category=...)`` 检索发现。
``psr4``（激发门四项移位规则）已注册、可通过 ``get_diff``/``resolve_diff`` 发现
与解析，但刻意不加入 ``select_diff`` 的自动优选偏好（仍是两项 ``psr`` 优先，
``psr4`` 需要调用方按线路生成元谱显式选择）。
"""

from __future__ import annotations

from typing import Any, Callable

from ..deriv import ad, auto, bdqng, dqng, fd, kqng, psr, psr4, qng, spsa, spsr
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


def circuit_shift_rule(circuit: Any) -> str | None:
    """按线路参数门的 GateSpec ``shift_rule`` 归类参数移位规则。

    返回 ``"two_term"``（全为两项旋转门，用 psr）、``"four_term"``（全为激发门
    等 {-1,0,1} 谱，用 psr4）、``"mixed"``（两者并存，无统一解析规则）或
    ``None``（无解析参数门）。只统计已注册且带 generator 或 shift_rule 的参数门；
    其余（如 u3/自定义 unitary）不影响 psr↔psr4 判定，跳过。
    """
    from ...gates import canonical_gate_name, gate_generator, gate_shift_rule
    from ...ir import circuit_instructions, instruction_name, instruction_parameter

    rules: set[str] = set()
    for gate in circuit_instructions(circuit):
        if instruction_parameter(gate) is None:
            continue
        name = canonical_gate_name(instruction_name(gate))
        rule = gate_shift_rule(name)
        if rule is not None:
            rules.add(rule)
        elif gate_generator(name) is not None:
            rules.add("two_term")  # 单生成元旋转门默认两项规则
    if not rules:
        return None
    if rules == {"two_term"}:
        return "two_term"
    if rules == {"four_term"}:
        return "four_term"
    return "mixed"


def select_diff(*, backend: Any = None, shots: Any = None, noisy: bool = False,
                circuit: Any = None) -> str:
    """按 NEXT.md §6 策略选择梯度方法名（纯函数）。

    过滤：``requires_torch`` 仅在 Torch 系后端保留；有 shots 丢弃
    ``supports_shots=False``；``noisy`` 丢弃 ``supports_noise=False``。
    偏好顺序：``auto -> psr -> fd``（``spsa``/``spsr`` 不参与自动优选）。

    生成元感知（传 ``circuit`` 时）：仅当优选落到两项规则 ``psr`` 时按线路门谱
    校正——全激发门 → 升级 ``psr4``（若兼容），两/四项混合 → 降到 ``fd``（psr/psr4
    均非逐参数正确）。``auto``（伴随 AD，对任意门精确）/``fd`` 不受影响。
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

    chosen = "fd"
    for name in _SELECT_PREFERENCE:
        spec = _REGISTRY.get(name)
        if spec is not None and compatible(spec):
            chosen = name
            break

    if circuit is not None and chosen == "psr":
        rule = circuit_shift_rule(circuit)
        if rule == "four_term":
            psr4_spec = _REGISTRY.get("psr4")
            if psr4_spec is not None and compatible(psr4_spec):
                return "psr4"
        elif rule == "mixed":
            return "fd"
    return chosen


# ---------------------------------------------------------------------------
# 内置微分方法（与 deriv.py 中函数一一对应）。capability 字段只为 fn_gradient
# 的 select_diff 服务；circuit_gradient/preconditioner 均从态向量求值，故标注
# supports_shots/noise=False。
# ---------------------------------------------------------------------------

_STANDARD_METHODS = (
    # fn_gradient: (fn, params) -> 梯度向量
    DiffMethod("psr", psr, exact=True),
    DiffMethod("psr4", psr4, exact=True),
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

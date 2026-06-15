# DiffMethod 策略注册表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 fn-based 梯度方法收敛为 `aicir.qml.diff` 策略注册表（查表 + 选择器），并接入 `optimizer/params.py`，使其能触达全部内置梯度方法。

**Architecture:** 新增子包 `aicir/qml/diff/`，镜像 `aicir/gates/` 的 frozen dataclass（`DiffMethod`）+ 模块级注册表习惯。注册表只引用 `deriv.py` 中已有函数（`deriv.py` 不改），保持 `psr` 单点出处。`select_diff` 把 NEXT.md §6 偏好表实现为纯函数并单测，本片不接入调用方（留给后续 QNode）。`params.py` 的 `_gradient_from_method` 改为通过 `resolve_diff` 分发。

**Tech Stack:** Python，`dataclasses`，pytest，numpy。`torch` 为可选依赖（select 仅按后端类名判定，不导入 torch）。

参考规范：`docs/superpowers/specs/2026-06-15-diffmethod-registry-design.md`

---

## 文件结构

- 新建 `aicir/qml/diff/__init__.py` —— 重导出 `DiffMethod` 与注册表 API。
- 新建 `aicir/qml/diff/spec.py` —— `DiffMethod` frozen dataclass。
- 新建 `aicir/qml/diff/registry.py` —— 注册表 dict + register/unregister/get/list/canonical/resolve/select + 内置注册。
- 修改 `aicir/qml/__init__.py` —— 重导出 diff API。
- 修改 `aicir/optimizer/params.py` —— `_gradient_from_method` 改用 `resolve_diff`；调整 import。
- 新建 `tests/qml/test_diff_registry.py` —— spec/注册表/resolve/select 测试。
- 修改/新建 `tests/optimizer/`（或并入既有测试）—— params.py 集成测试。

运行测试统一用：`PYTHONPATH=. pytest`（仓库根目录）。

---

## Task 1: `DiffMethod` spec

**Files:**
- Create: `aicir/qml/diff/spec.py`
- Create: `aicir/qml/diff/__init__.py`
- Test: `tests/qml/test_diff_registry.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/qml/test_diff_registry.py`：

```python
"""DiffMethod 策略注册表测试（NEXT.md §6 第一片）。"""

import pytest

from aicir.qml.diff import DiffMethod


def test_diff_method_normalizes_name_and_aliases():
    m = DiffMethod(name="  psr ", fn=lambda fn, p: p, aliases=["multipsr"])
    assert m.name == "psr"
    assert m.aliases == ("multipsr",)


def test_diff_method_empty_name_raises():
    with pytest.raises(ValueError):
        DiffMethod(name="  ", fn=lambda fn, p: p)


def test_diff_method_non_callable_fn_raises():
    with pytest.raises(TypeError):
        DiffMethod(name="psr", fn=123)


def test_diff_method_is_frozen():
    m = DiffMethod(name="psr", fn=lambda fn, p: p)
    with pytest.raises(Exception):
        m.name = "fd"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'aicir.qml.diff'`）

- [ ] **Step 3: Write minimal implementation**

创建 `aicir/qml/diff/spec.py`：

```python
"""DiffMethod：fn-based 梯度方法的单点描述（NEXT.md §6 第一片）。

每个梯度方法在此注册一次，统一契约为 ``(fn, params, **kw) -> 梯度向量``。
镜像 ``aicir.gates.GateSpec`` 的 frozen dataclass + 模块级注册表习惯。
仅覆盖 fn-based 方法；电路型 ``ad`` 与预条件 ``qng`` 不在其中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DiffMethod:
    """单个梯度方法的元信息。

    - ``name``：规范方法名，如 ``"psr"``。
    - ``fn``：对应可调用，契约 ``(fn, params, **kw) -> 梯度向量``。
    - ``aliases``：等价写法，如 ``("multipsr",)``。
    - ``exact``：是否精确（psr/auto 精确；fd/spsa/spsr 近似）。
    - ``stochastic``：是否随机（spsa、spsr）。
    - ``requires_torch``：是否需要 Torch 系后端（auto）。
    - ``supports_shots``：是否支持有限 shots（auto=False）。
    - ``supports_noise``：是否支持噪声线路（auto=False）。
    """

    name: str
    fn: Callable[..., Any]
    aliases: tuple[str, ...] = ()
    exact: bool = False
    stochastic: bool = False
    requires_torch: bool = False
    supports_shots: bool = True
    supports_noise: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("DiffMethod name cannot be empty")
        if not callable(self.fn):
            raise TypeError("DiffMethod fn must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", tuple(str(a) for a in self.aliases))
```

创建 `aicir/qml/diff/__init__.py`（本任务仅导出 spec，后续任务补全）：

```python
"""DiffMethod 策略注册表（NEXT.md §6 第一片）。"""

from .spec import DiffMethod

__all__ = ["DiffMethod"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add aicir/qml/diff/spec.py aicir/qml/diff/__init__.py tests/qml/test_diff_registry.py
git commit -m "feat(qml): add DiffMethod spec (NEXT.md §6 slice 1)"
```

---

## Task 2: 注册表 CRUD + resolve + 内置方法

**Files:**
- Create: `aicir/qml/diff/registry.py`
- Modify: `aicir/qml/diff/__init__.py`
- Test: `tests/qml/test_diff_registry.py`

- [ ] **Step 1: Write the failing test**

在 `tests/qml/test_diff_registry.py` 末尾追加：

```python
from aicir.qml import deriv
from aicir.qml.diff import (
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    unregister_diff,
)


def test_builtin_methods_registered():
    assert set(registered_diffs()) == {"psr", "fd", "auto", "spsa", "spsr"}


def test_resolve_returns_bound_function():
    assert resolve_diff("psr") is deriv.psr
    assert resolve_diff("auto") is deriv.auto


def test_resolve_unknown_raises_with_listing():
    with pytest.raises(ValueError) as exc:
        resolve_diff("nope")
    assert "psr" in str(exc.value)


def test_mpsr_not_registered():
    assert "mpsr" not in registered_diffs()
    assert get_diff("mpsr") is None
    with pytest.raises(ValueError):
        resolve_diff("mpsr")


def test_register_and_unregister_roundtrip():
    spec = DiffMethod("dummy", lambda fn, p: p)
    register_diff(spec)
    try:
        assert "dummy" in registered_diffs()
        assert resolve_diff("dummy") is spec.fn
    finally:
        unregister_diff("dummy")
    assert "dummy" not in registered_diffs()


def test_duplicate_register_raises():
    with pytest.raises(ValueError):
        register_diff(DiffMethod("psr", lambda fn, p: p))


def test_alias_resolution():
    spec = DiffMethod("dummy2", lambda fn, p: p, aliases=("dummy_alias",))
    register_diff(spec)
    try:
        assert canonical_diff("dummy_alias") == "dummy2"
        assert get_diff("dummy_alias") is spec
    finally:
        unregister_diff("dummy2")


def test_canonical_unknown_passthrough():
    assert canonical_diff("unknown_xyz") == "unknown_xyz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: FAIL（`ImportError: cannot import name 'register_diff' from 'aicir.qml.diff'`）

- [ ] **Step 3: Write minimal implementation**

创建 `aicir/qml/diff/registry.py`：

```python
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
```

把 `aicir/qml/diff/__init__.py` 替换为：

```python
"""DiffMethod 策略注册表（NEXT.md §6 第一片）。"""

from .registry import (
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    unregister_diff,
)
from .spec import DiffMethod

__all__ = [
    "DiffMethod",
    "canonical_diff",
    "get_diff",
    "register_diff",
    "registered_diffs",
    "resolve_diff",
    "unregister_diff",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: PASS（全部通过）

- [ ] **Step 5: Commit**

```bash
git add aicir/qml/diff/registry.py aicir/qml/diff/__init__.py tests/qml/test_diff_registry.py
git commit -m "feat(qml): add DiffMethod registry + resolve (NEXT.md §6 slice 1)"
```

---

## Task 3: `select_diff` 选择策略

**Files:**
- Modify: `aicir/qml/diff/registry.py`
- Modify: `aicir/qml/diff/__init__.py`
- Test: `tests/qml/test_diff_registry.py`

- [ ] **Step 1: Write the failing test**

在 `tests/qml/test_diff_registry.py` 末尾追加：

```python
from aicir.qml.diff import select_diff


class GPUBackend:  # noqa: N801 - 模拟 Torch 系后端类名
    pass


def test_select_prefers_auto_on_torch_noiseless_no_shots():
    assert select_diff(backend=GPUBackend()) == "auto"


def test_select_falls_back_to_psr_with_shots():
    assert select_diff(backend=GPUBackend(), shots=1024) == "psr"


def test_select_falls_back_to_psr_when_noisy():
    assert select_diff(backend=GPUBackend(), noisy=True) == "psr"


def test_select_psr_on_non_torch_backend():
    assert select_diff(backend=None) == "psr"


def test_select_never_returns_stochastic():
    for kwargs in ({}, {"shots": 1000}, {"noisy": True}, {"backend": GPUBackend()}):
        assert select_diff(**kwargs) not in {"spsa", "spsr"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: FAIL（`ImportError: cannot import name 'select_diff'`）

- [ ] **Step 3: Write minimal implementation**

在 `aicir/qml/diff/registry.py` 的 `resolve_diff` 之后、`_STANDARD_METHODS` 之前插入：

```python
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
```

在 `aicir/qml/diff/__init__.py` 的 registry import 列表中加入 `select_diff`，并加入 `__all__`。即把该文件改为：

```python
"""DiffMethod 策略注册表（NEXT.md §6 第一片）。"""

from .registry import (
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    select_diff,
    unregister_diff,
)
from .spec import DiffMethod

__all__ = [
    "DiffMethod",
    "canonical_diff",
    "get_diff",
    "register_diff",
    "registered_diffs",
    "resolve_diff",
    "select_diff",
    "unregister_diff",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: PASS（全部通过）

- [ ] **Step 5: Commit**

```bash
git add aicir/qml/diff/registry.py aicir/qml/diff/__init__.py tests/qml/test_diff_registry.py
git commit -m "feat(qml): add select_diff policy (NEXT.md §6 slice 1)"
```

---

## Task 4: `aicir.qml` 重导出

**Files:**
- Modify: `aicir/qml/__init__.py`
- Test: `tests/qml/test_diff_registry.py`

- [ ] **Step 1: Write the failing test**

在 `tests/qml/test_diff_registry.py` 末尾追加：

```python
def test_diff_api_reexported_from_qml():
    import aicir.qml as qml

    assert hasattr(qml, "DiffMethod")
    assert hasattr(qml, "resolve_diff")
    assert hasattr(qml, "select_diff")
    assert qml.resolve_diff("psr") is qml.psr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py::test_diff_api_reexported_from_qml -q`
Expected: FAIL（`AttributeError: module 'aicir.qml' has no attribute 'DiffMethod'`）

- [ ] **Step 3: Write minimal implementation**

把 `aicir/qml/__init__.py` 替换为（在 deriv 之后导入 diff，避免循环导入）：

```python
"""Quantum machine learning utilities."""

from .deriv import auto, psr, spsr, spsa, mpsr, fd, ad, qng, bdqng, kqng, dqng, rotosolve
from .diff import (
    DiffMethod,
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    select_diff,
    unregister_diff,
)

__all__ = [
    "auto",
    "psr",
    "spsr",
    "spsa",
    "mpsr",
    "fd",
    "ad",
    "qng",
    "bdqng",
    "kqng",
    "dqng",
    "rotosolve",
    "DiffMethod",
    "canonical_diff",
    "get_diff",
    "register_diff",
    "registered_diffs",
    "resolve_diff",
    "select_diff",
    "unregister_diff",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/qml/test_diff_registry.py -q`
Expected: PASS（全部通过）

- [ ] **Step 5: Commit**

```bash
git add aicir/qml/__init__.py tests/qml/test_diff_registry.py
git commit -m "feat(qml): re-export DiffMethod registry from aicir.qml"
```

---

## Task 5: 接入 `optimizer/params.py`

**Files:**
- Modify: `aicir/optimizer/params.py:11`（import）
- Modify: `aicir/optimizer/params.py:112-120`（`_gradient_from_method` 主体）
- Test: `tests/optimizer/test_params_diff_registry.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/optimizer/test_params_diff_registry.py`：

```python
"""params.py 通过 DiffMethod 注册表分发梯度方法。"""

import numpy as np
import pytest

from aicir.optimizer.params import Adam


def _cos_sum(params):
    # 参数移位规则对 cos 目标精确（梯度 = -sin），最小值在每个分量 = π。
    # 必须用 PSR 兼容目标，普通二次式会让 psr/spsr 算出错误"梯度"。
    return float(np.sum(np.cos(np.asarray(params, dtype=float))))


def test_adam_can_use_spsr_via_registry():
    # spsr 此前无法从 params.py 触达；现在应可运行并下降。
    init = np.array([1.0, 2.0])
    rng = np.random.default_rng(0)
    opt = Adam(
        gradient_method="spsr",
        gradient_kwargs={"rng": rng},
        learning_rate=0.15,
        max_iters=100,
    )
    result = opt.minimize(_cos_sum, init)
    assert result.best_fun < _cos_sum(init)


def test_adam_unknown_method_lists_registered():
    opt = Adam(gradient_method="bogus", max_iters=5)
    with pytest.raises(ValueError) as exc:
        opt.minimize(_cos_sum, np.array([1.0]))
    assert "psr" in str(exc.value)
```

> `Adam` 的实际 API（已核对 `aicir/optimizer/params.py:209-246`）：构造参数全为关键字（`gradient_method`/`gradient_kwargs`/`learning_rate`/`max_iters`…），梯度计算入口是 `minimize(fn, init_params)`，返回 `OptimizationResult`（含 `.best_fun`/`.fun`/`.x`）。无 `step()` 方法。

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/optimizer/test_params_diff_registry.py -q`
Expected: FAIL（`test_adam_can_use_spsr_via_registry` 因旧 `_gradient_from_method` 不识别 `"spsr"` 抛 `ValueError: gradient_method must be 'psr', 'fd', 'spsa', or a callable`）

- [ ] **Step 3: Write minimal implementation**

改 `aicir/optimizer/params.py` 第 11 行 import（`spsa` 仍被 SPSA 优化器在别处使用，保留；移除不再使用的 `psr`/`fd`）：

把
```python
from ..qml.deriv import fd, psr, spsa
```
改为
```python
from ..qml.deriv import spsa
from ..qml.diff import resolve_diff
```

把 `_gradient_from_method` 中的 if/elif（约 112-120 行）：
```python
    kwargs = dict(gradient_kwargs or {})
    method = str(gradient_method).strip().lower()
    if method == "psr":
        return psr(fn, params, **kwargs)
    if method == "fd":
        return fd(fn, params, **kwargs)
    if method == "spsa":
        return spsa(fn, params, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', 'spsa', or a callable")
```
替换为：
```python
    kwargs = dict(gradient_kwargs or {})
    method = str(gradient_method).strip().lower()
    grad_fn = resolve_diff(method)  # 未知名抛 ValueError（信息含已注册方法名）
    grad = grad_fn(fn, params, **kwargs)   # fn 为目标闭包，与 deriv.psr 签名一致
    return np.asarray(grad, dtype=float).reshape(params.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/optimizer/test_params_diff_registry.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: Run full suite to confirm no regression**

Run: `PYTHONPATH=. pytest -q`
Expected: 全绿（既有 VQE/SSVQE/VQD/QAS 的 `psr` 路径不变）

- [ ] **Step 6: Commit**

```bash
git add aicir/optimizer/params.py tests/optimizer/test_params_diff_registry.py
git commit -m "feat(optimizer): dispatch gradients via DiffMethod registry (NEXT.md §6 slice 1)"
```

---

## Task 6: 文档与 NEXT.md 当前状态

**Files:**
- Modify: `NEXT.md`（§6 追加「当前状态」段）
- Modify: `CHANGELOG.md`（按既有格式加 dated 条目）
- Create: `aicir/qml/diff/README.md`（可选，若 `aicir/gates/README.md` 模式要求子包带 README）

- [ ] **Step 1: 给 NEXT.md §6 追加当前状态**

在 `NEXT.md` 第 6 节（`### 6. 把梯度方法做成策略注册表`）正文末尾追加一段中文「当前状态」，说明：`aicir.qml.diff` 已落地 `DiffMethod` 富 spec、注册表 API（`register_diff`/`unregister_diff`/`get_diff`/`registered_diffs`/`canonical_diff`/`resolve_diff`）与纯函数选择器 `select_diff`（偏好 `auto -> psr -> fd`）；内置 5 个 fn-based 全梯度方法 `psr/fd/auto/spsa/spsr`，`mpsr` 因契约不符刻意排除；`optimizer/params.py` 已改为经 `resolve_diff` 分发；`select_diff` 已实现并单测，尚未接入调用方（留待 §5 QNode）；`ad`/`qng` 等非 fn-based 方法未纳入。

- [ ] **Step 2: 给 CHANGELOG.md 加条目**

按 `CHANGELOG.md` 既有 dated 格式（参照文件顶部最近条目的写法）新增一条，日期 2026-06-15，概述 `aicir.qml.diff` DiffMethod 策略注册表落地与 `params.py` 接入。

- [ ] **Step 3: Commit**

```bash
git add NEXT.md CHANGELOG.md
git commit -m "docs: record DiffMethod registry (NEXT.md §6 slice 1)"
```

---

## 完成校验

- [ ] `PYTHONPATH=. pytest -q` 全绿。
- [ ] `from aicir.qml import DiffMethod, resolve_diff, select_diff, registered_diffs` 可用。
- [ ] `registered_diffs()` 返回 `{psr, fd, auto, spsa, spsr}`，不含 `mpsr`。
- [ ] `Adam(gradient_method="spsr")` 可运行；未知方法报错且信息含已注册方法名。
- [ ] `aicir/qml/deriv.py` 未改动；`vqc`/`qas` 的 `psr` 导入路径不变。

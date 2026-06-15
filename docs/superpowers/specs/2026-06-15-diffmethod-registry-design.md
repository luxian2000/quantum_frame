# DiffMethod 策略注册表设计（NEXT.md §6 第一片）

## 背景与目标

NEXT.md 第 6 节提出把分散的梯度方法选择逻辑收敛为 `DiffMethod` 策略注册表。
当前问题：

- 梯度方法以函数形式散落在 `aicir/qml/deriv.py`（`psr`/`spsr`/`spsa`/`fd`/`mpsr`/`auto`/`ad`/`qng`…）。
- 唯一的选择逻辑是 `aicir/optimizer/params.py` 中 `_gradient_from_method` 的 if/elif，
  且只能触达 `psr`/`fd`/`spsa`，无法触达已存在的 `auto`/`spsr` 等方法。
- `vqc`（VQE/SSVQE/VQD）与 `qas` 各自硬编码 `from ..qml.deriv import psr`。

本片目标（已与用户确认的范围）：

1. 仅覆盖 **fn-based** 梯度方法（统一契约 `(fn, params) -> 梯度向量`）。
2. 提供查表（lookup/resolve）+ 显式提示选择器（`select`）。
3. 注册项采用与 `aicir/gates` 的 `GateSpec` 对称的富 spec 对象。
4. 把 `resolve()` 接入 `params.py`；`select()` 实现并单测，但本片不接入调用方
   （留给后续 QNode，NEXT.md §5）。

**非目标（明确排除）：** 电路型方法 `ad`（签名为 `(circuit, observable)`）、
预条件方法 `qng/bdqng/kqng/dqng`、`mpsr`（见下）、向优化器透传 backend/shots 上下文、
切换 `vqc`/`qas` 的默认梯度路径。

## 设计原则与兼容

- `deriv.py` **不改动**：注册表只是引用其函数。`psr` 等保持原导入路径，
  满足 CLAUDE.md「VQE/SSVQE/VQD/QAS 通过 `qml.deriv.psr` 单点出处」的约束。
- 公开入口保持轻量：`params.py` 仍接受字符串方法名与 callable。
- 镜像既有 `GateSpec` 习惯，保持两个注册表对称（frozen dataclass + 模块级 dict 注册表）。

## 模块布局

新增子包 `aicir/qml/diff/`，与 `aicir/gates/` 对称：

```text
aicir/qml/diff/
|-- __init__.py     # 重导出 DiffMethod 与注册表 API
|-- spec.py         # DiffMethod frozen dataclass
`-- registry.py     # 模块级注册表 dict + register/get/resolve/list/canonical/select
```

由 `aicir/qml/__init__.py` 重导出（与 `gates` 一致，**不**提升到顶层 `aicir`）。

## `DiffMethod` spec

```python
@dataclass(frozen=True)
class DiffMethod:
    name: str                      # 规范名，如 "psr"
    fn: Callable                   # deriv.py 中对应函数，契约 (fn, params, **kw) -> 梯度向量
    aliases: tuple[str, ...] = ()  # 等价写法，如 ("multipsr",)
    exact: bool = False            # psr/auto 精确；fd/spsa/spsr 近似
    stochastic: bool = False       # spsa、spsr
    requires_torch: bool = False   # auto（自动微分需 Torch 系后端）
    supports_shots: bool = True    # auto=False（需精确计算图）
    supports_noise: bool = True    # auto=False
```

校验（`__post_init__`，参照 `GateSpec`）：`name` 去空白且非空；`fn` 必须可调用；
`aliases` 规范化为字符串元组。无 `selectable` 字段——注册表内每一项都是真正的
全梯度方法（见 mpsr 决策）。

### 内置注册项

import 时注册 5 个 fn-based 全梯度方法：

| name   | aliases       | exact | stochastic | requires_torch | supports_shots | supports_noise |
|--------|---------------|-------|------------|----------------|----------------|----------------|
| `psr`  | —             | True  | False      | False          | True           | True           |
| `fd`   | —             | False | False      | False          | True           | True           |
| `auto` | —             | True  | False      | True           | False          | False          |
| `spsa` | —             | False | True       | False          | True           | True           |
| `spsr` | —             | False | True       | False          | True           | True           |

`fn` 字段分别绑定 `deriv.psr`/`fd`/`auto`/`spsa`/`spsr`。

### mpsr 决策（排除）

`mpsr` **不进入注册表**。理由：`mpsr(fn, params, parameter_indices=None, ...)` 返回
单个混合偏导（标量），不满足注册表 `(fn, params) -> 梯度向量` 的统一契约。若可
`resolve`，则 `Adam(gradient_method="mpsr")` 会在 `params.py` 的
`reshape(params.shape)` 处报错或静默产生错误结果。`mpsr` 属于另一类工具（高阶混合
导数），保持为 `qml.mpsr` 普通函数，导入路径不变。后果：`resolve()` 返回的每个方法
都是安全的全梯度，且无需 `selectable` 标志。

## 注册表 API（镜像 `gates/registry.py`）

模块级 `_REGISTRY: dict[str, DiffMethod] = {}`，别名映射 `_ALIASES: dict[str, str] = {}`。

- `register_diff_method(spec: DiffMethod, *, overwrite: bool = False) -> DiffMethod`
  ——重复名未 `overwrite` 时报错；同步登记别名。
- `unregister_diff_method(name: str) -> None`
- `get_diff_method(name: str) -> DiffMethod | None` ——别名归一后查找。
- `registered_diff_methods() -> tuple[str, ...]` ——规范名元组。
- `canonical_diff_name(name: str) -> str` ——别名→规范名（未知名透传，
  参照 `canonical_gate_name`）。
- `resolve_diff_method(name: str) -> Callable` ——返回绑定的 `fn`；未知名抛
  `ValueError`，错误信息列出 `registered_diff_methods()`。

## `select()` 选择策略（§6 表格的代码化）

```python
def select_diff_method(*, backend=None, shots=None, noisy: bool = False) -> str:
    ...
```

返回**方法名**（字符串），纯函数，无副作用。逻辑：

1. **过滤**已注册方法的兼容性：
   - `requires_torch=True` 的方法仅当 `backend` 为 Torch 系后端时保留；
   - `shots` 为真（非 None 且 > 0）时丢弃 `supports_shots=False` 的方法；
   - `noisy=True` 时丢弃 `supports_noise=False` 的方法。
2. **排序**偏好：`auto`（Torch 且无 shots/无噪声）→ `psr` → `fd`。
   `spsa`/`spsr` 仅在调用方显式请求时使用，不参与自动优选。
3. 返回排序后第一个兼容方法（最终 fallback 为 `fd`）。

Torch 系后端判定复用现有工具（`deriv._is_npu_family_backend` / `_torch_or_none`
所体现的判定思路；实现时确认可复用的 helper，避免重复逻辑）。

本片**不接入调用方**：`select()` 写好并单测，等待后续 QNode（NEXT.md §5）使用。

## `params.py` 接入（本片可见收益）

将 `aicir/optimizer/params.py` 的 `_gradient_from_method`（约 107–120 行）中的
if/elif 替换为：

```python
from ..qml.diff import resolve_diff_method
...
if callable(gradient_method):
    grad = gradient_method(params)
    return np.asarray(grad, dtype=float).reshape(params.shape)

kwargs = dict(gradient_kwargs or {})
method = str(gradient_method).strip().lower()
grad_fn = resolve_diff_method(method)   # 未知名抛 ValueError（信息含已注册方法名）
grad = grad_fn(fn, params, **kwargs)     # fn 为目标函数闭包，与 deriv.psr 签名一致
return np.asarray(grad, dtype=float).reshape(params.shape)
```

实现细节：`_gradient_from_method` 的首参 `fn` 即目标函数闭包；注册的方法
（`deriv.psr` 等）签名均为 `(fn, params, **kw)`，故调用形态不变。效果：
`GD`/`Adam`/`ScipyMinimize` 现在可触达
`auto`/`spsr`（以及 `psr`/`fd`/`spsa`），未知名错误信息列出已注册方法。

`spsr` 的额外 kwargs（如 `n_samples`/`rng`）通过现有 `gradient_kwargs` 透传，
无需新增参数。

## 测试

新增 `tests/qml/test_diff_registry.py`：

- spec 校验：空名/非可调用 `fn` 报错；别名规范化。
- 注册表：内置 5 方法均存在；`register`/`unregister`/`overwrite` 行为；
  重复注册未 `overwrite` 报错。
- `canonical_diff_name`：别名→规范名；未知名透传。
- `resolve_diff_method`：已知名返回正确 `fn`；未知名 `ValueError` 且信息列出方法。
- `select_diff_method` 真值表：
  - Torch 后端 + 无 shots + 无噪声 → `auto`；
  - 有 shots → `psr`（`auto` 被过滤）；
  - `noisy=True` → `psr`；
  - 非 Torch 后端 → `psr`/`fd`；
  - 任何情形都不会自动返回 `spsa`/`spsr`。

`params.py` 集成测试（可加入既有优化器测试或新建）：

- `Adam(gradient_method="spsr")` 能在小目标上运行并下降（或至少返回正确形状梯度）。
- 未知 `gradient_method` 抛 `ValueError` 且信息含已注册方法名。

## 验收标准

- `from aicir.qml.diff import DiffMethod, register_diff_method, get_diff_method,
  resolve_diff_method, registered_diff_methods, canonical_diff_name,
  select_diff_method` 全部可用，并经 `aicir.qml` 重导出。
- 内置注册 `{psr, fd, auto, spsa, spsr}`，`mpsr` 不在其中。
- `params.py` 通过 `resolve_diff_method` 分发，可触达全部 5 个方法。
- `select_diff_method` 真值表测试通过。
- `deriv.py` 与 `vqc`/`qas` 的现有 `psr` 导入路径不变，既有测试不回归。
- 全量 `PYTHONPATH=. pytest` 通过。

## 后续（不在本片）

- QNode（§5）调用 `select_diff_method` 实现 `diff_method="auto"`。
- 向优化器/QNode 透传 backend/shots/noisy 上下文以启用自动优选默认值。
- 将 `ad`、`qng` 等纳入更广义的微分策略体系（需先解决签名差异）。
```


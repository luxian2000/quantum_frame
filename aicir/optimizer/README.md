# aicir.optimizer

本模块只负责 **经典参数优化**：VQE/VQA 循环中迭代更新参数的优化器
（`Adam`、`SPSA`、`GD`、`ScipyMinimize`/`COBYLA`/`LBFGSB`、`minimize` 等）。

> **本模块不提供线路结构优化。** `optimize_circuit` / `optimize_basic` 以及
> 全部门级化简/重排/合并规则位于 [`aicir.transpile`](../transpile/README.md)；
> 请使用 `from aicir.transpile import optimize, optimize_basic, optimize_circuit`。

线路优化的统一入口是 `aicir.transpile.optimize(circuit)`。需要自定义 pass 顺序时使用
`aicir.transpile.PassManager`。

---

## 参数优化器

`aicir.optimizer.params` 面向 VQE/VQA 的经典参数优化循环。它与 `aicir.qml.deriv` 的职责不同：`qml.deriv` 提供梯度估计器或几何预条件方向，`optimizer.params` 负责迭代更新参数、记录 history、callback、best value 和统一结果对象。

### 公共接口

| 类 / 函数 | 说明 |
| --- | --- |
| `OptimizationResult` | 统一优化结果，包含 `x`、`fun`、`best_x`、`best_fun`、`history` 等 |
| `GD` | 固定步长梯度下降，支持 `psr` / `fd` / `spsa` / 自定义梯度 |
| `Adam` | Adam 参数优化器，适合 VQE 中配合 parameter-shift、finite-difference 或外部梯度 |
| `SPSA` | SPSA 优化循环，默认用 `qml.deriv.spsa` 估计梯度，也可传入 `gradient_fn=` 替换该估计 |
| `ScipyMinimize` | `scipy.optimize.minimize` 通用包装，保持参数原始 shape |
| `COBYLA` | `ScipyMinimize(method="COBYLA")` 便捷封装 |
| `LBFGSB` | `ScipyMinimize(method="L-BFGS-B")` 便捷封装 |
| `scipy_minimize` | 函数式 SciPy minimize 包装 |
| `minimize` | 接受 optimizer 对象或 SciPy method 名称的统一入口 |

### `Optimizer` 协议

`GD`/`Adam`/`SPSA`/`ScipyMinimize`（含 `COBYLA`/`LBFGSB`）都满足 `aicir.protocols.Optimizer` 这一运行时可检查协议：暴露统一签名

```text
minimize(fn, init_params, *, gradient_fn=None, callback=None) -> OptimizationResult
```

`gradient_fn(params) -> array` 是可选的外部梯度来源；未提供时各优化器按自身配置（`gradient_method`、内置 `spsa` 估计等）计算梯度。`callback(step, value, params)` 每步调用一次，可用于日志或早停判断。协议只检查 `minimize` 方法是否存在，不检查具体签名细节，因此自定义优化器只需实现同名方法即可注入到 `BasicVQE(optimizer=...)` 等消费点。

```python
from aicir.optimizer import Adam
from aicir.protocols import Optimizer

adam = Adam(max_iters=50, learning_rate=0.05, gradient_method="psr")
isinstance(adam, Optimizer)   # True
```

对于 `SPSA`，`gradient_fn=` 提供后会**替换**内置的同时扰动随机梯度估计（`qml.deriv.spsa`），改为直接使用外部梯度：

```python
import numpy as np
from aicir.optimizer import SPSA

def energy(theta):
    return float(np.cos(theta[0]))

def grad_fn(theta):
    return np.array([-np.sin(theta[0])])

spsa_opt = SPSA(max_iters=30, learning_rate=0.05, rng=7)
result = spsa_opt.minimize(energy, np.array([0.1]), gradient_fn=grad_fn)
```

### `OptimizationResult`

| 字段 / 属性 | 说明 |
| --- | --- |
| `x` / `parameters` | 最终参数；`.parameters` 是 `.x` 的别名（VQE 结果风格） |
| `fun` / `value` | 最终目标函数值；`.value` 是 `.fun` 的别名 |
| `best_x` / `best_fun` | 优化过程中记录的最优参数/目标值 |
| `nit` / `nfev` | 迭代步数 / 目标函数调用次数 |
| `success` / `message` | 是否收敛及说明 |
| `history` | `list[HistoryRecord]`，每步的 `step`/`fun`/`grad_norm`/`learning_rate`（以及各优化器特有的 `extras`，如 SPSA 的 `perturbation`） |

`HistoryRecord` 是具名字段的 dataclass，但仍支持旧版 dict 风格访问：`rec["fun"]`、`rec.get("grad_norm")`、`"perturbation" in rec` 均可用（具名字段优先命中，其余落到 `extras`）。

```python
result = adam.minimize(energy, np.array([0.1]))
print(result.value, result.parameters)          # 等价于 result.fun, result.x
print(result.history[0]["fun"])                 # dict 风格访问
print(result.history[0].get("grad_norm"))
```

### 示例

```python
import numpy as np
from aicir.optimizer import Adam, SPSA

def energy(theta):
    return float(np.cos(theta[0]))

adam = Adam(max_iters=100, learning_rate=0.05, gradient_method="psr")
result = adam.minimize(energy, np.array([0.1]))

spsa = SPSA(max_iters=100, learning_rate=0.05, perturbation=1e-2, rng=7)
result_spsa = spsa.minimize(energy, np.array([0.1]))
```

SciPy 优化器在调用时才导入 SciPy：

```python
from aicir.optimizer import COBYLA, LBFGSB

cobyla = COBYLA(options={"maxiter": 200})
result = cobyla.minimize(energy, np.array([0.1]))

def grad(theta):
    return np.array([-np.sin(theta[0])])

lbfgsb = LBFGSB(options={"maxiter": 100})
result = lbfgsb.minimize(energy, np.array([0.1]), gradient_fn=grad)
```

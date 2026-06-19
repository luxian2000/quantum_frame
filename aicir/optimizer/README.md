# aicir.optimizer

本模块只负责 **经典参数优化**：VQE/VQA 循环中迭代更新参数的优化器
（`Adam`、`SPSA`、`GD`、`ScipyMinimize`/`COBYLA`/`LBFGSB`、`minimize` 等）。

> **线路结构优化已迁出本模块。** `optimize_circuit` / `optimize_basic` 以及
> 全部门级化简/重排/合并规则现位于 [`aicir.transpile`](../transpile/README.md)。
> 旧的 `aicir.optimizer.circuit` 模块与 `aicir.optimizer.optimize_*` 导出已移除；
> 请改用 `from aicir.transpile import optimize, optimize_basic, optimize_circuit`。

线路优化的统一入口是 `aicir.transpile.optimize(circuit)`（等价于旧的
`default_optimization_pipeline().run(circuit)`）。需要自定义 pass 顺序时使用
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
| `SPSA` | 使用 `qml.deriv.spsa` 梯度估计的 SPSA 优化循环 |
| `ScipyMinimize` | `scipy.optimize.minimize` 通用包装，保持参数原始 shape |
| `COBYLA` | `ScipyMinimize(method="COBYLA")` 便捷封装 |
| `LBFGSB` | `ScipyMinimize(method="L-BFGS-B")` 便捷封装 |
| `scipy_minimize` | 函数式 SciPy minimize 包装 |
| `minimize` | 接受 optimizer 对象或 SciPy method 名称的统一入口 |

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

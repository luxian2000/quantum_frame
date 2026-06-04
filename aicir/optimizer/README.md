# aicir.optimizer

本模块包含两类优化器：

- `aicir.optimizer.circuit`：线路级本地化简与重写规则。
- `aicir.optimizer.params`：VQE/VQA 常用经典参数优化器。

`aicir.optimizer.basic` 已改名并移除；新代码应使用 `aicir.optimizer.circuit` 或包级导出的 `optimize_basic`。

## 概述

`aicir.optimizer.circuit.optimize_basic` 实现了一组简单的局部门级化简与合并规则，用于消除明显的冗余门并合并可合并的相邻旋转门，目的是减小导出后电路的体积并改善可读性。

这些优化可接受多种输入形式并保持输出类型：
- `Circuit` / `list[gate_dict]` / dict(`gates`, `n_qubits`)（记作 dict 路径）
- OpenQASM 文本（记作 qasm 路径）
- DAG 三元组/字典（记作 dag 路径）

所有路径均采用“固定点（fixed-point）优化”策略：迭代应用单轮规则直到电路/文本不再变化或达到最大轮数（当前实现的上限为 64 轮）。

## 新增旋转合并规则（rx/ry/rz）

为减少单量子比特旋转门的冗余，优化器新增以下规则：

- 若序列中存在相邻的同类单量子比特旋转门（`rx/ry/rz`），且它们作用在同一目标比特上且没有控制比特，则将它们合并为一个旋转门，合并后的角度为各旋转角度的代数和：

  - `rx(a); rx(b) -> rx(a+b)`
  - `ry(a); ry(b) -> ry(a+b)`
  - `rz(a); rz(b) -> rz(a+b)`

- 若合并后角度数值上接近 `0`（实现中使用 `numpy.isclose(..., atol=1e-15)`），则认为该门等效于恒等门，并直接删除。

- 仅合并**相邻**并且**连续无其他非可合并门**阻隔的旋转门；若两门间存在其他门（包括注释/空行在 qasm 路径中也视为屏障），则不会合并。

## qasm 路径的特殊说明

- QASM 文本解析支持解析 `rx(...)`、`ry(...)`、`rz(...)` 形式，角度表达式允许使用 `pi` 常量（例如 `pi/2`）。
- qasm 路径上的旋转合并与 dict 路径等价：优化器会解析角度、相加、并以尽量紧凑的格式重新输出（当角度恰好为 ±pi 会输出 `pi` / `-pi`，否则采用十进制表达）。

## 固定点收敛

- 为确保间接产生的可合并模式（例如多轮合并或合并后产生新的相邻可合并对）被充分处理，优化器对 dict/qasm/dag 路径均采用固定点迭代：
  - 持续应用单轮合并/消去规则；
  - 若本轮结果与上轮相同，则停止并返回结果；
  - 若循环次数达到实现中的阈值（默认 64 轮）仍未收敛，则返回当前结果以防死循环。

## 示例

- dict/Circuit 示例：

```py
gates = [
    {"type": "rx", "target_qubit": 0, "parameter": 0.1},
    {"type": "rx", "target_qubit": 0, "parameter": 0.2},
]
# 优化后 -> rx(0.3) on q0
```

- QASM 示例：

```
rx(0.1) q[0];
rx(0.2) q[0];

# 优化后 ->
rx(0.3) q[0];
```

## 与导出/演示的交互

- 由于 `aicir/core/io/qasm` 导出器在处理 `control_states=0` 时会在导出层面插入前后 `x` 门，导致导出后文本中可能产生新的相邻合并或消去机会（例如相邻 `x; x;` 可以被消去、`rz` 合并等）。
- 因此推荐在导出前对 `Circuit` 做一轮优化（电路级），并在导出后对生成的 QASM 文本再做一轮 qasm 优化（文本级）。当前 `aicir/encoder/demos/encode_1234_demo.py` 已采用此流程。

## 测试

- 新增/修改的测试位于 `tests/circuit/test_optimizer_basic.py`，覆盖了：
  - dict 路径的旋转合并；
  - qasm 路径的旋转合并（含 `pi` 表达式解析）；
  - 固定点迭代收敛性。

-## 实现细节指引

- 主要实现文件：`aicir/optimizer/circuit.py`。
- 合并逻辑仅处理“无控制”的单量子比特旋转；对受控旋转、三角函数等更复杂的恒等/合并关系不作尝试，以保持实现简单且高效。

如需将来扩展到更复杂的合并（例如等价变换、整数倍 pi 归约、模 2π 归约等），请在该 README 中记录新增规则并对应添加覆盖测试。

---

## 参数优化器

`aicir.optimizer.params` 面向 VQE/VQA 的经典参数优化循环。它与 `aicir.qml.deriv` 的职责不同：`qml.deriv` 提供梯度估计器或几何预条件方向，`optimizer.params` 负责迭代更新参数、记录 history、callback、best value 和统一结果对象。

### 公共接口

| 类 / 函数 | 说明 |
| --- | --- |
| `OptimizationResult` | 统一优化结果，包含 `x`、`fun`、`best_x`、`best_fun`、`history` 等 |
| `GradientDescentOptimizer` | 固定步长梯度下降，支持 `psr` / `fd` / `spsa` / 自定义梯度 |
| `AdamOptimizer` | Adam 参数优化器，适合 VQE 中配合 parameter-shift、finite-difference 或外部梯度 |
| `SPSAOptimizer` | 使用 `qml.deriv.spsa` 梯度估计的 SPSA 优化循环 |
| `ScipyOptimizer` | `scipy.optimize.minimize` 通用包装，保持参数原始 shape |
| `COBYLAOptimizer` | `ScipyOptimizer(method="COBYLA")` 便捷封装 |
| `LBFGSBOptimizer` | `ScipyOptimizer(method="L-BFGS-B")` 便捷封装 |
| `scipy_minimize` | 函数式 SciPy minimize 包装 |
| `minimize` | 接受 optimizer 对象或 SciPy method 名称的统一入口 |

### 示例

```python
import numpy as np
from aicir.optimizer import AdamOptimizer, SPSAOptimizer

def energy(theta):
    return float(np.cos(theta[0]))

adam = AdamOptimizer(max_iters=100, learning_rate=0.05, gradient_method="psr")
result = adam.minimize(energy, np.array([0.1]))

spsa = SPSAOptimizer(max_iters=100, learning_rate=0.05, perturbation=1e-2, rng=7)
result_spsa = spsa.minimize(energy, np.array([0.1]))
```

SciPy 优化器在调用时才导入 SciPy：

```python
from aicir.optimizer import COBYLAOptimizer, LBFGSBOptimizer

cobyla = COBYLAOptimizer(options={"maxiter": 200})
result = cobyla.minimize(energy, np.array([0.1]))

def grad(theta):
    return np.array([-np.sin(theta[0])])

lbfgsb = LBFGSBOptimizer(options={"maxiter": 100})
result = lbfgsb.minimize(energy, np.array([0.1]), gradient_fn=grad)
```

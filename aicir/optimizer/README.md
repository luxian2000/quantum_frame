# aicir.optimizer

本模块包含两类优化器：

- `aicir.optimizer.circuit`：线路级本地化简与重写规则。
- `aicir.optimizer.params`：VQE/VQA 常用经典参数优化器。

`aicir.optimizer.basic` 已改名并移除；新代码应使用 `aicir.optimizer.circuit` 或包级导出的 `optimize_basic` / `optimize_circuit`。

## 概述

`aicir.optimizer.circuit.optimize_basic` 实现了一组简单的局部门级化简与合并规则，用于消除明显的冗余门并合并可合并的旋转门，目的是减小导出后电路的体积并改善可读性。`optimize_circuit(circuit)` 是面向 `Circuit` 对象的专用入口，返回优化后的新 `Circuit`，保留 `n_qubits` 和已绑定 backend。

线路级优化正在迁移到 `aicir.transpile`。当前 `optimize_circuit(circuit)` 内部委托给 `aicir.transpile.default_optimization_pipeline()`；旧接口保持可用，新代码如果需要自定义 pass 顺序，建议直接使用 `aicir.transpile.PassManager`。

这些优化可接受多种输入形式并保持输出类型：
- `Circuit` / `list[gate_dict]` / dict(`gates`, `n_qubits`)（记作 dict 路径）
- OpenQASM 文本（记作 qasm 路径）
- DAG 三元组/字典（记作 dag 路径）

所有路径均采用“固定点（fixed-point）优化”策略：迭代应用单轮规则直到电路/文本不再变化或达到最大轮数（当前实现的上限为 64 轮）。

## 使用方法

### 优化 `Circuit` 对象

推荐对 `Circuit` 对象使用 `optimize_circuit`。它返回一个新的 `Circuit`，不会修改原线路，并会保留原线路的 `n_qubits` 和 backend。

```python
from aicir.core.circuit import Circuit, cx, hadamard, pauli_x, rx
from aicir.optimizer import optimize_circuit

circuit = Circuit(
    hadamard(0),
    hadamard(0),
    rx(0.1, target_qubit=1),
    pauli_x(0),
    rx(0.2, target_qubit=1),
    cx(target_qubit=1, control_qubits=[0]),
    cx(target_qubit=1, control_qubits=[0]),
    n_qubits=2,
)

optimized = optimize_circuit(circuit)

print(len(circuit.gates))     # 原线路不变
print(optimized.gates)        # 冗余 H、CNOT 被消去，rx 被合并
```

如需限制固定点轮数或安全回看距离，可显式传入参数：

```python
optimized = optimize_circuit(
    circuit,
    max_rounds=32,
    max_reorder_hops=4,
)
```

### 使用通用入口

`optimize_basic` 是通用入口，会根据输入类型选择 dict/Circuit、qasm 或 dag 路径，并保持输出类型不变。

```python
from aicir.optimizer import optimize_basic

optimized_circuit = optimize_basic(circuit)

optimized_gates = optimize_basic([
    {"type": "pauli_x", "target_qubit": 0},
    {"type": "pauli_x", "target_qubit": 0},
])

optimized_payload = optimize_basic({
    "n_qubits": 1,
    "gates": [
        {"type": "rz", "target_qubit": 0, "parameter": 0.1},
        {"type": "rz", "target_qubit": 0, "parameter": 0.2},
    ],
})
```

### 优化 QASM 文本

对导出的 OpenQASM 文本可继续使用 `optimize_basic`，优化器会删除可抵消门并合并可解析的 `rx/ry/rz` 角度。

```python
from aicir.optimizer import optimize_basic

qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(0.1) q[0];
rx(0.2) q[0];
x q[0];
x q[0];
"""

optimized_qasm = optimize_basic(qasm, input_type="qasm")
print(optimized_qasm)
```

### 典型导出流程

若目标是导出更简洁的 QASM，推荐先优化 `Circuit`，导出后再优化 QASM 文本。这样可以同时捕获线路级冗余和导出器生成文本后暴露出来的冗余。

```python
from aicir.core.io.qasm import circuit_to_qasm3
from aicir.optimizer import optimize_basic, optimize_circuit

optimized_circuit = optimize_circuit(circuit)
qasm3 = circuit_to_qasm3(optimized_circuit)
qasm3 = optimize_basic(qasm3, input_type="qasm")
```

## 新增旋转合并规则（rx/ry/rz）

为减少单量子比特旋转门的冗余，优化器新增以下规则：

- 若序列中存在可安全合并的同类单量子比特旋转门（`rx/ry/rz`），且它们作用在同一目标比特上且没有控制比特，则将它们合并为一个旋转门，合并后的角度为各旋转角度的代数和：

  - `rx(a); rx(b) -> rx(a+b)`
  - `ry(a); ry(b) -> ry(a+b)`
  - `rz(a); rz(b) -> rz(a+b)`

- 若合并后角度数值上接近 `0`（实现中使用 `numpy.isclose(..., atol=1e-15)`），则认为该门等效于恒等门，并直接删除。

- 相邻旋转门会直接合并；在 dict/Circuit 和 qasm 路径中，若中间只隔着可安全交换的门，也会通过有限回看完成合并或消去。

## 安全有限重排

为覆盖常见的浅层冗余，优化器会在有限范围内向前回看并消去/合并当前单比特门：

- 作用在不同量子比特上的单比特门可安全交换。
- `x/rx` 作用在 CNOT 目标比特时，可跨过该 CNOT。
- `z/s/sdg/rz` 作用在 CNOT 控制比特时，可跨过该 CNOT。
- 未知门、多比特未知门、同一比特上的不可合并门会作为屏障，优化不会跨过它们。

该规则是保守的局部优化，不尝试任意门等价变换或全局线路重综合。

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
  - dict 路径的安全有限重排；
  - qasm 路径的旋转合并（含 `pi` 表达式解析）；
  - 固定点迭代收敛性。

## 实现细节指引

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

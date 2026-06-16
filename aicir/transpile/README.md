# aicir.transpile

`aicir.transpile` 提供线路编译与结构优化的 pass pipeline。它是从 `aicir.optimizer.circuit` 迁出的新架构层，用于承载后续验证、规范化、门分解、布局、路由和调度等编译能力。

当前已提供第一批本地优化 pass：

| Pass | 作用 |
| --- | --- |
| `ValidatePass` | 通过当前 `Circuit` surface 检查线路可表示性 |
| `CanonicalizePass` | 通过 `Circuit` 构造器复制并规范化线路 |
| `CancelInversePass` | 消去相邻逆门，例如 `x/x`、`h/h`、`cx/cx`、`s/sdg` |
| `MergeRotationsPass` | 合并相邻同轴单比特旋转门 `rx/ry/rz` |
| `CommuteSingleQubitPass` | 通过有限安全回看消去或合并可交换的单比特门 |

## 使用方法

### 默认优化流水线

```python
from aicir import Circuit, hadamard, rx
from aicir.transpile import default_optimization_pipeline

circuit = Circuit(
    hadamard(0),
    hadamard(0),
    rx(0.1, 1),
    rx(0.2, 1),
    n_qubits=2,
)

optimized = default_optimization_pipeline().run(circuit)
```

### 自定义 PassManager

`PassManager` 接受 pass 对象，也接受内置 pass 的字符串名称。开启 `fixed_point=True` 后会重复运行 pass 序列，直到线路不再变化或达到 `max_rounds`。

```python
from aicir.transpile import PassManager

pm = PassManager(
    ["validate", "canonicalize", "cancel_inverse", "merge_rotations", "commute_single_qubit"],
    fixed_point=True,
    max_rounds=64,
)

optimized = pm.run(circuit)
```

### 与旧优化接口的关系

`aicir.optimizer.optimize_circuit(circuit)` 仍然可用，当前内部委托给 `default_optimization_pipeline()`。旧代码无需迁移；新代码若需要自定义编译流程，建议直接使用 `aicir.transpile.PassManager`。

# aicir.vqc — 变分量子算法编排

本模块提供 VQE、QAOA、VQD、SSVQE 等基础变分算法编排实现。ansatz（参数化线路模板：`hea`/`hea_ti`/`uccsd`）已独立为 `aicir.ansatze` 子系统（不依赖 `vqc`），详见其 README；本模块只消费 ansatz 产出的 `Circuit` 或 callable builder，不感知其内部结构。

---

## 1. 公共接口一览

| 函数 | 文件 | 返回 | 用途 |
| --- | --- | --- | --- |
| `BasicVQE` / `run_vqe` | `VQE.py` | `VQEResult` | VQE 编排：Hamiltonian、ansatz、backend、Measure、noise、optimizer |

---

## 2. VQE 编排

`BasicVQE` 支持两条路径：

- 不传 `ansatz`：保留原来的最小 dense_matrix RY/CNOT VQE。
- 传入 `Circuit` 或 `ansatz(params) -> Circuit`：走通用编排路径，接入 `Hamiltonian`、`Parameter`、`Measure`、backend、shots、density-matrix noise 和 `aicir.optimizer` 参数优化器。

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `hamiltonian` | dense_matrix 或 `aicir.operators.Hamiltonian` |
| `ansatz` | 可选 `Circuit` 模板或 callable builder |
| `backend` | `NumpyBackend` / `GPUBackend` / `NPUBackend` 等 |
| `optimizer` | 可选 `GD`、`Adam`、`SPSA`、`COBYLA`、`LBFGSB`、`ScipyMinimize` 等 |
| `shots` | 透传给 `Measure` 或 `PauliEstimator` 生成采样 counts |
| `noise_model` | 传入后自动走 density-matrix 测量路径 |
| `energy_estimator` | 默认 `"exact"`；也可传入 `PauliEstimator` 做有限 shots Pauli-term 能量估计 |
| `target` | 可选 `aicir.devices.Target`；未显式注入 `energy_estimator` 时经 `estimator_for_target` 按设备能力（statevector/shots/noisy）注入估计器，使能量求值走 primitives。显式 `energy_estimator` 优先 |
| `n_params` / `parameter_shape` | callable ansatz 无法从 `Circuit.parameters` 推断参数量时使用 |
| `initial_state` / `initial_density_matrix` | 自定义初态 |

### 示例：Circuit + Hamiltonian + optimizer

```python
import numpy as np
from aicir import Circuit, Hamiltonian, NumpyBackend, Parameter, ry
from aicir.optimizer import GD
from aicir.vqc import BasicVQE

theta = Parameter("theta")
ansatz = Circuit(ry(theta, 0), n_qubits=1)
hamiltonian = Hamiltonian([("Z", 1.0)])

solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    optimizer=GD(max_iters=80, learning_rate=0.15, gradient_method="psr"),
)
result = solver.run(init_params=np.array([0.1]))
print(result.energy, result.parameters)
```

### 示例：callable ansatz

```python
from aicir import Circuit, ry
from aicir.vqc import BasicVQE

def build(params):
    return Circuit(ry(params[0], 0), n_qubits=1)

solver = BasicVQE(hamiltonian, ansatz=build, n_params=1)
energy = solver.energy(np.array([np.pi]))
```

### 示例：shots 与 noise

```python
from aicir import BitFlipChannel, NoiseModel

noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=0.01))
solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    shots=1024,
    noise_model=noise,
)
energy = solver.energy(np.array([0.1]))
counts = solver._last_measurement.counts(-1)   # counts 现为方法，-1 取末端测量计数
```

### 示例：PauliEstimator 有限 shots 能量

```python
from aicir import NumpyBackend, PauliEstimator

solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    energy_estimator=PauliEstimator(NumpyBackend(), shots=4096),
)
energy = solver.energy(np.array([0.1]))
term_stats = solver._last_estimator_result.term_results
```

注意：`energy_estimator="exact"` 使用 full-matrix observable 精确期望；`energy_estimator=PauliEstimator(...)` 使用 Pauli 项拆分、测量基变换、shots 分配和 counts 统计估计能量。非 exact estimator 需要传入 `Circuit` 或 callable ansatz，不支持 legacy dense RY/CNOT 路径。

---

## 3. 验证命令

```bash
PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py
PYTHONPATH=. pytest tests/vqc/test_parameter_shift_uses_qml.py
```

ansatz 模板（`hea`/`hea_ti`/`uccsd`）的验证命令见 `aicir/ansatze/README.md`。

# quantum_sim 使用手册

`quantum_sim` 是一个可扩展的量子线路模拟器，支持：

- 状态矢量与密度矩阵两种执行模式
- NumPy / PyTorch 后端
- 期望值与方差计算
- 批量执行与参数扫描
- 噪声模型（Kraus 通道）
- JSON 与 OpenQASM 2.0/3.0 I/O

当前版本中，`quantum_sim` 已具备独立电路与量子门实现，不依赖根目录的 `Core.py` 与 `Circuit.py`。

## 1. 安装与环境

推荐 Python 3.10+。

最小依赖：

```bash
pip install numpy torch
```

如果你仅使用 `NumpyBackend`，理论上只需 `numpy`。

## 2. 核心模块总览

- `quantum_sim.circuit`
  - `Circuit` 电路类与门构造器（`hadamard`、`cnot`、`rx` 等）
- `quantum_sim.execution`
  - `ExecutionEngine` 统一执行入口
  - `ExecutionResult` 统一结果对象
- `quantum_sim.core.states`
  - `StateVector`、`DensityMatrix`
- `quantum_sim.core.operators`
  - `PauliOp`、`PauliString`、`Hamiltonian`
- `quantum_sim.core.noise`
  - `NoiseModel` 与常见噪声通道
- `quantum_sim.circuit.io`
  - JSON / OpenQASM 导入导出

## 3. 快速开始

### 3.1 Bell 态示例（状态矢量）

```python
from quantum_sim import Circuit, ExecutionEngine, TorchBackend, cnot, hadamard

backend = TorchBackend(device="cpu")
engine = ExecutionEngine(backend)

circ = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
)

result = engine.run(circ, shots=1024)

print(result.probabilities)
print(result.counts)
print(result.summary())
```

### 3.2 期望值与方差

```python
from quantum_sim import Circuit, ExecutionEngine, TorchBackend, cnot, hadamard
from quantum_sim.core.operators import Hamiltonian

backend = TorchBackend(device="cpu")
engine = ExecutionEngine(backend)

circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

# 可观测量: Z0 Z1
H = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0, 1]})
op = H.to_matrix(backend)

result = engine.run(circ, observables={"ZZ": op})
print("<ZZ> =", result.expectation_values["ZZ"])
print("Var(ZZ) =", result.expectation_variances["ZZ"])
```

## 4. 电路 API

### 4.1 电路对象

- `Circuit(*gates, n_qubits=...)`
- `append(gate)` / `extend(*gates)`
- `unitary()`：返回整体酉矩阵
- `matrix()`：`unitary()` 的别名

### 4.2 已支持门（与当前实现一致）

- 单比特门
  - `pauli_x` / `pauli_y` / `pauli_z`
  - `hadamard`
  - `s_gate` / `t_gate`
  - `rx` / `ry` / `rz`
  - `u2` / `u3`
- 受控门
  - `cx` / `cnot`
  - `cy` / `cz`
  - `crx` / `cry` / `crz`
- 多比特门
  - `swap`
  - `toffoli` / `ccnot`
- 其他
  - `identity`（通过 gate dict 使用）
  - `rzz`（通过 gate dict 使用）

门构造器示例：

```python
from quantum_sim import Circuit, crx, rx, toffoli

circ = Circuit(
    rx(0.3, 0),
    crx(1.2, target_qubit=1, control_qubits=[0]),
    toffoli(target_qubit=2, control_qubits=[0, 1]),
    n_qubits=3,
)
```

## 5. 执行引擎

### 5.1 状态矢量模式

- 接口：`ExecutionEngine.run(...)`
- 适合无噪声或近似纯态场景

关键参数：

- `shots`：采样次数（`None`/`0` 表示不采样）
- `initial_state`：可自定义初态
- `observables`：`{"name": operator_matrix}`

### 5.2 密度矩阵模式

- 接口：`ExecutionEngine.run_density_matrix(...)`
- 支持噪声模型

关键参数：

- `initial_density_matrix`
- `noise_model`

示例：

```python
from quantum_sim import (
    BitFlipChannel,
    Circuit,
    ExecutionEngine,
    NoiseModel,
    TorchBackend,
)

backend = TorchBackend(device="cpu")
engine = ExecutionEngine(backend)

circ = Circuit({"type": "identity", "n_qubits": 1}, n_qubits=1)
noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=1.0))

result = engine.run_density_matrix(circ, noise_model=noise)
print(result.probabilities)  # 期望接近 [0, 1]
```

### 5.3 批量执行与参数扫描

```python
import numpy as np
from quantum_sim import Circuit, ExecutionEngine, TorchBackend, ry

backend = TorchBackend(device="cpu")
engine = ExecutionEngine(backend)

def build(theta):
    return Circuit(ry(theta, 0), n_qubits=1)

results = engine.scan_parameters(
    circuit_builder=build,
    param_values=[0.0, np.pi / 2, np.pi],
    shots=None,
)

for r in results:
    print(r.metadata["scan_index"], r.metadata["scan_param"], r.probabilities)
```

## 6. 结果对象 `ExecutionResult`

`ExecutionResult` 主要字段：

- `probabilities`: 概率向量
- `counts`: 采样计数字典（例如 `|00>`）
- `expectation_values`
- `expectation_variances`
- `final_state`
- `metadata`

常用方法：

- `most_probable()`
- `variance(name)`
- `stddev(name)`
- `summary()`

## 7. 算符与哈密顿量

`quantum_sim.core.operators` 提供 Pauli 算符组合能力：

- `PauliOp('Z', qubit=0)`
- `PauliString({'Z': [0], 'X': [1]}, n_qubits=2, coefficient=0.5)`
- `Hamiltonian(n_qubits).add_term(coeff, pauli_dict)`

示例：

```python
from quantum_sim import TorchBackend
from quantum_sim.core.operators import Hamiltonian

backend = TorchBackend(device="cpu")
H = (Hamiltonian(n_qubits=2)
     .add_term(-0.5, {"Z": [0, 1]})
     .add_term(0.3, {"X": [0, 1]}))

H_mat = H.to_matrix(backend)
```

## 8. I/O：JSON 与 OpenQASM

### 8.1 JSON

```python
from quantum_sim import Circuit, hadamard, cnot
from quantum_sim.circuit.io.json_io import circuit_to_json, circuit_from_json

circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
text = circuit_to_json(circ)
new_circ = circuit_from_json(text)
```

### 8.2 OpenQASM 2.0 / 3.0

```python
from quantum_sim.circuit.io.qasm import circuit_to_qasm, circuit_from_qasm

qasm2 = circuit_to_qasm(new_circ, version="2.0")
qasm3 = circuit_to_qasm(new_circ, version="3.0")

parsed = circuit_from_qasm(qasm2)
```

已支持常见门：`x/y/z/h/s/t/rx/ry/rz/u2/u3/u/cx/cy/cz/swap/crx/cry/crz/ccx`。

## 9. 后端选择建议

- `TorchBackend`
  - 适合需要 GPU 或自动微分的场景
- `NumpyBackend`
  - 轻量、依赖更少，适合快速验证

你可以在不改电路定义的情况下切换后端。

## 10. 运行测试

在仓库根目录执行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 11. 迁移说明

如果你之前在项目里使用根目录 `Core.py` / `Circuit.py`：

- 新代码建议统一改为 `quantum_sim` 顶层导入
- `quantum_sim` 内部已完成独立门逻辑与电路实现
- 推荐入口：`Circuit + ExecutionEngine`

示例迁移：

```python
# old
# from Circuit import Circuit, hadamard, cnot

# new
from quantum_sim import Circuit, hadamard, cnot
```

---

如果你希望，我可以继续补一份 `README_en.md`（英文版）与一页 API 速查表。

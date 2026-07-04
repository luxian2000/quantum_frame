# aicir 使用手册

---

## 目录

- [1. 项目概览与安装](#1-项目概览与安装)
- [2. 模块导入](#2-模块导入)
- [3. 构建量子态](#3-构建量子态)
- [4. 量子线路的搭建](#4-量子线路的搭建)
- [5. 量子测量](#5-量子测量)
- [6. 构建哈密顿量](#6-构建哈密顿量)
- [7. 子包说明索引](#7-子包说明索引)

---

## 1. 项目概览与安装

A from-scratch quantum circuit simulator and quantum-algorithm framework for Python. Supports state vectors, density matrices, noise models, variational algorithms (VQE/QAOA/VQD/SSVQE), quantum architecture search, QML gradients, and OpenQASM I/O — with pluggable backends for CPU, GPU, and Ascend NPU.

### 1.1 Features（功能特性）


- **Unified quantum state** — `State` class handles pure states (amplitude vector) and mixed states (density matrix) with a consistent API
- **Rich gate library** — single-qubit, rotation, controlled, multi-target, multi-control, and particle-conserving excitation gates (`single_excitation`/Givens, `double_excitation`); typed `Operation` IR with construction-time validation
- **Flexible measurement** — in-circuit Pauli projection, terminal readout, shot sampling, exact mode, state snapshots, and partial traces
- **Classical control flow** — measurement-fed `ClassicalRegister`, `measure(creg=)`, and `if_`/`while_` (with `else`) evaluated per shot on the measurement trajectory (see §5.13)
- **Variational algorithms** — `BasicVQE`, `run_vqe`, QAOA, VQD, SSVQE with built-in ansatz templates (HEA, trapped-ion HEA-TI)
- **QML gradients** — parameter-shift (`psr`, `spsr`, `multipsr`, four-term `psr4` for excitation gates), finite-difference, SPSA, quantum natural gradient, and PyTorch `autograd`
- **Quantum architecture search** — weight-shared supernet, CRLQAS, PPR\_DQL, PPO\_RB (requires PyTorch)
- **Noise simulation** — depolarizing, bit/phase flip, amplitude damping, ion-trap noise via density-matrix evolution
- **OpenQASM I/O** — round-trip import/export for OpenQASM 2.0 and 3.0; Qiskit, PennyLane, and WuYue interop
- **Pluggable backends** — `NumpyBackend` (CPU), `GPUBackend` (PyTorch / CUDA), `NPUBackend` (Ascend) — swap with one line

### 1.2 Installation（安装）


```bash
# Core (NumPy only)
pip install aicir

# With optional extras
pip install "aicir[torch]"   # GPU/NPU backend + QAS
pip install "aicir[viz]"     # Circuit visualization (matplotlib)
pip install "aicir[sci]"     # Classical optimizers (scipy)
pip install "aicir[all]"     # Everything above
```

Or install editable from source:

```bash
git clone https://github.com/luxian2000/quantum_frame.git
cd quantum_frame
pip install -e ".[all]"
```

> `torch`, `matplotlib`, and `scipy` are optional. Core simulation works with NumPy alone.

### 1.3 Quick Start（快速上手）


```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot
from aicir.core import State

backend = NumpyBackend()

# Build a Bell-state circuit
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

# Run with 1024 shots
result = Measure(backend).run(cir, shots=1024)
print(result.counts)   # {'00': ~512, '11': ~512}

# Or evolve a state directly
psi = State.zero_state(2, backend)
psi1 = psi.evolve(cir.unitary())
print(psi1.ket)        # 1/√2|00> + 1/√2|11>
```

### 1.4 Backends（后端）


| Backend | Library | Device | Autograd |
| --- | --- | --- | --- |
| `NumpyBackend` | NumPy | CPU | No |
| `GPUBackend` | PyTorch | CPU / CUDA | Yes |
| `NPUBackend` | PyTorch + torch\_npu | Ascend NPU | Yes |

Switch backends by changing one line — no other code changes needed:

```python
from aicir import GPUBackend, NPUBackend

backend = GPUBackend(device="cuda:0")
# or
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
```

### 1.5 Project Structure（项目结构）


```text
aicir/
  backends/     # NumpyBackend, GPUBackend, NPUBackend
  core/         # State, Circuit, gates, I/O
  measure/      # Measure, Result, Estimator
  vqc/          # VQE, QAOA, VQD, SSVQE, ansatz templates
  qas/          # Quantum architecture search (requires torch)
  qml/          # Gradient methods
  noise/        # Noise channels and models
  ir/           # Typed Operation IR
  gates/        # GateSpec registry
  transpile/    # Pass-manager pipeline
  primitives/   # ShotSampler, StatevectorEstimator, ShotEstimator
  optimizer/    # Classical optimizers (Adam, COBYLA, LBFGS, …)
  chemistry/    # Preset qubit Hamiltonians (H2, H2-JW, H2-tapered)
  encoder/      # AmplitudeEncoder, AngleEncoder, BasisEncoder
  universal/    # Reusable primitives (QFT, …)
  visual/       # Circuit visualization
demos/          # Runnable end-to-end examples
tests/          # pytest test suite
```

### 1.6 Requirements（环境要求）


- Python ≥ 3.11
- NumPy (required)
- PyTorch (optional — GPU/NPU backend, QAS, autograd)
- matplotlib (optional — visualization)
- scipy (optional — classical optimizers)

---

## 2. 模块导入

所有常用类与函数均可从顶层 `aicir` 包一次性导入。

统一状态类 `State` 的规范模块路径为 `aicir.core`，同时也可从顶层 `aicir` 导入。它同时表示纯态（振幅向量形态）与混合态/密度矩阵（矩阵形态），取代了旧的 `StateVector` / `DensityMatrix`。

```python
# 后端
from aicir import GPUBackend, NumpyBackend, NPUBackend

# 量子态（规范路径）：统一的 State 类
from aicir.core import State

# 量子门（构造函数；签名与参数顺序不变，返回类型化 Operation，
# 支持旧门字典的只读访问与 == 比较，见 4.1 节说明）
from aicir import (
    pauli_x, pauli_y, pauli_z,
    hadamard,
    rx, ry, rz,
    s_gate, t_gate,
    cx, cnot, cy, cz,
    crx, cry, crz,
    swap, rzz, rxx, ms_gate,
    single_excitation, double_excitation,  # 粒子数守恒激发门（givens 为 single_excitation 别名）
    toffoli, ccnot,
    u2, u3,
    measure,     # 线路内联合 Pauli 投影测量标记，返回 Measurement
)

# 电路、 typed IR 与参数占位符
from aicir import Circuit, CircuitIR, Measurement, Observable, Operation, Parameter

# 测量
from aicir import Measure, Result

# 经典控制流（测量反馈的 if/while）
from aicir import ClassicalRegister, if_, while_
# measure(qubits, creg=/cbits=) 从 aicir.core.circuit 取用

# 哈密顿量
from aicir import PauliOp, PauliString, Hamiltonian

# 噪声
from aicir import (
    NoiseChannel, NoiseModel,
    DepolarizingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
)

# OpenQASM 互转
from aicir import (
    circuit_to_qasm, circuit_to_qasm3,
    circuit_from_qasm,
    load_circuit_qasm, save_circuit_qasm,
    save_circuit_qasm3,
    circuit_to_qiskit, circuit_from_qiskit,
    to_qiskit, from_qiskit,
    circuit_to_pennylane, circuit_from_pennylane,
    to_pennylane, from_pennylane,
    circuit_to_wuyue, circuit_from_wuyue,
    to_wuyue, from_wuyue,
)

# QML 梯度工具
from aicir.qml import psr, spsr, multipsr

# 线路编译与优化 pass pipeline
from aicir.transpile import PassManager, optimize, optimize_basic, optimize_circuit

# 门元信息注册表（GateSpec：num_qubits/num_params/num_controls/generator/decomposition）
from aicir.gates import GateSpec, get_gate_spec, register_gate, canonical_gate_name

# Sampler / Estimator primitives（统一执行入口）
from aicir.primitives import ShotSampler, StatevectorEstimator, ShotEstimator
```

---

## 3. 构建量子态

`aicir` 用统一的 `State` 类表示量子态：纯态以**振幅向量形态**存储，密度矩阵以**矩阵形态**存储，`state.is_density` 区分两者。`backend` 参数可选，省略时默认使用 `NumpyBackend`；数值数据保存在对应后端的张量对象中。

### 3.1 构建零态

最常用的初态是全零计算基态：

```python
from aicir.core import State

psi = State.zero_state(n_qubits=2)                  # 向量形态 |00>
rho = State.zero_state(n_qubits=2).to_density_matrix()  # 矩阵形态 |00><00|

print(psi.array)            # [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
print(rho.probabilities())  # [1. 0. 0. 0.]
print(rho.is_density)       # True
```

### 3.2 从数组构建纯态

`State.from_array(...)` 接收长度为 `2**n_qubits` 的数组，并会自动归一化。`n_qubits` 可省略（按数组长度推断），`backend` 也可省略。基态顺序采用大端序：二比特时下标依次对应 `|00>`、`|01>`、`|10>`、`|11>`。

```python
from aicir.core import State

psi = State.from_array([1, 0, 0, 1])   # n_qubits 自动推断为 2，backend 默认 NumpyBackend

print(psi.ket)  # 1/\sqrt{2}|00>+1/\sqrt{2}|11>
```

如果直接调用 `State(data, n_qubits, backend)`，构造器只检查形状，不会自动归一化；因此面向用户代码时优先使用 `from_array(...)`。

### 3.3 三种表示与打印

`State` 提供三个可直接打印的表示属性：

- `.array` —— 振幅向量（numpy `(2^n,)`）；**混合态返回 `None`**。
- `.matrix` —— 密度矩阵（numpy `(2^n, 2^n)`）；纯态会即时计算 `|ψ><ψ|`。
- `.ket` —— Dirac 记号字符串：纯态为 `Σaᵢ|i>` 叠加，混合态为 `Σρ_ij|i><j|` 算符展开。

`print(state)` 默认输出 `.ket`；需要控制端序或过滤极小振幅时调用 `state.format(...)`。

```python
from aicir.core import State

psi = State.from_array([1, 0, 0, 1])

print(psi)                  # 1/\sqrt{2}|00>+1/\sqrt{2}|11>
print(psi.ket)              # 1/\sqrt{2}|00>+1/\sqrt{2}|11>
print(psi.array)            # 态向量的一维复数数组
print(psi.matrix)           # 2^n x 2^n 密度矩阵
print(psi.probabilities())  # [0.5 0.  0.  0.5]
```

矩阵形态（混合态）下，`.array` 为 `None`，`.ket` 给出算符展开：

```python
import numpy as np
from aicir.core import State

rho = State.from_matrix(np.diag([0.5, 0.5, 0.0, 0.0]).astype(np.complex64))

print(rho.array)            # None（混合态无单一振幅向量）
print(rho.ket)              # 0.5|00><00|+0.5|01><01|
print(rho.matrix)           # 2^n x 2^n 复数矩阵
print(rho.probabilities())  # [0.5 0.5 0.  0. ]
print(rho.purity())         # 0.5（混合态）
```

### 3.4 从纯态构建密度矩阵

纯态可以直接转换为矩阵形态 `State`：

```python
from aicir.core import State

psi = State.from_array([1, 0, 0, 1])

rho = psi.to_density_matrix()

print(rho.is_density)  # True
print(rho.purity())    # 纯态密度矩阵的 purity 约为 1.0
```

也可以直接从矩阵或列表构建密度矩阵（`n_qubits` 由形状推断）：

```python
import numpy as np
from aicir.core import State

rho = State.from_matrix(
    np.diag([0.5, 0.5, 0.0, 0.0]).astype(np.complex64),
)
```

### 3.5 用指定初态执行线路

`Measure.run(...)` 可通过 `initial_state` 指定纯态初态；不需要采样时传 `shots=None`，返回值中的 `final_state` 就是线路演化后的完整末态。

```python
import numpy as np
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot
from aicir.core import State

backend = NumpyBackend()

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
)

psi0 = State.from_array(
    np.array([0, 1, 0, 0], dtype=np.complex64),  # |01>
    n_qubits=2,
    backend=backend,
)

result = Measure(backend).run(
    cir,
    shots=None,
    initial_state=psi0,
    return_state=True,
)

# result.final_state 是 numpy 态向量；重新封装为 State 后可按 ket 形式打印。
final_psi = State.from_array(
    result.final_state,
    n_qubits=result.n_qubits,
    backend=backend,
)
print(final_psi.ket)  # 1/\sqrt{2}|01>+1/\sqrt{2}|10>
```

密度矩阵路径可使用 `Measure.run_density_matrix(..., initial_density_matrix=...)`，适合噪声模型或混合态演化。

---

## 4. 量子线路的搭建

### 4.1 门构造函数速查

| 函数                        | 参数                             | 说明            |
| --------------------------- | -------------------------------- | --------------- |
| `pauli_x(q)`              | target_qubit                     | X 门            |
| `pauli_y(q)`              | target_qubit                     | Y 门            |
| `pauli_z(q)`              | target_qubit                     | Z 门            |
| `hadamard(q)`             | target_qubit                     | H 门            |
| `s_gate(q)`               | target_qubit                     | S 门            |
| `t_gate(q)`               | target_qubit                     | T 门            |
| `rx(θ, q)`               | 角度, target_qubit               | Rx 旋转门       |
| `ry(θ, q)`               | 角度, target_qubit               | Ry 旋转门       |
| `rz(θ, q)`               | 角度, target_qubit               | Rz 旋转门       |
| `u2(φ, λ, q)`           | phi, lambda, target_qubit        | U2 门           |
| `u3(θ, φ, λ, q)`       | theta, phi, lambda, target_qubit | U3 通用单比特门 |
| `cx(t, [c])`              | target（单个或列表）, control_list | CNOT（控制-X）；目标支持单个或多个（多目标等价于多个单目标 CX） |
| `cnot(t, [c])`            | target（单个或列表）, control_list | 同 cx           |
| `cy(t, [c])`              | target, control_list             | 控制-Y          |
| `cz(t, [c])`              | target, control_list             | 控制-Z          |
| `crx(θ, t, [c])`         | 角度, target, control_list       | 受控 Rx         |
| `cry(θ, t, [c])`         | 角度, target, control_list       | 受控 Ry         |
| `crz(θ, t, [c])`         | 角度, target, control_list       | 受控 Rz         |
| `swap(q1, q2)`            | qubit_1, qubit_2                 | SWAP 门         |
| `rzz(θ, q1, q2)`         | 角度, qubit_1, qubit_2           | ZZ 旋转门       |
| `rxx(θ, q1, q2)`         | 角度, qubit_1, qubit_2           | XX 旋转门       |
| `ms_gate(θ, q1, q2)`     | 角度, qubit_1, qubit_2           | `rxx` 的别名  |
| `single_excitation(θ, q1, q2)` | 角度, qubit_1, qubit_2     | 粒子数守恒 Givens 单激发门 |
| `givens(θ, q1, q2)`      | 角度, qubit_1, qubit_2           | `single_excitation` 的别名 |
| `double_excitation(θ, q1, q2, q3, q4)` | 角度, 四个 qubit  | 粒子数守恒双激发门（耦合 \|0011⟩↔\|1100⟩） |
| `toffoli(t, [c0,c1,...])` | target, control_list             | 多控制 X 门     |
| `ccnot(t, [c0,c1,...])`   | target, control_list             | 同 toffoli      |
| `measure(q...)`           | qubit list                       | 线路内测量标记  |
| `reset(q...)`             | qubit list                       | 测量后重置为零态 |

`toffoli` / `ccnot` 的矩阵构造与逐门执行路径支持任意数量控制位，也支持门字典中的 `control_states`；导出为 OpenQASM `ccx` 时仍只适用于两个控制位。

门构造函数的**签名与参数顺序与旧版完全一致**，但返回值已由裸门字典升级为类型化 `Operation`（`measure(...)` / `reset(...)` 返回 `Measurement`）：

- **构造期校验**：量子比特下标、控制位/控制态长度，以及按 GateSpec 注册表检查目标比特数、参数个数与控制位要求（见 4.6 节），错误在调用处立即报出。
- **旧字典只读兼容**：`gate["type"]`、`.get("parameter")`、`in`、`dict(gate)`、迭代等读取照常可用，且与旧门字典可直接 `==` 比较；写入（`gate[...] = ...`）会抛 `TypeError`（对象不可变）。
- **`Circuit` 内部存储不变**：仍是门字典列表（`circuit.gates`），下游代码无需改动。

### 4.2 构建电路

推荐的方式是在构造 `Circuit` 时直接传入门列表，并显式给出 `n_qubits`——这样电路宽度明确、不依赖自动推断，也不涉及后端继承等隐式行为：

```python
from aicir import Circuit, hadamard, cnot, rz

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),      # 目标比特=1，控制比特=[0]
    rz(0.5, 1),
    n_qubits=2,        # 显式指定电路宽度
)

# 获取电路酉矩阵（numpy complex64，2^n × 2^n）
U = cir.unitary()
print(U.shape)   # (4, 4)
```

同一种方式可容纳全部门类型。下面的例子覆盖单比特/旋转/通用单比特/受控/双比特旋转/SWAP/多控门，参数顺序与 4.1 节速查表一致：

```python
import math
from aicir import Circuit, rx, ry, u2, u3, crx, rzz, rxx, swap, toffoli, single_excitation, double_excitation

cir = Circuit(
    rx(math.pi / 2, 0),              # Rx(π/2) 作用在 qubit 0
    ry(math.pi / 4, 1),              # Ry(π/4) 作用在 qubit 1
    u2(math.pi / 3, math.pi / 5, 0), # U2 门
    u3(math.pi, 0, math.pi, 2),      # U3(π, 0, π) ≡ X 门，作用在 qubit 2
    crx(math.pi / 2, 2, [1]),        # 受控 Rx，控制=qubit1，目标=qubit2
    rzz(math.pi / 3, 0, 2),          # RZZ 作用在 qubit0 和 qubit2
    rxx(math.pi / 3, 0, 1),          # RXX / Mølmer-Sørensen 作用在 qubit0 和 qubit1
    swap(0, 1),                      # SWAP qubit0 和 qubit1
    toffoli(2, [0, 1]),              # Toffoli，控制=[0,1]，目标=qubit2
    single_excitation(0.3, 0, 1),    # 粒子数守恒 Givens 单激发，qubit0/qubit1
    n_qubits=3,
)
```

`double_excitation(θ, q1, q2, q3, q4)`（4 比特、粒子数守恒、耦合 |0011⟩↔|1100⟩）同样可直接构造，配合 HF 参考态做化学 VQE：

```python
from aicir import Circuit, double_excitation, pauli_x

cir = Circuit(
    pauli_x(1), pauli_x(3),          # 制备 HF 行列式 |0101⟩
    double_excitation(0.2, 0, 2, 1, 3),  # 双激发耦合 |0101⟩↔|1010⟩
    n_qubits=4,
)
```

门构造函数返回的是 typed IR 对象（`Operation`，`measure(...)` 为 `Measurement`），也可以直接用 `Operation(...)` 显式构造，程序化生成线路时更方便：

```python
from aicir import Circuit, Operation, Measurement

cir = Circuit(
    Operation("hadamard", qubits=(0,)),
    Operation("cx", qubits=(1,), controls=(0,)),
    Measurement((0, 1)),
    n_qubits=2,
)
```

`Circuit` 继续保留 `.gates` 门字典 surface（内部存储不变），同时提供 `.operations` 和 `.ir` typed IR 视图；`aicir.ir` 另有 `Observable`/`CircuitIR` 用于可观测量与线路级 IR。

### 4.3 让 Circuit 作用于 State

`aicir` 里“把线路作用到量子态上”有三种常用途径；它们底层都对应同一个数学对象，但适用场景不同。

**方式一：显式取酉矩阵，再用 `State.evolve(...)` 演化**

这是最直接、最适合教学或算法 demo 的写法。`Circuit.unitary()` 返回线路酉矩阵，`State.evolve(U)` 对纯态执行 `U|ψ>`，对密度矩阵执行 `UρU†`。`Circuit.matrix()` 只是 `unitary()` 的别名。

```python
from aicir import Circuit, NumpyBackend, hadamard, cnot
from aicir.core import State

backend = NumpyBackend()

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

psi0 = State.from_array([0, 1, 0, 0], backend=backend)  # |01>

U = cir.unitary()             # 或 cir.matrix()
psi1 = psi0.evolve(U)

print(psi1.ket)               # 1/\sqrt{2}|01>+1/\sqrt{2}|10>
print(psi1.probabilities())   # [0.  0.5 0.5 0. ]
```

如果初态已经是矩阵形态 `State`，同样可以直接演化：

```python
rho0 = psi0.to_density_matrix()
rho1 = rho0.evolve(cir.unitary())

print(rho1.is_density)   # True
print(rho1.matrix.shape) # (4, 4)
```

**方式二：通过 `Measure.run(..., initial_state=...)` 执行纯态线路**

如果你接下来本来就要测量、取概率、取期望值或读取 `Result`，可以直接把初态通过 `initial_state` 传给 `Measure.run(...)`。当 `shots=None` 或 `0` 时，不做采样，`result.state` / `result.final_state` 就是完整末态。

```python
from aicir import Measure

result = Measure(backend).run(
    cir,
    shots=None,
    initial_state=psi0,
    return_state=True,
)

psi1 = State.from_array(result.final_state, backend=backend)
print(psi1.ket)  # 1/\sqrt{2}|01>+1/\sqrt{2}|10>
```

这种方式的好处是线路执行与后续读出在同一个接口里完成；如果后面改成 `shots=1024`、加入 `observables=...` 或读取 `counts`，调用方式不需要重写。

**方式三：通过 `Measure.run_density_matrix(..., initial_density_matrix=...)` 执行矩阵形态初态**

噪声分析、混合态传播或显式密度矩阵初态，使用密度矩阵路径更合适。这里传入的 `State` 必须已经是矩阵形态（可由 `.to_density_matrix()` 得到）。

```python
rho0 = psi0.to_density_matrix()

result = Measure(backend).run_density_matrix(
    cir,
    shots=None,
    initial_density_matrix=rho0,
    return_state=True,
)

rho1 = State.from_matrix(result.final_state, backend=backend)
print(rho1.is_density)        # True
print(rho1.probabilities())   # [0.  0.5 0.5 0. ]
```

使用建议：

- 只想清楚地表达“线路作用到态上”时，优先用 `State.evolve(circuit.unitary())`。
- 后续还要测量、采样、算期望值时，优先用 `Measure.run(..., initial_state=...)`。
- 初态本身是混合态，或需要密度矩阵语义时，用 `run_density_matrix(..., initial_density_matrix=...)`。
- 若电路绑定了 `backend`，`unitary()` 与 `Measure.run(...)` 会优先沿该后端执行；对大线路通常比先在 CPU 组装再搬运更合适（见 `aicir/backends/README.md` 的 6.6 节）。

### 4.4 参数化量子线路

`Parameter` 可作为旋转门参数的符号占位符，用于构建量子神经网络、VQE、QAOA 等可训练线路模板。模板电路在绑定参数前只保存门字典，不会生成数值矩阵。

```python
from aicir import Circuit, Parameter, rx, ry, crz, cnot

theta0 = Parameter("theta0")
theta1 = Parameter("theta1")

template = Circuit(
    ry(theta0, 0),
    crz(theta1, target_qubit=1, control_qubits=[0]),
    cnot(1, [0]),
    n_qubits=2,
)

print(template.parameters)
# (Parameter(name='theta0'), Parameter(name='theta1'))

# 默认返回绑定后的新 Circuit，不修改模板
bound = template.bind_parameters({
    "theta0": 0.2,
    theta1: 0.5,   # key 也可以直接使用 Parameter 对象
})

U = bound.unitary()
print(U.shape)    # (4, 4)
```

也可以按 `template.parameters` 的顺序传入序列：

```python
bound = template.bind_parameters([0.2, 0.5])
```

如果希望原地更新模板，可使用：

```python
template.bind_parameters({"theta0": 0.2, "theta1": 0.5}, inplace=True)
```

注意事项：

- 未绑定参数的电路调用 `unitary()` 会报错，需要先 `bind_parameters(...)`。
- `allow_partial=True` 可做部分绑定，返回仍含未绑定参数的电路。
- `Parameter` 是符号占位符，不是自动微分张量；训练梯度可使用`aicir/qml/README.md` 的 parameter-shift 工具。
- 如果要使用 PyTorch autograd，可直接把 Torch 标量张量作为门参数，并调用 `Circuit.unitary(backend=GPUBackend(...))`。当前 `rx`/`ry`/`rz`/`u2`/`u3`、受控旋转门、`rzz`/`rxx` 和自定义 `unitary` 的 Torch 参数会保留计算图。
- 导出 QASM 前应先把所有符号参数绑定为数值。JSON 导出支持 `Parameter`、NumPy 标量/数组、复数和 Torch 张量数值；Torch 张量在 JSON 读回后会恢复为普通数值或列表，不会恢复为带计算图的 Tensor。

### 4.5 自定义 unitary、identity 与 Torch 自动微分

可以用门字典直接加入自定义酉矩阵：

```python
import numpy as np
from aicir import Circuit

custom = {
    "type": "unitary",
    "parameter": np.eye(4, dtype=np.complex64),
    "n_qubits": 2,
}

cir = Circuit(custom, n_qubits=3)
U = cir.unitary()   # 自定义 2-qubit unitary 会扩展到 3-qubit 电路宽度
```

`identity` 门也会按电路总宽度扩展；`Circuit.show()` 与 DAG 转换都能识别 `unitary` 门。

Torch 参数示例：

```python
import torch
from aicir import Circuit, GPUBackend, rx, rzz, rxx

backend = GPUBackend(device="cpu")
theta = torch.tensor(0.2, requires_grad=True)

cir = Circuit(
    rx(theta, 0),
    rzz(theta, 0, 1),
    rxx(theta, 0, 1),
    n_qubits=2,
)

U = cir.unitary(backend=backend)
loss = torch.real(U[0, 0])
loss.backward()
print(theta.grad)
```

### 4.6 GateSpec 门元信息注册表

`aicir.gates` 是门元信息的单一来源：每个门的目标比特数、参数个数、别名、QASM 导出名和绘图符号只在注册表里登记一次，`Operation` 构造期校验、`transpile` 的 `ValidatePass`/`CanonicalizePass`、QASM 导出、矩阵路径与绘图都从这里读取。

```python
from aicir.gates import GateSpec, get_gate_spec, register_gate, canonical_gate_name

get_gate_spec("rz")            # GateSpec(name="rz", num_qubits=1, num_params=1, symbol="Rz", ...)
get_gate_spec("X")             # 别名解析 → pauli_x 的 spec
canonical_gate_name("cnot")    # "cx"
get_gate_spec("not_a_gate")    # None：未注册门保持宽松（自定义门不受限）

# 注册自定义门：之后 Operation 构造会按 spec 校验，绘图直接使用 symbol
register_gate(GateSpec(name="my_iswap", num_qubits=2, num_params=0, symbol="iS"))
```

详见 [`aicir/gates/README.md`](aicir/gates/README.md)。

---

## 5. 量子测量

aicir 采用**统一测量模型**：线路内的 `measure`/`reset` 是操作序列中的正式操作，末端读出由 `measure_qubits` 控制；线路内操作与末端读出可共存。

`Measure` 对象绑定一个后端，`run()` 返回统一的 `Result` 对象。

### 5.1 运行接口与参数

```python
result = Measure(backend).run(
    circuit,
    shots=1,               # 采样次数；None/0 = exact 模式（不做末端测量，忽略 measure_qubits）
    measure_qubits=(),     # 末端读出：None=不测；空()=全部(默认)；[q0,…]=子集(保留顺序)
    snap=None,             # 记录完整态快照的操作下标列表
    sm="avg",              # 多轨迹聚合模式；当前仅支持 "avg"
    seed=None,             # 随机种子（用于复现）
    *,
    initial_state=None,          # 初始态（None = |0…0>）
    initial_density_matrix=None, # 初始密度矩阵（与 initial_state 互斥）
    observables=None,            # 可观测量字典 {name: matrix}
    return_state=True,           # 是否在结果中附带 state / final_state
)
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `shots` | `1` | 正整数/`0`/`None`；`0` 与 `None` 同义（exact 模式）；负数/非整数报 `ValueError` |
| `measure_qubits` | `()` | `None` = 不做末端测量；空 `()`/`[]` = 全部比特 `[0..n−1]`；`[q0,…]` = 子集（保留顺序）；exact 模式忽略本参数 |
| `snap` | `None` | 操作下标集合（0 起，`<` 操作总数 L），记录该操作完成后的完整态 |
| `sm` | `"avg"` | `"avg"`：多轨迹平均；`"shot"`/`"cond"` 待实现（传入报 `NotImplementedError`） |
| `seed` | `None` | 单个 `numpy.random` 种子，贯穿线路内 measure/reset/末端测量，相同参数可复现 |

### 5.2 线路与操作序列约定

线路 `cir` 包含操作序列 `[O₀, O₁, …, O_{L−1}]`，每个操作分配一个**操作下标**（0 起）：

- 普通量子门（`hadamard`、`cnot`、`rx`、…）按顺序执行酉演化。
- `measure(*qubits, basis="Z", id=None)`：线路内联合 Pauli 投影测量，占用一个操作下标。
- `reset(*qubits)`：重置信道，占用一个操作下标。
- 默认初态为 `|0…0>` 纯态；可通过 `initial_state` 或 `initial_density_matrix` 指定。

```python
from aicir import Circuit, hadamard, cnot, measure, reset

cir = Circuit(
    hadamard(0),   # op 0
    cnot(1, [0]), # op 1
    measure([0, 1]), # op 2 — 线路内联合 ZZ 投影测量
    reset(0),      # op 3 — 重置信道
    n_qubits=2,
)
```

### 5.3 线路内 measure：联合 Pauli 投影测量

`measure(qubits=None, *, basis="Z", id=None)` 对所列比特执行**两结果联合 Pauli 投影测量**：

- **投影**到 `λ=±1` 联合本征子空间，结果为 `λ∈{+1,−1}`。
- **非破坏性保留**：被测比特仍留在电路中，后续门可作用其上；子空间内相干被保留（不是逐比特坍缩）。
- `basis`：`"Z"`（默认）/ `"X"` / `"Y"`，同一 basis 作用于所列全部比特。
- `id`：可选字符串，在整条电路中唯一，使 `result.output("id")` 可用。

`result.output(i)` 按操作下标 `i` 或字符串 `id` 取该次测量结果；`result.counts(i)` / `result.prob(i)` 仅在 `shots≥1` 时可用。

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot, measure

# Bell 态用 ZZ 联合投影测量：Bell 态是 Z⊗Z 的 +1 本征态，恒返回 +1
cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1], id="zz"), n_qubits=2)
m = Measure(NumpyBackend())

# shots=None：单条精确轨迹，output 为标量
r = m.run(cir, shots=None, measure_qubits=None)
print(r.output("zz"))      # 1（+1，必然）
print(r.state.array)       # [0.707+0j, 0, 0, 0.707+0j]（Bell 态，仍相干）

# shots=8：多轨迹，output(2) 形状 (8, 1)，全为 +1
r8 = m.run(cir, shots=8, measure_qubits=None)
print(r8.output(2))        # [[1],[1],[1],[1],[1],[1],[1],[1]]
print(r8.counts(2))        # {1: 8, -1: 0} 或 {1: 8}
```

### 5.4 reset：重置信道

`reset(qubits)` 实施 `R_S(ρ) = |0⟩⟨0|_S ⊗ Tr_S(ρ)`，**无需事先 measure**，可出现在任意位置。`qubits` 为单个 int 或 list，多比特须用列表（`reset([0, 1])`）；`reset()` 重置全部比特：

- 被重置比特回到 `|0⟩`，与其他比特的关联被清除。
- 对**纠缠**目标比特施加 reset：该轨迹升级为密度矩阵（混合态），其他比特约化态不变。
- 对**可分**目标施加 reset：仍保持纯态。

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot, reset
import numpy as np

# Bell 态的 qubit 0 施加 reset → 混合态（密度矩阵）
cir = Circuit(hadamard(0), cnot(1, [0]), reset(0), n_qubits=2)
r = Measure(NumpyBackend()).run(cir, shots=None, snap=[2])

snap_after_reset = r.snap(2)
print(snap_after_reset.shape)   # (4, 4)  — 升级为密度矩阵
# 对角元约为 [0.5, 0.5, 0, 0]，表示 |00> 和 |01> 各占 0.5
# （reset(0) 把 Bell 态的 q0 重置为 |0>，q1 仍是最大混合 I/2）
print(np.real(np.diag(snap_after_reset)))
```

### 5.5 末端测量

电路全部显式操作执行完后，shot 模式（`shots≥1`）且 `measure_qubits` 非 `None` 时，对所列比特执行**逐比特 Z 基**投影测量：

- **输入顺序保留**：`measure_qubits=[1, 0]` 时 `output(-1)` 列顺序为 `[qubit1, qubit0]`，不做内部排序。
- `output(-1)` 形状：`shots=1` 时 `(1, k)`，`shots=M` 时 `(M, k)`（`k=len(measure_qubits)`，空时 `k=n`）。
- `measure_qubits=()`（默认）或 `[]`：读出全部 `n` 个比特。
- `measure_qubits=None`：不做末端测量。

```python
from aicir import Circuit, Measure, NumpyBackend, pauli_x
import numpy as np

# qubit1=|1>，qubit0=|0>；末端按 [1, 0] 顺序读出
cir = Circuit(pauli_x(1), n_qubits=2)
r = Measure(NumpyBackend()).run(cir, shots=1, measure_qubits=[1, 0])
print(r.output(-1))   # [[-1, 1]]  — qubit1=-1(|1>), qubit0=+1(|0>)
```

### 5.6 shots 语义

| `shots` | 模式 | 说明 |
|---|---|---|
| `None` 或 `0` | exact 模式 | 单条精确轨迹；忽略 `measure_qubits`、不执行末端测量；`output(-1)` / `counts(i)` / `prob(i)` 报错 |
| `M ≥ 1` | 采样模式 | M 条独立轨迹（线路内含 measure 时重跑完整电路，否则仅对末端重采样）；`output(i)` 形状 `(M,1)` / `output(-1)` 形状 `(M,k)` |

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot
import numpy as np

cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
m = Measure(NumpyBackend())

# exact 模式：不做末端测量
r = m.run(cir, shots=None)
assert np.allclose(r.final_state, r.state)
try:
    r.output(-1)
except ValueError:
    print("shots=None 时 output(-1) 报 ValueError")  # 预期输出

# shots=16：末端读出
r16 = m.run(cir, shots=16)
print(r16.output(-1).shape)  # (16, 2)
print(r16.counts(-1))        # {'00': N1, '11': N2}，N1+N2=16
```

### 5.7 末态与约化：state / final_state / snap / reduce

| 字段/方法 | `shots=None/0` | `shots=1` | `shots=M>1` |
|---|---|---|---|
| `result.state` | 本轨迹条件态（纯/DM） | 单 shot 条件态 | 平均 DM `(2^n,2^n)` |
| `result.final_state` | 等于 `state`（无末端测量） | 末端投影后的条件态 | 末端后平均 DM |
| `result.snap(t)` | 第 `t` 个操作后条件态 | 同左 | 第 `t` 个操作后平均 DM |
| `result.reduce(R, pos)` | 对 `state`/`final_state` 做偏迹，保留比特集 `R` | 同左 | 同左 |

- `shots>1` 时 `state` / `final_state` 一律为密度矩阵（`shape=(2^n, 2^n)`）。
- `sm="avg"` 时 `snap(t)` 为各轨迹第 `t` 步的平均 DM；`sm="shot"/"cond"` 待实现。
- `pos` 参数：`"final"`（默认，约化 `final_state`）/ `"state"`（约化 `state`）。

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot
import numpy as np

cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
m = Measure(NumpyBackend())

# snap 记录 Bell 态建立过程
r = m.run(cir, shots=None, snap=[0, 1])
print(r.snap(0).array)  # [0.707+0j, 0, 0.707+0j, 0] — H(0) 后
print(r.snap(1).array)  # [0.707+0j, 0, 0, 0.707+0j] — Bell 态

# shots>1 时 state 为密度矩阵；偏迹得单比特约化 DM
r16 = m.run(cir, shots=16)
print(np.asarray(r16.state).shape)  # (4, 4)
red = r16.reduce([0], pos="state")
print(red)   # [[0.5+0j, 0], [0, 0.5+0j]] — 约化为 I/2
```

### 5.8 期望值与 observables

```python
import numpy as np
from aicir import Circuit, Measure, NumpyBackend, pauli_x

Z = np.array([[1, 0], [0, -1]], dtype=complex)
cir = Circuit(pauli_x(0), n_qubits=1)   # |1> 态

result = Measure(NumpyBackend()).run(cir, shots=None, observables={"Z0": Z})
print(result.expectation_values)   # {'Z0': -1.0}
```

### 5.9 从 State 直接测量

```python
from aicir.core import State
from aicir import NumpyBackend

backend = NumpyBackend()
sv = State.zero_state(2, backend)

# 直接获取概率分布
probs = sv.probabilities()   # [1., 0., 0., 0.]

# 模拟 4 次测量，返回 {bitstring: count}
counts = sv.measure(shots=4)
print(counts)   # {'|00>': 4}
```

### 5.10 Result 字段/方法速查

**方法**（按需调用，带参数）：

| 方法 | 说明 |
|---|---|
| `output(i)` | 按操作下标 `i`（或字符串 `id`）取线路内 measure 结果；`output(-1)` 取末端结果 |
| `counts(i)` | 该 measure 的计数字典；仅 `shots≥1` 可用 |
| `prob(i)` | 该 measure 的相对频率；仅 `shots≥1` 可用 |
| `snap(t)` | 第 `t` 个操作完成后的完整态（需 `snap=[...,t,...]` 预注册） |
| `reduce(R, pos)` | 对 `pos` 态（`"final"`/`"state"`）保留比特集 `R` 求偏迹 |
| `most_probable()` | 返回 `(bitstring, prob)` 最高概率基态 |
| `summary()` | 单行摘要字符串 |

**字段**（直接访问）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `probabilities` | `np.ndarray` | `ρ_pre` 计算基对角元，shape `(2^n,)` |
| `state` | `np.ndarray` | 末端测量前完整态（SV 或 DM） |
| `final_state` | `np.ndarray` | 末端测量后的态；`measure_qubits=None`/exact 时与 `state` 相同 |
| `terminal_qubits` | `list[int]` | 末端测量比特列表（按输入顺序） |
| `measurement_specs` | `list[MeasureSpec]` | 线路内 measure 操作登记（`op_index`/`id`/`qubits`/`basis`） |
| `expectation_values` | `dict` | `{name: float}` 期望值 |
| `expectation_variances` | `dict` | `{name: float}` 方差 |
| `shots` | `int or None` | 实际采样次数（None = exact 模式） |
| `n_qubits` | `int` | 电路比特数 |
| `metadata` | `dict` | `state_mode` 等辅助信息 |

### 5.11 Sampler / Estimator primitives（统一执行入口）

`aicir.primitives` 把采样与期望值估计统一为 primitives，算法层无需各自处理测量、Hamiltonian 和 counts：

```python
from aicir import Circuit, Hamiltonian, hadamard, cx, pauli_x
from aicir.primitives import ShotSampler, StatevectorEstimator, ShotEstimator

bell = Circuit(hadamard(0), cx(1, [0]), n_qubits=2)
ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

sample = ShotSampler(shots=1024).run(bell)
# SampleResult(counts={'|00>': ..., '|11>': ...}, probs=..., shots=1024, measured_qubits=(0, 1))

exact = StatevectorEstimator().run(Circuit(pauli_x(0), n_qubits=1), ham)
# EstimateResult(value=-1.0, shots=None)  # 精确路径

noisy = ShotEstimator(shots=4096).run(Circuit(pauli_x(0), n_qubits=1), ham)
# EstimateResult(value≈-1.0, variance=..., shots=4096, term_results=(...))
```

约定：接收**已绑定参数**的电路；单个电路入参返回单个结果，序列入参返回结果列表；Estimator 支持单个可观测量广播到多个电路。`ShotEstimator` 包装 `PauliEstimator`（qubit-wise commuting 分组、基变换测量、shot 分配），并暴露 `estimate(circuit, hamiltonian)` 直通方法，可直接作为 `BasicVQE(energy_estimator=...)` 注入。详见 [`aicir/primitives/README.md`](aicir/primitives/README.md)。

### 5.12 输入检查与报错

| 条件 | 抛出 |
|---|---|
| `shots < 0` 或非整数（含 `bool`） | `ValueError` |
| `measure_qubits` / `measure(...)` / `reset(...)` 含越界比特 | `ValueError` |
| 单个列表内有重复比特 | `ValueError` |
| 跨 measure 操作的 `id` 重复 | `ValueError` |
| `snap` 含越界操作下标 | `ValueError` |
| `shots∈{None,0}` 且 `measure_qubits` 非 `None` | （exact 模式下静默忽略 `measure_qubits`，不报错） |
| `shots∈{None,0}` 且显式 `measure_qubits` 非空 | `ValueError` |
| `sm="shot"` 或 `"cond"` | `NotImplementedError`（待实现） |
| 产生混合态但后端不支持密度矩阵 | `ValueError` |

---

### 5.13 经典控制流（if/while）

依赖测量结果的经典控制流：把 Z 基测量写入**经典寄存器**，再用 `if_`/`while_` 按寄存器取值条件执行子线路。控制流是**每 shot 的运行时行为**，只在 `Measure.run` 的测量轨迹路径上执行——每条轨迹按自己的测量结果各自决定分支。

```python
import numpy as np
from aicir import Circuit, Measure, NumpyBackend, ClassicalRegister, hadamard, pauli_x, if_, while_
from aicir.core.circuit import measure

# 经典寄存器：ClassicalRegister(size, name)，reg[0] 为 LSB
c = ClassicalRegister(1, "c")

# H(0) → 测 q0 写入 c[0] → 若 c[0]==1 则翻转 q1
circ = Circuit(
    hadamard(0),
    measure(0, creg=c),                              # 每比特 Z 基投影，|0>→0 / |1>→1
    if_(c[0] == 1, Circuit(pauli_x(1), n_qubits=2)),  # else_body=... 可选
    n_qubits=2,
)
res = Measure(NumpyBackend()).run(circ, shots=400, seed=7, measure_qubits=[1])
res.classical_counts(c)   # {0: ~200, 1: ~200}：c 的整数取值分布（LSB=c[0]）
res.counts(-1)            # q1 末端读出与 c 完全关联（每 shot q1 == c[0]）
```

`while_` 必须提供 `max_iterations`；达上限仍满足条件时抛 `RuntimeError`（不静默截断）：

```python
r = ClassicalRegister(1, "r")
body = Circuit(pauli_x(0), measure(0, creg=r), n_qubits=1)   # 循环体须刷新 r，否则条件不变
loop = Circuit(
    pauli_x(0), measure(0, creg=r),                          # r[0]=1
    while_(r[0] == 1, body, max_iterations=5),               # 一步内收敛：X 后再测得 0
    n_qubits=1,
)
Measure(NumpyBackend()).run(loop, shots=10).classical_counts(r)   # {0: 10}
```

要点：

- **写经典位**：`measure(qubits, creg=reg)` 按序写 `reg` 的 0..k-1 位；`measure(qubits, cbits=[reg[i], ...])` 显式指定。有经典目标时仅支持 Z 基，`creg`/`cbits` 互斥。无经典目标时 `measure(...)` 保持原联合 Pauli 投影语义不变。
- **条件**：`reg[i] == v`（位，`v∈{0,1}`）或 `reg == N`（整个寄存器整数值）；支持 `==` 与 `!=`。
- **结构**：`if_(condition, body, else_body=None)`、`while_(condition, body, *, max_iterations)`；`body`/`else_body` 为 `Circuit`，`n_qubits` 须与外层一致，可任意嵌套（body 内可再含 measure→creg、`if_`/`while_`）。
- **读出**：`Result.classical_counts(reg)` 给出各 shot 末尾寄存器整数值的分布（reg 可传名字或 `ClassicalRegister`；从未写入 → `{}`）。
- **限制**：控制流只能经 `Measure.run` 执行；`Circuit.unitary()`、张量网络引擎（`aicir.simulator`）遇控制流一律抛 `ValueError`（语义上不可表示为酉矩阵）；QASM3 控制流导出暂未支持。含控制流+creg 的电路支持 JSON 往返（`circuit_to_json`/`circuit_from_json`），往返后执行结果一致。

---

## 6. 构建哈密顿量

aicir 提供三个层次的算符构建工具：`PauliOp`、`PauliString`、`Hamiltonian`。

### 6.1 单算符 PauliOp

```python
from aicir import PauliOp, GPUBackend

backend = GPUBackend()

# Z 作用在 qubit 0，在 2 比特空间里展开为 4×4 矩阵
Z0 = PauliOp('Z', qubit=0)
mat = Z0.to_matrix(n_qubits=2, backend=backend)
print(mat.shape)   # torch.Size([4, 4])
```

### 6.2 多体泡利串 PauliString

```python
from aicir import PauliString, GPUBackend

backend = GPUBackend()

# 0.5 × Z₀ ⊗ X₁（2 比特空间）
ps = PauliString("ZX", coefficient=0.5)
mat = ps.to_matrix(backend)
print(ps)   # PauliString(0.5+0j × Z⊗X)

# 也可以显式指定 n_qubits；字符串长度必须与 n_qubits 一致
ps_auto = PauliString("ZX", coefficient=0.5, n_qubits=2)
print(ps_auto)
```

### 6.3 哈密顿量 Hamiltonian

```python
from aicir import Hamiltonian, GPUBackend, Circuit, Measure, hadamard

backend = GPUBackend()

# H = -1.0 × Z₀Z₁  +  0.5 × X₀X₁  +  0.3 × Z₀
H = Hamiltonian([
    ("ZZ", -1.0),
    ("XX", 0.5),
    ("ZI", 0.3),
])

# ("ZZ", -2.0) 表示 -2.0 × Z₀Z₁；省略 qubit index 时默认为 [0, 1, ...]。
H_coeff = Hamiltonian([
    ("ZZ", -2.0),
])

# ("ZZ", [0, 3]) 表示 1.0 × Z₀Z₃；省略 coefficient 时默认为 1.0。
H_qubits = Hamiltonian([
    ("ZZ", [0, 3]),
])

# 完整显式形式可同时给出 qubit index 和 coefficient。
# 下面等价于 4 比特完整字符串 ("ZIIZ", -2.0)。
H03 = Hamiltonian([
    ("ZZ", [0, 3], -2.0),
])
# coefficient 和 qubit index 可以交换顺序；下面与 H03 等价。
H03_alt = Hamiltonian([
    ("ZZ", -2.0, [0, 3]),
])

# 多个局部 Pauli 项可以直接写在同一个 Hamiltonian 中。
H_pairs = Hamiltonian([
    ("ZZ", [0, 3], -2.0),
    ("XX", [1, 2], -0.5),
])

# 如果需要在更大的 Hilbert 空间中补 I，可传 n_qubits。
H_default = Hamiltonian(n_qubits=4, terms=[
    "ZZ",          # 等价于 ("ZZ", 1.0, [0, 1])
    ("X", [2]),    # 等价于 ("X", 1.0, [2])
])
# 也兼容 ("ZZ", -2.0, [0, 3]) 这种 Pauli-first 显式形式。

# 转为后端矩阵
H_mat = H.to_matrix(backend)
print(H_mat.shape)   # torch.Size([4, 4])

# 计算期望值（通过 State）
from aicir.core import State
sv = State.zero_state(2, backend)
print(H.expectation(sv, backend))   # 实数期望值

# 通过 Measure.run() 传入 observables 参数
measure = Measure(backend)
cir = Circuit(hadamard(0), n_qubits=2)
result = measure.run(cir, shots=None, observables={"H": H_mat})
print(result.expectation_values["H"])
```

### 6.4 噪声通道（开放量子系统）

```python
from aicir import (
    NoiseModel,
    DepolarizingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
    GPUBackend,
)
from aicir.core import State

backend = GPUBackend()
model = (NoiseModel()
         .add_channel(DepolarizingChannel(target_qubit=0, p=0.01))
         .add_channel(BitFlipChannel(target_qubit=1, p=0.02), after_gates=["hadamard"])
         .add_channel(PhaseFlipChannel(target_qubit=0, p=0.005)))

# 手动应用到密度矩阵
rho = State.zero_state(2, backend).to_density_matrix()
rho_noisy = model.apply(rho.data, n_qubits=2, backend=backend)
```

---

## 7. 子包说明索引

`aicir` 子目录中还包含更具体的说明文档：

| 子目录                      | 说明文档                                                                | 内容概要                                                                                   |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `aicir/backends`         | [`aicir/backends/README.md`](aicir/backends/README.md)                 | 计算后端选择与使用：NumpyBackend / GPUBackend / NPUBackend、NPU complex64 兼容、后端绑定、严格模式、分布式与端到端示例。 |
| `aicir/chemistry`         | [`aicir/chemistry/README.md`](aicir/chemistry/README.md)                 | 小型固定设置的分子 qubit Hamiltonian 预置，用于 VQE 示例、单元测试和算法原型验证。         |
| `aicir/core/io`           | [`aicir/core/io/README.md`](aicir/core/io/README.md)                     | OpenQASM 导出行为、受控旋转门和多控旋转门分解规则。                                        |
| `aicir/gates`             | [`aicir/gates/README.md`](aicir/gates/README.md)                         | GateSpec 门元信息注册表：目标比特数/参数个数/别名/QASM 名/绘图符号的单一来源。             |
| `aicir/metrics`           | [`aicir/metrics/README.md`](aicir/metrics/README.md)                     | 任务无关的量子线路评分指标，供 QAS、VQE ansatz 筛选等架构层任务复用。                      |
| `aicir/optimization/qubo` | [`aicir/optimization/qubo/README.md`](aicir/optimization/qubo/README.md) | QUBO 建模、Ising/Hamiltonian 转换、BasicQAOA 矩阵入口与结果解码。                          |
| `aicir/optimizer`         | [`aicir/optimizer/README.md`](aicir/optimizer/README.md)                 | VQE/VQA 经典参数优化器（`Adam`/`SPSA`/`minimize` 等）；线路结构优化已迁至 `aicir.transpile`。 |
| `aicir/primitives`        | [`aicir/primitives/README.md`](aicir/primitives/README.md)               | Sampler/Estimator primitives 统一执行入口与 `SampleResult`/`EstimateResult` 结果对象。 |
| `aicir/qas`               | [`aicir/qas/README.md`](aicir/qas/README.md)                             | 量子架构搜索模块、统一入口、配置工厂和各 QAS 方法说明。                                    |
| `aicir/qml`               | [`aicir/qml/README.md`](aicir/qml/README.md)                             | 量子机器学习梯度工具，包括参数移位、有限差分、伴随微分和自动微分等方法。                   |
| `aicir/transpile`         | [`aicir/transpile/README.md`](aicir/transpile/README.md)                 | 线路编译与优化流水线：`PassManager`、`optimize` 入口、多格式 `optimize_basic`/`optimize_circuit` 与本地化简 pass。 |
| `aicir/visual`           | [`aicir/visual/README.md`](aicir/visual/README.md)                     | 线路图、态向量/概率分布、密度矩阵热力图，以及 QAS / metrics 结果可视化。 |
| `aicir/vqc`               | [`aicir/vqc/README.md`](aicir/vqc/README.md)                             | VQE、QAOA、VQD、SSVQE 等基础变分算法实现，以及可复用的参数化线路 ansatz 模板。             |
| `demos`                   | [`demos/README.md`](demos/README.md)                                     | 演示 `aicir.visual` 模块的示例脚本，涵盖线路、态向量、密度矩阵和 QAS 结果可视化。        |

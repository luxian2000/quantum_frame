# aicir 使用手册

---

## 目录

- [1. 模块导入](#1-模块导入)
- [2. 构建量子态](#2-构建量子态)
- [3. 量子线路的搭建](#3-量子线路的搭建)
- [4. 量子测量](#4-量子测量)
- [5. 构建哈密顿量](#5-构建哈密顿量)
- [6. 计算后端的选择与使用](#6-计算后端的选择与使用)
- [7. 与 OpenQASM 2.0 / 3.0 互转](#7-与-openqasm-20--30-互转)
- [8. QML 梯度工具](#8-qml-梯度工具)
- [9. 可视化模块](#9-可视化模块)
- [10. 子包说明索引](#10-子包说明索引)

---

## 1. 模块导入

所有常用类与函数均可从顶层 `aicir` 包一次性导入。

统一状态类 `State` 的规范模块路径为 `aicir.core`，同时也可从顶层 `aicir` 导入。它同时表示纯态（振幅向量形态）与混合态/密度矩阵（矩阵形态），取代了旧的 `StateVector` / `DensityMatrix`。

```python
# 后端
from aicir import GPUBackend, NumpyBackend, NPUBackend

# 量子态（规范路径）：统一的 State 类
from aicir.core import State

# 量子门（构造函数；签名与参数顺序不变，返回类型化 Operation，
# 支持旧门字典的只读访问与 == 比较，见 3.1 节说明）
from aicir import (
    pauli_x, pauli_y, pauli_z,
    hadamard,
    rx, ry, rz,
    s_gate, t_gate,
    cx, cnot, cy, cz,
    crx, cry, crz,
    swap, rzz, rxx, ms_gate,
    toffoli, ccnot,
    u2, u3,
    measure,     # 线路内联合 Pauli 投影测量标记，返回 Measurement
)

# 电路、 typed IR 与参数占位符
from aicir import Circuit, CircuitIR, Measurement, Observable, Operation, Parameter

# 测量
from aicir import Measure, Result

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
from aicir.transpile import PassManager, default_optimization_pipeline

# 门元信息注册表（GateSpec）
from aicir.gates import GateSpec, get_gate_spec, register_gate, canonical_gate_name

# Sampler / Estimator primitives（统一执行入口）
from aicir.primitives import ShotSampler, StatevectorEstimator, ShotEstimator
```

---

## 2. 构建量子态

`aicir` 用统一的 `State` 类表示量子态：纯态以**振幅向量形态**存储，密度矩阵以**矩阵形态**存储，`state.is_density` 区分两者。`backend` 参数可选，省略时默认使用 `NumpyBackend`；数值数据保存在对应后端的张量对象中。

### 2.1 构建零态

最常用的初态是全零计算基态：

```python
from aicir.core import State

psi = State.zero_state(n_qubits=2)                  # 向量形态 |00>
rho = State.zero_state(n_qubits=2).to_density_matrix()  # 矩阵形态 |00><00|

print(psi.array)            # [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
print(rho.probabilities())  # [1. 0. 0. 0.]
print(rho.is_density)       # True
```

### 2.2 从数组构建纯态

`State.from_array(...)` 接收长度为 `2**n_qubits` 的数组，并会自动归一化。`n_qubits` 可省略（按数组长度推断），`backend` 也可省略。基态顺序采用大端序：二比特时下标依次对应 `|00>`、`|01>`、`|10>`、`|11>`。

```python
from aicir.core import State

psi = State.from_array([1, 0, 0, 1])   # n_qubits 自动推断为 2，backend 默认 NumpyBackend

print(psi.ket)  # 1/\sqrt{2}|00>+1/\sqrt{2}|11>
```

如果直接调用 `State(data, n_qubits, backend)`，构造器只检查形状，不会自动归一化；因此面向用户代码时优先使用 `from_array(...)`。

### 2.3 三种表示与打印

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

### 2.4 从纯态构建密度矩阵

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

### 2.5 用指定初态执行线路

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

## 3. 量子线路的搭建

### 3.1 门构造函数速查

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
| `cx(t, [c])`              | target, control_list             | CNOT（控制-X）  |
| `cnot(t, [c])`            | target, control_list             | 同 cx           |
| `cy(t, [c])`              | target, control_list             | 控制-Y          |
| `cz(t, [c])`              | target, control_list             | 控制-Z          |
| `crx(θ, t, [c])`         | 角度, target, control_list       | 受控 Rx         |
| `cry(θ, t, [c])`         | 角度, target, control_list       | 受控 Ry         |
| `crz(θ, t, [c])`         | 角度, target, control_list       | 受控 Rz         |
| `swap(q1, q2)`            | qubit_1, qubit_2                 | SWAP 门         |
| `rzz(θ, q1, q2)`         | 角度, qubit_1, qubit_2           | ZZ 旋转门       |
| `rxx(θ, q1, q2)`         | 角度, qubit_1, qubit_2           | XX 旋转门       |
| `ms_gate(θ, q1, q2)`     | 角度, qubit_1, qubit_2           | `rxx` 的别名  |
| `toffoli(t, [c0,c1,...])` | target, control_list             | 多控制 X 门     |
| `ccnot(t, [c0,c1,...])`   | target, control_list             | 同 toffoli      |
| `measure(q...)`           | qubit list                       | 线路内测量标记  |
| `reset(q...)`             | qubit list                       | 测量后重置为零态 |

`toffoli` / `ccnot` 的矩阵构造与逐门执行路径支持任意数量控制位，也支持门字典中的 `control_states`；导出为 OpenQASM `ccx` 时仍只适用于两个控制位。

门构造函数的**签名与参数顺序与旧版完全一致**，但返回值已由裸门字典升级为类型化 `Operation`（`measure(...)` / `reset(...)` 返回 `Measurement`）：

- **构造期校验**：量子比特下标、控制位/控制态长度，以及按 GateSpec 注册表检查目标比特数、参数个数与控制位要求（见 3.6 节），错误在调用处立即报出。
- **旧字典只读兼容**：`gate["type"]`、`.get("parameter")`、`in`、`dict(gate)`、迭代等读取照常可用，且与旧门字典可直接 `==` 比较；写入（`gate[...] = ...`）会抛 `TypeError`（对象不可变）。
- **`Circuit` 内部存储不变**：仍是门字典列表（`circuit.gates`），下游代码无需改动。

### 3.2 构建电路

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

同一种方式可容纳全部门类型。下面的例子覆盖单比特/旋转/通用单比特/受控/双比特旋转/SWAP/多控门，参数顺序与 3.1 节速查表一致：

```python
import math
from aicir import Circuit, rx, ry, u2, u3, crx, rzz, rxx, swap, toffoli

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
    n_qubits=3,
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

### 3.3 让 Circuit 作用于 State

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
- 若电路绑定了 `backend`，`unitary()` 与 `Measure.run(...)` 会优先沿该后端执行；对大线路通常比先在 CPU 组装再搬运更合适（见第 6.6 节）。

### 3.4 参数化量子线路

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
- `Parameter` 是符号占位符，不是自动微分张量；训练梯度可使用第 8 节的 parameter-shift 工具。
- 如果要使用 PyTorch autograd，可直接把 Torch 标量张量作为门参数，并调用 `Circuit.unitary(backend=GPUBackend(...))`。当前 `rx`/`ry`/`rz`/`u2`/`u3`、受控旋转门、`rzz`/`rxx` 和自定义 `unitary` 的 Torch 参数会保留计算图。
- 导出 QASM 前应先把所有符号参数绑定为数值。JSON 导出支持 `Parameter`、NumPy 标量/数组、复数和 Torch 张量数值；Torch 张量在 JSON 读回后会恢复为普通数值或列表，不会恢复为带计算图的 Tensor。

### 3.5 自定义 unitary、identity 与 Torch 自动微分

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

### 3.6 GateSpec 门元信息注册表

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

## 4. 量子测量

aicir 采用**统一测量模型**：线路内的 `measure`/`reset` 是操作序列中的正式操作，末端读出由 `tm`/`measure_qubits` 控制，二者可共存而非互斥。

`Measure` 对象绑定一个后端，`run()` 返回统一的 `Result` 对象。

### 4.1 运行接口与参数

```python
result = Measure(backend).run(
    circuit,
    shots=1,             # 采样次数；None/0 = exact 模式（覆盖 tm，不做末端测量）
    measure_qubits=None, # 显式末端读出比特，保留输入顺序；None = 全部比特
    snap=None,           # 记录完整态快照的操作下标列表
    tm=True,             # 是否在线路全部操作后执行末端测量
    sm="avg",            # 多轨迹聚合模式；当前仅支持 "avg"
    seed=None,           # 随机种子（用于复现）
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
| `measure_qubits` | `None` | `None` = 全部比特 `[0..n−1]`；`[]` = 不做末端测量，等价 `tm=False` |
| `snap` | `None` | 操作下标集合（0 起，`<` 操作总数 L），记录该操作完成后的完整态 |
| `tm` | `True` | `False` 时不执行末端测量（`final_state == state`）；exact 模式下被静默覆盖 |
| `sm` | `"avg"` | `"avg"`：多轨迹平均；`"shot"`/`"cond"` 待实现（传入报 `NotImplementedError`） |
| `seed` | `None` | 单个 `numpy.random` 种子，贯穿线路内 measure/reset/末端测量，相同参数可复现 |

### 4.2 线路与操作序列约定

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

### 4.3 线路内 measure：联合 Pauli 投影测量

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
r = m.run(cir, shots=None, tm=False)
print(r.output("zz"))      # 1（+1，必然）
print(r.state.array)       # [0.707+0j, 0, 0, 0.707+0j]（Bell 态，仍相干）

# shots=8：多轨迹，output(2) 形状 (8, 1)，全为 +1
r8 = m.run(cir, shots=8, tm=False)
print(r8.output(2))        # [[1],[1],[1],[1],[1],[1],[1],[1]]
print(r8.counts(2))        # {1: 8, -1: 0} 或 {1: 8}
```

### 4.4 reset：重置信道

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

### 4.5 末端测量

电路全部显式操作执行完后，`tm=True`（默认）时对 `measure_qubits` 所列比特执行**逐比特 Z 基**投影测量：

- **输入顺序保留**：`measure_qubits=[1, 0]` 时 `output(-1)` 列顺序为 `[qubit1, qubit0]`，不做内部排序。
- `output(-1)` 形状：`shots=1` 时 `(1, k)`，`shots=M` 时 `(M, k)`（`k=len(measure_qubits)`）。
- `measure_qubits=None`（默认）：读出全部 `n` 个比特。
- `measure_qubits=[]`：不做末端测量（等价 `tm=False`）。

```python
from aicir import Circuit, Measure, NumpyBackend, pauli_x
import numpy as np

# qubit1=|1>，qubit0=|0>；末端按 [1, 0] 顺序读出
cir = Circuit(pauli_x(1), n_qubits=2)
r = Measure(NumpyBackend()).run(cir, shots=1, measure_qubits=[1, 0])
print(r.output(-1))   # [[-1, 1]]  — qubit1=-1(|1>), qubit0=+1(|0>)
```

### 4.6 shots 语义

| `shots` | 模式 | 说明 |
|---|---|---|
| `None` 或 `0` | exact 模式 | 单条精确轨迹；覆盖 `tm`，不执行末端测量；`output(-1)` / `counts(i)` / `prob(i)` 报错 |
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

### 4.7 末态与约化：state / final_state / snap / reduce

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

### 4.8 期望值与 observables

```python
import numpy as np
from aicir import Circuit, Measure, NumpyBackend, pauli_x

Z = np.array([[1, 0], [0, -1]], dtype=complex)
cir = Circuit(pauli_x(0), n_qubits=1)   # |1> 态

result = Measure(NumpyBackend()).run(cir, shots=None, observables={"Z0": Z})
print(result.expectation_values)   # {'Z0': -1.0}
```

### 4.9 从 State 直接测量

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

### 4.10 Result 字段/方法速查

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
| `final_state` | `np.ndarray` | 末端测量后的态；`tm=False`/exact 时与 `state` 相同 |
| `terminal_qubits` | `list[int]` | 末端测量比特列表（按输入顺序） |
| `measurement_specs` | `list[MeasureSpec]` | 线路内 measure 操作登记（`op_index`/`id`/`qubits`/`basis`） |
| `expectation_values` | `dict` | `{name: float}` 期望值 |
| `expectation_variances` | `dict` | `{name: float}` 方差 |
| `shots` | `int or None` | 实际采样次数（None = exact 模式） |
| `n_qubits` | `int` | 电路比特数 |
| `metadata` | `dict` | `state_mode` 等辅助信息 |

### 4.11 Sampler / Estimator primitives（统一执行入口）

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

### 4.12 输入检查与报错

| 条件 | 抛出 |
|---|---|
| `shots < 0` 或非整数（含 `bool`） | `ValueError` |
| `measure_qubits` / `measure(...)` / `reset(...)` 含越界比特 | `ValueError` |
| 单个列表内有重复比特 | `ValueError` |
| 跨 measure 操作的 `id` 重复 | `ValueError` |
| `snap` 含越界操作下标 | `ValueError` |
| `tm=False` 且 `measure_qubits` 非空 | `ValueError` |
| `shots∈{None,0}` 且显式 `measure_qubits` 非空 | `ValueError` |
| `sm="shot"` 或 `"cond"` | `NotImplementedError`（待实现） |
| 产生混合态但后端不支持密度矩阵 | `ValueError` |

---

## 5. 构建哈密顿量

aicir 提供三个层次的算符构建工具：`PauliOp`、`PauliString`、`Hamiltonian`。

### 5.1 单算符 PauliOp

```python
from aicir import PauliOp, GPUBackend

backend = GPUBackend()

# Z 作用在 qubit 0，在 2 比特空间里展开为 4×4 矩阵
Z0 = PauliOp('Z', qubit=0)
mat = Z0.to_matrix(n_qubits=2, backend=backend)
print(mat.shape)   # torch.Size([4, 4])
```

### 5.2 多体泡利串 PauliString

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

### 5.3 哈密顿量 Hamiltonian

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

### 5.4 噪声通道（开放量子系统）

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

## 6. 计算后端的选择与使用

aicir 提供三种计算后端，都实现统一的 `Backend` 接口，可与 `Circuit` / `Measure` / `State` / `Hamiltonian` 无缝配合。它们的区别只在底层张量库与运行设备，业务代码无需改动即可切换：

| 后端                                    | 底层库                 | 运行设备                 | 自动微分 | 典型用途                                      |
| --------------------------------------- | ---------------------- | ------------------------ | -------- | --------------------------------------------- |
| `NumpyBackend`                        | NumPy                  | CPU                      | 否       | 小规模验证、算法原型、无 PyTorch 依赖环境     |
| `GPUBackend`（别名 `TorchBackend`） | PyTorch                | CPU / CUDA GPU           | 是       | 变分算法（VQE/QAOA/QML）、需要梯度或 GPU 加速 |
| `NPUBackend`                          | PyTorch +`torch_npu` | Ascend NPU（可回退 CPU） | 是       | 昇腾 NPU 上的仿真与训练                       |

> 命名说明：`TorchBackend` 是 `GPUBackend` 的**过时别名**，仅为向后兼容保留，新代码请使用 `GPUBackend`。`NPUBackend` 继承自 `GPUBackend`，复用其全部数学内核，并对 NPU 缺失的 `complex64` 算子做实部/虚部拆分回退（详见 6.4 / 6.5）。

三种后端的调用范式完全一致——构造一个后端实例，绑定到 `Circuit`（或传给 `Measure`）即可：

```python
from aicir import Circuit, Measure, NumpyBackend, GPUBackend, NPUBackend, hadamard, cnot

backend = GPUBackend(device="cpu")     # 换成 NumpyBackend() 或 NPUBackend() 完全等价
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name, result.counts)
```

> 后端可绑定在 `Circuit` 上，也可只传给 `Measure`；当两者都给定时以 `circuit.backend` 优先。推荐绑定到 `Circuit`，原因与逐门演化、减少主机↔设备搬运有关（详见 6.6）。

### 6.1 NumpyBackend（CPU 参考实现）

```python
from aicir import NumpyBackend

backend = NumpyBackend()               # 纯 CPU，默认 numpy complex64
backend = NumpyBackend(dtype="complex128")   # 可选：指定复数精度
```

- 依赖最少，无需安装 PyTorch。
- **不支持自动微分**，因此不能用于基于 autograd 的参数训练（如需梯度，请改用 `GPUBackend`，或配合第 8 节中参数移位 `psr` 等数值方法）。
- 适合教学、单元测试与小比特数验证。

### 6.2 GPUBackend（PyTorch，CPU / CUDA）

```python
import torch
from aicir import GPUBackend

backend = GPUBackend()                       # 默认设备：有 CUDA 用 cuda，否则 cpu
backend = GPUBackend(device="cpu")           # 强制 CPU
backend = GPUBackend(device="cuda:0")        # 指定某张 GPU
backend = GPUBackend(device=torch.device("cuda"), dtype=torch.complex128)  # 自定义设备与精度
```

构造参数：

| 参数       | 默认                              | 说明                                         |
| ---------- | --------------------------------- | -------------------------------------------- |
| `device` | 有 CUDA 用 `cuda`，否则 `cpu` | 接收字符串或 `torch.device`                |
| `dtype`  | `torch.complex64`               | 复数精度，可设 `torch.complex128` 提高精度 |

- **支持 PyTorch autograd**：把 Torch 标量张量作为门参数（`rx/ry/rz/u2/u3`、受控旋转、`rzz/rxx`、自定义 `unitary`）即可保留计算图，用于 VQE/QAOA/QML 训练（见 3.4 节与第 8 节）。
- **支持 CUDA GPU 加速**：把 `device` 指向 GPU 即可，门矩阵构造与态演化都在 GPU 上完成。

### 6.3 NPUBackend（Ascend NPU）

```python
from aicir import NPUBackend

backend = NPUBackend()                                       # 自动选 npu:0（不可用则回退 CPU）
backend = NPUBackend(device="npu:1")                         # 指定某张 NPU 卡
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)  # 严格模式：NPU 不可用直接报错
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)  # 多卡：按 LOCAL_RANK 自动绑卡
```

构造参数：

| 参数                | 默认                | 说明                                                                      |
| ------------------- | ------------------- | ------------------------------------------------------------------------- |
| `device`          | 自动 `npu:0`      | 目标 NPU 设备                                                             |
| `dtype`           | `torch.complex64` | 复数精度                                                                  |
| `fallback_to_cpu` | `True`            | NPU 不可用时是否回退 CPU；`False` 则抛 `RuntimeError`（用于平台验证） |

- 继承自 `GPUBackend`，API 完全一致，可直接替换其它后端。
- 依赖 `torch_npu`；在真正的 NPU 设备上会自动对缺失的 `complex64` 内核做兼容回退（见 6.4 / 6.5）。
- 严格模式见 6.7，多卡/分布式见 6.9，端到端示例见 6.10。

> 在 QAS supernet（`aicir/qas`）中无需手动构造后端：只要把 `device="npu:0"` 传入配置，框架会自动选用 `NPUBackend`（见 `aicir/qas/supernet.py` 的 `_make_backend`）；CPU/CUDA 设备则用 `GPUBackend`。

### 6.4 NPU 兼容性概述（complex64）

Ascend NPU 在不同版本的 `torch_npu` 组合下，对 `complex64` 的内核支持并不完整，某些复数算子会直接报错，例如：

- `aclnnMatmul ... DT_COMPLEX64 not implemented`
- `aclnnEye ... DT_COMPLEX64 not implemented`
- `aclnnAdd ... DT_COMPLEX64 not implemented`

`NPUBackend` 在后端层提供 NPU 专用兼容路径（workaround），核心思路：优先走后端抽象接口（`matmul/kron/trace/...`）而非业务层直接做 torch 复数运算；在 NPU 且输入为复数时，将计算拆成实部/虚部后重组，绕过缺失内核；并对常见初始化路径（如 `eye`、`|0...0>`）提供 NPU 安全实现。当前已覆盖的高频兼容算子包括：

- `matmul`, `apply_unitary`
- `kron`
- `dagger`, `trace`
- `inner_product`, `partial_trace`
- `expectation_sv`, `expectation_dm`
- `abs_sq`, `measure_probs`
- `eye`, `zeros_state`

此外，门矩阵构造层（`aicir/core/gates.py`）也避免对复数张量直接调用 `torch.exp`（`rz/rzz/u2/u3` 改用 `cos + i·sin` 构造），同样是为了绕过 NPU 缺失的复数内核。为保证**反向**同样可用（见下），参数化门矩阵的每个含梯度单元都构造为**独立**的复数张量、且不做复数乘法：早期 `rzz`/`rxx` 把同一个复数相位张量放进矩阵的多个位置，autograd 在累加其梯度时会触发 `aclnnAdd ... DT_COMPLEX64`；`u2`/`u3` 用复数乘法 `exp(i·x)·sin` 会触发 `aclnnMul`。现改为按实/虚部直接拼装、各单元互不复用，梯度累加因此落在**实数角度**上（NPU 支持的实数加法）。注意：这不代表 NPU 对所有复数算子都原生可用——若新增路径中出现“直接 torch 复数加减乘”，仍可能触发新报错（排查方法见 6.5）。

> **训练（autograd backward）**：`complex64` 的**反向**内核同样缺失——朴素实现中，复数张量的梯度累加/相乘会触发 `aclnnAdd` / `aclnnMul ... DT_COMPLEX64`，导致 `loss.backward()` 直接报错。要让 `loss.backward()` 在 NPU 上跑通，需要保证**整条反向路径里没有任何复数加/乘**，为此做了两层处理：
>
> 1. **线性代数层**：`matmul` 与 `expectation_sv` 用自定义 `torch.autograd.Function`（`_NpuMatmulFn` / `_NpuExpectationFn`）封装，实/虚部拆分只发生在 Function 的 forward/backward **内部**（不进入 autograd 计算图），因此计算图里只剩“线性使用”的复数节点，反向时不会发生复数梯度累加。
> 2. **门矩阵构造层**：如上一段所述，`rzz/rxx/u2/u3` 等参数化门按实/虚部独立拼装、各单元不复用、不做复数乘法，使来自门矩阵的复数梯度也不会触发复数加/乘。
>
> 两者配合后，梯度与原生复数 autograd 数值一致（已在 CPU 上以“禁用复数加/乘”的 dispatch 模式模拟 NPU 限制对齐验证，见 `tests/backends/test_npu_backend.py`），所以**在 NPU 上可以直接用 `loss.backward()` 训练**，QAS supernet 也因此走标准 autograd 路径（比参数移位快）。
>
> 仍可选用参数移位规则（`SupernetConfig(use_parameter_shift=True)`）作为对照或后备；它仅前向计算，对 `rx/ry/rz/rzz` 这类 Pauli 旋转门是精确梯度，但每个参数每步需 2 次前向求值，通常更慢。若自行在 NPU 上写训练循环并直接对复数张量做运算，请确保经过 `NPUBackend` 的封装方法（`matmul`/`expectation_sv` 等），否则裸的复数 backward 仍会报错。

### 6.5 NPU complex64 问题详解（建议先读）

#### 6.5.1 根因

- 问题不在量子算法本身，而在底层内核支持矩阵。
- 同样的 Python 代码在 CPU/CUDA 可运行，不代表在 NPU 复数路径可运行。

#### 6.5.2 典型触发点

- 前端构造电路矩阵时，直接对复数张量做 `+`、`*`、某些初始化操作。
- 绕过 `Backend` 接口，直接调用 torch 复数运算。

#### 6.5.3 处理原则

- 不需要把整个项目都改成“处处手工拆实虚部”。
- 只需要确保“在 NPU 上实际执行的复数运算”都经过后端封装或 NPU 专用回退。
- 若出现新报错，按栈定位到具体算子点，再做最小修复。

#### 6.5.4 快速排查清单

- 检查报错是否包含 `DT_COMPLEX64 not implemented`。
- 检查报错栈是否位于后端层之外（例如业务文件里直接做了 torch 复数加法）。
- 优先改为调用 `backend` 方法，必要时在 `NPUBackend` 增加拆分回退。

### 6.6 推荐方式：在 `Circuit` 绑定后端（也可在 `Measure` 指定）

推荐在构建电路时把目标后端绑定到 `Circuit`（即在 `Circuit(..., backend=...)` 或随后调用 `bind_backend()`）。

示例：

```python
from aicir import Circuit, Measure, NPUBackend, hadamard, cnot

backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

# Measure 也可以接收 backend，但会被 circuit.backend 优先覆盖
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name)
```

要点说明：

- **可以在两处指定 backend**：`Circuit` 或 `Measure` 都支持传入后端。
- **优先级**：`Measure` 会优先使用 `circuit.backend`（若存在），否则使用 `Measure` 自身的后端（见 `Measure._resolve_backend` 的实现）。因此将后端绑定到 `Circuit` 能避免回退到主机端拼装或与 Measure 中传入后端的混淆。
- **为什么推荐绑定到 `Circuit`**：当电路具有 `gates` 时，`Measure` 会逐门调用 `gate_to_matrix(..., backend=resolved_backend)` 在目标设备上构造并作用门矩阵，从而减少构造完整 2^n×2^n 矩阵的内存与主机→设备搬运；若 `unitary(backend=...)` 不被支持则会回退到无 backend 的 `unitary()`（在 CPU 上用 numpy 拼装整矩阵），然后再 `backend.cast` 到设备，这会引起大规模数据搬运。对于 `GPUBackend`，参数化门的 Torch 标量张量会通过 torch 运算构造矩阵，从而保留 autograd 计算图。

关于将多个电路合并（拼接）时的 backend 确定：

- 使用 `+` 操作符拼接两个 `Circuit`（`a + b`）时，新电路会按实现选择后端：优先采用左侧电路的 backend（`a._backend`）；若左侧没有，则采用右侧的 backend（`b._backend`）。这与 `Circuit.__add__` 的实现一致。
- 因此，若要把多个原本绑定到不同后端的 `Circuit` 连接成一个整体并在统一设备上运行，应在拼接后或拼接前显式统一后端：

```python
# 推荐做法：拼接后显式设置统一后端
full = part_a + part_b
full.bind_backend(common_backend)
result = Measure(common_backend).run(full)
```

- 如果不显式统一后端，拼接结果会继承左侧电路的 backend（若左侧没有则用右侧），这可能不是预期且可能导致在运行时出现回退或不一致的行为。

小结：将后端绑定到 `Circuit` 并在合并后或合并前统一后端，是既安全又高效的做法。

- 构建阶段只保存门描述: 调用 `hadamard(0)` 等构造的是门的描述字典（例如 `{"type": "hadamard", "target_qubit": 0}`），`Circuit.__init__` 只是把这些描述存起来，并不会在构建时把门转换成数值矩阵。
- 当前执行策略: `Measure.run`/`run_density_matrix` 在电路对象具备 `gates` 序列时，会优先走“逐门演化”路径（按门依次作用到态/密度矩阵），而不是先组装整条电路的全局矩阵后再一次性作用。
- 矩阵在组装时生成: 真正把门变为 2^n×2^n 的数值矩阵发生在调用 `Circuit.unitary(backend=...)` 或 `Measure` 等需要数值矩阵的地方。此时会调用 `gate_to_matrix(gate, cir_qubits, backend)` 来生成每个门的矩阵。
- backend 参数的作用: 当 `backend=None` 时，`gate_to_matrix` 会走 numpy 路径（例如调用 `_hadamard()` 等函数，在 CPU 上生成矩阵）；当传入 `backend` 时，`gate_to_matrix` 会使用后端分支（先构造 base 矩阵再通过 `_single_qubit_from_base_backend`/`_controlled_from_base_backend` 等路径调用 `backend.cast`、`backend.kron`、`backend.matmul` 等接口），从而在目标后端（CPU/GPU/NPU）上构造和组合张量。`rx`/`ry`/`rz`/`u2`/`u3`、受控旋转、`rzz`/`rxx` 和自定义 `unitary` 可在 `GPUBackend` 下保留 Torch 参数的梯度链路。
- 兼容回退路径: 若电路对象不提供 `gates` 序列，`Measure` 仍会回退到 `unitary()` 路径以兼容外部实现。
- 可能的设备搬运: 在 `unitary()` 回退路径中，`Measure` 现在优先直接 `backend.cast(unitary_raw)`，避免无必要的 `to_numpy` 主机往返。
- 性能建议: 对大 qubit 数，显式组装全矩阵会占用大量内存并产生迁移成本。若要最小化搬运，优先在构建时绑定后端（本节方式 B），或改为按门逐步在态上直接作用（逐门 apply），避免生成完整 2^n×2^n 矩阵；若需要彻底避免中间拷贝，可考虑修改 `Measure` 中的 `to_numpy` 使用点或直接在后端上逐门演化。

也可先构建再绑定：

```python
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
cir.bind_backend(backend)
```

适用场景：

- 你希望前端矩阵组装与执行严格在同一设备上
- 希望减少 CPU 和 XPU 之间的数据迁移

### 6.7 严格 NPU 模式（不允许回退）

```python
from aicir import NPUBackend

# NPU 不可用时直接抛 RuntimeError，用于验证平台
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)
```

### 6.8 运行示例

仓库示例脚本：`demos/demo_npu.py`

```bash
python demos/demo_npu.py
python demos/demo_npu.py --shots 2048 --allow-cpu-fallback
```

### 6.9 分布式环境（多卡/多节点）

使用环境变量 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 自动绑定对应卡：

```python
from aicir import NPUBackend

# 自动读取 LOCAL_RANK 决定 npu:LOCAL_RANK
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
print(backend.runtime_context)
# NPURuntimeContext(world_size=4, rank=0, local_rank=0, distributed=True)
```

启动方式：

```bash
# 单卡验证
python demos/demo_npu.py

# 允许 CPU 回退（本地调试用）
python demos/demo_npu.py --allow-cpu-fallback

# 多卡分布式启动
torchrun --nproc_per_node=4 your_script.py
```

### 6.10 完整端到端示例

```python
import math
from aicir import NPUBackend, Circuit, Measure, hadamard, cnot, rz

# 1. 构建后端
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

# 2. 构建电路
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

# 3. 测量
measure = Measure(backend)
result = measure.run(cir, shots=2048)

print(f"backend : {backend.name}")
print(f"probs   : {result.probabilities}")
print(f"counts  : {result.counts}")
print(f"summary : {result.summary()}")
```

### 6.11 runtime_context 字段说明

| 字段            | 说明                                      |
| --------------- | ----------------------------------------- |
| `world_size`  | 总进程数                                  |
| `rank`        | 全局进程编号                              |
| `local_rank`  | 本节点本地编号（对应 `npu:local_rank`） |
| `distributed` | `world_size > 1` 时为 True              |

### 6.12 远程 NPU 验证输出示例（新路径）

使用新路径 smoke 脚本进行全链路验证（单门、受控门、参数门、density matrix）：

```bash
python tests/smoke_npu_new_path.py --shots 512
```

示例输出：

```text
=== Smoke NPU New Path ===
backend: NPUBackend(dtype=torch.complex64, device=npu:0, npu_available=True)
runtime_context: NPURuntimeContext(world_size=1, rank=0, local_rank=0, distributed=False)
[PASS] single_gate
[PASS] controlled_gate
[PASS] parametric_gate
[PASS] density_matrix

Summary: PASS
```

### 6.13 BatchSV（批量态矢量路径，NPU 安全且可微）

`aicir` 主路径（`State` / `apply_gate_to_state`）一次只演化单个 `(2^n, 1)` 态矢量，且门参数为标量。深度学习场景（例如把变分量子线路当作神经网络的一层）需要：一次模拟一批态矢量；旋转门角度可逐样本（per-sample）不同（如数据编码角度依赖输入）；端到端可微（autograd）；以及 **Ascend NPU 安全**——NPU 缺少 `complex64` 的 `aclnnAdd` / `aclnnMul` 内核，因此 `BatchSV` 全程以实部/虚部两个实张量表示，只用实数乘加，反向传播不会触发复数累加。

`BatchSV`（规范路径 `aicir.core`，也可从顶层 `aicir` 导入）即为该批量路径。门矩阵与单态路径采用同一套定义（复用 `_single_qubit_base_for_gate`），保持单一事实来源；量子比特端序也与主路径一致（**qubit 0 为最高位**）。需要 `torch`。

```python
import torch
from aicir import BatchSV, GPUBackend, hadamard, ry, crz

backend = GPUBackend()                      # 或 NPUBackend；用于获取目标 device / dtype
bsv = BatchSV(n_qubits=3, batch_size=8, backend=backend)

# 逐样本数据编码角度：ry 的参数可以是 (batch,) 张量，逐样本不同。
enc = torch.randn(8, 3, requires_grad=True)
theta = torch.zeros(3, requires_grad=True)  # 0 维/标量参数按 batch 广播

for q in range(3):
    bsv.apply_gate(hadamard(q))
    bsv.apply_gate(ry(enc[:, q], q))        # 逐样本角度
bsv.apply_gate(crz(theta[0], 1, [0]))       # 受控旋转门同样支持
bsv.apply_gate(ry(theta[1], 2))

z = bsv.z_expectations()                     # (batch, n_qubits) 逐比特 <Z_q>
probs = bsv.probabilities()                  # (batch, 2^n) 基测量概率

loss = z.sum()
loss.backward()                              # 端到端可微，全程实张量
```

逐样本张量角度目前支持 `rx` / `ry` / `rz` 及其受控形式 `crx` / `cry` / `crz`；常量门与标量参数门复用单态路径定义，覆盖范围与之一致。`apply_gate` 返回自身以便链式调用。

---

## 7. 与 OpenQASM 2.0 / 3.0 互转

### 7.1 Circuit → QASM 字符串

```python
from aicir import Circuit, hadamard, cnot, rz, circuit_to_qasm, circuit_to_qasm3
import math

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

# 导出 OpenQASM 2.0
qasm2 = circuit_to_qasm(cir, version="2.0")
print(qasm2)
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# h q[0];
# cx q[0],q[1];
# rz(pi/4) q[1];

# 导出 OpenQASM 3.0
qasm3 = circuit_to_qasm3(cir)
print(qasm3)
```

### 7.2 QASM 字符串 → Circuit

```python
from aicir import circuit_from_qasm

qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
rz(pi/4) q[1];
"""

cir = circuit_from_qasm(qasm_str)
print(cir)          # Circuit(n_qubits=2, gates=[...])
print(cir.unitary().shape)  # (4, 4)
```

### 7.3 读写 QASM 文件

```python
from aicir import (
    save_circuit_qasm, load_circuit_qasm,
    save_circuit_qasm3,
)

# 保存为 .qasm 文件（OpenQASM 2.0）
save_circuit_qasm(cir, "my_circuit.qasm")

# 保存为 OpenQASM 3.0
save_circuit_qasm3(cir, "my_circuit_v3.qasm")

# 从文件读取（自动识别 2.0/3.0 版本头）
cir2 = load_circuit_qasm("my_circuit.qasm")
```

### 7.4 OpenQASM 3.0 格式差异

aicir 在 3.0 模式下的主要差异：

| 项目           | 2.0                       | 3.0                         |
| -------------- | ------------------------- | --------------------------- |
| 版本头         | `OPENQASM 2.0;`         | `OPENQASM 3.0;`           |
| 标准库         | `include "qelib1.inc";` | `include "stdgates.inc";` |
| 量子寄存器声明 | `qreg q[2];`            | `qubit[2] q;`             |
| U3 门名称      | `u3(θ,φ,λ)`          | `u(θ,φ,λ)`             |

### 7.5 支持的 QASM 门集

| QASM 门名                          | aicir 对应函数                        |
| ---------------------------------- | ------------------------------------- |
| `x`, `y`, `z`                | `pauli_x`, `pauli_y`, `pauli_z` |
| `h`                              | `hadamard`                          |
| `s`, `t`                       | `s_gate`, `t_gate`                |
| `rx`, `ry`, `rz`             | `rx`, `ry`, `rz`                |
| `rzz(θ)`                        | `rzz`                               |
| `rxx(θ)`                        | `rxx`                               |
| `u2(φ,λ)`                      | `u2`                                |
| `u3(θ,φ,λ)` / `u(θ,φ,λ)` | `u3`                                |
| `cx`                             | `cx` / `cnot`                     |
| `cy`, `cz`                     | `cy`, `cz`                        |
| `swap`                           | `swap`                              |
| `crx`, `cry`, `crz`          | `crx`, `cry`, `crz`             |
| `ccx`                            | `toffoli` / `ccnot`               |

> **当前不支持**：`if`、`reset`、`opaque`、自定义 `gate`、`cp`/`cu` 系列门。
> `measure` 和 `barrier` 语句在导入时会被跳过。

### 7.6 与 Qiskit 互转

`qiskit` 是可选依赖：导入 `aicir` 不要求安装 Qiskit，只有调用互转函数时才会检查。

```python
import math
from qiskit import QuantumCircuit
from aicir import Circuit, cnot, hadamard, rz, circuit_from_qiskit, circuit_to_qiskit

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

qc = circuit_to_qiskit(cir)      # aicir Circuit -> qiskit.QuantumCircuit
cir2 = circuit_from_qiskit(qc)   # qiskit.QuantumCircuit -> aicir Circuit

qc2 = QuantumCircuit(2)
qc2.h(0)
qc2.cx(0, 1)
cir3 = circuit_from_qiskit(qc2)
```

也可以使用 `NEXT.md` 第 8 节中的短别名：

```python
from aicir import to_qiskit, from_qiskit

qc = to_qiskit(cir)
cir2 = from_qiskit(qc)
```

当前 Qiskit 互操作支持基础单比特门、参数旋转门、`u2`/`u3`、`cx/cy/cz`、`crx/cry/crz`、`swap`、`rzz/rxx`、`ccx` 和线路内 `measure` 标记；暂不支持 Qiskit 自定义门和未绑定符号参数。

### 7.7 与 PennyLane 互转

`pennylane` 是可选依赖：导入 `aicir` 不要求安装 PennyLane，只有调用互转函数时才会检查。

```python
import math
import pennylane as qml
from aicir import Circuit, cnot, hadamard, rz, circuit_from_pennylane, circuit_to_pennylane

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

script = circuit_to_pennylane(cir)        # aicir Circuit -> pennylane QuantumScript
cir2 = circuit_from_pennylane(script)     # pennylane QuantumScript -> aicir Circuit

script2 = qml.tape.QuantumScript(
    [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])],
    [],
)
cir3 = circuit_from_pennylane(script2)
```

也可以使用 `NEXT.md` 第 8 节中的短别名：

```python
from aicir import to_pennylane, from_pennylane

script = to_pennylane(cir)
cir2 = from_pennylane(script)
```

当前 PennyLane 互操作支持基础单比特门、参数旋转门、`u2`/`u3`、`cx/cy/cz`、`crx/cry/crz`、`swap`、`rzz/rxx`（对应 PennyLane 的 `IsingZZ`/`IsingXX`）、`ccx` 和 `identity`。暂不支持 PennyLane 自定义门、未绑定符号参数和 aicir 线路内 `measure` 标记。

### 7.8 与 WuYue 互转

`wuyue` 是可选依赖：导入 `aicir` 不要求安装 WuYue SDK，只有调用互转函数时才会检查。

```python
import math
from wuyue.circuit.circuit import QuantumCircuit
from wuyue.element.gate import CX, H
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.register.quantumregister import QuantumRegister
from aicir import Circuit, cnot, hadamard, rz, circuit_from_wuyue, circuit_to_wuyue

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

wc = circuit_to_wuyue(cir)      # aicir Circuit -> WuYue QuantumCircuit
cir2 = circuit_from_wuyue(wc)   # WuYue QuantumCircuit -> aicir Circuit

qreg = QuantumRegister(2)
creg = ClassicalRegister(1)
wc2 = QuantumCircuit(qreg, creg)
wc2.add(H, qreg[0])
wc2.add(CX, qreg[1], control=qreg[0])
cir3 = circuit_from_wuyue(wc2)
```

也可以使用 `NEXT.md` 第 8 节中的短别名：

```python
from aicir import to_wuyue, from_wuyue

wc = to_wuyue(cir)
cir2 = from_wuyue(wc)
```

当前 WuYue 互操作支持基础单比特门、参数旋转门、`u2`/`u3`、`cx/cz`、`swap`、`rzz`（对应 WuYue `IsingZZ`）、`ccx`、`identity` 和线路内 `measure` 标记。受 WuYue 当前原生门集限制，暂不转换 `cy`、`crx/cry/crz`、`rxx`、自定义门和未绑定符号参数。

---

## 8. QML 梯度工具

`aicir.qml.deriv` 提供面向量子机器学习和变分量子线路的梯度与 gradient-free 工具。常用函数可直接从 `aicir.qml` 导入：

```python
from aicir.qml import psr, spsr, multipsr
```

这些函数都假设目标函数 `fn(params)` 返回标量，`params` 可以是标量或任意形状的 NumPy 数组。

### 8.1 标准 parameter-shift rule：`psr`

`psr(fn, params)` 对每个参数计算：

```text
0.5 * [fn(theta + pi/2) - fn(theta - pi/2)]
```

默认系数 `0.5` 和位移 `pi/2` 适用于常见 Pauli 旋转生成元。

```python
import numpy as np
from aicir.qml import psr

params = np.array([0.3, -0.4])

def loss(theta):
    return np.cos(theta[0]) + np.sin(theta[1])

grad = psr(loss, params)
print(grad)  # [-sin(0.3), cos(-0.4)]
```

结合参数化电路使用：

```python
import numpy as np
from aicir import Circuit, NumpyBackend, Parameter, State, ry
from aicir.qml import psr

theta = Parameter("theta")
template = Circuit(ry(theta, 0), n_qubits=1)
backend = NumpyBackend()
Z = np.diag([1.0, -1.0])

def expectation(values):
    circuit = template.bind_parameters({"theta": values[0]})
    state = State.zero_state(1, backend).evolve(circuit.unitary()).to_numpy().reshape(-1)
    return np.real(np.vdot(state, Z @ state))

grad = psr(expectation, np.array([0.5]))
```

### 8.2 stochastic parameter-shift rule：`spsr`

`spsr` 随机抽样部分参数坐标，只对抽中的参数做 shift 评估。默认 `unbiased=True`，会按参数总数和采样数缩放，使估计量在期望上等于完整 `psr` 梯度。

```python
from aicir.qml import spsr

grad_est = spsr(
    loss,
    params,
    n_samples=1,
    rng=42,
)
```

常用参数：

| 参数            | 说明                                  |
| --------------- | ------------------------------------- |
| `n_samples`   | 每次估计抽样的参数坐标数量            |
| `rng`         | 随机种子或 NumPy generator            |
| `replace`     | 是否允许重复抽样；默认 `False`      |
| `unbiased`    | 是否缩放为无偏估计；默认 `True`     |
| `shift`       | 参数位移；默认 `np.pi / 2`          |
| `coefficient` | shifted difference 系数；默认 `0.5` |

### 8.3 multivariate parameter-shift rule：`multipsr`

`multipsr` 用多参数符号求和公式计算选定参数的混合偏导。例如 `parameter_indices=[0, 1]` 表示计算：

```text
d² fn / d theta[0] d theta[1]
```

```python
from aicir.qml import multipsr

def objective(theta):
    return np.cos(theta[0]) * np.sin(theta[1])

mixed = multipsr(objective, np.array([0.4, -0.2]), parameter_indices=[0, 1])
print(mixed)  # -sin(theta[0]) * cos(theta[1])
```

对于多维参数数组，可使用 tuple index：

```python
params = np.array([[0.4, 0.1], [-0.2, 0.3]])

def objective_2d(theta):
    return np.cos(theta[0, 0]) * np.sin(theta[1, 0])

mixed = multipsr(objective_2d, params, parameter_indices=[(0, 0), (1, 0)])
```

如果省略 `parameter_indices`，`multipsr` 会对所有参数计算一个全参数混合偏导，此时需要 `2 ** params.size` 次函数调用，参数数量较大时成本会很高。

### 8.4 VQC 中的使用

`aicir.vqc` 中已有的 `BasicVQE.parameter_shift_gradient()`、`BasicSSVQE.parameter_shift_gradient()` 和 `BasicVQD.parameter_shift_gradient()` 已统一调用 `aicir.qml.deriv.psr`。因此自定义 QNN/VQC 模型时也建议复用 `psr`、`spsr` 和 `multipsr`，避免各模块重复实现 parameter-shift 逻辑。

---

## 9. 可视化模块

`aicir.visual` 提供第一阶段的轻量可视化工具，用于查看量子线路、量子态概率/振幅和密度矩阵。文本线路图与门统计不依赖额外图形库；绘图函数会在调用时按需导入 `matplotlib`。matplotlib 图中的 `reset` 会以与 `measure` 同色、标注 `Reset` 的虚线显示；`Reset` 字号与 `Rz` 门主标签一致，虚线 dash 间距更大。虚线连接 `measure(q)` 与后续同一比特量子门，没有后续量子门时延伸到线路末端，且不再叠加普通实线。若后续门没有完整方框，reset 虚线按虚拟方框左边界停止，虚拟方框内用普通实线。

```python
import numpy as np
from aicir import Circuit, hadamard, cnot, rzz
from aicir.visual import (
    circuit_to_text,
    draw_circuit,
    gate_histogram,
    plot,
    plot_state_probs,
    plot_state_amplitudes,
    plot_density_matrix,
)

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rzz(np.pi / 2, 0, 1),
    n_qubits=2,
)

print(circuit_to_text(cir))
print(gate_histogram(cir))  # {'cx': 1, 'hadamard': 1, 'rzz': 1}

# 保存彩色线路图。无 path 时，默认保存到调用它的 .py 文件所在目录，
# 文件名为 <脚本名>_<电路变量名>.png，例如 demos/demo_1.py 中的
# cir.plot() 会保存为 demos/demo_1_cir.png。
fig, ax = cir.plot()

# 显式相对路径也以调用它的 .py 文件所在目录为基准，而不是命令行 cwd。
# 例如在 demos/demo_1.py 中会保存为 demos/figures/h2.png。
fig, ax = cir.plot("figures/h2")

# 也可以继续使用函数式入口。
fig, ax = plot(cir, "figures/h2_function")

# draw_circuit 默认返回文本；output="mpl" 时返回 matplotlib 的 (fig, ax)
diagram = draw_circuit(cir)
fig, ax = draw_circuit(cir, output="mpl")

# 态向量或概率向量均可输入
state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
fig, ax = plot_state_probs(state)
fig, ax = plot_state_amplitudes(state)

# 密度矩阵热力图，part 可取 "abs"、"real"、"imag"、"phase"
rho = np.outer(state, np.conjugate(state))
fig, ax = plot_density_matrix(rho, part="abs")
```

当前已实现的公共函数：

- `circuit_to_text(circuit)`：返回 ASCII 线路图
- `draw_circuit(circuit, output="text" | "mpl")`：统一线路图入口
- `gate_histogram(circuit)`：按门 `type` 统计数量
- `plot(circuit, path=None, ...)` / `Circuit.plot(path=None, ...)`：保存彩色线路图，返回 `(fig, ax)`；未提供 `path` 时默认保存到调用它的 `.py` 文件所在目录
- `plot_state_probs(state_or_probs)`：绘制计算基概率柱状图
- `plot_state_amplitudes(state)`：绘制振幅实部、虚部和模长
- `plot_state_phase(state)`：绘制振幅相位
- `plot_density_matrix(rho, part=...)`：绘制密度矩阵热力图
- `plot_density_real_imag(rho)`：并排绘制密度矩阵实部和虚部

### 9.1 QAS 与 metrics 可视化

`aicir.visual` 也可以直接消费 `aicir.qas` 的 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec`，用于对比架构搜索候选、查看四类 objective group 分数，以及把线路图和指标放在一个 summary 图中。

```python
from aicir.backends.numpy_backend import NumpyBackend
from aicir.qas import ArchitectureSearch, SearchConfig
from aicir.visual import (
    qas_scores_to_rows,
    plot_search_history,
    plot_architecture_metrics,
    compare_architectures,
    plot_qas_summary,
)

backend = NumpyBackend()
search = ArchitectureSearch(backend=backend)
result = search.run(SearchConfig(n_qubits=3, candidate_layers=1, n_samples=8))

# 转成扁平行，便于打印、保存或接入其它分析流程
rows = qas_scores_to_rows(result)
print(rows[0])

# 按 rank 展示候选架构的 weighted score 和分组指标
fig, ax = plot_search_history(
    result,
    metrics=["weighted_score", "expressibility", "trainability", "hardware_efficiency"],
)

# 对比多个候选
fig, ax = compare_architectures(
    result.scores[:5],
    metrics=["weighted_score", "n_gates", "two_qubit_gate_count"],
)

# 查看单个候选的指标，或生成带线路图的 summary
best = result.best
fig, ax = plot_architecture_metrics(best)
fig, axes = plot_qas_summary(best, metrics=["weighted_score", "trainability", "hardware_efficiency"])
```

QAS/metrics 相关公共函数：

- `qas_scores_to_rows(data)`：将 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec` 或 dict 记录转为扁平行
- `plot_search_history(history, metrics=None, x=None)`：绘制搜索过程或候选 ranking 的指标曲线
- `plot_architecture_metrics(item, metrics=None)`：绘制单个候选的指标条形图
- `compare_architectures(data, metrics=None)`：对比多个候选架构或评分
- `plot_qas_summary(item, metrics=None)`：左侧线路图、右侧指标图的组合视图

---

## 10. 子包说明索引

`aicir` 子目录中还包含更具体的说明文档：

| 子目录                      | 说明文档                                                                | 内容概要                                                                                   |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `aicir/chemistry`         | [`aicir/chemistry/README.md`](aicir/chemistry/README.md)                 | 小型固定设置的分子 qubit Hamiltonian 预置，用于 VQE 示例、单元测试和算法原型验证。         |
| `aicir/core/io`           | [`aicir/core/io/README.md`](aicir/core/io/README.md)                     | OpenQASM 导出行为、受控旋转门和多控旋转门分解规则。                                        |
| `aicir/gates`             | [`aicir/gates/README.md`](aicir/gates/README.md)                         | GateSpec 门元信息注册表：目标比特数/参数个数/别名/QASM 名/绘图符号的单一来源。             |
| `aicir/metrics`           | [`aicir/metrics/README.md`](aicir/metrics/README.md)                     | 任务无关的量子线路评分指标，供 QAS、VQE ansatz 筛选等架构层任务复用。                      |
| `aicir/optimization/qubo` | [`aicir/optimization/qubo/README.md`](aicir/optimization/qubo/README.md) | QUBO 建模、Ising/Hamiltonian 转换、BasicQAOA 矩阵入口与结果解码。                          |
| `aicir/optimizer`         | [`aicir/optimizer/README.md`](aicir/optimizer/README.md)                 | `aicir.optimizer.circuit` 的线路化简、旋转门合并和固定点优化策略。                       |
| `aicir/primitives`        | [`aicir/primitives/README.md`](aicir/primitives/README.md)               | Sampler/Estimator primitives 统一执行入口与 `SampleResult`/`EstimateResult` 结果对象。 |
| `aicir/qas`               | [`aicir/qas/README.md`](aicir/qas/README.md)                             | 量子架构搜索模块、统一入口、配置工厂和各 QAS 方法说明。                                    |
| `aicir/qml`               | [`aicir/qml/README.md`](aicir/qml/README.md)                             | 量子机器学习梯度工具，包括参数移位、有限差分、伴随微分和自动微分等方法。                   |
| `aicir/transpile`         | [`aicir/transpile/README.md`](aicir/transpile/README.md)                 | 线路编译与优化流水线，包含 `PassManager` 和本地线路化简 pass。                           |
| `aicir/vqc`               | [`aicir/vqc/README.md`](aicir/vqc/README.md)                             | VQE、QAOA、VQD、SSVQE 等基础变分算法实现，以及可复用的参数化线路 ansatz 模板。             |
| `demos`                   | [`demos/README.md`](demos/README.md)                                     | 演示 `aicir.visual` 模块的示例脚本，涵盖线路、态向量、密度矩阵和 QAS 结果可视化。        |

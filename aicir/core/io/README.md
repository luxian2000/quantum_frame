# aicir.core.io

量子线路的序列化、反序列化与第三方框架互操作。

## 目录

| 文件 | 说明 |
| --- | --- |
| `qasm.py` | OpenQASM 2.0 / 3.0 导入导出 |
| `qiskit_io.py` | Qiskit `QuantumCircuit` 互转（可选依赖） |
| `pennylane_io.py` | PennyLane `QuantumScript` 互转（可选依赖） |
| `wuyue_io.py` | WuYue `QuantumCircuit` 互转（可选依赖） |
| `json_io.py` | Circuit 的 JSON 序列化与反序列化 |
| `dag.py` | Circuit → DAG 图表示（节点特征 + 邻接矩阵） |

> **依赖策略**：导入 `aicir` 不会强制导入 Qiskit / PennyLane / WuYue SDK。只有在调用对应的互转函数时才会检查并导入第三方库。

---

## 1  OpenQASM 互操作

### 1.1  公共 API

| 函数 | 说明 |
| --- | --- |
| `circuit_to_qasm(circuit, version="2.0")` | Circuit → OpenQASM 字符串 |
| `circuit_to_qasm3(circuit)` | Circuit → OpenQASM 3.0 字符串（等价于 `circuit_to_qasm(circuit, version="3.0")`） |
| `circuit_from_qasm(qasm_text)` | OpenQASM 字符串 → Circuit（自动识别 2.0 / 3.0） |
| `save_circuit_qasm(circuit, path, version="2.0")` | 保存为 `.qasm` 文件 |
| `save_circuit_qasm3(circuit, path)` | 保存为 OpenQASM 3.0 文件 |
| `load_circuit_qasm(path)` | 从 `.qasm` 文件加载（自动识别版本） |

### 1.2  支持的门集

| QASM 门名 | aicir 对应函数 |
| --- | --- |
| `x` / `y` / `z` | `pauli_x` / `pauli_y` / `pauli_z` |
| `h` | `hadamard` |
| `s` / `t` | `s_gate` / `t_gate` |
| `rx(θ)` / `ry(θ)` / `rz(θ)` | `rx` / `ry` / `rz` |
| `rzz(θ)` / `rxx(θ)` | `rzz` / `rxx` |
| `u2(φ,λ)` | `u2` |
| `u3(θ,φ,λ)` / `u(θ,φ,λ)` | `u3` |
| `cx` | `cx` / `cnot` |
| `cy` / `cz` | `cy` / `cz` |
| `crx(θ)` / `cry(θ)` / `crz(θ)` | `crx` / `cry` / `crz` |
| `swap` | `swap` |
| `ccx` | `toffoli` / `ccnot` |
| `p(θ)` / `u1(θ)`（仅导入） | `rz` |

> 门名与 QASM 发射名的对应关系以 `aicir.gates`（GateSpec 注册表）中每个门的 `qasm_name` 为单一来源；`qasm.py` 只在此基础上区分“单比特无参 / 带参 / 双引用 / 三引用”几类发射形态，不单独维护第二份门名表。
>
> **当前不支持**：`if`、`reset`、`opaque`、自定义 `gate`、`cp`/`cu` 系列门、未绑定符号参数（导出前必须先调用 `circuit.bind_parameters(...)` 完成绑定，否则抛出 `TypeError`）。`measure` 和 `barrier` 语句在导入时会被跳过（不会重建为 Circuit 中的测量指令）。
>
> **`measure` 导出**：单比特 Z 基、不带 `id` 的 `measure(...)` 标记可导出为标准 QASM——`circuit_to_qasm` 会按需自动声明经典寄存器（2.0 为 `creg c[N]`，3.0 为 `bit[N] c`）并追加对应的 `measure`/`c[i] = measure` 语句；联合多比特测量、非 Z 基测量或带 `id` 的测量无法用标准 QASM 表达，导出时抛出 `NotImplementedError`（请改用 JSON 格式）。由于导入方向会跳过 `measure` 语句，QASM 往返对 `measure` 标记不是双向保真的。

### 1.3  2.0 vs 3.0 格式差异

| 项目 | 2.0 | 3.0 |
| --- | --- | --- |
| 版本头 | `OPENQASM 2.0;` | `OPENQASM 3.0;` |
| 标准库 | `include "qelib1.inc";` | `include "stdgates.inc";` |
| 量子寄存器声明 | `qreg q[2];` | `qubit[2] q;` |
| U3 门名称 | `u3(θ,φ,λ)` | `u(θ,φ,λ)` |

### 1.4  多控旋转门分解（仅 QASM 3.0）

对多控 `crx` / `cry` / `crz`（≥ 2 个控制位），导出器自动执行以下分解：

1. 申请辅助寄存器 `anc`（数量按全电路所需最大值一次性声明）。
2. 使用 `ccx` 链把多个控制位"聚合"到一个辅助位。
3. 在聚合后的辅助位上执行单控旋转门。
4. 按逆序执行 `ccx` 反计算，确保辅助位回到 `|0⟩`。

**设计约束**：

- OpenQASM 2.0 下**不做**多控旋转门分解，仅支持单控制。
- 单控制 `crx` / `cry` / `crz` 保持原样导出，不会被拆分。
- 分解依赖 `ccx` 和辅助位，导出文本中的总量子位数量可能高于原逻辑数据位数量。

**控制态**：`control_states` 支持 `0` / `1`。遇到 `0` 控制态时，导出器会自动在对应控制位前后插入 `x` 门，等价转换为 `|1⟩` 控制逻辑。

### 1.5  代码示例

**导出**

```python
from aicir import Circuit, hadamard, cnot, rz, circuit_to_qasm, circuit_to_qasm3
import math

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

# OpenQASM 2.0
qasm2 = circuit_to_qasm(cir, version="2.0")
print(qasm2)
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# h q[0];
# cx q[0],q[1];
# rz(pi/4) q[1];

# OpenQASM 3.0
qasm3 = circuit_to_qasm3(cir)
print(qasm3)
```

**导入**

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
print(cir)                   # Circuit(n_qubits=2, gates=[...])
print(cir.unitary().shape)   # (4, 4)
```

**文件读写**

```python
from aicir import save_circuit_qasm, load_circuit_qasm, save_circuit_qasm3

save_circuit_qasm(cir, "my_circuit.qasm")       # OpenQASM 2.0
save_circuit_qasm3(cir, "my_circuit_v3.qasm")   # OpenQASM 3.0
cir2 = load_circuit_qasm("my_circuit.qasm")     # 自动识别版本
```

---

## 2  Qiskit 互操作

### 2.1  公共 API

| 函数 | 短别名 | 说明 |
| --- | --- | --- |
| `circuit_to_qiskit(circuit)` | `to_qiskit` | aicir Circuit → `qiskit.QuantumCircuit` |
| `circuit_from_qiskit(qc)` | `from_qiskit` | `qiskit.QuantumCircuit` → aicir Circuit |

### 2.2  支持的门集

| aicir 门 | Qiskit 门 |
| --- | --- |
| `x` / `y` / `z` / `h` / `s` / `t` | `x` / `y` / `z` / `h` / `s` / `t` |
| `rx` / `ry` / `rz` | `rx` / `ry` / `rz` |
| `u2` / `u3` | `u`（Qiskit 侧以 `u` 发射） |
| `cx` / `cy` / `cz` | `cx` / `cy` / `cz` |
| `crx` / `cry` / `crz` | `crx` / `cry` / `crz` |
| `swap` / `rzz` / `rxx` | `swap` / `rzz` / `rxx` |
| `ccx` | `ccx` |
| `measure(...)` | `measure` |

> **当前不支持**：Qiskit 自定义门和未绑定符号参数。

### 2.3  代码示例

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

qc = circuit_to_qiskit(cir)       # aicir → Qiskit
cir2 = circuit_from_qiskit(qc)    # Qiskit → aicir

# 从 Qiskit 原生线路导入
qc2 = QuantumCircuit(2)
qc2.h(0)
qc2.cx(0, 1)
cir3 = circuit_from_qiskit(qc2)
```

**短别名**

```python
from aicir import to_qiskit, from_qiskit

qc = to_qiskit(cir)
cir2 = from_qiskit(qc)
```

---

## 3  PennyLane 互操作

### 3.1  公共 API

| 函数 | 短别名 | 说明 |
| --- | --- | --- |
| `circuit_to_pennylane(circuit)` | `to_pennylane` | aicir Circuit → PennyLane `QuantumScript` |
| `circuit_from_pennylane(script)` | `from_pennylane` | PennyLane `QuantumScript` → aicir Circuit |

### 3.2  支持的门集

| aicir 门 | PennyLane 门 |
| --- | --- |
| `x` / `y` / `z` | `PauliX` / `PauliY` / `PauliZ` |
| `h` / `s` / `t` | `Hadamard` / `S` / `T` |
| `rx` / `ry` / `rz` | `RX` / `RY` / `RZ` |
| `u2` / `u3` | `U2` / `U3` |
| `cx` / `cy` / `cz` | `CNOT` / `CY` / `CZ` |
| `crx` / `cry` / `crz` | `CRX` / `CRY` / `CRZ` |
| `swap` | `SWAP` |
| `rzz` / `rxx` | `IsingZZ` / `IsingXX` |
| `ccx` | `Toffoli` |
| `identity` | `Identity` |

> **当前不支持**：PennyLane 自定义门、未绑定符号参数和 aicir 线路内 `measure` 标记。

### 3.3  代码示例

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

script = circuit_to_pennylane(cir)        # aicir → PennyLane
cir2 = circuit_from_pennylane(script)     # PennyLane → aicir

# 从 PennyLane 原生对象导入
script2 = qml.tape.QuantumScript(
    [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])],
    [],
)
cir3 = circuit_from_pennylane(script2)
```

**短别名**

```python
from aicir import to_pennylane, from_pennylane

script = to_pennylane(cir)
cir2 = from_pennylane(script)
```

---

## 4  WuYue 互操作

### 4.1  公共 API

| 函数 | 短别名 | 说明 |
| --- | --- | --- |
| `circuit_to_wuyue(circuit)` | `to_wuyue` | aicir Circuit → WuYue `QuantumCircuit` |
| `circuit_from_wuyue(wc)` | `from_wuyue` | WuYue `QuantumCircuit` → aicir Circuit |

### 4.2  支持的门集

| aicir 门 | WuYue 门 |
| --- | --- |
| `x` / `y` / `z` | `X` / `Y` / `Z` |
| `h` / `s` / `t` | `H` / `S` / `T` |
| `rx` / `ry` / `rz` | `RX` / `RY` / `RZ` |
| `u2` / `u3` | `U2` / `U3` |
| `cx` / `cz` | `CX` / `CZ` |
| `swap` | `SWAP` |
| `rzz` | `IsingZZ` |
| `ccx` | `TOFFOLI` |
| `identity` | 每个量子位上的 `I` |
| `measure(...)` | `MEASURE` |

> **当前不支持**：`cy`、`crx` / `cry` / `crz`、`rxx`、自定义门和未绑定符号参数（受 WuYue 原生门集限制）。
>
> **特殊行为**：WuYue SDK 不接受 0 位经典寄存器。当 aicir 线路没有测量时，导出的 `QuantumCircuit` 会保留 1 位经典寄存器占位。

### 4.3  代码示例

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

wc = circuit_to_wuyue(cir)       # aicir → WuYue
cir2 = circuit_from_wuyue(wc)    # WuYue → aicir

# 从 WuYue 原生线路导入
qreg = QuantumRegister(2)
creg = ClassicalRegister(1)
wc2 = QuantumCircuit(qreg, creg)
wc2.add(H, qreg[0])
wc2.add(CX, qreg[1], control=qreg[0])
cir3 = circuit_from_wuyue(wc2)
```

**短别名**

```python
from aicir import to_wuyue, from_wuyue

wc = to_wuyue(cir)
cir2 = from_wuyue(wc)
```

---

## 5  JSON 序列化

### 5.1  公共 API

| 函数 | 说明 |
| --- | --- |
| `circuit_to_json(circuit, indent=2)` | Circuit → JSON 字符串 |
| `circuit_from_json(json_text)` | JSON 字符串 → Circuit |
| `save_circuit_json(circuit, path, indent=2)` | 保存为 `.json` 文件 |
| `load_circuit_json(path)` | 从 `.json` 文件加载 |

格式版本：`1.0`。支持 `numpy.ndarray`、`complex`、`torch.Tensor`（可选）和 `Parameter` 的自动序列化与反序列化。

### 5.2  代码示例

```python
from aicir import circuit_to_json, circuit_from_json, save_circuit_json, load_circuit_json

json_str = circuit_to_json(cir)
cir2 = circuit_from_json(json_str)

save_circuit_json(cir, "my_circuit.json")
cir3 = load_circuit_json("my_circuit.json")
```

---

## 6  DAG 图表示

### 6.1  公共 API

| 函数 | 说明 |
| --- | --- |
| `circuit_to_dag(circuit, gate_types)` | Circuit → `(X, A, type_onehot)` |

**参数**：

- `circuit`：aicir `Circuit` 对象。
- `gate_types`：`list[str]`，所有可能的门类型集合（如 `['hadamard', 'rx', 'ry', 'rz', 'cx', 'cz']`）。

**返回值**：

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `X` | `(N+2, F_type + n_qubits)` | 节点特征矩阵（门类型 one-hot + 量子位位置向量） |
| `A` | `(N+2, N+2)` | 有向邻接矩阵 |
| `type_onehot` | `(N+2, F_type)` | 门类型 one-hot 子矩阵（START / END 节点行全零） |

其中 `N` 为门节点数量，索引 `0` = START，`N+1` = END。

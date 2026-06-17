# aicir.core.io

本文档说明 `aicir.core.io` 中 OpenQASM、Qiskit、PennyLane 与 WuYue 互操作的当前行为。

## QASM 导出行为概览

- 支持 `OpenQASM 2.0` 与 `OpenQASM 3.0`。
- 对 `crx/cry/crz`：
- 单控制（1 个控制位）保持原样导出。
- 多控制（>=2 个控制位）仅在 `OpenQASM 3.0` 下自动分解。
- 控制态 `control_states` 支持 `0/1`：
- 遇到 `0` 控制态时，导出器会自动在对应控制位前后插入 `x` 门，将其转换为等价的 `|1>` 控制逻辑。

## 多控旋转门分解（QASM 3.0）

对多控 `crx/cry/crz`，导出器会：

1. 申请辅助寄存器 `anc`（数量按全电路所需最大值一次性声明）。
2. 使用 `ccx` 链把多个控制位“聚合”到一个辅助位。
3. 在聚合后的辅助位上执行单控旋转门（`crx/cry/crz`）。
4. 按逆序执行 `ccx` 反计算，确保辅助位回到 `|0>`。

等价结构示意（以多控 `cry(theta)` 为例）：

- 计算：`ccx ... -> anc`
- 执行：`cry(theta) anc, target`
- 反计算：`ccx ...`（逆序）

## 设计约束

- `OpenQASM 2.0` 下不做多控 `crx/cry/crz` 分解，仍按“仅支持单控制”处理。
- 自动分解只发生在多控制情形，不会拆分单控制 `crx/cry/crz`。
- 分解依赖 `ccx` 和辅助位，导出文本中的总量子位数量可能高于原逻辑数据位数量。

## 与 Demo 的关系

`aicir/encoder/demos/encode_1234_demo.py` 直接调用 `circuit_to_qasm3`。若 BasisEncoder 生成了多控旋转门，导出结果会自动包含 `anc` 辅助寄存器与 `ccx` 分解序列，这是预期行为。

## Qiskit 互操作

`qiskit_io.py` 提供 `circuit_to_qiskit` / `circuit_from_qiskit`，以及短别名 `to_qiskit` / `from_qiskit`。`qiskit` 是可选依赖，导入 `aicir` 不会强制导入 Qiskit；只有调用这些函数时才检查依赖。

当前支持门集与 QASM 第一批互操作面保持一致：

- 基础单比特门：`x/y/z/h/s/t`
- 参数旋转：`rx/ry/rz`
- 通用单比特门：`u2/u3`（Qiskit 侧以 `u` 发射）
- 受控门：`cx/cy/cz/crx/cry/crz/ccx`
- 双比特门：`swap/rzz/rxx`
- 线路内测量标记：Qiskit `measure` 与 aicir `measure(...)` 互转

暂不支持 Qiskit 自定义门和未绑定符号参数。

## PennyLane 互操作

`pennylane_io.py` 提供 `circuit_to_pennylane` / `circuit_from_pennylane`，以及短别名 `to_pennylane` / `from_pennylane`。`pennylane` 是可选依赖，导入 `aicir` 不会强制导入 PennyLane；只有调用这些函数时才检查依赖。

当前支持门集与 QASM/Qiskit 第一批互操作面基本一致：

- 基础单比特门：`PauliX/PauliY/PauliZ/Hadamard/S/T`
- 参数旋转：`RX/RY/RZ`
- 通用单比特门：`U2/U3`
- 受控门：`CNOT/CY/CZ/CRX/CRY/CRZ/Toffoli`
- 双比特门：`SWAP/IsingZZ/IsingXX`（对应 aicir `swap/rzz/rxx`）
- 整体恒等门：`Identity`

暂不支持 PennyLane 自定义门、未绑定符号参数和 aicir 线路内 `measure` 标记。

## WuYue 互操作

`wuyue_io.py` 提供 `circuit_to_wuyue` / `circuit_from_wuyue`，以及短别名 `to_wuyue` / `from_wuyue`。`wuyue` 是可选依赖，导入 `aicir` 不会强制导入 WuYue SDK；只有调用这些函数时才检查依赖。

当前支持 WuYue 原生门集中与 aicir 基础门直接对应的部分：

- 基础单比特门：`X/Y/Z/H/S/T`
- 参数旋转：`RX/RY/RZ`
- 通用单比特门：`U2/U3`
- 受控门：`CX/CZ/TOFFOLI`
- 双比特门：`SWAP/IsingZZ`（对应 aicir `swap/rzz`）
- 整体恒等门：aicir `identity` 导出为每个量子位上的 WuYue `I`
- 线路内测量标记：WuYue `MEASURE` 与 aicir `measure(...)` 互转

受 WuYue 当前原生门集限制，暂不支持 `cy`、`crx/cry/crz`、`rxx`、自定义门和未绑定符号参数。WuYue SDK 不接受 0 位经典寄存器；当 aicir 线路没有测量时，导出的 WuYue `QuantumCircuit` 会保留 1 位经典寄存器占位。

---

# OpenQASM 互转教程

## 7.1 Circuit → QASM 字符串

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

## 7.2 QASM 字符串 → Circuit

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

## 7.3 读写 QASM 文件

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

## 7.4 OpenQASM 3.0 格式差异

aicir 在 3.0 模式下的主要差异：

| 项目           | 2.0                       | 3.0                         |
| -------------- | ------------------------- | --------------------------- |
| 版本头         | `OPENQASM 2.0;`         | `OPENQASM 3.0;`           |
| 标准库         | `include "qelib1.inc";` | `include "stdgates.inc";` |
| 量子寄存器声明 | `qreg q[2];`            | `qubit[2] q;`             |
| U3 门名称      | `u3(θ,φ,λ)`          | `u(θ,φ,λ)`             |

## 7.5 支持的 QASM 门集

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

## 7.6 与 Qiskit 互转

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

## 7.7 与 PennyLane 互转

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

## 7.8 与 WuYue 互转

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

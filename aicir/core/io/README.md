# aicir.core.io

本文档说明 `aicir.core.io` 中 OpenQASM、Qiskit 与 PennyLane 互操作的当前行为。

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

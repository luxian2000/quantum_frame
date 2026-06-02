# aicir.core.io

本文档说明 `aicir.core.io.qasm` 在 OpenQASM 导出时对受控旋转门的处理规则，尤其是多控门自动分解逻辑。

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

`aicir/encoder/demo/encode_1234_demo.py` 直接调用 `circuit_to_qasm3`。若 BasisEncoder 生成了多控旋转门，导出结果会自动包含 `anc` 辅助寄存器与 `ccx` 分解序列，这是预期行为。

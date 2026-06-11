# aicir.gates

门元信息注册表（GateSpec），NEXT.md 第 7 节的第一片落地。

## 概述

每个门类型的元信息只在此注册一次：

```python
from aicir.gates import GateSpec, get_gate_spec, register_gate

spec = get_gate_spec("rz")        # GateSpec(name="rz", num_qubits=1, num_params=1, ...)
get_gate_spec("X")                 # 别名解析 → pauli_x 的 spec
get_gate_spec("not_a_gate")        # None：未注册门保持宽松

from aicir.gates import canonical_gate_name
canonical_gate_name("cnot")        # "cx"；未注册名称原样返回
```

字段约定：

- `num_qubits`：目标比特数（不含控制位）；`None` 表示可变
  （`unitary`、`measure`、可作用于整个寄存器的 `identity`）。
- `num_params`：参数个数，符号 `Parameter` 计为一个；`None` 表示可变
  （`unitary` 的矩阵参数在绘图占位场景可缺省）。
- `controlled`：是否必须携带至少一个控制位（`cx`/`cy`/`cz`/`crx`/`cry`/`crz`/`toffoli`）。
- `aliases`：等价的门字典 `type` 写法（`"X"`、`"cnot"`、`"ccnot"` 等）。
- `qasm_name`：OpenQASM 导出名（`core/io/qasm.py` 的导出表由此派生）。

## 当前消费方

- `aicir.ir.Operation` 构造期校验：已注册门检查目标比特数/参数个数/控制位，
  未注册门名不受限（自定义门、实验性门可自由使用）。
- `aicir.transpile.ValidatePass`：结合 `n_qubits` 做越界/冲突/重复比特检查。
- `aicir.transpile.CanonicalizePass`：把别名门名（`X`/`cnot`/`ccnot`）重写为规范名。
- `aicir.core.io.qasm`：QASM 导出名以 `GateSpec.qasm_name` 为单一来源，
  别名经 `canonical_gate_name` 归一，导入表由导出表反推。

## 注册自定义门

```python
register_gate(GateSpec(name="my_iswap", num_qubits=2, num_params=0))
```

重复注册（含别名冲突）默认报错，`register_gate(spec, overwrite=True)` 显式覆盖；
`unregister_gate(name)` 可移除。

## 后续方向（尚未实现）

`matrix`/`generator`/`decomposition` 字段与让 `gate_to_matrix`、
`visual`、`qml`、`qas` 改为从注册表读取元信息，见 NEXT.md 第 7 节。

# aicir.gates

门元信息注册表——每个门类型的规范名、比特数、参数个数、别名与显示符号在此**单点注册**，供 IR 校验、转译 Pass、QASM 导出、矩阵构造和绘图等模块统一消费。

## 目录

| 文件 | 说明 |
| --- | --- |
| `spec.py` | `GateSpec` 数据类定义 |
| `registry.py` | 注册表实现 + 内置门集 `_STANDARD_GATES` |
| `__init__.py` | 公共 API 导出 |

---

## 1  GateSpec 字段说明

`GateSpec` 是一个 `frozen` 数据类，字段含义如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | 规范门名，与门字典的 `type` 字段一致 |
| `num_qubits` | `int \| None` | 目标比特数（不含控制位）；`None` = 可变（`identity` / `unitary` / `measure`） |
| `num_params` | `int \| None` | 参数个数（符号 `Parameter` 计为 1）；`None` = 可变（`unitary` 占位场景可缺省） |
| `aliases` | `tuple[str, ...]` | 等价的 `type` 写法，如 `"X"` / `"cnot"` / `"ccnot"` |
| `controlled` | `bool` | 是否必须携带至少一个控制位 |
| `num_controls` | `int` | 控制位数量；满足 `controlled == (num_controls > 0)` |
| `qasm_name` | `str \| None` | OpenQASM 导出名（`core/io/qasm.py` 的导出表由此派生）；`None` = 暂未约定 |
| `symbol` | `str \| None` | ASCII / matplotlib 绘图显示符号（受控门为目标位符号）；`None` = 特殊绘制或退回通用 fallback |
| `generator` | `str \| None` | 单参数旋转门的 Pauli 生成元标签（`U = exp(-i θ G / 2)`），如 `rx`→`"X"`、`rzz`→`"ZZ"`；受控旋转记目标位生成元。`None` = 非标准 Pauli 旋转。供 QML 自省能否用解析参数移位 |
| `shift_rule` | `str \| None` | 参数移位规则类别，`"two_term"` / `"four_term"`；`None` = 未指定（标准 Pauli 旋转走默认两项规则）。供后续按门选择移位规则 |
| `decomposition` | `Callable \| None` | 分解到更基础门集的规则，签名 `(qubits, controls, control_states, params) -> list[dict] \| None`（返回 `None` 表示当前形态不适用）。供 `transpile.DecomposePass` 驱动；`None` = 无内置规则 |
| `matrix` | `Callable \| None` | 构造该门**局部**稠密幺正矩阵的后端感知可调用，签名 `(params, backend) -> 局部矩阵`，作用于 `num_qubits` 个目标比特（比特顺序 = 门的比特列表顺序；无参门忽略 `params`）。标准门的构造器由 `aicir.core.gates` 在导入时经 `set_gate_matrix` 附加；`None` = 无局部矩阵（如受控门、`measure`/`reset`） |

---

## 2  公共 API

| 函数 | 说明 |
| --- | --- |
| `get_gate_spec(name)` | 按规范名或别名查询 `GateSpec`；未注册返回 `None` |
| `canonical_gate_name(name)` | 把别名解析为规范名；未注册的名称原样返回 |
| `register_gate(spec, *, overwrite=False)` | 注册一个 `GateSpec`；冲突时默认报错 |
| `unregister_gate(name)` | 移除已注册门（含全部别名）；未注册时静默返回 |
| `registered_gate_names()` | 返回全部已注册门的规范名元组 |
| `gate_generator(name)` | 返回门的 Pauli 生成元标签（`rx`→`"X"`、`rzz`→`"ZZ"`）；无生成元或未注册返回 `None` |
| `gate_shift_rule(name)` | 返回门的参数移位规则类别（`"two_term"` / `"four_term"` / `None`）；未注册返回 `None` |
| `parametric_pauli_gates()` | 返回所有带 Pauli 生成元的门规范名集合（即解析参数移位适用门） |
| `gate_decomposition(name)` | 返回门的分解规则可调用；无规则或未注册返回 `None` |
| `gate_matrix(name, params=(), backend=None)` | 返回门的局部稠密幺正矩阵；`backend` 非空时返回后端张量；无构造器或未注册返回 `None`。首次调用前若构造器尚未附加，会惰性触发 `aicir.core.gates` 导入 |
| `set_gate_matrix(name, builder)` | 为已注册门附加局部矩阵构造器（写入 `GateSpec.matrix`）；`GateSpec` 为冻结数据类，内部经 `dataclasses.replace` 重注册；未注册门静默忽略 |

```python
from aicir.gates import GateSpec, get_gate_spec, canonical_gate_name

spec = get_gate_spec("rz")       # GateSpec(name="rz", num_qubits=1, num_params=1, ...)
get_gate_spec("X")                # 别名解析 → pauli_x 的 spec
get_gate_spec("not_a_gate")       # None：未注册门保持宽松

canonical_gate_name("cnot")       # "cx"
canonical_gate_name("ccnot")      # "toffoli"
canonical_gate_name("my_gate")    # "my_gate"（未注册，原样返回）
```

---

## 3  内置门集

下表列出导入时自动注册的全部内置门（`_STANDARD_GATES`）：

### 3.1  基础单比特门

| 规范名 | 别名 | 比特数 | 参数数 | QASM 名 | 绘图符号 |
| --- | --- | :---: | :---: | --- | :---: |
| `pauli_x` | `X` | 1 | 0 | `x` | X |
| `pauli_y` | `Y` | 1 | 0 | `y` | Y |
| `pauli_z` | `Z` | 1 | 0 | `z` | Z |
| `hadamard` | `H` | 1 | 0 | `h` | H |
| `s_gate` | `S` | 1 | 0 | `s` | S |
| `t_gate` | `T` | 1 | 0 | `t` | T |
| `identity` | `I` | 可变 | 0 | `id` | I |

### 3.2  参数旋转门

| 规范名 | 比特数 | 参数数 | QASM 名 | 绘图符号 | 生成元 |
| --- | :---: | :---: | --- | :---: | :---: |
| `rx` | 1 | 1 | `rx` | Rx | X |
| `ry` | 1 | 1 | `ry` | Ry | Y |
| `rz` | 1 | 1 | `rz` | Rz | Z |
| `u2` | 1 | 2 | `u2` | U2 | — |
| `u3` | 1 | 3 | `u3` | U3 | — |

### 3.3  受控门

| 规范名 | 别名 | 目标比特数 | 参数数 | QASM 名 | 目标位符号 | 生成元 / 分解 |
| --- | --- | :---: | :---: | --- | :---: | :---: |
| `cx` | `cnot` | 可变 | 0 | `cx` | X | — |
| `cy` | — | 1 | 0 | `cy` | Y | 分解 `rz·cx·rz` |
| `cz` | — | 1 | 0 | `cz` | Z | 分解 `h·cx·h` |
| `crx` | — | 1 | 1 | `crx` | Rx | 生成元 X |
| `cry` | — | 1 | 1 | `cry` | Ry | 生成元 Y |
| `crz` | — | 1 | 1 | `crz` | Rz | 生成元 Z |
| `toffoli` | `ccnot` | 1 | 0 | `ccx` | X | — |

> `cx` / `cnot` 的 `num_qubits=None`：支持单目标或多目标写法（多目标等价于多个单目标 CX）。

### 3.4  双比特门

| 规范名 | 比特数 | 参数数 | QASM 名 | 绘图符号 | 生成元 / 分解 |
| --- | :---: | :---: | --- | :---: | :---: |
| `swap` | 2 | 0 | `swap` | （专用形状） | 分解 `3×cx` |
| `rzz` | 2 | 1 | `rzz` | （专用形状） | 生成元 ZZ |
| `rxx` | 2 | 1 | `rxx` | （专用形状） | 生成元 XX |

### 3.4b  激发门

| 规范名 | 别名 | 比特数 | 参数数 | QASM 名 | 移位规则 |
| --- | --- | :---: | :---: | --- | :---: |
| `single_excitation` | `givens` | 2 | 1 | — | `four_term` |
| `double_excitation` | — | 4 | 1 | — | `four_term` |

> `double_excitation` 作用于 4 个比特，不属于双比特门；两者均无 `decomposition`（`GateSpec.decomposition` 为 `None`），`DecomposePass` 不会对其展开。

### 3.5  特殊指令

| 规范名 | 别名 | 比特数 | 参数数 | 绘图符号 |
| --- | --- | :---: | :---: | :---: |
| `unitary` | — | 可变 | 可变 | U |
| `measure` | `measurement` | 可变 | 0 | （专用形状） |
| `reset` | — | 可变 | 0 | \|0⟩ |

---

## 4  注册自定义门

```python
from aicir.gates import GateSpec, register_gate, unregister_gate

# 注册
register_gate(GateSpec(name="my_iswap", num_qubits=2, num_params=0))

# 重复注册默认报错；显式覆盖需传 overwrite=True
register_gate(
    GateSpec(name="my_iswap", num_qubits=2, num_params=0, symbol="iSW"),
    overwrite=True,
)

# 移除
unregister_gate("my_iswap")
```

未注册的门名在 IR 构造和转译中保持**宽松**——自定义门、实验性门可自由使用，不会被校验拒绝。

---

## 5  消费方一览

| 模块 | 用法 |
| --- | --- |
| `aicir.ir.Operation` | 构造期校验：已注册门检查目标比特数 / 参数个数 / 控制位 |
| `aicir.transpile.ValidatePass` | 结合 `n_qubits` 做越界 / 冲突 / 重复比特检查 |
| `aicir.transpile.CanonicalizePass` | 把别名（`X` / `cnot` / `ccnot`）重写为规范名 |
| `aicir.core.io.qasm` | QASM 导出名以 `GateSpec.qasm_name` 为单一来源；别名经 `canonical_gate_name` 归一 |
| `aicir.core.gates`（`gate_to_matrix` 等） | 入口经 `canonical_gate_name` 归一后按规范名分发，别名与规范名共享矩阵缓存；导入时经 `set_gate_matrix` 为标准门附加局部矩阵构造器（`GateSpec.matrix`），供 `gate_matrix()` 访问器读取 |
| `aicir.core.circuit`（ASCII 绘图） | 显示符号由 `GateSpec.symbol` 与规范名派生 |
| `aicir.visual.plot`（matplotlib 绘图） | 配色族与显示符号由规范名派生；注册自定义门时携带 `symbol` 即可直接显示 |
| `aicir.transpile.DecomposePass` | 分解规则由 `gate_decomposition`（`GateSpec.decomposition`）驱动；注册自定义门携带 `decomposition` 即被自动识别 |
| `aicir.qml.deriv` | 可微门集与生成元由 `parametric_pauli_gates()` / `gate_generator()` 派生；注册新 Pauli 旋转门即自动可伴随微分 |

> 分解规则置于自包含的 `decompositions.py`（仅构造纯门字典，不依赖 `aicir.ir` / `aicir.transpile`），以避开 `gates`↔`ir` 循环导入。

---

## 6  后续方向（尚未实现）

- `metrics` / `qas` 评分中的别名容忍集合（`DEFAULT_NATIVE_GATES`、双比特门判定等）属评分语义，留待单独处理。

# 经典控制流（if/while）设计（Spec）

日期：2026-07-03
状态：已批准设计，待实现

## 目标

给 aicir 加**经典控制流**：基于测量结果的条件 `if`（含 `else`）与 `while` 循环，运行在测量/轨迹执行路径上。补齐 `compare.md` 中标注为 WuYue-only 的 `qif/qwhile` + 经典寄存器能力。

核心语义：控制流是**每 shot 的运行时概念**——每条轨迹按其测量结果各自决定分支，天然是随机采样，无法用单个酉矩阵表示。因此实现落在 `run_trajectory` 的逐操作执行循环，而非 `Circuit.unitary()`。

## 关键决策（已确认）

1. **经典数据模型**：`ClassicalRegister`/`Bit` 抽象（类 Qiskit/WuYue），非复用 measure id。
2. **条件比较**：仅 `==` 与 `!=`（位或整个寄存器整数值）；不做 `<`/`>` 等不等式。
3. **while 上限**：`max_iterations` **必填**；达上限仍满足条件时抛 `RuntimeError`（不静默截断）。
4. **架构**：控制流作为电路指令，携带嵌套 body `Circuit`，由递归轨迹执行器执行（方案 A）。
5. **else**：`if_` 含可选 `else_body`。
6. **序列化**：仅 JSON 往返。**QASM3 控制流导出为非目标（推迟）。**

## 非目标

- QASM3 的 if/while/for 导出（大表面，后续独立 spec）。
- 不等式比较（`<`/`>`/`<=`/`>=`）。
- `for` 循环、`switch`/多路分支、break/continue。
- 控制流下的密度矩阵/含噪路径优化（含噪路径若与控制流并用，按现有逐门噪声语义在 body 内照常施加，不做特殊处理；本 spec 主验纯态 shot 采样路径）。
- 控制流参与 `unitary()`/张量网络引擎（这些路径遇控制流指令**报错**，因语义上不可表示）。
- 经典寄存器上的算术（加减、位运算）；只支持测量写入 + ==/!= 读取。

## 架构与组件

### 组件 1：经典数据模型 `aicir/core/classical.py`（新建）

```python
class ClassicalRegister:
    def __init__(self, size: int, name: str): ...
    # size>0；name 非空、作为轨迹经典 store 的键
    def __getitem__(self, index: int) -> "Bit": ...   # 0<=index<size
    def __len__(self) -> int: ...
    # 条件糖：整个寄存器整数值比较
    def __eq__(self, value: int) -> "Condition": ...
    def __ne__(self, value: int) -> "Condition": ...

class Bit:
    # (register, index) 引用；不可变
    register: ClassicalRegister
    index: int
    def __eq__(self, value: int) -> "Condition": ...  # value ∈ {0,1}
    def __ne__(self, value: int) -> "Condition": ...

class Condition:
    # target: Bit | ClassicalRegister；op: "==" | "!="；value: int
    target; op; value
    def evaluate(self, store: Mapping[str, list[int]]) -> bool: ...
```

- **寄存器整数值约定**：`creg[0]` 为 **LSB**，即 `value = sum(bits[i] << i)`。
- `Bit == v` 要求 `v ∈ {0,1}`，否则 `ValueError`。
- `__hash__`：`ClassicalRegister`/`Bit` 需可哈希（`__eq__` 被重载为返回 `Condition`，故显式定义 `__hash__` 基于 `id`/`(name,index)`，避免放进 set/dict 时出错）。
- `Condition.evaluate(store)`：从轨迹经典 store（`{reg_name: [bit值]}`）读目标当前值，按 op 与 value 比较返回 bool。

顶层从 `aicir` 再导出 `ClassicalRegister`（`Bit`/`Condition` 通常经糖法构造，不强制导出，但可从 `aicir.core.classical` 取用）。

### 组件 2：measure → 经典寄存器

复用现有 `Measurement`（已含 `classical_bits: tuple[int,...]` 字段，`__post_init__` 校验 `len(classical_bits)==len(qubits)`）。扩展：

- `measure()` 工厂新增经典目标入参：
  - `measure(qubits, creg=reg)`：把每个测量比特按序写入 `reg` 的 0..k-1 号位（要求 `len(qubits) <= len(reg)`）。
  - `measure(qubits, cbits=[reg[i], reg[j], ...])`：显式指定每个比特写入的 `Bit`（`len(cbits)==len(qubits)`，且所有 `Bit` 属同一 register——跨寄存器写入为非目标）。
- `Measurement` 增加可选字段 `classical_register: str | None`（目标寄存器名）；`classical_bits` 存该寄存器内的位下标（沿用现有字段与长度校验）。无经典目标时两者为空/None，走原有 joint-Pauli/`id=` 行为，**完全向后兼容**。
- **测量语义（有经典目标时）**：按 **per-qubit Z 基投影测量**（每比特独立坍缩），比特 i 的结果 `|0>→0 / |1>→1` 写入对应 cbit。复用 `projector.terminal_z_measure` 的逐比特 Born 采样逻辑（但作为线路内非末端操作，态继续演化）。这与无经典目标时的 joint-Pauli 单本征值语义不同——有经典目标即声明"要逐位经典比特"。basis 固定 Z（有经典目标时传非 Z basis 抛 `ValueError`）。

### 组件 3：控制流指令 `if_` / `while_`

`aicir/core/circuit.py` 新增工厂，返回控制流指令对象（新类型，类比 `Measurement` 那样的 `LegacyGateView` 子类，携带嵌套 body）：

```python
def if_(condition: Condition, body: Circuit, else_body: Circuit | None = None): ...
def while_(condition: Condition, body: Circuit, *, max_iterations: int): ...
```

- body / else_body 必须是 `Circuit`，且 `n_qubits` 与父电路一致（构建期校验；不一致抛 `ValueError`）。
- `while_` 的 `max_iterations` 必填、正整数。
- 指令类型（`name`）：`"if"` / `"while"`。存 `condition`、`body`（Circuit）、`else_body`（Circuit|None）、`max_iterations`（while）。
- `Circuit.__init__`/`append`/`extend` 经 `normalize_gate` 接受这些指令；`n_qubits` 推断：控制流指令作用于全部父比特（body 决定的比特已受 n_qubits 约束）。
- **body 可嵌套**：body 内可含 measure→creg、其它 `if_`/`while_`——递归执行器自然处理。

### 组件 4：递归轨迹执行器

重构 `aicir/measure/trajectory.py`：把 `run_trajectory` 的逐操作循环体抽成递归函数

```python
def _exec_ops(ops, state, classical, backend, n, *, rng, noise_model, snap_ops, incircuit, snaps, op_offset=0) -> State
```

- `classical`：`{reg_name: [bit值]}`，轨迹级可变 store。
- 逐 op：
  - unitary 门 → 演化（同现状）。
  - measure（无经典目标）→ joint-Pauli，记 `incircuit[op_index]`（同现状）。
  - measure（有经典目标）→ 逐比特 Z 投影，写 `classical[reg][bit]`。
  - reset → 现状。
  - `if` → `condition.evaluate(classical)` 真则 `_exec_ops(body.gates, ...)`，否则 else_body（若有）。
  - `while` → 循环：eval 条件为真则执行 body，计数；`iterations > max_iterations` 且条件仍为真 → `RuntimeError("while 超过 max_iterations=N 仍满足条件")`。
- snap/op_index：控制流使 op 数动态，`snap` 语义仅对顶层线性 op 定义（body 内 op 不参与顶层 snap 下标）；实现保证顶层 snap 行为不变。
- `run_trajectory` 变为薄封装：初始化 `classical`（所有出现过的寄存器置 0），调 `_exec_ops`，末端测量照旧，`TrajectoryResult` 增加 `classical` 字段（该轨迹终态经典 store 拷贝）。

### 组件 5：Result 经典读出

- `TrajectoryResult` 增 `classical: dict[str, list[int]]`。
- `aicir/measure/result.py` 的 `Result` 聚合各轨迹经典 store：
  - `result.classical_counts(register)` → `{寄存器整数值: 次数}`（按 shot 统计，register 可传名字或 `ClassicalRegister`）。
  - 单 shot / exact 场景下可取代表性经典态；多 shot 返回 counts。
- 不改现有 `output`/`counts`（measure id 路径）。

### 组件 6：unitary / TN 守卫

- `Circuit.unitary()`：遇 `if`/`while` 指令抛 `ValueError("控制流指令无法表示为酉矩阵；请用 Measure.run 执行")`（`ignore_nonunitary=True` 也不跳过——控制流不是可跳过的非酉标记，而是语义不可表示）。
- 张量网络引擎（`aicir/simulator`）遇控制流指令抛 `ValueError`（同理由）。

### 组件 7：JSON 往返

`aicir/core/io/json_io.py`：控制流指令与经典寄存器的 `to_dict`/`from_dict`：
- 指令序列化：`{"type":"if","condition":{...},"body":[...],"else_body":[...]}`、`{"type":"while","condition":{...},"body":[...],"max_iterations":N}`。
- condition 序列化：`{"target":{"register":name,"index":i|null},"op":"==","value":v}`（index=null 表示整个寄存器）。
- body 递归为 gate dict 列表。
- 往返后 `n_qubits`、条件、嵌套结构、max_iterations 保持。

## 数据流

```
构建：
  reg = ClassicalRegister(2, "c")
  body = Circuit(pauli_x(1), n_qubits=2)
  circ = Circuit(hadamard(0), measure(0, creg=reg), if_(reg[0]==1, body), n_qubits=2)

执行（Measure.run）：
  每 shot: |00> → H → 测 q0 写 c[0] → 若 c[0]==1 执行 body(X on q1) → 末端读出
  → Result: classical_counts(reg) 给出 c 的分布；measure_qubits 给量子读出
```

## 错误处理

| 情形 | 行为 |
| --- | --- |
| `while_` 缺 `max_iterations` | `TypeError`（keyword-only 必填） |
| while 达上限仍满足条件 | `RuntimeError` |
| body/else_body 非 Circuit 或 n_qubits 不符 | `ValueError` |
| `Bit == v`，`v∉{0,1}` | `ValueError` |
| measure 有经典目标但 basis≠Z | `ValueError` |
| `cbits` 跨多个寄存器 | `ValueError` |
| `len(qubits)` 与经典目标位数不符 | `ValueError` |
| `unitary()` / TN 引擎遇控制流 | `ValueError` |
| 条件引用从未被测量写入的寄存器 | store 中该寄存器默认全 0，正常求值（不报错） |

## 测试

1. **classical.py 单测**：寄存器整数值（LSB 约定）、`Bit==/!=`、`ClassicalRegister==/!=` 生成正确 `Condition`；`evaluate` 各分支；哈希可用；`v∉{0,1}` 报错。
2. **measure→creg**：确定性电路（如 `X(0)` 后测 q0 写 c[0]）→ c[0]==1；`H(0)` 多 shot → c[0] 分布 ~50/50；basis≠Z 报错；位数不符报错。
3. **if/else 执行**：`H(0); measure(0→c[0]); if_(c[0]==1, X(1))` → 末态 q1 与 c[0] 完全关联（每 shot q1==c[0]）；带 else 分支验证两路都走到；`!=` 条件。
4. **while**：制备确定性收敛循环（如"测到 1 就 X 翻回 0 再测，直到测到 0"——构造必然在有限步收敛的 body）验证正常终止；构造条件恒真的 body + 小 max_iterations 验证 `RuntimeError`。
5. **嵌套**：if 内含 while、while body 含 measure→creg + if，验证递归执行正确。
6. **守卫**：含控制流的 Circuit 调 `unitary()` 报错；传给 `tn_statevector` 报错。
7. **JSON 往返**：含 if/else/while + creg 的电路 `to_json`→`from_json` 后结构、条件、max_iterations、n_qubits 一致；往返后执行结果与原电路一致。
8. **Result**：`classical_counts` 在多 shot 下给出正确分布；与量子读出关联正确。
9. 全量回归基线不变（现有 measure/unitary 路径无经典目标时行为完全不变）。

torch 可选：控制流执行走纯态 shot 采样，`NumpyBackend` 即可；不强制 torch。

## 文档

- 新建 `aicir/core/classical.py` 的模块 docstring + 在合适 README（`aicir/measure/README.md` 或新 `aicir/core` 说明）补控制流用法。
- `CHANGELOG.md`：2026-07-03 Added 条目。
- `compare.md`：§3 能力矩阵「经典控制流 if/while」「经典寄存器」两行由 ❌ 改为 ✅ 并注模块；§5 电路构建模型「控制流」行更新；§19「仅 WuYueSDK 具备」移除经典控制流条目、并入「两者都具备」。
- `CLAUDE.md`：在 core 或 measure 子系统描述补经典控制流与 `ClassicalRegister`。

## 影响的现有约定

- `Measurement` 增字段（`classical_register`），默认 None，向后兼容。
- `run_trajectory` 内部重构为递归执行器；`TrajectoryResult` 增 `classical` 字段（新增，不破坏现有读取）。
- `Circuit.unitary()` 对控制流指令新增报错分支（此前不存在此类指令，无回归）。

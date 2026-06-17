# 设计：反转 `Measure.run` 的 `measure_qubits` 语义并移除 `tm`

日期：2026-06-17

## 背景

`Measure.run(...)`（`aicir/measure/measure.py`）当前用 `measure_qubits` 与独立的
`tm` 布尔标志共同控制末端读出，语义为：

- `measure_qubits=None`（默认）→ 读出全部比特（需 `tm=True`）
- `measure_qubits=[]` → 不做末端测量（等价 `tm=False`）
- `measure_qubits=[q0, …]` → 读出指定子集

这套语义不直观（空列表表示“不测”、`None` 表示“全测”），且 `tm` 与
`measure_qubits` 职责重叠。

## 目标

把 `measure_qubits` 作为末端读出的**唯一**控制项，并反转空/None 语义：

| 取值 | 含义 |
|---|---|
| `None` | **不做**末端测量 |
| `[]`（默认） | 读出**全部**比特 `[0..n−1]` |
| `[q0, q1, …]` | 读出该子集，保留输入顺序 |

并**移除 `tm` 参数**。

## 决策

1. **默认值 = 字面 `[]`**。`run(cir, shots≥1)` 仍默认读出全部比特，与今日默认行为一致。
2. **移除 `tm`**。原 `tm=False` 改写为 `measure_qubits=None`；原 `tm=True`（全测）即默认 `[]`。
3. **exact 模式（`shots`∈`{None,0}`）从不执行末端测量，且完全忽略 `measure_qubits`、不报错**：
   `None` / `[]` / `[q0,…]` 在 exact 模式下一律为 no-op（off）。
   - 因此无需哨兵默认值，也无需 exact 模式的冲突校验。
4. **shot 模式（`shots≥1`）正常解析 `measure_qubits`**：`None`→off，`[]`→全部，`[list]`→子集。

### 行为示例

| 调用 | 结果 |
|---|---|
| `run(cir, shots=8)` | shot 模式，读出全部 |
| `run(cir, shots=8, measure_qubits=None)` | shot 模式，不做末端测量 |
| `run(cir, shots=8, measure_qubits=[0])` | shot 模式，仅读出 qubit 0 |
| `run(cir, shots=None)` | exact，off |
| `run(cir, shots=None, measure_qubits=[])` | exact，off（无错误） |
| `run(cir, shots=None, measure_qubits=[0])` | exact，off（无错误） |
| `run(cir, measure_qubits=None)` | shot 模式默认 shots=1，不做末端测量 |

## 实现要点（`aicir/measure/measure.py`）

- `run(...)` 签名：删除 `tm`；`measure_qubits=[]` 作为默认值（注意可变默认值，
  函数内不得就地修改它——用归一化生成新列表即可，现有 `_normalize_measure_qubits`
  已返回新列表，安全）。
- 末端读出解析逻辑重写：
  - exact → `do_terminal=False`，`terminal_qubits=None`，不解析 measure_qubits。
  - shot 模式：
    - `measure_qubits is None` → `do_terminal=False`，`terminal_qubits=None`。
    - 否则归一化：空列表 → `terminal_qubits=list(range(n))`；非空 → 该子集（保留顺序）。
      `do_terminal=True`。
  - 删除原 `mq_explicit` / `tm` / exact 冲突分支（line 161–174 区域）。
- 越界/重复校验仍由 `_normalize_measure_qubits` 负责（对 `[]` 直接返回 `[]`）。
- 更新 `run` docstring。

## 连带改动

- `aicir/primitives/sampler.py` — `ShotSampler.run(...)` 的 `measure_qubits`
  默认值由 `None` 改为 `[]`（采样器必须读出，否则无 counts）。
- `aicir/measure/result.py:63` — `output(-1)` 报错文案：把
  “`tm=False` / `measure_qubits=[]`” 更新为 “`measure_qubits=None` / `shots∈{None,0}`”。
- `README.md` §4.1 参数表、§4.5 末端测量、exact 模式表 — 重写 `measure_qubits`
  说明、删除 `tm` 行与示例中的 `tm=` 用法。
- `CHANGELOG.md` — 新增 2026-06-17 接口变更条目。

## 不改动

- `aicir/measure/trajectory.py` 维持原状。`measure.py` 始终向 `run_trajectory`
  传入**具体比特列表**（或 `tm=False`），其内部 `measure_qubits is None → 全部`
  的低层约定不会与新上层语义冲突，且有独立测试覆盖。

## 测试迁移

- `tests/measure/test_unified_run.py` — `tm=False` → `measure_qubits=None`；
  删除 `tm=False + measure_qubits` 冲突用例；删除 exact + 显式 `measure_qubits`
  冲突用例，改为断言其为 off。
- `tests/measure/test_measure_run_semantics.py` — 同步 `tm` 与冲突相关用例；
  更新模块顶部语义注释。
- `tests/measure/test_measure.py` — `test_explicit_measure_qubits_with_all_qubits`
  可改用 `measure_qubits=[]` 验证“全测”；其余子集用例不变。
- `tests/docs/test_readme_measure_examples.py` — `tm=False` → `measure_qubits=None`；
  与 README 示例保持同步。
- 新增：`measure_qubits=[]`（shot 模式）== 全测、`measure_qubits=None` == off、
  exact 模式忽略 `measure_qubits` 的断言。

## 验收

- `PYTHONPATH=. pytest tests/measure tests/docs/test_readme_measure_examples.py`
  全绿。
- README 示例与代码行为一致。

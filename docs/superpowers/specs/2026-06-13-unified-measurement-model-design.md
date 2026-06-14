# 统一量子测量模型 — 设计文档

- 日期：2026-06-13
- 状态：已评审（待用户最终确认 → writing-plans）
- 来源：用户上传的《量子测量设计规则》（共 60 条，编号有跳号 8/9/12/13/17 等缺失）+ 本次对话中的澄清与修正。
  > 上传原文在传输中出现编码问题（UTF-8 被按 Latin-1 解码的乱码），故本设计文档作为本次工作的**权威记录**。如需在仓库内保留干净原文，请另行提供。

---

## 1. 目标与动机

把 README 第 4 节中相互独立、且**互斥**的两套测量机制合并为**一个统一测量模型**，并同步实现到代码：

- **机制一**（外部 `measure_qubits` 决定末端读出）与 **机制二**（线路内嵌 `measure()` 标记）不再是二选一的两套语义，而是同一模型里的两个正交部分：
  - **线路内测量** `measure(...)`：电路演化过程中的**投影式（破坏性坍缩、但非破坏性保留比特）联合 Pauli 测量**操作；
  - **末端测量** `tm` / `measure_qubits`：电路全部显式操作执行完后，对指定比特做的**逐比特 Z 基**读出。
- 线路内 `measure()` 的语义由"非坍缩的边缘采样读出标记"**改为投影测量**（这是本次最核心的行为变更）。
- `reset(...)` 由"要求此前已 measure"的标记，改为**无前置条件的重置量子信道**。
- 采用**轨迹法**执行引擎，噪声完整纳入。

本次为**干净重构**：按新 API 重写核心，并**同步修复所有下游调用方**，最终 `PYTHONPATH=. pytest` 全绿。

### 1.1 与上传规则的有意偏差（用户在对话中确认）

| 项 | 上传规则 | 本设计采用 | 原因 |
|---|---|---|---|
| `shots=0` | 第 23 条：非法 | **等价于 `shots=None`** | 用户明确要求保留现状语义 |
| `shots=None`/`0` 与末端测量 | 第 36 条：`shots=None & tm=True` 仍取一个末端结果 | **`shots=0`/`None` 覆盖 `tm`：不执行末端测量**（`final_state==state`，`output(-1)` 报错） | 用户明确要求 |
| `base` 参数名 | `base="Z"` | 重命名为 **`basis`** | 用户要求 |
| 末端输出顺序 | 第 22 条：随 `measure_qubits` 顺序 | 随 **`measure_qubits` 原始输入顺序**，**不做内部排序** | 用户强调；现有 `_normalize_measure_qubits` 会排序，必须去掉 |
| `sm` 模式 | 第 50/53/54 条：`avg`/`shot`/`cond` | 本次**仅实现 `avg`**；`shot`/`cond` 标注待实现并在传入时报 `NotImplementedError` | 用户选择分期 |
| 噪声 | 几乎未涉及 | **完整纳入轨迹模型** | 用户选择 |
| QASM | 未涉及 | 联合/非 Z/带 id 的 measure **导出时报 `NotImplementedError`**（不实现自定义 pragma） | 用户选择 |
| `circuit.unitary()` 遇 measure/reset | 规则未约束（现状：measure 静默跳过、reset 报错） | **默认对 measure 与 reset 一律报错**，提供 `ignore_nonunitary=True` 显式选项 | 用户要求 |

> **`shots=0` / `shots=None` 的精确含义**（消歧，含对规则 36 的偏差）：二者同义，表示**单条精确轨迹**（exact 模式）——不做 shot 维度、不产生采样统计（`counts(i)`/`prob(i)` 报错）。**`shots=0`/`None` 覆盖 `tm`：不执行末端测量**（即便 `tm=True`），故 `final_state==state`、`output(-1)` 不存在（调用报错）。这**偏离上传规则 36**（规则 36 让 `shots=None & tm=True` 仍取一个末端结果），按用户要求改为关闭末端测量。
> 注意：覆盖只针对**末端**测量；**线路内** `measure()` 是投影操作，与 `tm` 无关——`shots=None`/`0` 时它仍按 Born 概率走**一条**随机轨迹并坍缩（规则 26/31/55），`output(i)` 为标量 `λ`，`result.state` 为该轨迹条件态 `ρ_pre`。要在末端也读出结果，请用 `shots≥1`。

---

## 2. 执行引擎（Approach A：统一逐操作轨迹引擎）

### 2.1 模块划分

- `aicir/measure/projector.py`（新）— 纯函数，作用于 SV 与 DM（按 `State.is_density` 分支）：
  - `pauli_basis_change(state, qubits, basis, inverse=False) -> State`：把列出比特从 `basis` 旋转到 Z（或逆旋转）。
  - `joint_parity_probs(state, qubits) -> (p_plus, p_minus)`：旋转后计算偶/奇宇称两个子空间的概率。
  - `project_parity(state, qubits, lam) -> State`：把状态投影到宇称 `lam` 子空间（清零异宇称分量并归一化），**保持子空间内相干**。
  - `reset_channel(state, qubits) -> State`：重置信道。
- `aicir/measure/trajectory.py`（新）— `run_trajectory(instructions, init_state, *, tm, measure_qubits, snap_ops, rng, noise_model) -> TrajectoryResult`：一次走完操作序列，返回该轨迹的记录（见 2.4）。
- `aicir/measure/aggregate.py`（新）— `aggregate_avg(trajectories, ...) -> Result-字段`：`sm="avg"` 的聚合。
- `aicir/measure/measure.py`（重写）— `Measure.run(...)` 编排：校验 → 选表示 → 定 shot 策略 → 循环 → 聚合 → 构造 `Result`。
- `aicir/measure/result.py`（重写）— 新 `Result` 容器（见第 4 节）。

旧的 `run_density_matrix` / `run_batch` / `_resolve_post_measurement` / `_marginal_counts` / `_collapse_*` 等被新引擎取代（保留必要的偏迹工具，迁入 projector/aggregate）。

### 2.2 状态表示（每次 run 选定一次）

- 纯电路且初态为向量 → **态矢 `State`**（规则 6）。
- 提供 `NoiseModel` 或初态为密度矩阵 → **密度矩阵 `State`**。
- 引擎内的 measure/reset/旋转工具按 `State.is_density` 自动分支。
- **轨迹内升级**：SV 路径上对**纠缠**比特做 `reset` 会产生混合态，则该轨迹在 reset 处升级为 DM（引擎已支持逐轨迹 DM）。

### 2.3 shot 策略

- 线路内含 `measure`（唯一的线路中途随机源）→ **独立重跑完整电路 M 次**，每个 shot 重置初态（规则 24；禁止"对已坍缩末态连续测 M 次"）。
- 线路内**无** `measure` → 计算 `ρ_pre` 一次，再对**末端测量**采样 M 次（物理等价；这是让 VQE/Sampler 类调用保持廉价的性能路径）。
- `shots=None` 或 `shots=0` → 单条精确轨迹（不做 shot 维度）；**覆盖 `tm`，不执行末端测量**（`final_state==state`，无 `output(-1)`）；线路内 `measure` 仍走一条随机轨迹并坍缩。
- 噪声：在循环内每个门之后按 `noise_model.apply(...)` 施加（沿用现 `_evolve_density_matrix_gatewise` 的做法），measure/reset 作用在 DM 上。

### 2.4 聚合（`sm="avg"`）

- `state = (1/M) Σ ρ_pre^(s)`，`final_state = (1/M) Σ ρ_post^(s)`；`shots>1` 时一律为**密度矩阵**。
- `shots∈{None,0,1}` 不平均，返回单条轨迹的条件态（纯则向量、混则 DM）。
- 逐 shot 的 ±1 结果堆叠为 `(M,·)` 数组；`counts(i)`/`prob(i)` 在聚合期统计。

### 2.5 随机种子

单个 `numpy.random.Generator(seed)` 贯穿：线路内 measure、末端 measure、reset 的随机实现（规则 27）。相同入参可复现。

---

## 3. `measure` / `reset` 语义与门构造器

### 3.1 `measure(*qubits, basis="Z", id=None)` —— 联合 Pauli 投影测量

- **物理语义**（规则 7/10）：对 `P = P_a⊗P_b⊗…⊗P_c`（`P_j = basis ∈ {X,Y,Z}`，同一 `basis` 作用于所列全部比特）做**单次两结果投影**到 `λ=±1` 联合本征子空间 `Π_λ=(I+λP)/2`，返回一个 `λ∈{+1,−1}`。
- **非破坏性保留**：被测比特**仍留在电路中**，后续门可作用其上；测量后**只有联合可观测量 `P` 是确定的**，单个比特 `a` 未必处于 `P_a` 的本征态。
- **明确禁止**：不得以"对每个目标比特各自测量再相乘"实现——那会把态过度坍缩到各自本征子空间。
- **实现（保留基变换法，旋转顺序明确）**：
  1. 旋转 `P_j → Z_j`（每个所列比特）：
     - X：测前 `H`；
     - Y：测前 `Sdg` 然后 `H`；
     - Z：不变。
  2. 旋转系下的**联合宇称投影**（`P_rot = Z_a⊗…⊗Z_c`）：`p(+1)=` 所列比特子集上**偶宇称**基态的总概率，`p(−1)=` 奇宇称；按 `rng` 采样 `λ`；投影=清零异宇称分量（DM 同时清行与列）并归一化，即 `Π_λ=(I+λP_rot)/2`，是真正的两结果**子空间**投影，**保持子空间内相干**（非逐比特坍缩）。
  3. 逆旋转（逆序）：
     - X：测后 `H`；
     - Y：测后 `H` 然后 `S`；
     - Z：不变。
  4. 在该操作下标记录 `λ`；比特保留。
- `id`：可选字符串，**在整条电路 measure 操作中唯一**（否则 `ValueError`）；使 `result.output("m0")` 可用。
- `basis`：校验 `{Z,X,Y}`（大小写不敏感）。

### 3.2 IR 字段与序列化

- `aicir/ir/measurement.py` 的 `Measurement` 增加**一等字段** `basis: str = "Z"` 与 `id: str | None = None`（**不**塞进 `metadata`）。
- `to_dict`/`from_dict` 显式序列化 `basis`/`id`，并带**向后兼容默认值**（缺省字段的旧字典仍可解析）。**JSON 往返保真**。
- `aicir/core/circuit.py` 的 `measure(*qubits, basis="Z", id=None)` 工厂相应扩展。

### 3.3 `reset(*qubits)` —— 重置信道

- 语义（规则 14–16）：`R_S(ρ)=|0⟩⟨0|_S ⊗ Tr_S(ρ)`，Kraus `K0=|0⟩⟨0|, K1=|0⟩⟨1|`。
- **删除前置 `measure` 要求**（移除 `_assert_reset_allowed` 及其两处检查），`reset` 可出现在任意位置。
- 效果：目标比特回到 `|0⟩`，其与其他比特的关联被清除，其他比特的约化态不变。
- SV 路径上对**纠缠**目标 reset → 该轨迹升级为 DM；对**可分**目标 reset → 仍为向量。
- 画图（matplotlib）：保持现有 reset 虚线渲染，去掉"必须跟在 measure 之后"的文案。

### 3.4 `Circuit.unitary()` / `matrix()`

- **默认对 measure 与 reset 一律 `raise`**（现状 measure 静默跳过、reset 报错——本次统一为都报错）。
- 增加显式选项 `ignore_nonunitary: bool = False`：为 `True` 时丢弃 measure/reset，只返回酉部分。
- 审计并调整把含 measure 电路传给 `unitary()` 的调用点（仅在确需取酉部分处传 `ignore_nonunitary=True`）。

### 3.5 QASM（诚实声明）

- 标准 QASM `measure q -> c` 无法表达联合 Pauli / 非 Z 基 / 带 id 的测量。
- 本次**不实现自定义 pragma**：JSON 为无损格式；QASM 导出遇到**联合（多比特）/ 非 Z / 带 id** 的 measure 时**报 `NotImplementedError`**；普通单比特 Z `measure` 仍按标准 QASM 导出。

---

## 4. `Result` API 与精确形状

`output`/`counts`/`prob` 改为**按操作下标的方法**；`state`/`final_state`/`probabilities`/`expectation_values` 保持字段。`Result` 内部维护**测量登记表**：有序的线路内 `measure` 操作 `(op_index, id, qubits, basis)` + 末端测量描述符，使 `output(i)`/`output("m0")`/`output(-1)` 可无歧义解析。

### 4.1 `result.output(target)` —— `target` = 操作下标 `i` / 字符串 `id` / `-1`（末端）

| 模式 | 线路内 `output(i)` | 末端 `output(-1)`（k 比特） | 取值 |
|---|---|---|---|
| `shots=M>0` | 形状 `(M, 1)` | 形状 `(M, k)` | `±1` |
| `shots=None`/`0` | 标量 `λ` | 不适用（不做末端测量；`output(-1)` 报错） | `±1` |

- 末端逐比特顺序 = `measure_qubits` **原始输入顺序**（不排序）。
- `output(i)` 指向非 measure 操作 → `ValueError`；`tm=False`、`measure_qubits=[]`、或 `shots∈{None,0}` 时 `output(-1)` → `ValueError`（未执行末端测量）。

### 4.2 `result.counts(target)` / `result.prob(target)` —— 仅采样模式

- 线路内：结果 `{+1: N₊, −1: N₋}`；末端：结果为 k 位比特串。
- `prob = counts / M`。
- `shots=None`/`0` → 报 `"单轨迹模式不支持统计结果"`（规则 26/44）。

### 4.3 `result.state`（= `ρ_pre`，所有显式操作后、末端测量前；规则 28–31）

- `shots=None`/`0` → 本轨迹条件态（纯则向量、混则 DM）。
- `shots=1` → 单 shot 条件态（纯则向量）。
- `shots>1` → **平均密度矩阵** `(1/M)Σρ_pre^(s)`（一律 DM）。

### 4.4 `result.final_state`（= `ρ_post`，末端测量后；被测比特保留、不自动求迹——规则 37）

- `tm=False`、`measure_qubits=[]`、或 `shots∈{None,0}` → 等于 `result.state`（规则 33；`shots=0`/`None` 覆盖 `tm`）。
- `shots=1` → 条件坍缩态 `ρ_b`；`shots>1` → 平均 DM。

### 4.5 `result.snap(t)` —— 第 `t` 个操作完成后的完整 n 比特态（操作下标 `0≤t<L`，含 measure 坍缩 / reset 效果；规则 46–49）

- `shots∈{None,0,1}` → 本轨迹 op-`t` 后条件态；`shots>1` 且 `sm="avg"` → op-`t` 后平均 DM。
- `sm∈{"shot","cond"}` → `NotImplementedError`（本次待实现）。
- 索引校验 `0≤t<L`，去重。

### 4.6 `result.reduce(R, pos="final")`（规则 38）

- 保留比特集 `R` 求偏迹；`pos="state"` 约化 `result.state`，`pos="final"` 约化 `result.final_state`。
- 返回约化密度矩阵（保留 `R` 的原始入参顺序）。

### 4.7 保留（便利 / 兼容）

`probabilities`（`ρ_pre` 的计算基分布=对角元）、`expectation_values`/`expectation_variances`（observables 路径）、`metadata`、`shots`、`n_qubits`、`most_probable()`、`summary()`。

---

## 5. `run()` 签名与输入校验

```python
def run(self, circuit, shots=1, measure_qubits=None, snap=None,
        tm=True, sm="avg", seed=None,
        *, initial_state=None, observables=None, return_state=True) -> Result
```

- `tm`（默认 `True`）：是否做末端测量（规则 18）。`tm=False` → `final_state==state`，无 `output(-1)`。
- `sm`（默认 `"avg"`）：`shots>1` 时 snap 聚合模式；当前仅 `"avg"`，`"shot"/"cond"` → `NotImplementedError`。
- `seed`：单一 RNG 种子（规则 27）。
- `measure_qubits`：末端读出比特，**保留输入顺序**；`None` → 全部 `[0..n−1]`（规则 19）；`[]` → 不做末端测量，等价 `tm=False`（规则 21）。
- `initial_state`/`observables`/`return_state`：保持现有语义（与测量模型正交）。

**合并要点**：删除原 `measure_qubits` 与线路内 `measure()` 的互斥报错——二者现在可共存（线路内 measure=演化中的投影操作；`measure_qubits`/`tm`=末端 Z 读出）。这就是机制一与机制二的合并。

**校验（规则 56–60 + shots）**：
- `measure_qubits` / `measure(...)` / `reset(...)` 内每个比特 `0≤q<n`，否则 `ValueError`（越界）。
- 单个 `measure_qubits` / `measure(...)` / `reset(...)` 列表内不得有重复比特。
- 跨 measure 操作的 `id` 重复 → `ValueError`。
- `snap` 索引 `0≤t<L`（L=操作数），去重。
- `tm=False` 且 `measure_qubits` 非空 → `ValueError`（冲突，规则 59），不静默忽略。
- `shots∈{None,0}` 且显式传入**非空** `measure_qubits` → `ValueError`（exact 模式覆盖 `tm`、不做末端测量，与显式末端读出请求冲突；请改用 `shots≥1`）。同理与 rule 59 一脉相承，不静默忽略。注：`shots∈{None,0}` 与 `tm=True`（默认或显式）**不**报错——`tm` 被静默覆盖为不测量。
- `shots`：正整数、`0` 或 `None`；`shots<0` 或非整数 → `ValueError`。`0` 与 `None` 同义（对上传规则 23 的有意偏差）。
- 产生混合态但后端只支持态矢 → 明确 `ValueError`（规则 60），绝不静默坍缩。

---

## 6. 下游迁移与测试策略

实现采用 **TDD**（按单元先写失败测试，再写实现）。

### 6.1 需迁移的调用方（干净重构、保持测试全绿）

| 文件 | 改动 |
|---|---|
| `aicir/primitives/sampler.py` | `result.counts`(字段) → `result.counts(-1)`；`probabilities` 字段保留 |
| `aicir/measure/estimator.py` | `result.counts` → 末端 `counts(-1)`（全比特）或 `probabilities`（精确） |
| `aicir/primitives/estimator.py` | 审计；对齐保留的 `expectation_values` / 新 counts |
| `aicir/vqc/VQE.py`,`QAOA.py`,`SSVQE.py`,`VQD.py` | 读 `expectation_values`（保留）；核对不依赖旧 `counts`/`output` 字段 |
| `aicir/optimization/qubo/qaoa.py`,`aicir/transpile/passmanager.py`,`aicir/optimizer/circuit.py`,`aicir/qas/demos/_np_ising_utils.py` | 审计 `.run()`/`unitary()`；仅在确需取酉部分处传 `ignore_nonunitary=True` |
| demos：`demo_1`,`grover_demo`,`qft_demo`,`reset_demo`,`snap_demo`,`test_1`,NPU demos | 更新为方法式 `output(i)`/`counts(i)`、投影 measure 语义、无前置 reset |

### 6.2 新增 / 更新测试（工作重心）

- **投影器正确性**：Bell 态测 ZZ → 恒 `+1` 且两比特仍相干纠缠（未坍缩到基态）；XX / YY 联合测量；断言单比特**未**被强制到 `P_a` 本征态而联合 `P` 确定；测后施加门可正常工作。
- **重置信道**：reset 纠缠目标 → 混合态、关联被清、其他比特约化 DM 不变；reset 可分目标 → 仍纯；无前置 measure。
- **shots 语义**：`None`/`0`（标量 `output`、条件 `state`、`counts` 报错）；`1`（`(1,·)` 形状、可统计）；`M>1`（平均 DM 的 `state`/`final_state`、`counts(i)`/`prob(i)`）。
- **末端**：逐比特 Z、`output(-1)` 形状 `(M,k)`/`(k,)`、**输入顺序保留**；`tm=False`+`measure_qubits` 冲突报错。
- **可复现**：相同 `seed` ⇒ measure/末端/reset 采样一致。
- **snap**（`avg`）；**reduce**；**校验报错**（规则 56–60 + shots）；**`unitary()`** 对 measure/reset 报错、`ignore_nonunitary` 选项；**JSON** 往返 `basis`/`id`；**QASM** 对联合/非 Z/带 id measure 报错；**observables** 路径不变。

### 6.3 验证门槛

`PYTHONPATH=. pytest` 全绿后方可宣称完成。

---

## 7. README §4 整节重构 + CHANGELOG

### 7.1 README 新结构（删除"机制一/机制二/互斥"框架）

- **4. 量子测量** — 统一模型导言；线路内 `measure`/`reset` 为操作序列中的操作，末端读出由 `tm`/`measure_qubits` 控制。
- **4.1 运行接口与参数** — `run(...)` 签名 + 参数表。
- **4.2 线路与操作序列约定** — `cir=[O₀…O_{L−1}]`、操作下标、默认初态、measure/reset 占用操作下标。
- **4.3 线路内 `measure`：联合 Pauli 投影测量** — 非破坏保留、`basis`、`id`、`output(i)/counts(i)/prob(i)`、示例（Bell→ZZ 仍纠缠）。
- **4.4 `reset`：重置信道** — 信道语义、无前置、示例、画图说明。
- **4.5 末端测量** — `tm`、逐比特 Z（输入顺序保留）、`output(-1)`。
- **4.6 `shots` 语义** — `None`/`0` 单条精确轨迹；`M` 独立重跑；示例。
- **4.7 末态与约化：`state`/`final_state`/`snap`/`reduce`** — 各 shots 模式形状表；`sm="avg"`（注 `shot`/`cond` 待实现）。
- **4.8 期望值与 `observables`** · **4.9 从 State 直接测量** · **4.10 Result 字段/方法速查**（更新：方法 vs 字段）· **4.11 Sampler/Estimator primitives**（更新）· **4.12 输入检查与报错**（规则 56–60 + shots）。

### 7.2 CHANGELOG.md（`2026-06-13` 条目）

列出破坏性变更：统一测量模型；投影式线路内 `measure(basis,id)`；`reset` 信道（无前置）；新增 `tm/sm/seed`；`output/counts/prob` → 方法；`state`/`final_state` 语义 + `shots>1` 为 DM；`shots=0≡None` 保留（对上传规则 23 的显式偏差）且 `shots=0`/`None` 覆盖 `tm`、不做末端测量（对规则 36 的显式偏差）；`unitary()` 对非酉操作报错 + `ignore_nonunitary`；QASM 对联合/非 Z/带 id measure 报错；JSON `basis`/`id` 往返；`sm="shot"/"cond"` 待实现。

---

## 8. 文件改动地图（概览）

- 新增：`aicir/measure/projector.py`、`aicir/measure/trajectory.py`、`aicir/measure/aggregate.py`。
- 重写：`aicir/measure/measure.py`、`aicir/measure/result.py`。
- 修改：`aicir/ir/measurement.py`（`basis`/`id` 字段）、`aicir/core/circuit.py`（`measure` 工厂 + `unitary()`/`matrix()` 报错与 `ignore_nonunitary`；删除 reset 前置校验）。
- 迁移：`aicir/primitives/{sampler,estimator}.py`、`aicir/measure/estimator.py`、`aicir/vqc/{VQE,QAOA,SSVQE,VQD}.py`、`aicir/optimization/qubo/qaoa.py`、`aicir/transpile/passmanager.py`、`aicir/optimizer/circuit.py`、`aicir/qas/demos/_np_ising_utils.py`。
- I/O：`aicir/core/io/`（QASM 对不可表达 measure 报错；JSON `basis`/`id`）。
- demos：`demo_1`、`grover_demo`、`qft_demo`、`reset_demo`、`snap_demo`、`test_1`、NPU demos。
- 文档：`README.md` §4、`CHANGELOG.md`。
- 测试：`tests/measure/`、`tests/primitives/`、`tests/vqc/`、`tests/noise/`、`tests/circuit/`、`tests/transpile/`、`tests/visual/` 等的相关用例。

---

## 9. 待实现 / 后续项（明确 out-of-scope）

- `sm="shot"` 与 `sm="cond"`（按 shot 独立态列表 / 按测量历史分组的条件态）。
- QASM 自定义 pragma 以无损表达联合 Pauli 测量。
- 噪声与线路内投影 measure 的更高效实现（当前为逐轨迹 DM）。

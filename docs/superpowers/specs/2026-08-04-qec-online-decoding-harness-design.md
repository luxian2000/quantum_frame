# QEC 在线实时解码实验平台 M1（Spec）

日期：2026-08-04
状态：已批准设计，待实现

## 目标

新增 `aicir.qec` 子系统。**核心需求是为新型在线（online）实时（real-time）纠错与解码算法提供实验平台**，并支持接入 Stim / PyMatching 等既有算法库。

内置码只是**参考实现与验证语料**，不是产品本身。产品是那个平台：用户能插入自己设计的（或论文里的）**码**、**syndrome 提取调度**和**在线解码器**，三者都不需要改模块内部代码。

本 spec 只覆盖 **M1（骨架：码 + 在线循环）**。M2/M3 见文末「里程碑划分」。

## 关键决策（已确认）

1. **分三个里程碑**，各自独立 spec → plan → 实现：M1 骨架 → **M3 可视化** → M2 规模化与互操作。M2/M3 都消费 M1 的记录结构与 `DetectorLayout`，故 M1 必须先行。
2. **数据模型：detector / observable 为主，raw syndrome 并存**。采用 Stim 语义，使 M2 的互操作成为格式转换而非语义翻译。raw syndrome 一并保留，因为 M3 要画错误链，而「差分」表达不出错误链。
3. **解码器接口是流式的**，不是批式的。`reset` / `update(round, events)` / `flush`，带解码窗口、滞后提交与声明代价。
4. **因果性由架构保证，而非约定**。运行器逐轮交错「模拟 ↔ 解码」，解码器物理上无法看到未来轮次——因为未来轮次尚未被模拟。
5. **实时性 = 因果性 + 建模延迟预算**。解码器声明每轮代价（`cost_of`），用户提供 `cost_to_seconds` 硬件模型；运行器按单服务台排队模型算 backlog。**不用 Python wall-clock 冒充硬件时间**（Python 比 FPGA 慢约 10⁴ 倍），wall-clock 单独记录、字段分离。
6. **修正模式两种**：`frame`（默认，经典 Pauli frame，真实实验做法，M2 的 Stim 路径只能支持这种）与 `active`（真正施加修正门，M3 可视化用）。
7. **误差模型：逐 shot 随机 Pauli 采样**，态始终保持纯态矢量。不走密度矩阵（现有噪声路径一挂 `NoiseModel` 就升密度矩阵，17 比特需 275 TB，不可行）。
8. **测量误差作用在经典记录上**，不作用在量子态上（与 Stim 的 `X_ERROR` before `MR` 等价，更省且传播语义正确）。
9. **外部库一律可选依赖**，`qec` 核心只依赖 numpy，与仓库「从零实现、不引入外部量子 SDK」的定位一致。M1 完全不含外部库代码。
10. **放弃线路内 `if_` feedforward**。见「已被取代的早期决策」。

## 非目标（M1）

- DEM（detector error model）构造、任何 Stim / PyMatching 代码、`.stim` / `.dem` 读写 —— 属 M2。
- MWPM、union-find、任何外部解码器适配 —— 属 M2。
- `benchmark()` 参数扫描 API 与阈值估计 —— 属 M2。
- 全部可视化（逐 shot 错误轨迹 / detection 时间线 / 晶格 / 线路标注 / 在线解码时间线）—— 属 M3。
- 稳定子 tableau 模拟器（Clifford O(n²) 引擎）—— 属 Phase 2，独立 spec。
- 子系统码 / gauge 码（Bacon-Shor 等）。M1 只做稳定子码。
- 线路内 `if_` 条件反馈修正（见下）。
- NPU / 分布式后端专项适配。`qec` 经由现有 `Circuit`/`run_trajectory` 抽象天然后端无关，但 M1 不新增 NPU 脚本或专项验证。

## 已被取代的早期决策

设计对话早期选定「线路级 `if_` feedforward 修正」。**在线解码需求出现后该决策被取代**：任意新型解码器无法编译成 `if_` 条件树（`if_` 只能表达查表解码器，且多轮空时综合征的条件数随轮数指数增长）。M1 的修正在轮间由 Python 侧计算。

因此 M1 的运行器**不使用** `ControlFlow`，与 `Circuit.unitary()` / 张量网络引擎拒绝 `ControlFlow` 的既有限制无交集。

若后续需要「可编译解码器 → 线路内 feedforward」作为独立模式，另开 spec。

## 架构与组件

### 目录结构

```text
aicir/qec/
├── __init__.py        # 公开 API 再导出
├── README.md          # 中文使用手册
├── code.py            # StabilizerCode：GF(2) 辛表示核心（无线路、无后端）
├── detectors.py       # Detector / Observable / DetectorLayout
├── errors.py          # PauliErrorModel：逐 shot 采样 Pauli 错误位置
├── schedules/
│   ├── __init__.py    # Schedule 协议 + register_schedule / resolve_schedule
│   └── bare.py        # 裸 ancilla syndrome 提取（默认实现）
├── decoders/
│   ├── __init__.py    # OnlineDecoder 协议 + DecodeStep + register_decoder / resolve_decoder
│   └── lookup.py      # LookupDecoder
├── record.py          # QECShotRecord / QECResult
├── runner.py          # 交错 simulate↔decode 主循环 + TimingModel + backlog 递推
└── codes/
    ├── __init__.py    # register_code / get_code / CODES
    ├── repetition.py  # 任意奇数 d，bit-flip / phase-flip
    ├── five_qubit.py  # [[5,1,3]]
    ├── steane.py      # [[7,1,3]]
    ├── shor.py        # [[9,1,3]]
    └── surface.py     # rotated surface d=3
```

目录约定直接沿用仓库既有惯例：`codes/` + `register_code`/`CODES` 对应 `chemistry/molecules/` + `register_molecule`/`MOLECULES`；解码器与调度注册表对应 `qml.diff` 的 `register_diff`/`resolve_diff`；`qec.run(...)` 对应 `qas.run(...)`。

### 数据流与依赖方向

```text
StabilizerCode (GF(2) 辛表示)
      │
      ├──> Schedule ──> Circuit(ancilla, measure→creg, reset) + DetectorLayout
      │                                          │
PauliErrorModel ──(逐 shot 采样)──> 错误图样        │
      │                                          │
      └──> Runner: 逐轮 run_trajectory ──────────┴──> 每轮 detection events
                        │                                    │
                        │                          OnlineDecoder.update()
                        │<── DecodeStep(frame 翻转 / active 门, 声明代价)
                        ▼
                  QECShotRecord ──> QECResult
```

依赖单向。**承重约束：运行器只经 `DetectorLayout` + detection event 流向 `OnlineDecoder` 供给信息**，绝不传 `Circuit`、`StabilizerCode`、后端或量子态。这正是「M1 写的解码器在 M2 能不加修改地跑 Stim 采样数据」的前提，也是 detector 模型作为主模型的全部理由。（解码器作者可在**构造时**自行注入额外知识，代价是失去跨码可移植性——见组件 5 对 `LookupDecoder` 的说明。）

### 组件 1：`code.py` —— `StabilizerCode`

n 比特 Pauli 表示为 GF(2) 向量 `(x | z) ∈ F₂^{2n}`，符号单独记。numpy `uint8` 按行打包：

```python
class StabilizerCode:
    n: int                    # 物理比特数
    k: int                    # 逻辑比特数 = n − rank(generators)
    generators: np.ndarray    # (m, 2n) uint8，行 = 稳定子生成元 (X|Z)
    signs: np.ndarray         # (m,) uint8，0 = +1，1 = −1
    logical_x: np.ndarray     # (k, 2n)
    logical_z: np.ndarray     # (k, 2n)
    coords: dict[int, tuple]  # 可选，比特几何坐标（M3 画晶格用）
    name: str
```

公开 I/O 是 Pauli 字符串，用户不接触二进制矩阵：

```python
StabilizerCode.from_paulis(
    ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
    logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
)
```

接受 `aicir.core.operators.PauliString` 或纯 `str`。

内部用 GF(2) 而非 `PauliString`：码距搜索与查表构造是对 ~C(n,w)·3^w 个 Pauli 的内层循环，位打包 numpy 行可向量化，字符串操作不行；且这正是 Phase 2 tableau 模拟器需要的表示。

| 方法 | 语义 |
| --- | --- |
| `validate()` | 生成元两两对易 · 生成元独立（GF(2) 秩 = m）· logical 与所有生成元对易 · `logical_x[i]` 与 `logical_z[i]` 反对易、与 `logical_z[j≠i]` 对易。失败抛 `ValueError` 并**指名违规的那一对** |
| `symplectic_product(a, b)` | `a_x·b_z + a_z·b_x mod 2`；0 = 对易，1 = 反对易 |
| `syndrome(error)` | 与各生成元的辛积 → (m,) 比特 |
| `distance(max_weight=None)` | 按权重升序搜索「与所有生成元对易但不属于稳定子群」的最小权 Pauli。`None` = 不设上限、搜到为止（n ≲ 15 可行）；给定整数则搜到该权重为止，未找到时抛 `ValueError`。**精确值，或抛错——绝不返回下界猜测** |
| `logical_class(residual)` | 残余 Pauli 落在哪个逻辑陪集。返回 (k, 2) 比特数组：第 i 行 = 第 i 个逻辑比特上的 (X 分量, Z 分量)，由 residual 与 `logical_z[i]` / `logical_x[i]` 的辛积给出。全零 = 残余在稳定子群内 |

`validate()` 是模块里单位价值最高的东西：用户插入新码时**免费**得到它，而手推生成元的错误绝大多数会被它当场抓住。

### 组件 2：`detectors.py` —— 面向解码器的契约

沿用 Stim 语义（这是 M2 互操作的契约）：

- **measurement record**：一条 shot 内所有线路中测量的**扁平有序列表**；下标 i = 第 i 个执行的 `measure`。
- **`Detector`**：一组 measurement record 下标，其**奇偶在无噪声线路中确定为 0**。轮式提取下即：轮 t 的稳定子 s 读数 XOR 轮 t−1 的稳定子 s 读数；t=0 时与初态蕴含的参考值 XOR。
- **`Observable`**：一组 measurement record 下标，其奇偶给出某逻辑算符的取值。
- **`DetectorLayout`**：解码器**唯一**被允许知道的东西——detector 数、轮数、(稳定子, 轮) ↔ detector 映射、可选 `coords`。解码器在 `reset()` 时拿到它，此后只收 detection event 比特向量流。

公开辅助入口：

```python
qec.verify_schedule(code, schedule, rounds)   # 无噪声运行，断言每个 detector 恒为 0
```

detector 确定性是 syndrome 提取调度**唯一最有力的结构性检验**——它抓 CNOT 顺序错、漏掉 ancilla reset、轮 0 参考值推错。公开它，意味着用户验证**自己写的调度**时享有与内置调度同等的保障。

### 组件 3：`schedules/` —— 码 → 线路

```python
class Schedule(Protocol):
    def build_encode(self, code, logical_state: str) -> Circuit: ...
    def build_round(self, code, round_index, *, creg_name="syn") -> RoundCircuit: ...
    def build_readout(self, code, logical_state: str) -> ReadoutCircuit: ...
```

三个方法而非两个：末端逻辑读出也由调度提供（它决定在哪个基下测 data 比特、以及 `Observable` 由哪些 measurement record 下标构成）。`ReadoutCircuit` 携带该读出线路与对应的 `Observable` 定义。

`logical_state` 是长度 k 的字符串，每个字符取自 `{"0", "1", "+", "-"}`，逐位指定各逻辑比特的初态；k=1 时即 `"0"` / `"+"` 等。`build_encode` 与 `build_readout` 都接受它——读出基必须与制备基匹配（`"0"`/`"1"` → 逻辑 Z 基，`"+"`/`"-"` → 逻辑 X 基），不匹配时 `build_readout` 抛 `ValueError`。

按**轮**构建，不构建单一整体线路——运行器必须执行轮 t、暂停解码、再继续。`run_trajectory(circuit, init_state, …)` 返回 `.pre`，运行器据此在轮间串联量子态，**无需新增任何引擎机制**。

`RoundCircuit` 携带：该轮 `Circuit`、data/ancilla 比特下标、该轮各 measurement 在全局 measurement record 中的下标、该轮产生的 detector 定义。

`bare.py`（默认）：每个生成元一个 ancilla；H → 从 ancilla 向生成元 support 上各比特施加受控 P（CX/CY/CZ）→ H → `measure(creg=…)` → `reset`。轮间靠 `reset` 复用 ancilla（轨迹引擎已支持），这是 surface d=3 保持在约 10 比特而非 17 比特的原因。

**测量误差作用在经典记录上**：误差模型在测量后翻转记录比特，不作用于量子态。与 Stim 等价、更省，且传播语义正确——解码器随后基于被污染的综合征动作，这正是被测行为。

### 组件 4：`errors.py` —— `PauliErrorModel`

逐 shot、逐轮采样具体 Pauli 错误图样：

- data 错误：每个 data 比特每轮以概率 p 施加 X / Y / Z（bit-flip / phase-flip / depolarizing 三种预设）。
- measurement 错误：每个 ancilla 读数每轮以概率 p_m 翻转经典记录比特。
- 采样输出 `list[ErrorEvent]`，`ErrorEvent = (round, qubit, pauli, source)`，`source ∈ {"data", "measurement"}`。

运行器把 data 部分拼接进该轮线路的操作序列，measurement 部分在读数后施加。

### 组件 5：`decoders/` —— 在线解码契约

```python
@dataclass
class DecodeStep:
    frame_flips: np.ndarray | None      # 已提交的逻辑 frame 翻转（每逻辑比特 X/Z）
    corrections: list[tuple[int, str]] | None   # active 模式：(qubit, pauli) 门列表
    committed_through: int              # 已最终确定的轮次（含）；−1 = 尚无提交
    cost: float                         # 本次调用的声明代价

class OnlineDecoder(Protocol):
    name: str
    window: int        # 解码窗口（轮数）；1 = 逐轮即时提交
    commit_lag: int    # 窗口内滞后提交的轮数（sliding-window 的 commit region）

    def reset(self, layout: DetectorLayout) -> None: ...
    def update(self, round_index: int, events: np.ndarray) -> DecodeStep: ...
    def flush(self) -> DecodeStep: ...        # 线路结束，强制提交所有未决
    def cost_of(self, round_index: int, events: np.ndarray) -> float: ...
```

**因果性是结构性的，不是约定性的。** `update(t, …)` 是唯一输入通道，按序每轮恰调用一次；解码器不持有线路、码、量子态或未来轮次的任何引用。它无法偷看未来，因为**轮 t+1 尚未被模拟**。批式后处理平台只能靠自律保证这一点，这里它是架构性质。这是「交错循环」相对「批式运行」的具体收益。

滑窗语义：解码器缓存若干轮，在 `[t−window+1, t]` 上解码，只提交最旧的 `window − commit_lag` 轮，其余保持暂定。`committed_through` 是契约字段。运行器断言其**单调不减**，且 `flush()` 后无未决。

注册表：`register_decoder` / `resolve_decoder`，对应 `qml.diff`。

M1 内置 `LookupDecoder(code, t=None)`：构造时枚举权重 ≤ t（默认 `t = ⌊(code.distance()−1)/2⌋`）的错误，建 syndrome → 最小权修正查表；`window=1`、`commit_lag=0`，逐轮即时提交。

**这里要澄清与「承重约束」的边界，否则会读成矛盾**：约束的是**运行器交给解码器什么**——运行器只经 `reset(layout)` + `update(round, events)` 传递信息，不传线路、码或量子态。解码器**作者**当然可以在构造时注入任何额外知识；`LookupDecoder` 就是构造时吃进 `code` 来建表的。区别在于：`LookupDecoder` 因此**只能**用于它构造时那个码，而一个只依赖 `layout` 的解码器可以跑任何码、包括 M2 里 Stim 采样出的事件流。

M2 的 DEM 正是用来消除这个特例的：DEM 把「错误机制 → 翻转哪些 detector/observable + 概率」编码进 `DetectorLayout` 本身，届时查表解码器也能只靠 layout 建表。M1 的构造时注入是**已知的临时做法**，不是设计缺陷，但也不应被后续实现当作范例推广。

**M1 的诚实局限**：只有 `LookupDecoder` 时，M1 是**逐轮**解码——对多轮空时综合征建查表的规模随轮数指数增长。故 M1 验证的是在线**管道**（因果性、窗口、提交、backlog），而非优质多轮解码；后者随 M2 的 MWPM 到位。

### 组件 6：实时模型

```python
@dataclass
class TimingModel:
    round_duration: float                        # 硬件一轮时长（秒），如 1e-6
    cost_to_seconds: Callable[[float], float]    # 声明代价 → 秒
```

确定性到达（每 `round_duration` 一次）的单服务台排队：

```text
backlog[t] = max(0, backlog[t−1] + decode_time[t] − round_duration)
```

逐 shot 报告：

- `commit_latency[t]` = 排队延迟 + 解码时长 + `commit_lag × round_duration`
- `max_backlog`
- `backlog_growth` = 线性拟合斜率。**斜率为正 = 吞吐失败模式**：解码器永久落后。
- `budget_violations` = `decode_time > round_duration` 的轮数

**用声明代价而非 Python wall-clock**：Python 解码器比 FPGA 慢约 10⁴ 倍，wall-clock 对实时可行性毫无意义。`cost_of()` 让解码器声明自己的复杂度度量（如「union-find 合并次数」「松弛的边数」），`cost_to_seconds` 映射到用户掌控的硬件模型。wall-clock 也照常记录（免费），但在**独立字段**里，绝不与建模时间混同。

`TimingModel` 默认 `None` → 所有 timing 字段为 `None`。**未指定机器时不编造数字。**

### 组件 7：修正模式

- **`"frame"`（默认）**：解码器提交逻辑 frame 翻转，运行器把它 XOR 进末端 observable 读数。不施加任何门。真实实验做法，也是 M2 中 Stim 路径唯一能支持的模式。
- **`"active"`**：运行器在轮间把已提交的 Pauli 修正作为真实门施加，量子态真正回到码空间——M3 要画的就是这个。

**必须在设计里处理、而非日后踩坑的细节**：active 模式下施加修正会把**原始**稳定子读数复位，于是下一轮的朴素差分会放出一个**虚假 detection event**。因此运行器必须追踪已施加修正自身对各稳定子的贡献，在形成 detection event 前扣除。处理正确时，**两种模式交给解码器的事件流逐字节相同**。

由此得到一个廉价而高价值的测试：同码、同种子、两种模式 → 断言 detection event 流与逻辑判定完全一致。该测试的存在就是为了守住这个细节。

### 组件 8：`runner.py` —— 交错主循环

```python
qec.run(code, *, schedule="bare", errors, decoder, rounds, shots,
        logical_state="0", correction_mode="frame", timing=None,
        backend=None, seed=None, keep_records=100, keep_failures=100) -> QECResult
```

逐 shot：

1. `state ← run_trajectory(schedule.build_encode(code, logical_state), |0…0⟩, tm=False).pre`
2. `for t in range(rounds)`：
   - 从 `PauliErrorModel` 采样轮 t 的错误图样
   - 取 `schedule.build_round(code, t)`，把 data 错误门拼接进操作序列
   - `res = run_trajectory(round_circuit, state, tm=False, …)`；`state = res.pre`
   - 读 `res.classical[creg]` → 原始稳定子读数
   - 施加 measurement 误差的比特翻转
   - 形成 detection event（与上一轮差分；active 模式下扣除已施加修正的贡献）
   - `step = decoder.update(t, events)`；累计代价与 backlog
   - active 模式且 `step.corrections` 非空 → 作为门施加，更新 `state`
3. 末端逻辑读出：在逻辑基下测量 data 比特 → observable 奇偶；`decoder.flush()`；XOR 已提交 frame
4. 经 `code.logical_class(residual)` 判定 verdict：全零 → `"corrected"`；否则 k=1 时为 `"logical_x"` / `"logical_z"` / `"logical_y"`，k>1 时为 `"logical_q{i}_{x|y|z}"` 形式、多个逻辑比特出错时按下标升序以 `"+"` 连接（如 `"logical_q0_x+logical_q2_z"`）。`logical_error_rate` 统计的是 verdict ≠ `"corrected"` 的比例

**不新增任何模拟器机制**——`run_trajectory` 本就接受 `init_state`、返回 `.pre` 与逐 shot `classical`，正是串联轮次所需。

### 组件 9：`record.py` —— 记录结构

```python
@dataclass
class QECShotRecord:
    shot: int
    seed: int
    injected_errors: list[ErrorEvent]       # (round, qubit, pauli, source)
    raw_syndromes: np.ndarray               # (rounds, m) 原始稳定子读数
    detection_events: np.ndarray            # (rounds, m) 探测事件
    decode_steps: list[DecodeStep]
    commit_latency: np.ndarray | None
    backlog: np.ndarray | None
    wall_clock: np.ndarray
    observable_raw: np.ndarray              # (k,) 未修正的逻辑读数
    frame_flips: np.ndarray                 # 解码器提交的 frame
    verdict: str                            # 见运行器步骤 4 的 verdict 约定
```

`raw_syndromes` 与 `detection_events` **并存**（关键决策 2）：M3 要画错误链，而差分表达不出错误链。

```python
@dataclass
class QECResult:
    code_name: str; decoder_name: str; schedule_name: str
    rounds: int; shots: int
    records: list[QECShotRecord]
    failure_records: list[QECShotRecord]
    logical_error_rate: float
    logical_error_rate_stderr: float        # sqrt(p(1−p)/N)
    verdict_counts: dict[str, int]
    max_backlog: float | None
    mean_commit_latency: float | None
    budget_violations: int | None
    def summary(self) -> str: ...
```

**内存**：10 万 shot 的完整记录是 GB 级。`keep_records` 保留前 N 条 shot；`keep_failures` **单独**保留前 N 条失败 shot——失败样本才是真正要看的，而它们恰恰在最需要时最稀有。聚合量始终覆盖全部 shot。

### 组件 10：`codes/` —— 内置码

五个 `register_code` 条目：repetition（任意奇数 d，bit-flip / phase-flip）、`[[5,1,3]]`、Steane `[[7,1,3]]`、Shor `[[9,1,3]]`、rotated surface d=3（提供 `coords` 供 M3）。

比特预算（statevector + ancilla 复用）：全部 ≤ 17 比特，surface d=3 复用后约 10 比特。

## 错误处理

- `StabilizerCode.validate()`：抛 `ValueError`，指名违规的生成元对 / logical 对。
- `distance()`：在 `max_weight` cutoff 处抛错，不返回下界猜测。
- `verify_schedule()`：抛错并列出哪些 (detector, 轮) 触发。
- 运行器拒绝：`correction_mode="active"` 但解码器只产出 frame；`rounds < 1`；`committed_through` 回退；用户调度线路中存在未绑定 `Parameter`。

## 测试计划（`tests/qec/`）

| 文件 | 断言 |
| --- | --- |
| `test_code_algebra.py` | 对全部内置码参数化：`validate()` 通过、`distance()` 等于文档标称 d、权重 1 错误综合征非零、`logical_class` 往返一致。负例：故意破坏的生成元集合抛出正确消息 |
| `test_detectors.py` | `verify_schedule` 覆盖内置码 × 轮数 ∈ {1,3,5}：无噪声下每个 detector 恒为 0 |
| `test_lookup_decoder.py` | 穷举：每个权重 ≤ ⌊(d−1)/2⌋ 的 data 错误，单轮，判定必须为 `corrected` |
| `test_online_protocol.py` | `committed_through` 单调不减；`flush()` 后无未决；spy 解码器记录其收到的全部输入，证明不存在通往未来轮次的通道 |
| `test_correction_modes.py` | frame vs active，同种子 → detection event 流与逻辑判定**完全一致**（守住组件 7 的细节） |
| `test_timing.py` | backlog 递推与手算序列一致；`timing=None` → timing 字段为 `None` |
| **`test_custom_plugin.py`** | **在测试文件内定义一个新码（`[[4,2,2]]`）、一个新调度、一个新解码器，注册三者并端到端运行** |

`test_custom_plugin.py` 是 M1 最重要的测试——它是唯一真正证明三个扩展点为实、而非停留在设想的测试。其余测试验证内置码，那是容易的情形。

全部 statevector、≤17 比特、每测试数十 shot，套件应以秒计，留在默认 `PYTHONPATH=. pytest` 中。

## 交付物

- `aicir/qec/` 全部模块（见目录结构）
- `aicir/qec/README.md`（中文，遵循子系统手册惯例）
- `tests/qec/`（上表七个文件）
- `demos/qec_online_demo.py`
- `CHANGELOG.md` 新增一条 dated 条目
- `CONTENTS.md` 目录树补 `aicir/qec/`
- `CLAUDE.md` 子系统清单补 `aicir/qec/` 条目

## 里程碑划分

| | 里程碑 | 内容 | 完成判据 |
| --- | --- | --- | --- |
| **M1** | 骨架：码 + 在线循环 | 本 spec 全部内容 | 能写一个新型在线解码器，对任意稳定子码端到端跑通 |
| **M3** | 可视化 | 逐 shot 错误轨迹、detection 时间线热图、码布局/晶格图、QEC 标注线路图，外加实时模型才画得出的**在线解码时间线**（综合征到达 vs 提交、窗口滑动、backlog） | 能看清解码器在 aicir 采样与 Stim 采样的 shot 上分别在做什么 |
| **M2** | 规模化与互操作 | DEM 构造；Stim 导出/导入（`.stim`/`.dem`）；Stim 采样后端；PyMatching / fusion-blossom 适配；自研免依赖 MWPM；streaming union-find；`benchmark()` 扫描 API 与 opt-in 脚本 | 你的解码器能在 d=5/7 上与已发表基线正面对比 |

实现顺序 **M1 → M3 → M2**。M3 与 M2 在 M1 之后彼此独立。

`stim` / `pymatching` 在 M2 落地时加入 `pyproject.toml` 的可选 extra（与 `torch`/`scipy`/`cotengra` 同等地位），`qec` 核心始终只依赖 numpy。

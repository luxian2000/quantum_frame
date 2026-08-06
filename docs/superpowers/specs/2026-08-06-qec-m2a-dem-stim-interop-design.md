# QEC M2a：DEM 构造与 Stim/PyMatching 互操作（Spec）

日期：2026-08-06
状态：已批准设计，待实现

## 目标

给 `aicir.qec` 补上两件 M1 缺失、且互为因果的能力：

1. **一个真正的多轮在线解码器。** M1 只有 `LookupDecoder`，逐轮解码——多轮空时综合征
   的查表规模随轮数指数增长，因此 M1 交付的平台**还不能做它立身的那件事**。适配
   PyMatching 能以远低于自研 Blossom 的代价立刻补上这个缺口。
2. **一个外部正确性 oracle。** M1 开发期出现五处静默错误（active 模式逻辑错误率恒为
   零、`commit_latency` 恰好低估一个 `round_duration`、轮 0 参考值错配等），共同点是
   **一切都在拿「我自己推导的期望」校验「我自己推导的实现」**。Stim 是独立于本仓库的
   第三方实现，能把这个闭环打开。

两者共用同一个前置件：**DEM（Detector Error Model）**。

## 关键决策（已确认）

1. **M2 拆分，本 spec 只做 M2a。** M2a = DEM + Stim 导出/导入 + Stim 采样后端 +
   PyMatching 适配。自研免依赖 MWPM、streaming union-find、fusion-blossom 适配、
   `benchmark()` 扫描归入 **M2b**，另开 spec。理由：PyMatching 一旦可用，自研 Blossom
   就不在关键路径上了；而 Blossom 是整个路线图里最硬的算法件，不该压在 oracle 前面。
2. **DEM 由 aicir 自己推导，Stim 的 DEM 作为 oracle 交叉验证**，而不是从 Stim 取回。
   保住 `aicir.qec` 核心的 numpy-only 契约，同时让「两套独立推导必须一致」成为 M1
   那类静默错误的克星。
3. **DEM 描述 M1 既有的唯象（phenomenological）误差模型**——逐轮 data Pauli + 测量
   记录翻转——不是门级（circuit-level）故障。若描述另一套误差模型，oracle 比对就失去
   意义；且门级噪声需要同时扩展 `PauliErrorModel`，属范围蔓延。
4. **DEM 用解析推导，不用逐故障模拟。** 见「组件 1」的推导规则。模拟法不仅慢，更是
   **循环论证**：用正在被验证的引擎去构造用来验证它的模型。
5. **`Sampler` 协议 + 注册表**，与既有的 codes / schedules / decoders 三处注册表同构，
   不引入新概念。`qec.run(..., sampler="stim")`。
6. **PyMatching 适配走滑窗**，`window` / `commit_lag` 可配，使其成为真正的在线解码器
   而非「批式解码套一层在线外衣」。窗口覆盖全部轮次即退化为批式。
7. **oracle 以采样统计为主、DEM 结构比对为辅**（见「组件 5」的风险说明）。

## 非目标（M2a）

- 自研免依赖 MWPM（Blossom）、streaming union-find、fusion-blossom 适配 —— 属 M2b。
- `benchmark()` 参数扫描与阈值估计 —— 属 M2b。
- 全部可视化 —— 属 M3。
- 门级（circuit-level）噪声与相应 DEM。
- 稳定子 tableau 模拟器：Stim 采样后端由 Stim 自己完成，本仓库不实现 Clifford 引擎。
- 消费 `code.signs`：M1 已在构造期拒绝非 +1 号生成元（commit `6b7eb13`），M2a 维持该
  限制不变。

## 架构与组件

### 目录结构

```text
aicir/qec/
├── dem.py                   # ErrorMechanism / DetectorErrorModel / build_dem（numpy only）
├── samplers/
│   ├── __init__.py          # Sampler 协议 + register_sampler / resolve_sampler
│   ├── aicir.py             # AicirSampler：从 _run_one_shot 抽出的既有采样循环
│   └── stim.py              # StimSampler：可选依赖，import 受保护
├── interop/
│   ├── __init__.py
│   ├── export.py            # aicir → stim.Circuit / .stim 文本
│   └── import_.py           # .stim / .dem → DetectorLayout + DetectorErrorModel
└── decoders/
    └── matching.py          # PyMatchingDecoder：DEM 匹配图上的滑窗解码
```

依赖分层：`dem.py` 只依赖 numpy，无 Stim 也能用；只有 `samplers/stim.py`、`interop/`、
`decoders/matching.py` 触碰可选包，各自按 `aicir.qml` 守护 torch 的同样方式做保护性导入。

### 组件 1：`dem.py` —— DEM 的解析构造

```python
@dataclass(frozen=True)
class ErrorMechanism:
    probability: float
    detectors: tuple[int, ...]      # 该机制翻转的 detector 全局下标
    observables: tuple[int, ...]    # 该机制翻转的 observable 下标
    source: str                     # "data" | "measurement"，供诊断与可视化
    location: tuple                 # (round, qubit, pauli) 或 (round, stabilizer)

@dataclass(frozen=True)
class DetectorErrorModel:
    n_detectors: int
    n_observables: int
    mechanisms: tuple[ErrorMechanism, ...]

def build_dem(code, schedule, errors, rounds, *, logical_state="0") -> DetectorErrorModel: ...
```

**推导规则（唯象模型下为闭式，无需模拟）：**

- **data Pauli `P` 落在比特 `q`、轮 `t`（`t >= 1`）**：错误在此后持续存在，故 `raw[s]`
  从轮 `t` 起翻转，而 detector 取相邻差分 → **只在 `(s, t)` 触发**，其中 `s` 取遍与 `P`
  反对易的生成元（即 `code.syndrome(P)` 的非零位）。observable `i` 翻转当且仅当 `P` 与
  `logical_z[i]` 反对易（Z 基读出；X 基读出改用 `logical_x[i]`）。
- **测量翻转落在稳定子 `s`、轮 `t`（`t >= 1`）**：只翻转 `raw[s][t]` 这一个读数 →
  **在 `(s, t)` 与 `(s, t+1)` 各触发一次**（`t+1 == rounds` 时只有前者）。不影响
  observable。
- **轮 0 不产生任何机制**：M1 的运行器在轮 0 不注入任何错误（轮 0 是投影式制备轮）。
- **概率**：`p_data` 按 `PauliErrorModel.channel` 均分到各 Pauli（`bit_flip` → X 独占
  `p_data`；`depolarizing` → X/Y/Z 各 `p_data/3`）；测量机制取 `p_measure`。

推导全部走 GF(2)，复用 `code.syndrome()` 与既有的辛积，不构造也不执行任何线路。

**已知前提（沿用 M1，不在 M2a 修）：** observable 的翻转判据假设 `logical_z[i]` 是纯 Z
型（`logical_x[i]` 纯 X 型），与 `runner._residual_from_readout` 现有实现一致。五个内置码
均满足。用户若提供含 Y 分量的逻辑算符，M1 的运行器与 M2a 的 DEM 会以同样的方式失准——
这是既有限制，M2a 记录之，修复留待后续。

### 组件 2：`samplers/` —— 采样与解码循环解耦

```python
@dataclass
class ShotContext:
    """采样一条 shot 所需的全部输入（不含解码器——采样器不认识解码器）。"""
    code; schedule; errors; rounds: int; layout
    logical_state: str; backend; rng; shot: int; seed: int

@dataclass
class ShotSamples:
    """一条 shot 的采样产物；解码循环只消费这些字段。"""
    raw_syndromes: np.ndarray        # (rounds, m) uint8；Stim 路径无原始读数时为 None
    detection_events: np.ndarray     # (rounds, m) uint8
    observable_flips: np.ndarray     # (k,) uint8
    injected_errors: list            # ErrorEvent 列表；Stim 路径为空（Stim 不回报故障位置）

class Sampler(Protocol):
    name: str
    supports_active_mode: bool
    def sample_shot(self, ctx: ShotContext) -> ShotSamples: ...
```

**注意两处 Stim 路径的信息缺口**，它们决定了哪些 M1 记录字段在 Stim 采样下不可用：
Stim 的 detector sampler 只回报 detection events 与 observable flips，**不回报原始稳定子
读数，也不回报注入了哪些故障**。因此 `QECShotRecord.raw_syndromes` 与 `injected_errors`
在 Stim 路径下分别为 `None` 与空列表——**必须是 `None`/空而非编造的零**，与 M1
`TimingModel=None → 字段为 None` 的原则一致。M3 的逐 shot 错误轨迹图因此只能画 aicir
采样的 shot，这一点要写进 README。

`active` 模式下 `AicirSampler` 需要在轮间接受修正门，故采样器额外暴露
`apply_corrections(state, pairs)` 钩子；`StimSampler` 不实现它（`supports_active_mode`
为 `False`）。

`AicirSampler` 是把 `runner._run_one_shot` 里的采样部分原样抽出——逐轮建线路、注入错误、
`run_trajectory`、读经典寄存器。**交错的解码循环留在 `runner.py`**，两者目前纠缠在同一个
函数里，本次借机分开：这是本工作证成的定向清理，不是无关重构。

`StimSampler` 把 code/schedule/errors 导出成 `stim.Circuit`，用 Stim 的
`compile_detector_sampler()` 批量采样 detection events 与 observable flips，再逐 shot
喂给同一套在线解码器。

**Stim 路径不支持 `active` 模式**（Stim 不保留量子态，无从施加物理修正）：
`supports_active_mode = False`，`run()` 在 `sampler="stim"` 且 `correction_mode="active"`
时抛 `ValueError` 指名二者不兼容。

### 组件 3：`interop/export.py`

把 (code, schedule, errors, rounds, logical_state) 写成 `stim.Circuit`：每轮 `H`/受控门/
`MR`（measure+reset），data 错误位置写 `X_ERROR`/`Y_ERROR`/`Z_ERROR` 或 `DEPOLARIZE1`，
测量误差写 `MR` 前的 `X_ERROR`（与 M1「测量误差作用在经典记录上」严格等价），detector
写 `DETECTOR rec[-1] rec[-1-m]`，末端写 `OBSERVABLE_INCLUDE`。同时提供 `.stim` 文本落盘。

### 组件 4：`interop/import_.py`

读 `.stim` / `.dem`，产出 `DetectorLayout` 与 `DetectorErrorModel`，使已发表的
artifact 能直接进本平台跑自己的在线解码器。导入的 layout 没有 aicir 侧的 code/schedule，
因此 `round0_stabilizers` 与 `coords` 尽力而为地从文件推断，推断不出时留空并记录。

### 组件 5：oracle 策略与其风险

**主检验——采样统计一致。** 同一 code/schedule/噪声下，`AicirSampler` 与 `StimSampler`
各采 N shot，逐 detector 比对触发频率是否落在统计容差内（二项分布，按 N 与 p 定阈）。
端到端、与双方 DEM 的内部表示无关，是最稳的检验。

**辅检验——DEM 结构比对。** 本仓库的 `build_dem` 输出 vs
`stim.Circuit.detector_error_model()`。

> **风险，现在就写明：** Stim 会合并重复机制，并可能把误差分解成 graphlike 片段，因此
> **集合级严格相等很可能对不上，而且未必是 bug**。故此项以 `decompose_errors=False`
> 降低分歧，比对「诱导出的 detector 翻转集合」与「按类汇总的概率」，且**失配判定为
> 「待查」而非「失败」**——只有当采样统计检验同时不一致时才升级为缺陷。若一开始就宣称
> 「两个 DEM 必须严格相等」，一旦结构上做不到，看起来会像 oracle 坏了，其实没坏。

### 组件 6：`decoders/matching.py` —— `PyMatchingDecoder`

由 DEM 构造 PyMatching 的匹配图（每个机制一条超边，概率转权重）。滑窗：缓存 `window`
轮 detection events，在窗上解码，提交最旧的 `window - commit_lag` 轮，其余保持暂定；
`flush()` 强制提交剩余。`cost_of` 声明代价按窗内 detection event 数计，接入 M1 的
`TimingModel`。窗口覆盖全部轮次即退化为批式解码。

## 错误处理

- `build_dem` 在 `rounds < 2` 时抛错（与 `run()` 一致：轮 0 不产生机制，`rounds=1` 的
  DEM 必然为空，是个无意义对象）。
- `sampler="stim"` 而 `stim` 未安装 → `ImportError` 指明安装 `pip install "aicir[qec]"`。
- `sampler="stim"` + `correction_mode="active"` → `ValueError` 指名不兼容。
- `PyMatchingDecoder` 在 `pymatching` 缺失时构造即抛 `ImportError`。
- 导入的 `.dem` 与当前 layout 的 detector 数不符 → `ValueError` 给出两侧数值。

## 测试计划（`tests/qec/`）

| 文件 | 断言 |
| --- | --- |
| `test_dem.py` | 手算小例逐条核对机制：data X 在 (q,t) 只产生 `(s,t)`；测量翻转产生 `(s,t)` 与 `(s,t+1)`；末轮测量翻转只产生一个；轮 0 无机制；概率按 channel 均分；observable 翻转判据 |
| `test_stim_export.py` | 导出文本可被 `stim.Circuit(...)` 解析；detector 数与 `build_layout` 一致；无噪声导出电路的 detector 在 Stim 下恒为 0（与 `verify_schedule` 同一断言，换 Stim 验） |
| `test_stim_oracle.py` | **主 oracle**：aicir 与 Stim 采样的逐 detector 触发频率在统计容差内一致（固定种子）；辅：DEM 结构比对，失配只 warn |
| `test_stim_import.py` | 往返：导出再导入得到等价 layout/DEM；detector 数不符时报错 |
| `test_samplers.py` | `Sampler` 注册表；`AicirSampler` 与 M1 直跑 `run()` 结果逐字节一致（重构无回归）；stim+active 组合被拒 |
| `test_pymatching_decoder.py` | 滑窗提交语义（`committed_through` 单调、`flush` 收尾）；窗口覆盖全程时与批式解码结果一致；**多轮收益**见下方说明 |

`stim`/`pymatching` 缺失时相关文件整体 `pytest.importorskip`，与仓库既有 torch/scipy
可选依赖测试的处理方式一致。

**关于「多轮解码收益」这条断言的写法。** 直觉上 `PyMatchingDecoder` 在
`rounds >= 4` 且测量噪声不可忽略时应当明显优于逐轮的 `LookupDecoder`——那正是做 M2a 的
理由。但这是**统计性断言**，直接写成「A 的错误率低于 B」会得到一个随种子飘的脆弱测试，
而脆弱测试的结局通常是被放宽到失去意义。故按如下方式落地：
- 用固定种子 + 足够 shot 数（≥400），并选一个测量误差主导的配置（`p_measure` 与
  `p_data` 同量级，`rounds=6`）——这是逐轮解码结构性吃亏、而空时匹配结构性占优的区间。
- 断言留出明确余量（`lookup_rate - matching_rate > 0.05`），而不是断言严格小于。
- 若该测试失败，**先怀疑适配器接线（权重、匹配图、滑窗提交），而不是放宽阈值**。
真正的量化对比属于 M2b 的 `benchmark()` 扫描，本测试只作方向性回归守卫。

**必须同时修改的既有测试：** `tests/qec/test_public_api.py::test_qec_core_has_no_optional_dependencies`
目前 grep 整个 `aicir/qec/`，M2a 之后必然失败。改为只断言**核心模块**
（`code.py`/`codes/`/`detectors.py`/`errors.py`/`record.py`/`runner.py`/`schedules/`/
`dem.py`/`decoders/__init__.py`/`decoders/lookup.py`）无可选依赖，允许
`samplers/stim.py`、`interop/`、`decoders/matching.py` 触碰。

## 交付物

- `aicir/qec/dem.py`、`samplers/`、`interop/`、`decoders/matching.py`
- 上表六个测试文件 + `test_public_api.py` 的范围收窄
- `pyproject.toml` 新增 `qec = ["stim", "pymatching"]` extra，并入 `all` / `dev`
- `aicir/qec/README.md` 增补：DEM、Stim 互操作、PyMatching 解码器、oracle 的用法与局限
- `scripts/npu/qec_probe.py` 不改（DEM/Stim 路径与 NPU 无关，纯 CPU/numpy）
- `CHANGELOG.md` dated 条目

## 与其他里程碑的关系

| | 里程碑 | 状态 |
| --- | --- | --- |
| M1 | 骨架：码 + 在线循环 | 已交付（`9e03d88` + 修复 `6b7eb13`），单卡 NPU 实测 7/7 |
| **M2a** | **DEM + Stim/PyMatching 互操作** | **本 spec** |
| M2b | 自研 MWPM、union-find、fusion-blossom、`benchmark()` 扫描 | 待 spec |
| M3 | 可视化（含在线解码时间线） | 待 spec |

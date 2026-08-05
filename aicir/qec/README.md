# aicir.qec 量子纠错实验平台

`aicir.qec` 是一个量子纠错（Quantum Error Correction）在线实时解码实验平台，
面向**新型在线实时纠错/解码算法**的研究：如何在综合征（syndrome）逐轮到达的
约束下，用一个只能看到"已经发生"的信息、且有硬实时预算的解码器，把逻辑错误率
压到可用水平。

三处显式插拔点——码、syndrome 提取调度、在线解码器——均可替换，且**不需要
改动 `aicir/qec/` 内部任何一行代码**：新增一个码只是调用
`StabilizerCode.from_paulis` 再 `register_code`；新增一个调度只是实现三个
方法再 `register_schedule`；新增一个解码器只是实现四个方法再
`register_decoder`。内置的 repetition / five_qubit /
steane / shor / surface(d=3) 五个码是参考实现与验证语料（用来在开发新调度/新
解码器时有稳定的地基可站），**不是这个平台要交付的产品**——产品是这套插拔
架构本身。

当前为 **M1（骨架）**：GF(2) 码代数、detector/observable 数据模型、在线解码
协议、交错运行器、实时预算模型全部就绪，但只有一个内置解码器（逐轮查表）、
一个内置调度（裸 ancilla）。M2（DEM/Stim/PyMatching 互操作、`benchmark()`
批量扫描）与 M3（可视化）见文末「里程碑」一节。

## 快速开始

```python
from aicir.qec import get_code, PauliErrorModel, LookupDecoder, run

code = get_code("steane")                       # 取一个内置码
errors = PauliErrorModel(p_data=0.02, p_measure=0.01, channel="depolarizing")
decoder = LookupDecoder(code)                    # 逐轮查表解码器，构造时吃进 code

result = run(code, errors=errors, decoder=decoder, rounds=5, shots=500, seed=0)
print(result.summary())
```

```text
码 steane · 调度 bare · 解码器 lookup
轮数 5 · shots 500
逻辑错误率 0.074 ± 0.0117
判定分布 {'corrected': 463, 'logical_x': 37}
```

`run(...)` 逐 shot 交错「模拟 ↔ 解码」：制备逻辑初态 → 逐轮（注入错误 → 提取
综合征 → 喂给解码器）→ 末端逻辑读出 → 与解码器声称已修正的部分比较，判定
是否发生了不可纠的逻辑错误。`result.summary()` 给出逻辑错误率及其标准误、
判定分布，以及（若传了 `timing=`）实时预算的聚合统计。`result.failure_records`
保留了失败 shot 的完整记录，可用来复盘具体错误链（见 demo）。

## 三个扩展点

### 1. 插入新码

用 `StabilizerCode.from_paulis(...)` 从 Pauli 串直接构造，再用 `register_code`
挂进全局注册表，之后就能用 `get_code(名字)` 取用——与内置码完全等价：

```python
from aicir.qec.code import StabilizerCode
from aicir.qec.codes import register_code, get_code

def build_422() -> StabilizerCode:
    """[[4,2,2]] 检测码：两个权重 4 的生成元，两个逻辑比特。"""
    return StabilizerCode.from_paulis(
        ["XXXX", "ZZZZ"],
        logical_x=["XXII", "XIXI"],
        logical_z=["ZIZI", "ZZII"],
        name="detection_422",
        coords={q: (0, q) for q in range(4)},
    )

register_code("detection_422", build_422)
code = get_code("detection_422")
code.validate()          # 见下——这一行是免费拿到的最有力保障
```

`code.validate()` 是新码定义时**最应该第一时间调用**的方法：它逐一检验生成元
两两对易、生成元线性无关（GF(2) 秩满）、`n − rank == k`、`logical_x`/`logical_z`
都落在 normalizer 内（与全部生成元对易）、以及逻辑算符间的辛积满足
`{logical_x[i], logical_z[j]} = δ_ij`。任何一项不满足都会抛 `ValueError` 并
指名具体是哪个生成元/哪对逻辑算符出了问题——手推 Pauli 串时最容易犯的符号、
对易性错误，`validate()` 一次调用全覆盖，不需要额外写测试去分别验证这些代数
性质。

### 2. 插入新调度

调度把一个 `StabilizerCode` 编译成逐轮线路。实现 `Schedule` 协议的三个方法——
`build_encode(code, logical_state)`（制备逻辑初态）、
`build_round(code, round_index, *, creg_name="syn")`（一轮综合征提取，返回
`RoundCircuit`）、`build_readout(code, logical_state)`（末端逻辑读出，返回
`ReadoutCircuit`）——再用 `register_schedule` 注册：

```python
from aicir.core.circuit import Circuit, cx, cy, cz, hadamard, measure, reset
from aicir.core.classical import ClassicalRegister
from aicir.qec.code import gf2_to_pauli
from aicir.qec.schedules import (BareAncillaSchedule, RoundCircuit,
                                 register_schedule, verify_schedule)
from aicir.qec.codes import get_code

_CONTROLLED = {"X": cx, "Y": cy, "Z": cz}

class ReversedOrderSchedule(BareAncillaSchedule):
    """与内置裸 ancilla 调度相同，但受控门按比特降序施加。"""
    name = "reversed"

    def build_round(self, code, round_index, *, creg_name="syn") -> RoundCircuit:
        data = tuple(range(code.n))
        ancilla = tuple(range(code.n, code.n + code.m))
        reg = ClassicalRegister(code.m, creg_name)
        cir = Circuit(n_qubits=code.n + code.m)
        for j in range(code.m):
            anc = ancilla[j]
            labels = gf2_to_pauli(code.generators[j])
            cir.append(hadamard(anc))
            for q in range(code.n - 1, -1, -1):        # 降序——与内置调度相反
                if labels[q] != "I":
                    cir.append(_CONTROLLED[labels[q]](q, [anc]))
            cir.append(hadamard(anc))
        cir.append(measure(list(ancilla), creg=reg))
        cir.append(reset(list(ancilla)))
        return RoundCircuit(circuit=cir, creg_name=creg_name, ancilla_qubits=ancilla,
                            data_qubits=data, record_offset=int(round_index) * code.m)

register_schedule("reversed", ReversedOrderSchedule)
verify_schedule(get_code("steane"), ReversedOrderSchedule(), rounds=3)   # 见下
```

**`verify_schedule(code, schedule, rounds)` 是新调度定义时必须跑的检验**，而
且应该在把调度用到任何真实实验之前就跑：它无噪声运行若干个 shot，断言每个
detector（相邻两轮同一稳定子读数的差分，轮 0 与固定参考值的差分）严格恒为
0。这是提取调度**唯一最有力的结构性检验**——它能抓住 CNOT 顺序写错、漏掉
ancilla reset、轮 0 "确定生成元集合"推错这类会让 detector 在无噪声下也不为
0 的 bug。不满足时它会抛 `ValueError` 并指名具体是哪个 shot、哪一轮、哪个
稳定子的 detector 被触发。公开这个函数，使自定义调度享有与内置 `bare`
调度完全相同等级的保障。

### 3. 插入新在线解码器

实现 `reset(layout)` / `update(round_index, events) -> DecodeStep` /
`flush() -> DecodeStep` / `cost_of(round_index, events) -> float` 四个方法，
声明 `name` / `window` / `commit_lag` 三个属性，再用 `register_decoder`
注册：

```python
import numpy as np
from aicir.qec.decoders import DecodeStep, register_decoder

class SlidingMajorityDecoder:
    """滑窗解码器：缓存 window 轮，滞后 commit_lag 轮才提交——不是好解码器，
    只是用来证明"带窗口与滞后提交的在线解码器"能被平台正确驱动。"""
    name = "sliding_majority"

    def __init__(self, window: int = 3, commit_lag: int = 1):
        self.window, self.commit_lag = int(window), int(commit_lag)

    def reset(self, layout) -> None:
        self._buffer, self._seen, self._committed = [], [], -1

    def cost_of(self, round_index, events) -> float:
        return float(self.window)          # 声明代价：不是 wall-clock

    def update(self, round_index, events) -> DecodeStep:
        self._buffer.append(np.asarray(events, dtype=np.uint8))
        self._seen.append(int(round_index))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        target = int(round_index) - self.commit_lag
        self._committed = max(self._committed, target)
        return DecodeStep(frame_flips=None, corrections=None,
                          committed_through=self._committed,
                          cost=self.cost_of(round_index, events))

    def flush(self) -> DecodeStep:
        if self._seen:
            self._committed = max(self._committed, max(self._seen))
        return DecodeStep(committed_through=self._committed, cost=0.0)

register_decoder("sliding_majority", SlidingMajorityDecoder)
```

`window` 是解码器一次决策依赖的轮数上限（纯参考信息，运行器不会据此截断
喂给它的数据）；`commit_lag` 是解码器允许把"最终判定"滞后多少轮才给出（例如
本例中窗口解码器要等到看过后续轮次才敢对某一轮下结论）；`committed_through`
是解码器在每次 `update`/`flush` 返回时**必须单调不减**地报告"我已经对哪一轮
及之前给出了最终判定"——运行器会在它回退时立即抛 `ValueError`。`update` 的
`frame_flips`/`corrections` 可以为 `None`（表示这一步没有新的最终判定要
提交，正常发生在 `commit_lag > 0` 的窗口解码器身上）；`flush()` 在线路跑完
后调用一次，用来强制冲掉所有还没提交的窗口状态。

## 在线与实时语义

**因果性是结构性的，不是约定性的。** `update(round_index, events)` 是解码器
获得信息的唯一通道，运行器按序每轮恰好调用一次，此时轮 `round_index + 1`
及之后的综合征、线路、量子态**根本还没有被模拟出来**——不是"解码器答应不
偷看"，而是"要偷看的东西此刻在计算机内存里还不存在"。批式后处理平台（先把
全部轮次的综合征模拟完，再一次性喂给解码器）只能靠开发者自律来保证这一点，
这里它是运行器主循环的执行顺序保证的架构性质。

`TimingModel` 建模实时预算用的是解码器通过 `cost_of(...)` **声明的代价**，
而不是 Python 解释器实测的 wall-clock。原因很直接：这个平台里的解码器是
Python 对象，跑在 CPython 解释器里；真实的在线解码器跑在 FPGA/ASIC 上，
两者的时钟数量级相差约 **10⁴ 倍**。如果直接拿 Python `time.perf_counter()`
的实测秒数去判断"能不能追上综合征到达速度"，结论会完全没有意义——一个
Python 查表函数实测几十微秒，看起来"来不及"，但同样的查表逻辑烧到 FPGA 上
可能只要几纳秒。所以解码器作者要在 `cost_of` 里给出一个与硬件无关的复杂度
度量（本仓库的 `LookupDecoder` 就固定返回 `1.0`，表示"一次哈希查表"），
研究者再通过 `TimingModel(round_duration=..., cost_to_seconds=lambda c: ...)`
把这个度量映射到自己假想的目标硬件上（例如"每单位代价对应多少纳秒"）。
两套时间各自留痕：`QECShotRecord.wall_clock` 记录 Python 实测秒数（免费拿到，
但仅供参考），`QECShotRecord.decode_times`/`backlog`/`commit_latency` 记录
按 `TimingModel` 建模出的目标硬件时间，两者绝不混同、绝不互相替代。

**backlog 斜率为正即吞吐失败模式。** `backlog_sequence` 实现的是单服务台
排队的 Lindley 递推：`backlog[t] = max(0, backlog[t-1] + decode[t] - 轮时长)`。
只要解码器某一轮偶尔算得慢一点，只要 `round_duration` 留有余量，backlog
会在后续轮次里被追平、回落到 0——这是正常抖动。但如果解码器**平均**每轮
花费的（建模）时间超过了 `round_duration`，backlog 会随轮数线性增长、永不
回落——这才是真正的吞吐失败：无论线路跑多久，解码器都会越落越远，最终
撑爆任何有限的缓冲区。`result.max_backlog` 报告观测到的峰值，但真正该看的
是 `record.backlog` 这条序列的**趋势**：单轮偶尔超预算（`budget_violations`
计数的对象）是可以容忍的抖动，backlog 呈线性上升趋势才是不可用的信号。

## 两种修正模式

`run(..., correction_mode=)` 支持两种解码器如何"生效"的语义。

**`"frame"`（默认）**：解码器从不真的对量子态施加任何门，只在内部记账"我
认为哪些逻辑算符已经被翻转了"（`frame_flips`），运行器逐轮异或累积这份账本，
最后拿它去和末端读出结果比较。这是文献里"Pauli frame tracking"的标准做法：
对纠错码，同一个逻辑等价类中的所有物理修正效果相同，没必要真的施加门，
记账即可。默认使用它是因为它更快（不用额外跑修正门的线路）、且解码器只需
要实现 `frame_flips` 这一半协议。

**`"active"`**：解码器每轮通过 `DecodeStep.corrections` 返回一份门列表
（`[(qubit, pauli), ...]`），运行器把它当作真实的门**物理施加**到量子态上。
这更贴近某些真实实验（尤其是需要中途读出逻辑值、或后续操作依赖修正后物理
态的场景），代价是每轮多跑一次修正线路。选用 `active` 时解码器**必须**在
每轮 `update` 里填充 `corrections`（`frame_flips` 可以省略）——若某轮返回
`corrections=None`，运行器会立即抛 `ValueError` 而不是静默退化。

**⚠️ 开发中发现过的问题，务必了解**：`active` 模式一度存在重复计数 bug——
物理修正已经被烧进了态与读数里，但解码器的 frame 记账又被叠加应用了一次，
两者刚好抵消，使 `active` 模式报出的逻辑错误率**恒为 0**，而这个 0 是假的、
不是"解码器很好"的信号。根因有两处：(1) 施加物理修正会把原始稳定子读数
复位，若不从下一轮的朴素差分里扣除"刚施加的修正贡献"，就会放出一个虚假
detection event；(2) 末端读出已经把累积施加的物理修正烙进了态里，若再和
frame 记账的判定逻辑叠加，就会把同一份修正数了两次、正负抵消。两处已修复
（详见 `runner.py` 的 `_applied_syndrome_delta` 与 `_run_one_shot` 中末端
读出的回退修正），现在 `frame` 与 `active` 两种模式对同一份错误注入必须给出
逐字节相同的 detection event 流与完全一致的判定（`test_correction_modes.py`
把这当作核心回归测试）。**任何在此修复之前跑出来的 `active` 模式逻辑错误率
数字都不可信**——不是"结果偏低"，而是恒为零，看起来像一个好得不像话的结果，
这正是基准测试工具最危险的失败模式：不报错、不崩溃，只是安静地给出错误答案。
持有旧结果的人应当重新跑一遍。

## 已知局限

1. **`LookupDecoder` 只做逐轮解码，不做多轮空时（spacetime）解码。** 它把
   每一轮的综合征独立查表、独立提交修正，不利用相邻轮次之间的关联信息。
   一个真正的多轮解码器需要在"轮 × 稳定子"这个二维空时图上找最优修正
   （例如 MWPM 在 detector graph 上找最小权完美匹配），而查表法要覆盖多轮
   空时综合征空间，表的大小相对于轮数是**指数**增长的，构造时枚举权重 ≤t
   的错误在 M1 单轮场景下已经勉强可行，多轮场景不可行。好的多轮在线解码
   等 M2 引入 MWPM 后端。

2. **重复码的 `distance()` 返回 1，不是构造时传入的 `d`。** 重复码的全部
   稳定子生成元都是同一类型（`basis="Z"` 时全是 `Z_iZ_{i+1}`），因此在
   *未受保护* 的那个基（`basis="Z"` 对应的 X 基）上存在权重 1 的逻辑算符——
   `logical_x = X^{⊗d}` 权重是 `d`，但 `logical_z = Z_0` 权重只有 1，且
   `distance()` 取两者的最小权重。`d` 只是它针对*受保护*那个基（本例中是
   Z 型稳定子能检测的 X 型噪声）的**有效距离**，不是这个码在一般 Pauli
   噪声下的真实码距。这不是 bug，是重复码本身只保护单一噪声类型的物理
   事实。**后果是 `LookupDecoder(code)` 对重复码不能用默认的
   `t=(distance()-1)//2`**（会算出 `t=0`，无意义，构造时直接抛
   `ValueError`）——必须显式传 `t` 与 `error_basis`，例如
   `LookupDecoder(get_code("repetition", d=5, basis="Z"), t=2, error_basis="X")`。

3. **`LookupDecoder` 在构造时就吃进了 `code`，因此绑定到那一个码上。**
   这不违反"解码器运行期承重约束"——运行器与解码器之间在运行期仍然只经
   `reset(layout)` + `update(round_index, events)` 传递信息，`LookupDecoder`
   在 `reset` 时也只是把已经建好的表挂上去，并不会再向运行器索要 `code`
   以外的东西。但这是 **M1 的一个临时取巧做法**：一个只依赖 `layout`
   （不在构造时预先知道具体码）的解码器可以跑任何码，包括 M2 里由 Stim
   采样出来、平台本身并不知道其码定义细节的事件流；而 `LookupDecoder`
   为了在 M1 免费换来一个能用的解码器，选择了"构造时就知道码是什么，
   照着它把表建出来"这条捷径。**不应该把这个模式当作后续解码器实现的
   范例照抄**——M2 引入的 DEM（Detector Error Model）正是要把"解码器
   需要知道码的哪些细节"这件事从"构造时硬编码"变成"运行期通过标准化的
   DEM 描述传递"，从根本上消除这个特例。

4. **Z 基运行观测不到逻辑 Z 错误。** `run(..., logical_state="0")` 在
   Z 基下制备、读出：末端读出给出的是逻辑 **Z** 算符的本征值，因此只有
   翻转了逻辑 Z 本征值的错误（即带有 X 分量的逻辑错误：`logical_x`、
   `logical_y`）才会被观测到；一个纯逻辑 Z 错误对 Z 基读出完全透明——
   不是没发生，是这次实验的读出方式看不见它。这是**物理上正确**的
   行为，真实的容错实验也是分别跑 X 基与 Z 基两轮独立实验来分别刻画
   两类逻辑错误率的；`logical_state="+"` 对应的 X 基运行覆盖的正好是
   互补的另一半（观测逻辑 X 本征值，能看见带 Z 分量的逻辑错误，看不见
   纯逻辑 X 错误）。**因此不要把某一次 Z 基运行报出的逻辑错误率当作
   "这个码/这个解码器的"逻辑错误率**——它只是其中一半，完整的刻画需要
   两个基都跑。

5. **轮 0 是投影式制备，不是一轮纠错。** `run(...)` 的默认 `rounds=2`
   而不是 1，原因是：`build_encode` 制备出的 `|0…0⟩`（或 `|+…+⟩`）
   在一般的稳定子码里并不在码空间内——只有该码里"制备基下确定"的那部分
   生成元（`|0⟩` 制备对应纯 Z 型生成元，`deterministic_round0` 给出具体
   下标集合）本征值确定为 +1，其余生成元的本征值是 50/50 随机的；正是
   *第一轮*综合征提取把态**投影**进了码空间的一个确定分支。因此 M1 的
   错误注入从轮 1 开始，轮 0 不注入任何错误——轮 0 探测的是制备是否
   正确，不是"第一轮纠错能力"。任何想看到"注入错误 → 被纠正"这个完整
   闭环的实验都至少需要 `rounds >= 2`。连带的一点：并非每个生成元都在
   轮 0 有 detector，只有 `deterministic_round0` 给出的那些才有——对
   非 CSS 的 `[[5,1,3]]`（five-qubit）码，这个集合是**空的**（0/4，
   因为它没有任何纯 Z 型生成元），意味着该码的轮 0 完全没有 detector，
   `DetectorLayout.round0_stabilizers` 是空元组。

6. **`active` 模式下，`flush()` 给出的修正不会被物理施加。** 运行器只对
   逐轮 `update(...)` 返回的 `DecodeStep.corrections` 调门去修正量子态；
   `flush()` 返回的 `corrections`（如果有的话）只会被并入 `frame` 记账
   （若非 `None`），**不会**再补一次物理施加。对本仓库自带的
   `LookupDecoder`（逐轮即时提交，`flush()` 恒无新增判定）这无影响。
   但如果未来实现一个 `commit_lag > 0` 的窗口解码器、并选择在
   `flush()` 里才吐出这些滞后判定对应的物理修正，这些修正会被**静默
   丢弃**而不是报错——这是 `active` 修正模式当前的一处协议缺口，使用
   窗口/滞后解码器搭配 `active` 模式前请先确认这一点是否影响你的实验
   结论（`frame` 模式不受影响，它从不依赖物理施加）。

## 里程碑

**M1（本次交付）**：`StabilizerCode` 的 GF(2) 辛表示代数（生成元对易性、秩、
`distance()`、`logical_class`/`verdict` 判定）；`Detector`/`Observable`/
`DetectorLayout` 沿用 Stim 语义的数据模型；`Schedule` 协议 + 裸 ancilla 调度
（`BareAncillaSchedule`）+ `verify_schedule` 结构性检验；`PauliErrorModel`
逐 shot 随机 Pauli 错误注入（数据错误作用于态、测量错误翻转经典读数位）；
`OnlineDecoder` 流式协议（`reset`/`update`/`flush`/`cost_of`，因果性由运行
顺序结构性保证）+ 参考实现 `LookupDecoder`；交错运行器 `run(...)`（frame /
active 双修正模式）；`TimingModel` 声明代价实时预算模型（backlog、提交延迟、
预算超限计数）；`repetition`/`five_qubit`/`steane`/`shor`/`surface(d=3)`
五个内置码；三处均已验证可插拔（`test_custom_plugin.py` 端到端跑通自定义码、
自定义调度与自定义在线解码器三者组合）。

**M2（规划中）**：Detector Error Model（DEM）导出，使解码器不再需要在构造
时知道具体码的细节（消除本文档「已知局限」第 3 点的临时做法）；与 Stim/
PyMatching 的互操作（本模块的 detector/observable 数据模型已经按 Stim 语义
设计，为此做准备）；真正的多轮空时最小权完美匹配（MWPM）解码器（覆盖
「已知局限」第 1 点）；`benchmark()` 批量扫描入口（跨噪声率/轮数/码距的
逻辑错误率曲线，用于估计阈值）；更大规模的表面码（当前只内置 d=3）。

**M3（规划中）**：可视化——detector 图/错误链的图形化展示，逻辑错误率 vs.
物理错误率的阈值曲线绘制，backlog/提交延迟随时间的实时监控图。

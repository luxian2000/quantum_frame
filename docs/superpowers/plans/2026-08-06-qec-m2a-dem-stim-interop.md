# QEC M2a：DEM 与 Stim/PyMatching 互操作 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `aicir.qec` 补上 DEM（Detector Error Model）构造、Stim 导出/导入、Stim 采样后端与 PyMatching 滑窗解码器——即「一个真正的多轮在线解码器」与「一个外部正确性 oracle」。

**Architecture:** DEM 在唯象误差模型下**解析推导**（纯 GF(2)，复用 `code.syndrome()`，不构造也不执行线路）；Stim 只作为独立第三方实现用于交叉验证与规模化采样，绝不作为 DEM 的来源。`Sampler` 协议把「采样」从「交错解码循环」里切出来，`StimSampler` 因此可以完全绕开 aicir 模拟器。

**Tech Stack:** Python、numpy（核心唯一硬依赖）；`stim` 1.15 与 `pymatching` 为可选依赖（本机已装）。

**Spec:** `docs/superpowers/specs/2026-08-06-qec-m2a-dem-stim-interop-design.md`（commit `6fbc683`）

## Global Constraints

- 仓库根目录运行；测试 `PYTHONPATH=. pytest`。包未 pip 安装。
- **`aicir/qec/` 的核心模块只依赖 numpy**：`code.py`、`codes/`、`detectors.py`、`errors.py`、`record.py`、`runner.py`、`schedules/`、**`dem.py`**、`decoders/__init__.py`、`decoders/lookup.py`。只有 `samplers/stim.py`、`interop/`、`decoders/matching.py` 可触碰可选包，且必须做保护性导入。
- 注释 / docstring / README 用**中文**，测试文件同样适用。
- **提交信息中绝对不得出现 `Co-Authored-By` 或任何 Anthropic/Claude 署名。**
- GF(2) 约定不变：`(x|z)`，x 块在前各宽 n；辛积 0=对易、1=反对易；数组 `uint8`。
- 比特编号不变：data `0..n-1`，ancilla `n..n+m-1`，ancilla j 测生成元 j。
- **轮 0 是投影式制备轮，不产生任何误差机制**；`rounds >= 2`（`run()` 已在 `6b7eb13` 强制）。
- 每个 Task 结束提交一次。

## 已验证的外部 API（照抄，勿臆测）

```python
import stim, pymatching
c = stim.Circuit(text)                      # 从文本解析
c.num_detectors ; c.num_observables
dem = c.detector_error_model(decompose_errors=False)   # -> stim.DetectorErrorModel
dem.num_errors                              # 可迭代；每条 ins.type=="error"
ins.args_copy()      # -> [probability]
ins.targets_copy()   # -> targets，str(t) 形如 "D0" / "L0"
det, obs = c.compile_detector_sampler().sample(shots=N, separate_observables=True)
# det: (N, num_detectors) bool ; obs: (N, num_observables) bool

pymatching.Matching.from_check_matrix(H, weights=w, faults_matrix=F)
m.decode(syndrome_1d)        # -> np.ndarray, 长度 num_observables
m.decode_batch(syndrome_2d)  # -> (N, num_observables) uint8
```

**注意 `from_check_matrix` 的存在改变了分层**：`PyMatchingDecoder` 可以直接从**我们自己的 DEM** 建图，**不需要 stim**。故 `decoders/matching.py` 只依赖 `pymatching`，不依赖 `stim`。

## M1 既有 API（本计划要消费的）

```python
from aicir.qec.code import StabilizerCode, pauli_to_gf2, gf2_to_pauli, symplectic_product
from aicir.qec.codes import get_code
from aicir.qec.detectors import Detector, DetectorLayout, Observable
from aicir.qec.schedules import (BareAncillaSchedule, build_layout, resolve_schedule,
                                 deterministic_round0, verify_schedule)
from aicir.qec.errors import ErrorEvent, PauliErrorModel
from aicir.qec.decoders import DecodeStep, register_decoder, resolve_decoder
from aicir.qec.record import QECShotRecord, QECResult
from aicir.qec.runner import run, TimingModel, collect_noiseless_syndromes, _read_creg
```

- `build_layout(code, schedule, rounds, *, logical_state="0") -> DetectorLayout`；`DetectorLayout` 有 `.n_detectors`、`.n_rounds`、`.n_stabilizers`、`.detectors`（每个 `Detector` 有 `.index`/`.records`/`.stabilizer`/`.round_index`）、`.observables`、`.round0_stabilizers`、`.detection_events(raw, t, reference)`。
- `code.syndrome(vec) -> (m,) uint8`；`code.logical_x`/`logical_z` 形状 `(k, 2n)`。
- `PauliErrorModel(p_data, p_measure, channel)`，`channel ∈ {"bit_flip","phase_flip","depolarizing"}`。
- `DecodeStep(frame_flips, corrections, committed_through, cost)`；运行器对 `frame_flips` 做 `frame ^= ...`，故它是**增量**。

---

### Task 1: `dem.py` —— DEM 的解析构造

**Files:**
- Create: `aicir/qec/dem.py`
- Modify: `aicir/qec/errors.py`（新增 `PauliErrorModel.paulis` 属性）、`aicir/qec/__init__.py`（追加导出）
- Test: `tests/qec/test_dem.py`

**Interfaces:**
- Consumes: `build_layout`、`code.syndrome`、`symplectic_product`、`pauli_to_gf2`
- Produces:
  - `ErrorMechanism(probability: float, detectors: tuple[int,...], observables: tuple[int,...], source: str, location: tuple)`
  - `DetectorErrorModel(n_detectors: int, n_observables: int, mechanisms: tuple[ErrorMechanism,...])`，方法 `.check_matrix() -> np.ndarray (n_detectors, n_mech)`、`.faults_matrix() -> np.ndarray (n_observables, n_mech)`、`.weights() -> np.ndarray (n_mech,)`
  - `build_dem(code, schedule, errors, rounds, *, logical_state="0") -> DetectorErrorModel`
  - `PauliErrorModel.paulis -> tuple[str, ...]`

**推导规则（本任务的全部要点，务必逐条落实）：**
- data Pauli `P` 在比特 `q`、轮 `t`（`1 <= t < rounds`）：错误**持续存在**，`raw[s]` 自轮 t 起翻转，detector 取相邻差分 → **只在 `(s, t)` 触发**，`s` 取 `code.syndrome(P)` 的非零位。
- 测量翻转在稳定子 `s`、轮 `t`：只翻转 `raw[s][t]` 一个读数 → **在 `(s,t)` 与 `(s,t+1)` 各触发一次**；`t+1 == rounds` 时只剩前者。
- **轮 0 不产生任何机制。**
- observable `i` 翻转判据：`symplectic_product(P, logical_z[i])`（Z 基）/ `logical_x[i]`（X 基）。测量机制不影响 observable。
- 概率：`p_data` 按 channel 均分（`depolarizing` 每个 Pauli `p_data/3`）；测量机制取 `p_measure`。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_dem.py`：

```python
import numpy as np
import pytest

from aicir.qec.code import pauli_to_gf2
from aicir.qec.codes import get_code
from aicir.qec.dem import DetectorErrorModel, ErrorMechanism, build_dem
from aicir.qec.errors import PauliErrorModel
from aicir.qec.schedules import BareAncillaSchedule, build_layout


def _dem(code, rounds=3, **kw):
    errors = PauliErrorModel(**{"p_data": 0.01, "p_measure": 0.02, **kw})
    return build_dem(code, BareAncillaSchedule(), errors, rounds)


def test_round_zero_produces_no_mechanisms():
    """轮 0 是投影式制备轮，运行器不在其中注入任何错误 → DEM 也不能有轮 0 机制。"""
    code = get_code("repetition", d=3, basis="Z")
    dem = _dem(code, rounds=3, channel="bit_flip")
    assert all(mech.location[0] >= 1 for mech in dem.mechanisms)


def test_data_error_fires_exactly_one_detector_per_flipped_stabilizer():
    """data 错误持续存在 → 只在注入那一轮的差分里触发，之后各轮不再触发。"""
    code = get_code("repetition", d=3, basis="Z")
    dem = _dem(code, rounds=3, channel="bit_flip")
    layout = build_layout(code, BareAncillaSchedule(), 3)
    idx = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}

    # d=3 重复码：X 错误在比特 1 上与两个生成元 Z0Z1、Z1Z2 都反对易
    mech = next(m for m in dem.mechanisms
                if m.source == "data" and m.location == (1, 1, "X"))
    assert set(mech.detectors) == {idx[(0, 1)], idx[(1, 1)]}


def test_boundary_data_error_fires_single_detector():
    """比特 0 上的 X 只与生成元 0 (Z0Z1) 反对易 → 只触发一个 detector。"""
    code = get_code("repetition", d=3, basis="Z")
    dem = _dem(code, rounds=3, channel="bit_flip")
    layout = build_layout(code, BareAncillaSchedule(), 3)
    idx = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}
    mech = next(m for m in dem.mechanisms
                if m.source == "data" and m.location == (2, 0, "X"))
    assert set(mech.detectors) == {idx[(0, 2)]}


def test_measurement_error_fires_two_consecutive_detectors():
    """测量翻转只污染一个读数 → 相邻两轮差分各触发一次。"""
    code = get_code("repetition", d=3, basis="Z")
    dem = _dem(code, rounds=4, channel="bit_flip")
    layout = build_layout(code, BareAncillaSchedule(), 4)
    idx = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}
    mech = next(m for m in dem.mechanisms
                if m.source == "measurement" and m.location == (1, 0))
    assert set(mech.detectors) == {idx[(0, 1)], idx[(0, 2)]}


def test_final_round_measurement_error_fires_only_one_detector():
    """末轮的测量翻转没有「下一轮」可比对 → 只剩一个 detector。"""
    code = get_code("repetition", d=3, basis="Z")
    rounds = 4
    dem = _dem(code, rounds=rounds, channel="bit_flip")
    layout = build_layout(code, BareAncillaSchedule(), rounds)
    idx = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}
    mech = next(m for m in dem.mechanisms
                if m.source == "measurement" and m.location == (rounds - 1, 0))
    assert set(mech.detectors) == {idx[(0, rounds - 1)]}


def test_depolarizing_splits_probability_three_ways():
    code = get_code("steane")
    dem = _dem(code, rounds=3, p_data=0.03, p_measure=0.0, channel="depolarizing")
    data = [m for m in dem.mechanisms if m.source == "data"]
    assert {m.location[2] for m in data} == {"X", "Y", "Z"}
    assert all(abs(m.probability - 0.01) < 1e-12 for m in data)


def test_bit_flip_channel_emits_only_x():
    code = get_code("steane")
    dem = _dem(code, rounds=3, p_data=0.02, p_measure=0.0, channel="bit_flip")
    data = [m for m in dem.mechanisms if m.source == "data"]
    assert {m.location[2] for m in data} == {"X"}
    assert all(abs(m.probability - 0.02) < 1e-12 for m in data)


def test_observable_flip_follows_logical_operator():
    """Z 基读出下，只有与 logical_z 反对易的错误（带 X 分量）翻转 observable。"""
    code = get_code("steane")     # logical_z = ZZZZZZZ，全支持
    dem = _dem(code, rounds=3, p_data=0.01, p_measure=0.0, channel="depolarizing")
    for mech in dem.mechanisms:
        if mech.source != "data":
            continue
        _, q, p = mech.location
        expect = 1 if p in ("X", "Y") else 0      # Z 与 logical_z 对易 → 不翻转
        assert (0 in mech.observables) == bool(expect), mech.location


def test_matrices_have_consistent_shapes():
    code = get_code("steane")
    dem = _dem(code, rounds=3)
    n_mech = len(dem.mechanisms)
    assert dem.check_matrix().shape == (dem.n_detectors, n_mech)
    assert dem.faults_matrix().shape == (dem.n_observables, n_mech)
    assert dem.weights().shape == (n_mech,)
    # 权重为 log((1-p)/p)，p<0.5 时为正
    assert np.all(dem.weights() > 0)


def test_zero_probability_yields_no_mechanisms():
    code = get_code("steane")
    dem = _dem(code, rounds=3, p_data=0.0, p_measure=0.0)
    assert dem.mechanisms == ()


def test_rounds_below_two_rejected():
    code = get_code("steane")
    with pytest.raises(ValueError, match="rounds"):
        build_dem(code, BareAncillaSchedule(), PauliErrorModel(p_data=0.01), 1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_dem.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.dem'`

- [ ] **Step 3: 给 `PauliErrorModel` 加 `paulis` 属性**

在 `aicir/qec/errors.py` 的 `PauliErrorModel` 类中追加（`_CHANNELS` 已存在于该模块）：

```python
    @property
    def paulis(self) -> tuple[str, ...]:
        """该 channel 会产生的 Pauli 集合。公开出来供 dem.build_dem 复用，
        避免下游模块去 import 私有的 _CHANNELS。"""
        return _CHANNELS[self.channel]
```

- [ ] **Step 4: 实现 `aicir/qec/dem.py`**

```python
"""Detector Error Model：把误差模型翻译成「哪个故障翻转哪些 detector / observable」。

**解析推导，不做模拟。** 唯象误差模型（逐轮 data Pauli + 测量记录翻转）下这是闭式的：

- data Pauli P 落在比特 q、轮 t：错误**持续存在**，raw[s] 自轮 t 起一直翻着，而 detector
  取相邻轮差分，故**只在 (s, t) 触发一次**，之后各轮差分里它自己抵消掉。s 取遍与 P
  反对易的生成元。
- 测量翻转落在稳定子 s、轮 t：只污染 raw[s][t] 这**一个**读数，故在 (s,t) 与 (s,t+1)
  两处差分里各触发一次（末轮没有下一轮，只剩前者）。
- 轮 0 不产生任何机制：运行器在轮 0（投影式制备轮）不注入错误。

之所以不用「逐故障注入 + 跑无噪声线路 + 比对 detector」的模拟法：那不仅慢，更是**循环
论证**——用正在被验证的引擎去构造用来验证它的模型。本模块因此只依赖 numpy 与 GF(2)
代数，与 aicir 模拟器完全无关。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .code import pauli_to_gf2, symplectic_product


@dataclass(frozen=True)
class ErrorMechanism:
    """一个独立故障：以 probability 发生，翻转 detectors 与 observables。

    source / location 只用于诊断与可视化，不参与解码。
    location 为 ("data" 时) (round, qubit, pauli) 或 ("measurement" 时) (round, stabilizer)。
    """
    probability: float
    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    source: str
    location: tuple


@dataclass(frozen=True)
class DetectorErrorModel:
    """一组独立故障机制。matching 解码器由此建图。"""
    n_detectors: int
    n_observables: int
    mechanisms: tuple[ErrorMechanism, ...] = ()

    def check_matrix(self) -> np.ndarray:
        """(n_detectors, n_mechanisms) uint8：列 j 是机制 j 翻转的 detector 集合。"""
        h = np.zeros((self.n_detectors, len(self.mechanisms)), dtype=np.uint8)
        for j, mech in enumerate(self.mechanisms):
            for d in mech.detectors:
                h[d, j] = 1
        return h

    def faults_matrix(self) -> np.ndarray:
        """(n_observables, n_mechanisms) uint8：列 j 是机制 j 翻转的 observable 集合。"""
        f = np.zeros((self.n_observables, len(self.mechanisms)), dtype=np.uint8)
        for j, mech in enumerate(self.mechanisms):
            for o in mech.observables:
                f[o, j] = 1
        return f

    def weights(self) -> np.ndarray:
        """匹配图边权 log((1−p)/p)：概率越小权重越大，最小权匹配即最大似然。"""
        p = np.array([m.probability for m in self.mechanisms], dtype=float)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return np.log((1.0 - p) / p)


def build_dem(code, schedule, errors, rounds: int, *, logical_state: str = "0") -> DetectorErrorModel:
    """由 (码, 调度, 误差模型, 轮数) 解析构造 DEM。"""
    from .schedules import build_layout, resolve_schedule

    rounds = int(rounds)
    if rounds < 2:
        raise ValueError(
            f"rounds 必须 ≥2，收到 {rounds}：轮 0 是投影式制备轮、不产生任何误差机制，"
            f"rounds=1 的 DEM 必然为空"
        )
    schedule = resolve_schedule(schedule)
    layout = build_layout(code, schedule, rounds, logical_state=logical_state)
    det_index = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}

    logicals = code.logical_z if logical_state in ("0", "1") else code.logical_x
    paulis = errors.paulis
    p_each = float(errors.p_data) / len(paulis)
    p_meas = float(errors.p_measure)

    mechanisms: list[ErrorMechanism] = []
    for t in range(1, rounds):                      # 轮 0 不产生机制
        if p_each > 0.0:
            for q in range(code.n):
                for label in paulis:
                    vec = pauli_to_gf2("I" * q + label + "I" * (code.n - q - 1), code.n)
                    syn = code.syndrome(vec)
                    dets = tuple(det_index[(int(s), t)] for s in np.nonzero(syn)[0]
                                 if (int(s), t) in det_index)
                    obs = tuple(i for i in range(code.k)
                                if int(symplectic_product(vec, logicals[i])))
                    if dets or obs:
                        mechanisms.append(ErrorMechanism(
                            p_each, dets, obs, "data", (t, q, label)))
        if p_meas > 0.0:
            for s in range(code.m):
                dets = tuple(det_index[(s, tt)] for tt in (t, t + 1)
                             if (s, tt) in det_index)
                if dets:
                    mechanisms.append(ErrorMechanism(
                        p_meas, dets, (), "measurement", (t, s)))

    return DetectorErrorModel(
        n_detectors=layout.n_detectors, n_observables=code.k,
        mechanisms=tuple(mechanisms),
    )
```

在 `aicir/qec/__init__.py` 追加：

```python
from .dem import DetectorErrorModel, ErrorMechanism, build_dem
```
并把 `"DetectorErrorModel"`, `"ErrorMechanism"`, `"build_dem"` 加进 `__all__`。

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_dem.py -q`
Expected: PASS（11 passed）

若 `test_data_error_fires_exactly_one_detector_per_flipped_stabilizer` 失败，先核对是否误把 detector 建在了 `(s, t)` 与 `(s, t+1)` 两处——那是**测量**机制的行为，data 机制只有一处。这两条规则搞反是本任务最容易犯的错。

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/dem.py aicir/qec/errors.py aicir/qec/__init__.py tests/qec/test_dem.py
git commit -m "feat(qec): DEM 解析构造（唯象模型下闭式，纯 GF(2)）"
```

---

### Task 2: `samplers/` —— 把采样从解码循环里切出来

**Files:**
- Create: `aicir/qec/samplers/__init__.py`、`aicir/qec/samplers/aicir.py`
- Modify: `aicir/qec/runner.py`（改用 `AicirSampler`）、`aicir/qec/__init__.py`
- Test: `tests/qec/test_samplers.py`

**Interfaces:**
- Produces:
  - `ShotContext(code, schedule, errors, rounds, layout, logical_state, backend, rng, shot, seed)`
  - `ShotSamples(raw_syndromes, detection_events, observable_flips, injected_errors)`
  - `Sampler` 协议：`name: str`、`supports_active_mode: bool`、`sample_shot(ctx) -> ShotSamples`
  - `register_sampler(name, factory)` / `resolve_sampler(name_or_obj)`
  - `AicirSampler`

**本任务是纯重构，行为必须逐字节不变。** 判据：同种子下 `run()` 的输出与重构前完全一致。

**重要范围界定：** `active` 模式需要在轮间往量子态上施加修正门，这与「采样器一次性产出整条 shot」的形状冲突。**M2a 不把 active 模式搬进 `Sampler`**——`runner._run_one_shot` 保留现有的逐轮循环用于 `correction_mode="active"`，只有 `frame` 模式走 `Sampler`。这样重构面积最小、且 active 模式那套刚在 M1 修好的精细逻辑不被触碰。`AicirSampler.supports_active_mode = False`，`StimSampler` 同样为 `False`；`run()` 在 `correction_mode="active"` 时直接走原有内联路径。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_samplers.py`：

```python
import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.samplers import (AicirSampler, ShotSamples, register_sampler,
                                resolve_sampler)


def test_registry_resolves_name_and_instance():
    assert isinstance(resolve_sampler("aicir"), AicirSampler)
    inst = AicirSampler()
    assert resolve_sampler(inst) is inst
    with pytest.raises(KeyError, match="aicir"):
        resolve_sampler("no_such_sampler")


def test_aicir_sampler_declares_no_active_support():
    """active 模式留在 runner 的内联路径，不经采样器。"""
    assert AicirSampler().supports_active_mode is False


@pytest.mark.parametrize("name,kwargs", [("steane", {}), ("surface", {"d": 3})])
def test_refactor_is_byte_identical_to_m1_behaviour(name, kwargs):
    """同种子下 frame 模式结果必须与重构前完全一致——本任务是纯重构。"""
    code = get_code(name, **kwargs)
    common = dict(errors=PauliErrorModel(p_data=0.06, p_measure=0.02),
                  rounds=4, shots=8, seed=11)
    a = run(code, decoder=LookupDecoder(code), sampler="aicir", **common)
    b = run(code, decoder=LookupDecoder(code), **common)          # 默认采样器
    assert a.verdict_counts == b.verdict_counts
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.raw_syndromes, rb.raw_syndromes)
        assert np.array_equal(ra.detection_events, rb.detection_events)
        assert ra.verdict == rb.verdict


def test_active_mode_still_works_after_refactor():
    """active 模式走内联路径，重构不得破坏它。"""
    code = get_code("steane")
    common = dict(errors=PauliErrorModel(p_data=0.08), rounds=4, shots=8, seed=17)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    assert a.verdict_counts == b.verdict_counts
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.detection_events, rb.detection_events)


def test_unknown_sampler_rejected():
    code = get_code("steane")
    with pytest.raises(KeyError):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
            rounds=2, shots=1, sampler="nope")


def test_shot_samples_shape():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=3, shots=2, seed=1, sampler="aicir")
    rec = result.records[0]
    assert rec.raw_syndromes.shape == (3, code.m)
    assert rec.detection_events.shape == (3, code.m)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_samplers.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.samplers'`

- [ ] **Step 3: 实现 `aicir/qec/samplers/__init__.py`**

```python
"""采样器：把「怎么拿到 detection events」与「拿到之后怎么在线解码」分开。

M1 里这两件事纠缠在 runner._run_one_shot 一个函数里。切开之后，Stim 可以作为一个
平级的采样器接进来——它自己编译并采样线路，完全绕开 aicir 模拟器，而下游的在线
解码器、记录结构、实时模型一概不变。

**active 模式不走采样器**：它需要在轮间把修正作为真实门施加回量子态，与「一次性
产出整条 shot」的形状冲突。runner 保留原有的逐轮内联路径处理 active，采样器只服务
frame 模式。故所有采样器的 supports_active_mode 均为 False。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass
class ShotContext:
    """采样一条 shot 所需的全部输入。注意其中没有解码器——采样器不认识解码器。"""
    code: object
    schedule: object
    errors: object
    rounds: int
    layout: object
    logical_state: str
    backend: object
    rng: object
    shot: int
    seed: int


@dataclass
class ShotSamples:
    """一条 shot 的采样产物；在线解码循环只消费这些字段。

    raw_syndromes / injected_errors 在 Stim 路径下分别为 None 与空列表——Stim 的
    detector sampler 不回报原始稳定子读数，也不回报注入了哪些故障。**必须是 None /
    空，而不是编造的零**：「没测到」和「测出来是零」是两回事，与 M1
    `TimingModel=None → 字段为 None` 同一原则。
    """
    detection_events: np.ndarray               # (rounds, m) uint8
    observable_flips: np.ndarray               # (k,) uint8
    raw_syndromes: np.ndarray | None = None    # (rounds, m) uint8；Stim 路径为 None
    injected_errors: list = field(default_factory=list)


class Sampler(Protocol):
    name: str
    supports_active_mode: bool

    def sample_shot(self, ctx: ShotContext) -> ShotSamples: ...


SAMPLERS: dict[str, Callable[[], Sampler]] = {}


def register_sampler(name: str, factory: Callable[[], Sampler]) -> None:
    SAMPLERS[str(name)] = factory


def resolve_sampler(name_or_obj) -> Sampler:
    """名字 → 采样器实例；已经是实例则原样返回。"""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in SAMPLERS:
        raise KeyError(f"未知采样器 {name_or_obj!r}；可用：{sorted(SAMPLERS)}")
    return SAMPLERS[name_or_obj]()


from .aicir import AicirSampler  # noqa: E402  自注册

__all__ = ["ShotContext", "ShotSamples", "Sampler", "SAMPLERS",
           "register_sampler", "resolve_sampler", "AicirSampler"]
```

`aicir/qec/samplers/aicir.py`：

```python
"""AicirSampler：M1 既有采样循环的原样封装（逐轮建线路 → run_trajectory → 读寄存器）。"""

from __future__ import annotations

import numpy as np

from aicir.core.state import State
from aicir.measure.trajectory import run_trajectory

from . import ShotContext, ShotSamples, register_sampler


class AicirSampler:
    """用 aicir 自己的态矢量引擎逐轮采样。"""

    name = "aicir"
    supports_active_mode = False        # active 走 runner 的内联路径，见本包 docstring

    def sample_shot(self, ctx: ShotContext) -> ShotSamples:
        from ..runner import _error_gates, _read_creg

        code, schedule, rng = ctx.code, ctx.schedule, ctx.rng
        n_total = code.n + code.m
        reference = np.zeros(code.m, dtype=np.uint8)

        state = run_trajectory(
            schedule.build_encode(code, ctx.logical_state),
            State.zero_state(n_total, ctx.backend), ctx.backend,
            tm=False, measure_qubits=None, snap_ops=set(), rng=rng,
        ).pre

        raw = np.zeros((ctx.rounds, code.m), dtype=np.uint8)
        events = np.zeros((ctx.rounds, code.m), dtype=np.uint8)
        injected = []

        for t in range(ctx.rounds):
            # 轮 0 是投影式制备轮，不注入错误
            round_errors = [] if t == 0 else ctx.errors.sample_round(t, code.n, code.m, rng)
            injected.extend(round_errors)

            rc = schedule.build_round(code, t)
            rc.circuit.gates[:0] = _error_gates(round_errors)
            res = run_trajectory(rc.circuit, state, ctx.backend, tm=False,
                                 measure_qubits=None, snap_ops=set(), rng=rng)
            state = res.pre
            bits = _read_creg(res.classical, rc.creg_name, code.m)
            for e in round_errors:                       # 测量误差翻转经典记录
                if e.source == "measurement":
                    bits[e.qubit] ^= 1
            raw[t] = bits
            events[t] = ctx.layout.detection_events(raw, t, reference)

        ro = schedule.build_readout(code, ctx.logical_state)
        ro_res = run_trajectory(ro.circuit, state, ctx.backend, tm=False,
                                measure_qubits=None, snap_ops=set(), rng=rng)
        readout = _read_creg(ro_res.classical, ro.creg_name, code.n)

        return ShotSamples(detection_events=events, observable_flips=readout,
                           raw_syndromes=raw, injected_errors=injected)


register_sampler("aicir", AicirSampler)
```

- [ ] **Step 4: 让 `runner.run` 接受 `sampler=`**

在 `aicir/qec/runner.py` 的 `run(...)` 签名里追加 `sampler="aicir"`；在 `decoder is None` 校验之后加：

```python
    from .samplers import ShotContext, resolve_sampler
    sampler_obj = resolve_sampler(sampler)
    if correction_mode == "active" and not sampler_obj.supports_active_mode:
        # active 需要在轮间把修正作为真实门施加回量子态，采样器一次性产出整条 shot
        # 的形状容不下它。frame 模式走采样器，active 走 runner 的内联逐轮路径。
        if getattr(sampler_obj, "name", "") != "aicir":
            raise ValueError(
                f"采样器 {getattr(sampler_obj, 'name', '?')!r} 不支持 correction_mode='active'"
                f"（它不保留量子态，无从施加物理修正）；请用 correction_mode='frame'"
            )
```

`_run_one_shot` 增加 `sampler_obj` 形参。在其开头分流：

```python
    if correction_mode == "frame":
        samples = sampler_obj.sample_shot(ShotContext(
            code=code, schedule=schedule, errors=errors, rounds=rounds, layout=layout,
            logical_state=logical_state, backend=backend, rng=rng, shot=shot, seed=shot_seed,
        ))
        raw = samples.raw_syndromes
        events_log = samples.detection_events
        injected = list(samples.injected_errors)
        readout = samples.observable_flips
        steps, wall = [], np.zeros(rounds, dtype=float)
        frame = np.zeros(2 * code.k, dtype=np.uint8)
        committed = -1
        for t in range(rounds):
            t0 = time.perf_counter()
            step = decoder.update(t, events_log[t])
            wall[t] = time.perf_counter() - t0
            if step.committed_through < committed:
                raise ValueError(
                    f"解码器 {getattr(decoder, 'name', '?')} 的 committed_through 回退："
                    f"轮 {t} 报 {step.committed_through}，此前已提交到 {committed}"
                )
            committed = step.committed_through
            steps.append(step)
            if step.frame_flips is not None:
                frame ^= np.asarray(step.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]
        final = decoder.flush()
        if final.committed_through < committed:
            raise ValueError("flush() 的 committed_through 不得低于此前已提交轮次")
        if final.frame_flips is not None:
            frame ^= np.asarray(final.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]
        residual = _residual_from_readout(code, readout, frame, logical_state)
        record = QECShotRecord(
            shot=shot, seed=shot_seed, injected_errors=injected, raw_syndromes=raw,
            detection_events=events_log, decode_steps=steps, wall_clock=wall,
            observable_raw=readout, frame_flips=frame, verdict=code.verdict(residual),
        )
        if timing is not None:
            _fill_shot_timing(record, decoder, steps, timing, rounds)
        return record
    # 以下保留 M1 原有的 active 逐轮内联路径，一字不改
```

**`decoder.reset(layout)` 必须在分流之前调用**（两条路径都需要）。

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_samplers.py -q` 然后 `PYTHONPATH=. pytest tests/qec/ -q`
Expected: 全部通过；**既有测试数不得减少**（这是纯重构）。

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/samplers/ aicir/qec/runner.py aicir/qec/__init__.py tests/qec/test_samplers.py
git commit -m "refactor(qec): 抽出 Sampler 协议与 AicirSampler，采样与解码循环解耦"
```

---

### Task 3: `interop/export.py` —— 导出为 Stim 线路

**Files:**
- Create: `aicir/qec/interop/__init__.py`、`aicir/qec/interop/export.py`
- Modify: `pyproject.toml`（新增 `qec` extra）
- Test: `tests/qec/test_stim_export.py`

**Interfaces:**
- Produces：`to_stim_text(code, schedule, errors, rounds, *, logical_state="0") -> str`、`to_stim_circuit(...) -> "stim.Circuit"`、`write_stim(path, ...) -> None`

**导出格式要点：**
- 每轮：`H` 打在 ancilla → 受控门（`CX`/`CY`/`CZ`，控制位是 ancilla）→ `H` → `MR`（measure+reset 合一，正是 M1 调度的 measure+reset 语义）。
- data 错误写在该轮提取**之前**（与 M1 的 `cir.gates[:0] = _error_gates(...)` 一致）；`bit_flip`→`X_ERROR(p)`、`phase_flip`→`Z_ERROR(p)`、`depolarizing`→`DEPOLARIZE1(p)`。
- 测量误差写成 `MR` **之前**的 `X_ERROR(p_measure)` 打在 ancilla 上——与 M1「翻转经典记录」严格等价（读数前翻 ancilla ⇔ 翻读数）。
- **轮 0 不写任何噪声。**
- detector：轮 0 只对 `deterministic_round0` 内的稳定子写 `DETECTOR rec[-(m-s)]`；轮 `t>=1` 对每个 s 写 `DETECTOR rec[-(m-s)] rec[-(2m-s)]`。
- 末端 `M` 打在 data 比特上，`OBSERVABLE_INCLUDE(i)` 收 `logical_z[i]`（或 X 基下 `logical_x[i]`）支持上的那些 `rec`。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_stim_export.py`：

```python
import pytest

stim = pytest.importorskip("stim")

from aicir.qec.codes import get_code
from aicir.qec.errors import PauliErrorModel
from aicir.qec.interop.export import to_stim_circuit, to_stim_text
from aicir.qec.schedules import BareAncillaSchedule, build_layout

CASES = [("repetition", {"d": 3, "basis": "Z"}), ("steane", {}), ("surface", {"d": 3})]


@pytest.mark.parametrize("name,kwargs", CASES)
def test_export_parses_as_stim_circuit(name, kwargs):
    code = get_code(name, **kwargs)
    text = to_stim_text(code, BareAncillaSchedule(),
                        PauliErrorModel(p_data=0.01, p_measure=0.01, channel="bit_flip"), 3)
    stim.Circuit(text)          # 解析失败会抛异常


@pytest.mark.parametrize("name,kwargs", CASES)
def test_detector_count_matches_aicir_layout(name, kwargs):
    """导出的 detector 数必须与 aicir 自己的 build_layout 一致——两套实现的第一处对表。"""
    code = get_code(name, **kwargs)
    rounds = 3
    circuit = to_stim_circuit(code, BareAncillaSchedule(),
                              PauliErrorModel(p_data=0.01), rounds)
    layout = build_layout(code, BareAncillaSchedule(), rounds)
    assert circuit.num_detectors == layout.n_detectors
    assert circuit.num_observables == code.k


@pytest.mark.parametrize("name,kwargs", CASES)
def test_noiseless_export_has_all_detectors_deterministic(name, kwargs):
    """无噪声导出线路在 Stim 下 detector 必须恒为 0。

    这与 M1 的 verify_schedule 是同一条断言，但换成 Stim 来验——若 aicir 的调度
    与导出器对 detector 的理解有任何分歧，这里立刻暴露。
    """
    code = get_code(name, **kwargs)
    circuit = to_stim_circuit(code, BareAncillaSchedule(), PauliErrorModel(), 4)
    det, _ = circuit.compile_detector_sampler().sample(shots=16, separate_observables=True)
    assert not det.any(), f"{name} 无噪声导出线路有 detector 触发"


def test_noise_produces_some_detections():
    """有噪声时必须真的触发 detector，否则上一条测试是空断言。"""
    code = get_code("steane")
    circuit = to_stim_circuit(code, BareAncillaSchedule(),
                              PauliErrorModel(p_data=0.15, p_measure=0.15), 4)
    det, _ = circuit.compile_detector_sampler().sample(shots=64, separate_observables=True)
    assert det.any()


def test_round_zero_carries_no_noise():
    """轮 0 是制备轮：导出文本里第一个 MR 之前不得出现噪声指令。"""
    code = get_code("steane")
    text = to_stim_text(code, BareAncillaSchedule(),
                        PauliErrorModel(p_data=0.1, p_measure=0.1), 3)
    head = text.split("MR")[0]
    assert "X_ERROR" not in head and "DEPOLARIZE1" not in head
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_export.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.interop'`

- [ ] **Step 3: 实现导出器**

`aicir/qec/interop/__init__.py`：

```python
"""与外部 QEC 生态（Stim / PyMatching）的互操作层。

**本子包依赖可选包**，与 aicir.qec 的 numpy-only 核心分开：核心模块无 stim 也能用，
只有显式使用互操作功能时才需要 `pip install "aicir[qec]"`。
"""

from __future__ import annotations

from .export import to_stim_circuit, to_stim_text, write_stim

__all__ = ["to_stim_text", "to_stim_circuit", "write_stim"]
```

`aicir/qec/interop/export.py`：

```python
"""aicir 的 (码, 调度, 误差模型) → Stim 线路文本。

导出必须与 M1 运行器的语义严格对齐，否则 oracle 比对就成了比对两个不同的实验：
- data 错误写在该轮提取**之前**（对应 runner 的 `cir.gates[:0] = _error_gates(...)`）
- 测量误差写成 MR 之前打在 ancilla 上的 X_ERROR。这与 M1「翻转经典记录」严格等价：
  读数前翻 ancilla 与翻转读数本身给出同一分布，也正是 Stim 自己的惯用写法。
- **轮 0 不写任何噪声**（轮 0 是投影式制备轮）
- 轮 0 只对「制备基下确定」的生成元写 DETECTOR，与 build_layout 的 round0_stabilizers 一致
"""

from __future__ import annotations

from ..code import gf2_to_pauli

_CONTROLLED = {"X": "CX", "Y": "CY", "Z": "CZ"}
_DATA_NOISE = {"bit_flip": "X_ERROR", "phase_flip": "Z_ERROR", "depolarizing": "DEPOLARIZE1"}


def to_stim_text(code, schedule, errors, rounds: int, *, logical_state: str = "0") -> str:
    """生成 Stim 线路文本。"""
    from ..schedules import build_layout, deterministic_round0, resolve_schedule

    rounds = int(rounds)
    if rounds < 2:
        raise ValueError(f"rounds 必须 ≥2，收到 {rounds}")
    schedule = resolve_schedule(schedule)
    build_layout(code, schedule, rounds, logical_state=logical_state)   # 校验参数合法
    round0 = set(deterministic_round0(code, logical_state))

    n, m = code.n, code.m
    data = list(range(n))
    anc = [n + j for j in range(m)]
    lines = [f"R {' '.join(str(q) for q in range(n + m))}"]
    if logical_state in ("+", "-"):
        lines.append(f"H {' '.join(str(q) for q in data)}")

    noise_op = _DATA_NOISE[errors.channel]
    for t in range(rounds):
        lines.append("TICK")
        if t >= 1:                                   # 轮 0 是制备轮，不加噪声
            if errors.p_data > 0.0:
                lines.append(f"{noise_op}({errors.p_data}) {' '.join(str(q) for q in data)}")
        for j in range(m):
            labels = gf2_to_pauli(code.generators[j])
            lines.append(f"H {anc[j]}")
            for q, ch in enumerate(labels):
                if ch != "I":
                    lines.append(f"{_CONTROLLED[ch]} {anc[j]} {q}")
            lines.append(f"H {anc[j]}")
        if t >= 1 and errors.p_measure > 0.0:
            # MR 之前翻 ancilla ⇔ 翻转读数本身
            lines.append(f"X_ERROR({errors.p_measure}) {' '.join(str(a) for a in anc)}")
        lines.append(f"MR {' '.join(str(a) for a in anc)}")
        for s in range(m):
            if t == 0:
                if s in round0:
                    lines.append(f"DETECTOR rec[{-(m - s)}]")
            else:
                lines.append(f"DETECTOR rec[{-(m - s)}] rec[{-(2 * m - s)}]")

    lines.append("TICK")
    if logical_state in ("+", "-"):
        lines.append(f"H {' '.join(str(q) for q in data)}")
    lines.append(f"M {' '.join(str(q) for q in data)}")
    logicals = code.logical_z if logical_state in ("0", "1") else code.logical_x
    for i in range(code.k):
        block = logicals[i][n:] if logical_state in ("0", "1") else logicals[i][:n]
        recs = [f"rec[{-(n - q)}]" for q in range(n) if block[q]]
        lines.append(f"OBSERVABLE_INCLUDE({i}) {' '.join(recs)}")
    return "\n".join(lines) + "\n"


def to_stim_circuit(code, schedule, errors, rounds: int, *, logical_state: str = "0"):
    """生成 `stim.Circuit`。需要安装 stim。"""
    try:
        import stim
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "Stim 互操作需要可选依赖：pip install \"aicir[qec]\""
        ) from exc
    return stim.Circuit(to_stim_text(code, schedule, errors, rounds,
                                     logical_state=logical_state))


def write_stim(path, code, schedule, errors, rounds: int, *, logical_state: str = "0") -> None:
    """把线路写成 .stim 文件，供任意外部工具消费。"""
    from pathlib import Path
    Path(path).write_text(
        to_stim_text(code, schedule, errors, rounds, logical_state=logical_state),
        encoding="utf-8",
    )
```

在 `pyproject.toml` 的 `[project.optional-dependencies]` 中新增，并把两项并入 `all` 与 `dev`：

```toml
# QEC 互操作：Stim 采样/DEM 交叉验证 + PyMatching 解码
qec = ["stim", "pymatching"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_export.py -q`
Expected: PASS（11 passed）

`test_noiseless_export_has_all_detectors_deterministic` 失败即说明**导出器与 aicir 调度对 detector 的理解不一致**——这正是 oracle 要抓的第一类分歧。修导出器或修轮 0 集合，**不要放宽断言**。

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/interop/ pyproject.toml tests/qec/test_stim_export.py
git commit -m "feat(qec): 导出为 Stim 线路，并以 Stim 复验 detector 确定性"
```

---

### Task 4: `StimSampler` 与 oracle

**Files:**
- Create: `aicir/qec/samplers/stim.py`
- Modify: `aicir/qec/samplers/__init__.py`（注册）
- Test: `tests/qec/test_stim_oracle.py`

**Interfaces:**
- Produces：`StimSampler`（`name="stim"`，`supports_active_mode=False`），注册名 `"stim"`

**这是 M2a 的核心交付。** oracle 分两层，主次分明：
- **主检验（采样统计一致）**：aicir 与 Stim 各采 N shot，逐 detector 触发频率落在统计容差内。端到端、与双方 DEM 内部表示无关。
- **辅检验（DEM 结构比对）**：失配只 `warn` 不 `fail`——Stim 会合并/分解误差机制，结构级严格相等很可能对不上**且未必是 bug**。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_stim_oracle.py`：

```python
import numpy as np
import pytest

stim = pytest.importorskip("stim")

from aicir.qec.codes import get_code
from aicir.qec.dem import build_dem
from aicir.qec.errors import PauliErrorModel
from aicir.qec.interop.export import to_stim_circuit
from aicir.qec.samplers import ShotContext, resolve_sampler
from aicir.qec.schedules import BareAncillaSchedule, build_layout


def _aicir_detection_rates(code, errors, rounds, shots, seed=0):
    sampler = resolve_sampler("aicir")
    from aicir.backends import NumpyBackend
    layout = build_layout(code, BareAncillaSchedule(), rounds)
    backend = NumpyBackend()
    total = np.zeros(layout.n_detectors, dtype=float)
    idx = {(d.stabilizer, d.round_index): d.index for d in layout.detectors}
    for shot in range(shots):
        ctx = ShotContext(code=code, schedule=BareAncillaSchedule(), errors=errors,
                          rounds=rounds, layout=layout, logical_state="0",
                          backend=backend, rng=np.random.default_rng(seed * 7919 + shot),
                          shot=shot, seed=seed * 7919 + shot)
        ev = sampler.sample_shot(ctx).detection_events
        for (s, t), i in idx.items():
            total[i] += int(ev[t, s])
    return total / shots


def _stim_detection_rates(code, errors, rounds, shots, seed=0):
    circuit = to_stim_circuit(code, BareAncillaSchedule(), errors, rounds)
    det, _ = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True)
    return det.mean(axis=0)


@pytest.mark.parametrize("name,kwargs", [
    ("repetition", {"d": 3, "basis": "Z"}), ("steane", {}),
])
def test_detection_rates_agree_with_stim(name, kwargs):
    """**主 oracle**：两套独立实现的逐 detector 触发频率必须统计一致。

    这是 M1 缺失的那个外部参照——此前一切都在拿「我推导的期望」校验「我推导的实现」。
    """
    code = get_code(name, **kwargs)
    errors = PauliErrorModel(p_data=0.03, p_measure=0.03, channel="bit_flip")
    rounds, shots = 4, 4000
    a = _aicir_detection_rates(code, errors, rounds, shots, seed=1)
    b = _stim_detection_rates(code, errors, rounds, shots, seed=1)
    # 二项标准差上界 0.5/sqrt(N)；给 5σ 余量，避免种子敏感
    tol = 5.0 * 0.5 / np.sqrt(shots)
    worst = float(np.max(np.abs(a - b)))
    assert worst < tol, f"{name} 最大逐 detector 频率偏差 {worst:.4f} 超过容差 {tol:.4f}"


def test_noiseless_agrees_trivially():
    code = get_code("steane")
    errors = PauliErrorModel()
    a = _aicir_detection_rates(code, errors, 3, 32)
    b = _stim_detection_rates(code, errors, 3, 32)
    assert not a.any() and not b.any()


def test_dem_structural_comparison_is_advisory(recwarn):
    """辅检验：与 Stim 的 DEM 比对，**失配只告警不失败**。

    Stim 会合并重复机制、并可能把误差分解成 graphlike 片段，故结构级严格相等
    很可能对不上且未必是 bug。真正的判据是上面的采样统计一致。
    """
    from aicir.qec.interop.compare import compare_dem_with_stim

    code = get_code("repetition", d=3, basis="Z")
    errors = PauliErrorModel(p_data=0.02, p_measure=0.02, channel="bit_flip")
    report = compare_dem_with_stim(code, BareAncillaSchedule(), errors, rounds=3)
    assert set(report) >= {"ours_n_mechanisms", "stim_n_mechanisms",
                           "detector_sets_match", "mismatches"}
    assert report["ours_n_mechanisms"] > 0
    assert report["stim_n_mechanisms"] > 0


def test_stim_sampler_runs_end_to_end():
    from aicir.qec import run
    from aicir.qec.decoders.lookup import LookupDecoder

    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.02, p_measure=0.02),
                 decoder=LookupDecoder(code), rounds=3, shots=16, seed=5, sampler="stim")
    assert sum(result.verdict_counts.values()) == 16
    rec = result.records[0]
    # Stim 不回报原始综合征与故障位置 —— 必须是 None / 空，而不是编造的零
    assert rec.raw_syndromes is None
    assert rec.injected_errors == []
    assert rec.detection_events.shape == (3, code.m)


def test_stim_sampler_rejects_active_mode():
    from aicir.qec import run
    from aicir.qec.decoders.lookup import LookupDecoder

    code = get_code("steane")
    with pytest.raises(ValueError, match="active"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
            rounds=3, shots=1, sampler="stim", correction_mode="active")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_oracle.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.interop.compare'`

- [ ] **Step 3: 实现 `aicir/qec/samplers/stim.py`**

```python
"""StimSampler：用 Stim 编译并采样，完全绕开 aicir 模拟器。

它是 M2a 的两个目的之一——外部正确性 oracle。同一 (码, 调度, 噪声) 下，本采样器与
AicirSampler 的逐 detector 触发频率必须统计一致；不一致就说明两套独立实现里至少有
一套错了，而这正是 M1 全程缺失的那个外部参照。

**信息缺口（重要）**：Stim 的 detector sampler 只回报 detection events 与 observable
flips，不回报原始稳定子读数，也不回报注入了哪些故障。因此 ShotSamples.raw_syndromes
为 None、injected_errors 为空列表——**不是编造的零**。M3 的逐 shot 错误轨迹图因此
只能画 aicir 采样的 shot。
"""

from __future__ import annotations

import numpy as np

from . import ShotContext, ShotSamples, register_sampler


class StimSampler:
    """把线路交给 Stim 编译采样。批量采样后按 shot 逐条取出。"""

    name = "stim"
    supports_active_mode = False        # Stim 不保留量子态，无从施加物理修正

    def __init__(self):
        self._cache_key = None
        self._det = None
        self._obs = None
        self._cursor = 0

    def sample_shot(self, ctx: ShotContext) -> ShotSamples:
        from ..interop.export import to_stim_circuit

        code = ctx.code
        key = (id(code), ctx.rounds, ctx.logical_state, id(ctx.errors), ctx.seed - ctx.shot)
        if key != self._cache_key:
            # 一次编译、批量采样，再按 shot 取用——逐 shot 编译会慢一个数量级
            circuit = to_stim_circuit(code, ctx.schedule, ctx.errors, ctx.rounds,
                                      logical_state=ctx.logical_state)
            n_batch = max(1024, ctx.shot + 1)
            sampler = circuit.compile_detector_sampler(seed=int(ctx.seed) & 0xFFFFFFFF)
            det, obs = sampler.sample(shots=n_batch, separate_observables=True)
            self._cache_key, self._det, self._obs, self._cursor = key, det, obs, 0
        if self._cursor >= self._det.shape[0]:
            raise RuntimeError("Stim 批量采样耗尽；请增大批大小")

        flat = self._det[self._cursor].astype(np.uint8)
        obs_flips = self._obs[self._cursor].astype(np.uint8)
        self._cursor += 1

        # 把扁平的 detector 向量摊回 (rounds, m)：轮 0 只有确定生成元有 detector
        events = np.zeros((ctx.rounds, code.m), dtype=np.uint8)
        for d in ctx.layout.detectors:
            events[d.round_index, d.stabilizer] = flat[d.index]

        return ShotSamples(detection_events=events, observable_flips=obs_flips,
                           raw_syndromes=None, injected_errors=[])


register_sampler("stim", StimSampler)
```

在 `aicir/qec/samplers/__init__.py` 的自注册区追加：

```python
try:                                    # stim 可选
    from .stim import StimSampler       # noqa: F401,E402
except ImportError:                     # pragma: no cover
    StimSampler = None                  # type: ignore[assignment]
```

**`_residual_from_readout` 对 Stim 路径不适用**（Stim 直接给 observable flips，不给逐比特读数）。在 `runner._run_one_shot` 的 frame 分支里，改为：若 `samples.raw_syndromes is None`，则 `parity` 直接取 `samples.observable_flips[i]`，跳过按支持求奇偶那一步。为此把判定逻辑抽成：

```python
def _verdict_from_observable_flips(code, obs_flips, frame, logical_state):
    """由 observable 翻转位与已提交 frame 定判定（Stim 路径：无逐比特读数）。"""
    residual = np.zeros(2 * code.n, dtype=np.uint8)
    for i in range(code.k):
        flip = int(frame[2 * i]) if logical_state in ("0", "1") else int(frame[2 * i + 1])
        if int(obs_flips[i]) ^ flip:
            residual ^= code.logical_x[i] if logical_state in ("0", "1") else code.logical_z[i]
    return residual
```

- [ ] **Step 4: 实现 `aicir/qec/interop/compare.py`**

```python
"""DEM 结构比对：把我们自己推导的 DEM 与 Stim 的放在一起看。

**这是辅助检验，失配不等于缺陷。** Stim 会合并重复机制，并可能把误差分解成 graphlike
片段，因此集合级严格相等很可能对不上，而那多半是表示差异不是 bug。真正的判据是
tests/qec/test_stim_oracle.py 里的采样统计一致性。本函数只产出报告供人判断。
"""

from __future__ import annotations

from collections import Counter

from ..dem import build_dem
from .export import to_stim_circuit


def compare_dem_with_stim(code, schedule, errors, rounds: int, *,
                          logical_state: str = "0") -> dict:
    """返回一份比对报告（不抛异常、不断言）。"""
    ours = build_dem(code, schedule, errors, rounds, logical_state=logical_state)
    circuit = to_stim_circuit(code, schedule, errors, rounds, logical_state=logical_state)
    stim_dem = circuit.detector_error_model(decompose_errors=False)

    def _ours_key(mech):
        return (tuple(sorted(mech.detectors)), tuple(sorted(mech.observables)))

    stim_keys = Counter()
    for ins in stim_dem:
        if ins.type != "error":
            continue
        dets, obs = [], []
        for tgt in ins.targets_copy():
            s = str(tgt)
            if s.startswith("D"):
                dets.append(int(s[1:]))
            elif s.startswith("L"):
                obs.append(int(s[1:]))
        stim_keys[(tuple(sorted(dets)), tuple(sorted(obs)))] += 1

    ours_keys = Counter(_ours_key(m) for m in ours.mechanisms)
    mismatches = sorted(set(ours_keys) ^ set(stim_keys))
    return {
        "ours_n_mechanisms": len(ours.mechanisms),
        "stim_n_mechanisms": stim_dem.num_errors,
        "detector_sets_match": not mismatches,
        "mismatches": mismatches,
        "note": "结构失配未必是缺陷：Stim 会合并/分解误差机制。判据以采样统计一致为准。",
    }
```

在 `aicir/qec/interop/__init__.py` 追加 `from .compare import compare_dem_with_stim` 并加入 `__all__`。

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_oracle.py -q`
Expected: PASS（7 passed）

**`test_detection_rates_agree_with_stim` 是本里程碑最重要的测试。** 若它失败，说明 aicir 的采样路径与 Stim 的对同一实验给出了不同的统计——**先怀疑导出器的语义对齐**（噪声位置、轮 0 是否加噪、`MR` 与 measure+reset 的对应、detector 的 `rec[]` 偏移），再怀疑 M1 的采样路径。**绝不要放宽容差**：5σ 已经很宽，真失配会远超它。

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/samplers/stim.py aicir/qec/samplers/__init__.py \
        aicir/qec/interop/compare.py aicir/qec/interop/__init__.py \
        aicir/qec/runner.py tests/qec/test_stim_oracle.py
git commit -m "feat(qec): Stim 采样后端与外部 oracle（采样统计为主、DEM 结构比对为辅）"
```

---

### Task 5: `interop/import_.py` —— 读入外部 `.stim` / `.dem`

**Files:**
- Create: `aicir/qec/interop/import_.py`
- Modify: `aicir/qec/interop/__init__.py`
- Test: `tests/qec/test_stim_import.py`

**Interfaces:**
- Produces：`dem_from_stim_text(text) -> DetectorErrorModel`、`dem_from_dem_file(path) -> DetectorErrorModel`、`layout_from_stim_text(text) -> DetectorLayout`

导入的 layout 没有 aicir 侧的 code/schedule，故 `round0_stabilizers` 与 `coords` 留空、`n_stabilizers` 取 detector 数（每轮一个的退化视角）；这是刻意的降级，不是遗漏。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_stim_import.py`：

```python
import pytest

stim = pytest.importorskip("stim")

from aicir.qec.codes import get_code
from aicir.qec.dem import build_dem
from aicir.qec.errors import PauliErrorModel
from aicir.qec.interop.export import to_stim_text
from aicir.qec.interop.import_ import (dem_from_dem_file, dem_from_stim_text,
                                       layout_from_stim_text)
from aicir.qec.schedules import BareAncillaSchedule


def _text(code, rounds=3):
    return to_stim_text(code, BareAncillaSchedule(),
                        PauliErrorModel(p_data=0.02, p_measure=0.02, channel="bit_flip"),
                        rounds)


def test_dem_from_stim_text_roundtrip_shapes():
    code = get_code("repetition", d=3, basis="Z")
    dem = dem_from_stim_text(_text(code))
    ours = build_dem(code, BareAncillaSchedule(),
                     PauliErrorModel(p_data=0.02, p_measure=0.02, channel="bit_flip"), 3)
    assert dem.n_detectors == ours.n_detectors
    assert dem.n_observables == ours.n_observables
    assert len(dem.mechanisms) > 0


def test_imported_mechanisms_have_valid_indices():
    code = get_code("steane")
    dem = dem_from_stim_text(_text(code))
    for mech in dem.mechanisms:
        assert all(0 <= d < dem.n_detectors for d in mech.detectors)
        assert all(0 <= o < dem.n_observables for o in mech.observables)
        assert 0.0 < mech.probability < 1.0


def test_layout_from_stim_text_detector_count():
    code = get_code("steane")
    layout = layout_from_stim_text(_text(code))
    assert layout.n_detectors == stim.Circuit(_text(code)).num_detectors
    assert layout.round0_stabilizers == ()       # 外部文件推断不出，刻意留空


def test_dem_file_roundtrip(tmp_path):
    code = get_code("repetition", d=3, basis="Z")
    circuit = stim.Circuit(_text(code))
    path = tmp_path / "x.dem"
    path.write_text(str(circuit.detector_error_model(decompose_errors=False)))
    dem = dem_from_dem_file(path)
    assert dem.n_detectors == circuit.num_detectors
    assert len(dem.mechanisms) > 0


def test_malformed_dem_rejected(tmp_path):
    path = tmp_path / "bad.dem"
    path.write_text("this is not a detector error model\n")
    with pytest.raises(ValueError):
        dem_from_dem_file(path)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_import.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.interop.import_'`

- [ ] **Step 3: 实现**

```python
"""读入外部 .stim / .dem，使已发表的 artifact 能直接进本平台跑自己的在线解码器。

导入得到的 layout 没有 aicir 侧的 code/schedule，故 round0_stabilizers 与 coords
留空——这是刻意的降级（外部文件推断不出这些），不是遗漏。
"""

from __future__ import annotations

from pathlib import Path

from ..dem import DetectorErrorModel, ErrorMechanism
from ..detectors import Detector, DetectorLayout, Observable


def _require_stim():
    try:
        import stim
    except ImportError as exc:                       # pragma: no cover
        raise ImportError('Stim 互操作需要可选依赖：pip install "aicir[qec]"') from exc
    return stim


def _dem_to_model(stim_dem) -> DetectorErrorModel:
    mechanisms = []
    for ins in stim_dem:
        if ins.type != "error":
            continue
        prob = float(ins.args_copy()[0])
        dets, obs = [], []
        for tgt in ins.targets_copy():
            s = str(tgt)
            if s.startswith("D"):
                dets.append(int(s[1:]))
            elif s.startswith("L"):
                obs.append(int(s[1:]))
        mechanisms.append(ErrorMechanism(
            prob, tuple(sorted(dets)), tuple(sorted(obs)), "imported", ()))
    n_det = int(stim_dem.num_detectors)
    n_obs = int(stim_dem.num_observables)
    return DetectorErrorModel(n_det, n_obs, tuple(mechanisms))


def dem_from_stim_text(text: str) -> DetectorErrorModel:
    """由 .stim 线路文本导出并转换 DEM。"""
    stim = _require_stim()
    circuit = stim.Circuit(text)
    return _dem_to_model(circuit.detector_error_model(decompose_errors=False))


def dem_from_dem_file(path) -> DetectorErrorModel:
    """直接读 .dem 文件。"""
    stim = _require_stim()
    raw = Path(path).read_text(encoding="utf-8")
    try:
        stim_dem = stim.DetectorErrorModel(raw)
    except Exception as exc:
        raise ValueError(f"无法解析为 Stim DetectorErrorModel：{path}") from exc
    return _dem_to_model(stim_dem)


def layout_from_stim_text(text: str) -> DetectorLayout:
    """由 .stim 线路文本构造一个退化的 DetectorLayout。

    外部文件没有「稳定子 / 轮」的结构信息，故把每个 detector 视作独立一项，
    round0_stabilizers 与 coords 留空。
    """
    stim = _require_stim()
    circuit = stim.Circuit(text)
    n_det = int(circuit.num_detectors)
    detectors = tuple(
        Detector(index=i, records=(), stabilizer=i, round_index=0) for i in range(n_det)
    )
    observables = tuple(
        Observable(index=i, records=()) for i in range(int(circuit.num_observables))
    )
    return DetectorLayout(
        n_detectors=n_det, n_rounds=1, n_stabilizers=n_det,
        detectors=detectors, observables=observables, coords={}, round0_stabilizers=(),
    )
```

在 `aicir/qec/interop/__init__.py` 追加三个名字的导入与 `__all__` 条目。

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_stim_import.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/interop/import_.py aicir/qec/interop/__init__.py tests/qec/test_stim_import.py
git commit -m "feat(qec): 读入外部 .stim / .dem"
```

---

### Task 6: `PyMatchingDecoder` —— 滑窗在线匹配解码

**Files:**
- Create: `aicir/qec/decoders/matching.py`
- Modify: `aicir/qec/decoders/__init__.py`（可选注册）、`aicir/qec/__init__.py`
- Test: `tests/qec/test_pymatching_decoder.py`

**Interfaces:**
- Consumes：Task 1 的 `build_dem` 与 `DetectorErrorModel.check_matrix/faults_matrix/weights`
- Produces：`PyMatchingDecoder(code, schedule, errors, rounds, *, window=None, commit_lag=0, logical_state="0")`，注册名 `"pymatching"`

**关键设计——为什么 `frame_flips` 是增量：** M1 运行器对 `frame_flips` 做 `frame ^= ...`，故解码器必须吐**增量**。PyMatching 每次返回的是「到目前为止的全量预测」，因此本解码器保存上次已提交的预测，提交时吐 `当前预测 ^ 上次已提交预测`。**直接吐全量会导致 frame 反复自我抵消**，这与 M1 里 active 模式重复计数那个 bug 是同一类错误。

**因果性：** 每轮只把「已到达轮次」的 detection events 填进综合征向量，未来轮次保持 0。解码器物理上拿不到未来数据——`update(t, ...)` 是唯一入口，而轮 t+1 尚未被采样。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_pymatching_decoder.py`：

```python
import numpy as np
import pytest

pytest.importorskip("pymatching")

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.decoders.matching import PyMatchingDecoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.schedules import BareAncillaSchedule


def _dec(code, rounds, **kw):
    return PyMatchingDecoder(code, BareAncillaSchedule(),
                             PauliErrorModel(p_data=0.03, p_measure=0.03), rounds, **kw)


def test_committed_through_is_monotone():
    code = get_code("repetition", d=5, basis="Z")
    rounds = 6
    result = run(code, errors=PauliErrorModel(p_data=0.03, p_measure=0.03),
                 decoder=_dec(code, rounds, commit_lag=1), rounds=rounds, shots=4, seed=3)
    committed = [s.committed_through for s in result.records[0].decode_steps]
    assert committed == sorted(committed)


def test_flush_commits_everything():
    code = get_code("repetition", d=5, basis="Z")
    rounds = 5
    dec = _dec(code, rounds, commit_lag=2)
    result = run(code, errors=PauliErrorModel(p_data=0.03), decoder=dec,
                 rounds=rounds, shots=2, seed=1)
    assert result.records[0].decode_steps[-1].committed_through <= rounds - 1


def test_noiseless_run_is_clean():
    code = get_code("repetition", d=5, basis="Z")
    rounds = 4
    result = run(code, errors=PauliErrorModel(), decoder=_dec(code, rounds),
                 rounds=rounds, shots=8, seed=0)
    assert result.logical_error_rate == 0.0


def test_full_window_matches_batch_decoding():
    """窗口覆盖全部轮次时应退化为批式解码，与 commit_lag=0 的全窗结果一致。"""
    code = get_code("repetition", d=5, basis="Z")
    rounds = 5
    common = dict(errors=PauliErrorModel(p_data=0.04, p_measure=0.04),
                  rounds=rounds, shots=16, seed=9)
    a = run(code, decoder=_dec(code, rounds, window=rounds), **common)
    b = run(code, decoder=_dec(code, rounds, window=None), **common)
    assert a.verdict_counts == b.verdict_counts


def test_beats_per_round_lookup_when_measurement_noise_dominates():
    """多轮空时匹配相对逐轮查表的实际收益 —— 做 M2a 的理由本身。

    配置刻意选在逐轮解码结构性吃亏的区间：测量噪声与数据噪声同量级、轮数够多。
    断言留明确余量而非严格小于，避免变成随种子飘的脆弱测试。
    **失败时先怀疑适配器接线（权重、匹配图、滑窗提交），不要放宽阈值。**
    """
    code = get_code("repetition", d=5, basis="Z")
    rounds, shots = 6, 400
    errors = PauliErrorModel(p_data=0.05, p_measure=0.05, channel="bit_flip")
    lookup = run(code, errors=errors, decoder=LookupDecoder(code, t=2, error_basis="X"),
                 rounds=rounds, shots=shots, seed=7)
    matching = run(code, errors=errors,
                   decoder=PyMatchingDecoder(code, BareAncillaSchedule(), errors, rounds),
                   rounds=rounds, shots=shots, seed=7)
    assert lookup.logical_error_rate - matching.logical_error_rate > 0.05, (
        f"lookup={lookup.logical_error_rate:.4f} matching={matching.logical_error_rate:.4f}"
    )


def test_missing_pymatching_raises_at_construction(monkeypatch):
    import aicir.qec.decoders.matching as mod
    monkeypatch.setattr(mod, "_require_pymatching",
                        lambda: (_ for _ in ()).throw(ImportError("no pymatching")))
    code = get_code("steane")
    with pytest.raises(ImportError):
        PyMatchingDecoder(code, BareAncillaSchedule(), PauliErrorModel(p_data=0.01), 3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_pymatching_decoder.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.decoders.matching'`

- [ ] **Step 3: 实现 `aicir/qec/decoders/matching.py`**

```python
"""PyMatchingDecoder：DEM 匹配图上的滑窗在线解码。

由**我们自己的 DEM** 建图（check matrix + 权重 + faults matrix），因此只依赖
pymatching，不依赖 stim。

**滑窗而非批式**：每轮只把已到达轮次的 detection events 填进综合征，未来轮次保持 0，
再在图上求最小权匹配；提交滞后 commit_lag 轮。因果性是结构性的——update(t, …) 是
唯一入口，而轮 t+1 此刻尚未被采样。窗口覆盖全部轮次即退化为批式解码。

**frame_flips 必须是增量。** 运行器对它做 `frame ^= ...`，而 PyMatching 每次给的是
「到目前为止的全量预测」。因此本类保存上次已提交的预测，提交时吐
`当前预测 ^ 上次已提交预测`。直接吐全量会让 frame 反复自我抵消——与 M1 里
active 模式重复计数是同一类错误。
"""

from __future__ import annotations

import numpy as np

from ..dem import build_dem
from . import DecodeStep, register_decoder


def _require_pymatching():
    try:
        import pymatching
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            'PyMatchingDecoder 需要可选依赖：pip install "aicir[qec]"'
        ) from exc
    return pymatching


class PyMatchingDecoder:
    """滑窗最小权匹配在线解码器。"""

    name = "pymatching"

    def __init__(self, code, schedule, errors, rounds: int, *, window=None,
                 commit_lag: int = 0, logical_state: str = "0"):
        pymatching = _require_pymatching()
        self._code = code
        self._rounds = int(rounds)
        self.window = int(rounds) if window is None else int(window)
        self.commit_lag = int(commit_lag)
        self._dem = build_dem(code, schedule, errors, rounds, logical_state=logical_state)
        if not self._dem.mechanisms:
            raise ValueError("DEM 为空（误差概率全为 0），无法构造匹配图")
        self._matching = pymatching.Matching.from_check_matrix(
            self._dem.check_matrix(),
            weights=self._dem.weights(),
            faults_matrix=self._dem.faults_matrix(),
        )
        self._layout = None
        self._syndrome = None
        self._committed = -1
        self._last_prediction = None

    def reset(self, layout) -> None:
        self._layout = layout
        self._syndrome = np.zeros(self._dem.n_detectors, dtype=np.uint8)
        self._committed = -1
        self._last_prediction = np.zeros(self._code.k, dtype=np.uint8)

    def cost_of(self, round_index: int, events) -> float:
        """声明代价按窗内 detection event 数计——匹配的实际工作量随缺陷数增长。"""
        return float(np.count_nonzero(self._syndrome)) + 1.0

    def _predict(self) -> np.ndarray:
        pred = self._matching.decode(self._syndrome)
        return np.asarray(pred, dtype=np.uint8).ravel()[:self._code.k]

    def update(self, round_index: int, events) -> DecodeStep:
        # 只填已到达的轮次；未来轮次保持 0（因果性）
        for d in self._layout.detectors:
            if d.round_index == int(round_index):
                self._syndrome[d.index] = np.asarray(events, dtype=np.uint8)[d.stabilizer]

        target = int(round_index) - self.commit_lag
        cost = self.cost_of(round_index, events)
        if target <= self._committed:
            return DecodeStep(frame_flips=None, corrections=None,
                              committed_through=self._committed, cost=cost)

        prediction = self._predict()
        delta = (prediction ^ self._last_prediction).astype(np.uint8)
        self._last_prediction = prediction
        self._committed = target
        # frame 布局是 [X_0, Z_0, X_1, Z_1, ...]；Z 基读出下逻辑 X 错误占 X 分量
        frame = np.zeros(2 * self._code.k, dtype=np.uint8)
        frame[0::2] = delta
        return DecodeStep(frame_flips=frame, corrections=None,
                          committed_through=self._committed, cost=cost)

    def flush(self) -> DecodeStep:
        """线路结束：把窗内剩余的滞后判定一次性提交。"""
        prediction = self._predict()
        delta = (prediction ^ self._last_prediction).astype(np.uint8)
        self._last_prediction = prediction
        self._committed = max(self._committed, self._rounds - 1)
        frame = np.zeros(2 * self._code.k, dtype=np.uint8)
        frame[0::2] = delta
        return DecodeStep(frame_flips=frame, corrections=None,
                          committed_through=self._committed, cost=0.0)


register_decoder("pymatching", PyMatchingDecoder)
```

在 `aicir/qec/__init__.py` 里以保护性导入追加 `PyMatchingDecoder`（缺 pymatching 时为 `None`，与 `aicir.qml` 守护 torch 同款写法）。

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_pymatching_decoder.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/decoders/matching.py aicir/qec/decoders/__init__.py \
        aicir/qec/__init__.py tests/qec/test_pymatching_decoder.py
git commit -m "feat(qec): PyMatching 滑窗在线匹配解码器"
```

---

### Task 7: 依赖门禁收窄、README 与 CHANGELOG

**Files:**
- Modify: `tests/qec/test_public_api.py`、`aicir/qec/README.md`、`CHANGELOG.md`
- Test: `tests/qec/test_public_api.py`

**必须改的既有测试：** `test_qec_core_has_no_optional_dependencies` 目前 grep 整个 `aicir/qec/`，M2a 之后必然失败。

- [ ] **Step 1: 收窄依赖门禁**

把该测试改为只覆盖核心模块，并**显式列出允许触碰可选依赖的文件**：

```python
def test_qec_core_has_no_optional_dependencies():
    """qec **核心**只能依赖 numpy。

    M2a 起，samplers/stim.py、interop/、decoders/matching.py 允许触碰 stim/pymatching
    （均为保护性导入）。除此之外一律不得引入可选依赖——尤其 dem.py 必须保持纯净，
    因为「DEM 由我们自己推导」正是 oracle 能成立的前提。
    """
    import pathlib
    import re

    allowed = {
        "samplers/stim.py", "interop/export.py", "interop/import_.py",
        "interop/compare.py", "interop/__init__.py", "decoders/matching.py",
    }
    root = pathlib.Path("aicir/qec")
    banned = re.compile(r"^\s*(?:import|from)\s+(torch|scipy|matplotlib|stim|pymatching)\b",
                        re.MULTILINE)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowed:
            continue
        hits = banned.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} 引入了禁止的依赖：{hits}"


def test_dem_module_is_importable_without_stim():
    """dem.py 必须无 stim 也能用——否则 oracle 就成了拿 Stim 验 Stim。"""
    import pathlib
    src = pathlib.Path("aicir/qec/dem.py").read_text(encoding="utf-8")
    assert "stim" not in src and "pymatching" not in src
```

同时把 `test_public_api.py` 的公开名集合补上 M2a 新增的 `DetectorErrorModel`、`ErrorMechanism`、`build_dem`。

- [ ] **Step 2: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/ -q`
Expected: 全部通过。

- [ ] **Step 3: 增补 README**

在 `aicir/qec/README.md` 中新增章节，覆盖：

1. **DEM**：`build_dem` 用法；两条推导规则（data 错误只触发一处、测量翻转触发相邻两处）；为什么是解析推导而非模拟（循环论证）。
2. **Stim 互操作**：`to_stim_text`/`write_stim`/`dem_from_dem_file`；`sampler="stim"`；`pip install "aicir[qec]"`。
3. **oracle**：主检验（采样统计）与辅检验（DEM 结构比对）的分工，以及**为什么结构失配不等于缺陷**。
4. **PyMatchingDecoder**：滑窗语义、`window`/`commit_lag`、与 `LookupDecoder` 的适用区间差别。
5. **已知局限增补**（接在 M1 那六条之后）：
   - Stim 路径不回报 `raw_syndromes` 与 `injected_errors`（为 `None`/空），故 M3 的逐 shot 错误轨迹图只能画 aicir 采样的 shot。
   - Stim 路径不支持 `correction_mode="active"`（Stim 不保留量子态）。
   - DEM 只描述唯象误差模型，不含门级故障。
   - observable 翻转判据假设 `logical_z` 为纯 Z 型（`logical_x` 纯 X 型），与 M1 运行器同一前提；含 Y 分量的用户逻辑算符会失准。

- [ ] **Step 4: CHANGELOG**

在 `CHANGELOG.md` 顶部加 dated 条目，记录 M2a 的新增能力，并**点明 oracle 的意义**：这是 M1 全程缺失的外部参照，此前一切都在拿自推导的期望校验自推导的实现。

- [ ] **Step 5: 全量回归**

Run: `PYTHONPATH=. pytest tests/qec/ -q` 与 `PYTHONPATH=. pytest -q`
Expected: 全部通过。

- [ ] **Step 6: 提交**

```bash
git add tests/qec/test_public_api.py aicir/qec/README.md CHANGELOG.md
git commit -m "docs(qec): M2a 依赖门禁收窄、README 与 CHANGELOG"
```

---

## Self-Review

**1. Spec 覆盖检查**

| Spec 组件 | 对应 Task |
| --- | --- |
| 组件 1 `dem.py` 解析构造 | Task 1 |
| 组件 2 `samplers/`（Sampler 协议 + AicirSampler） | Task 2 |
| 组件 3 `interop/export.py` | Task 3 |
| 组件 4 `interop/import_.py` | Task 5 |
| 组件 5 oracle 策略（主/辅两层） | Task 4（`test_stim_oracle.py` + `interop/compare.py`） |
| 组件 6 `decoders/matching.py` | Task 6 |
| `pyproject.toml` 的 `qec` extra | Task 3 |
| 依赖门禁收窄 | Task 7 |
| README / CHANGELOG | Task 7 |
| 六个 spec 测试文件 | 全部落地，另加 `test_public_api.py` 的收窄 |

无遗漏。

**2. 与 spec 的一处有意偏离（已在计划中说明）**

spec 的 `Sampler` 协议提到 `apply_corrections` 钩子以支持 active 模式。**本计划改为：active 模式不走采样器**，保留 runner 的内联逐轮路径。理由是重构面积最小，且 M1 刚修好的 active 精细逻辑（相邻差扣除、末端读出回退）不被触碰。两个采样器的 `supports_active_mode` 均为 `False`，`StimSampler` 与 active 的组合仍按 spec 要求报错。

**3. 占位符扫描**

无 TBD / TODO / "类似 Task N" / "适当处理错误"。每个 Step 都含可直接运行的代码或命令。Task 7 Step 3 的 README 以五点提纲给出（散文体手册），每点的必含内容已列明。

**4. 类型一致性**

- `ErrorMechanism` / `DetectorErrorModel` 字段在 Task 1 定义，Task 4/5/6 使用一致。
- `ShotContext` / `ShotSamples` 字段在 Task 2 定义，Task 4 的 `StimSampler` 与测试使用一致。
- `build_dem(code, schedule, errors, rounds, *, logical_state)` 签名在 Task 1 定义，Task 4/6 调用一致。
- `to_stim_text/to_stim_circuit(code, schedule, errors, rounds, *, logical_state)` 在 Task 3 定义，Task 4/5 调用一致。
- `DecodeStep` 字段沿用 M1，Task 6 的 `frame_flips` 按 M1 的 `frame ^= ...` 语义吐**增量**。

**5. 实现者需注意的风险**

- **Task 3/4 的语义对齐是本里程碑成败所系。** 导出器与 M1 运行器必须描述同一个实验：噪声位置（提取之前）、轮 0 不加噪、`MR` 对应 measure+reset、`rec[]` 偏移、轮 0 detector 只对确定生成元。`test_detection_rates_agree_with_stim` 就是这条的守卫。
- **Task 6 的 `frame_flips` 增量语义**极易写成全量，且写错后噪声较小时仍可能"看起来对"。`test_beats_per_round_lookup_when_measurement_noise_dominates` 是它的间接守卫。
- **Task 2 是纯重构**，判据是行为逐字节不变；任何测试数减少都说明改坏了。

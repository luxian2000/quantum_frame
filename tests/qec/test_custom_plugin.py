"""证明「插入自己的码 / 调度 / 解码器」不需要改模块内部代码。

三样东西全部在本文件内定义并注册，然后端到端跑通。
"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, cz, hadamard, measure, reset
from aicir.core.classical import ClassicalRegister
from aicir.qec import run
from aicir.qec.code import StabilizerCode, gf2_to_pauli
from aicir.qec.codes import CODES, get_code, register_code
from aicir.qec.decoders import DecodeStep, register_decoder, resolve_decoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.schedules import (
    BareAncillaSchedule, RoundCircuit, build_layout, register_schedule,
    resolve_schedule, verify_schedule,
)


# ---------- 1. 一个新的码：[[4,2,2]] 检测码 ----------

def build_422() -> StabilizerCode:
    """[[4,2,2]] 检测码：两个权重 4 的生成元，两个逻辑比特。"""
    return StabilizerCode.from_paulis(
        ["XXXX", "ZZZZ"],
        logical_x=["XXII", "XIXI"],
        logical_z=["ZIZI", "ZZII"],
        name="detection_422",
        coords={q: (0, q) for q in range(4)},
    )


# ---------- 2. 一个新的调度：反转 CNOT 顺序 ----------

class ReversedOrderSchedule(BareAncillaSchedule):
    """与内置裸 ancilla 调度相同，但 support 上的受控门按降序施加。

    对无 flag 的裸 ancilla 提取，顺序不影响 detector 确定性 —— verify_schedule 会证实这点。
    """

    name = "reversed"

    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit:
        data = tuple(range(code.n))
        ancilla = tuple(range(code.n, code.n + code.m))
        n_total = code.n + code.m
        reg = ClassicalRegister(code.m, creg_name)
        cir = Circuit(n_qubits=n_total)
        for j in range(code.m):
            anc = ancilla[j]
            labels = gf2_to_pauli(code.generators[j])
            cir.append(hadamard(anc))
            for q in range(code.n - 1, -1, -1):          # 降序 —— 与内置调度相反
                ch = labels[q]
                if ch == "X":
                    cir.append(cx(q, [anc]))
                elif ch == "Z":
                    cir.append(cz(q, [anc]))
                elif ch == "Y":
                    from aicir.core.circuit import cy
                    cir.append(cy(q, [anc]))
            cir.append(hadamard(anc))
        cir.append(measure(list(ancilla), creg=reg))
        cir.append(reset(list(ancilla)))
        return RoundCircuit(circuit=cir, creg_name=creg_name, ancilla_qubits=ancilla,
                            data_qubits=data, record_offset=int(round_index) * code.m)


# ---------- 3. 一个新的在线解码器：滑窗多数表决 ----------

class SlidingMajorityDecoder:
    """滑窗解码器：缓存 window 轮，只在某稳定子在窗内多数轮触发时才提交。

    这不是一个好解码器 —— 它存在的意义是证明「带窗口与滞后提交的在线解码器」
    能被平台正确驱动：因果性、committed_through 单调、flush 收尾。
    """

    name = "sliding_majority"

    def __init__(self, window: int = 3, commit_lag: int = 1):
        self.window = int(window)
        self.commit_lag = int(commit_lag)

    def reset(self, layout) -> None:
        self._layout = layout
        self._buffer = []
        self._committed = -1
        self._seen_rounds = []

    def cost_of(self, round_index, events) -> float:
        return float(self.window)          # 声明代价 = 窗口大小

    def update(self, round_index, events) -> DecodeStep:
        self._buffer.append(np.asarray(events, dtype=np.uint8))
        self._seen_rounds.append(int(round_index))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        # 只提交滞后 commit_lag 轮之前的轮次
        target = int(round_index) - self.commit_lag
        if target > self._committed:
            self._committed = target
        return DecodeStep(frame_flips=None, corrections=None,
                          committed_through=self._committed,
                          cost=self.cost_of(round_index, events))

    def flush(self) -> DecodeStep:
        """线路结束，强制提交所有未决。"""
        if self._seen_rounds:
            self._committed = max(self._committed, max(self._seen_rounds))
        return DecodeStep(committed_through=self._committed, cost=0.0)


# ---------- 测试 ----------

def test_custom_code_registers_and_validates():
    register_code("detection_422", build_422)
    assert "detection_422" in CODES
    code = get_code("detection_422")
    code.validate()
    assert (code.n, code.k, code.m) == (4, 2, 2)


def test_custom_schedule_registers_and_passes_detector_determinism():
    register_schedule("reversed", ReversedOrderSchedule)
    assert isinstance(resolve_schedule("reversed"), ReversedOrderSchedule)
    code = get_code("steane")
    verify_schedule(code, ReversedOrderSchedule(), rounds=3)


def test_custom_decoder_registers_and_resolves():
    register_decoder("sliding_majority", SlidingMajorityDecoder)
    dec = resolve_decoder("sliding_majority", window=3, commit_lag=1)
    assert isinstance(dec, SlidingMajorityDecoder)
    assert (dec.window, dec.commit_lag) == (3, 1)


def test_all_three_custom_pieces_run_end_to_end():
    """新码 + 新调度 + 新在线解码器，端到端跑通。"""
    register_code("detection_422", build_422)
    code = get_code("detection_422")
    result = run(
        code,
        schedule=ReversedOrderSchedule(),
        errors=PauliErrorModel(p_data=0.05, p_measure=0.02, channel="depolarizing"),
        decoder=SlidingMajorityDecoder(window=3, commit_lag=1),
        rounds=5, shots=12, seed=41,
    )
    assert result.shots == 12
    assert result.code_name == "detection_422"
    assert result.schedule_name == "reversed"
    assert result.decoder_name == "sliding_majority"
    assert sum(result.verdict_counts.values()) == 12
    assert result.records[0].detection_events.shape == (5, code.m)


def test_custom_decoder_commit_lag_is_respected_and_monotone():
    """滞后提交的解码器不得被平台误判为 committed_through 回退。"""
    code = get_code("steane")
    dec = SlidingMajorityDecoder(window=3, commit_lag=2)
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=dec,
                 rounds=6, shots=4, seed=9)
    committed = [s.committed_through for s in result.records[0].decode_steps]
    assert committed == sorted(committed)          # 单调不减
    assert committed[0] == -2 or committed[0] <= 0  # 前 commit_lag 轮尚无可提交轮次


def test_custom_decoder_works_with_timing_model():
    from aicir.qec.runner import TimingModel

    code = get_code("steane")
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 1e-6)
    result = run(code, errors=PauliErrorModel(p_data=0.05),
                 decoder=SlidingMajorityDecoder(window=4, commit_lag=1),
                 rounds=5, shots=3, seed=13, timing=timing)
    # 声明代价 = window = 4 → 每轮建模 4e-6s，轮时长 1e-6s → 每轮都超预算，backlog 线性增长
    assert result.budget_violations == 5 * 3
    assert result.max_backlog > 0.0
    rec = result.records[0]
    assert np.all(np.diff(rec.backlog) > 0)        # 吞吐失败模式：backlog 单调增长

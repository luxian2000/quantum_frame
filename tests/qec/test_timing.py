"""Task 8：实时模型（TimingModel）、backlog 递推与提交延迟统计。"""

import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.runner import TimingModel, backlog_sequence, commit_latency_sequence


def test_backlog_recurrence_matches_hand_computed_sequence():
    """backlog[t] = max(0, backlog[t-1] + decode_time[t] - round_duration)"""
    decode = [3.0, 0.5, 2.0, 0.25]
    got = backlog_sequence(decode, round_duration=1.0)
    # 轮0: max(0, 0+3-1)=2 ; 轮1: max(0, 2+0.5-1)=1.5 ; 轮2: max(0, 1.5+2-1)=2.5 ; 轮3: max(0,2.5+0.25-1)=1.75
    assert got == pytest.approx([2.0, 1.5, 2.5, 1.75])


def test_backlog_stays_zero_when_decoder_keeps_up():
    got = backlog_sequence([0.2, 0.2, 0.2], round_duration=1.0)
    assert got == pytest.approx([0.0, 0.0, 0.0])


def test_commit_latency_uses_previous_round_backlog_not_current():
    """提交延迟是 FIFO 单服务台的 sojourn time：第 t 轮的排队延迟是 backlog[t-1]
    （t=0 时前面无历史，取 0），不是 backlog[t]——backlog[t] 是第 t 轮处理完之后
    留给下一轮的积压。用错索引会把每个拥堵轮次的延迟低估恰好一个 round_duration。

    与 test_backlog_recurrence_matches_hand_computed_sequence 共用同一组手算输入，
    便于两个数列对照检查：backlog=[2.0,1.5,2.5,1.75]，
    真实 FIFO sojourn time=[3.0,2.5,3.5,2.75]（= decode_time[t] + backlog[t-1]）。
    """
    decode = [3.0, 0.5, 2.0, 0.25]
    got = commit_latency_sequence(decode, round_duration=1.0, commit_lag=0)
    assert got == pytest.approx([3.0, 2.5, 3.5, 2.75])


def test_commit_latency_adds_commit_lag_times_round_duration():
    """commit_lag>0 时，每轮延迟额外加上 commit_lag × round_duration。"""
    decode = [3.0, 0.5, 2.0, 0.25]
    got = commit_latency_sequence(decode, round_duration=1.0, commit_lag=2)
    # 在 test_commit_latency_uses_previous_round_backlog_not_current 的结果上
    # 逐轮加 2 × 1.0 = 2.0。
    assert got == pytest.approx([5.0, 4.5, 5.5, 4.75])


def test_timing_fields_populate_when_model_given():
    code = get_code("steane")
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 1e-7)
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=4, shots=4, seed=0, timing=timing)
    assert result.max_backlog is not None
    assert result.mean_commit_latency is not None
    assert result.budget_violations is not None
    rec = result.records[0]
    assert rec.commit_latency.shape == (4,)
    assert rec.backlog.shape == (4,)


def test_budget_violation_counted_when_decode_exceeds_round_duration():
    code = get_code("steane")
    # 每轮声明代价 1.0，映射成 10s，远超 1e-6s 的轮时长 → 每轮都超预算
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 10.0)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=2, seed=0, timing=timing)
    assert result.budget_violations == 3 * 2


def test_no_violation_when_decoder_is_fast_enough():
    code = get_code("steane")
    timing = TimingModel(round_duration=1.0, cost_to_seconds=lambda c: c * 1e-9)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=2, seed=0, timing=timing)
    assert result.budget_violations == 0
    assert result.max_backlog == pytest.approx(0.0)


def test_wall_clock_is_recorded_separately_from_modeled_time():
    """wall-clock 与建模时间必须是分开的字段，绝不混同。"""
    code = get_code("steane")
    timing = TimingModel(round_duration=1.0, cost_to_seconds=lambda c: c * 5.0)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=2, shots=1, seed=0, timing=timing)
    rec = result.records[0]
    assert rec.wall_clock.shape == (2,)
    assert np.all(rec.wall_clock >= 0.0)
    # 建模的解码时长是 5s/轮，wall-clock 实测远小于此 —— 二者不相等即证明未混用
    assert rec.wall_clock.max() < 1.0


def test_timing_model_rejects_bad_round_duration():
    with pytest.raises(ValueError, match="round_duration"):
        TimingModel(round_duration=0.0)

import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep
from aicir.qec.errors import PauliErrorModel


class SpyDecoder:
    """记录自己被喂了什么——用来证明不存在通往未来轮次的通道。"""

    name = "spy"
    window = 1
    commit_lag = 0

    def __init__(self):
        self.seen = []
        self.reset_args = []
        self.flushed = False
        self._committed = -1

    def reset(self, layout):
        self.reset_args.append(layout)
        self.seen = []
        self.flushed = False
        self._committed = -1

    def update(self, round_index, events):
        self.seen.append((round_index, np.array(events, copy=True)))
        self._committed = int(round_index)
        return DecodeStep(committed_through=self._committed, cost=1.0)

    def flush(self):
        self.flushed = True
        return DecodeStep(committed_through=self._committed, cost=0.0)

    def cost_of(self, round_index, events):
        return 1.0


class RegressingDecoder(SpyDecoder):
    """轮 0 正常提交，轮 1 起明确回退到比此前已提交值更小的 committed_through。

    注：brief 原稿里 update() 始终返回 committed_through=0——轮 0 汇报的也是 0，
    之后各轮同样汇报 0，相对「此前已提交值」从未真正变小（0 不小于 0），
    因此从不触发回退检查，pytest.raises 断言会失败（DID NOT RAISE）。
    这里改为轮 0 之后返回 -1，确保真的比此前提交值（0）更小，
    以实际测试 runner 的 committed_through 单调不减不变量。
    """

    name = "regressing"

    def update(self, round_index, events):
        super().update(round_index, events)
        if round_index == 0:
            return DecodeStep(committed_through=0, cost=1.0)
        return DecodeStep(committed_through=-1, cost=1.0)   # 第二轮起回退


def test_decoder_is_called_once_per_round_in_order():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=4, shots=1, seed=0)
    assert [r for r, _ in spy.seen] == [0, 1, 2, 3]


def test_decoder_receives_only_layout_and_events():
    """解码器拿到的只有 DetectorLayout 与事件向量——没有线路、码、量子态、后端。"""
    from aicir.qec.detectors import DetectorLayout

    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(p_data=0.1), decoder=spy, rounds=2, shots=1, seed=0)
    assert all(isinstance(a, DetectorLayout) for a in spy.reset_args)
    for _, events in spy.seen:
        assert events.shape == (code.m,)
        assert events.dtype == np.uint8


def test_flush_is_called_at_end_of_every_shot():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=2, shots=3, seed=0)
    assert spy.flushed


def test_reset_is_called_once_per_shot():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=2, shots=5, seed=0)
    assert len(spy.reset_args) == 5


def test_runner_rejects_regressing_committed_through():
    code = get_code("steane")
    with pytest.raises(ValueError, match="committed_through"):
        run(code, errors=PauliErrorModel(), decoder=RegressingDecoder(),
            rounds=3, shots=1, seed=0)

"""Task 9：active 修正模式与 detector 参考值扣除。

守卫的是 spec 里点名的那处细节——active 模式下施加修正会把**原始**稳定子读数
复位，若不扣除已施加修正自身的综合征贡献，下一轮的朴素差分就会放出一个
**虚假 detection event**。处理正确时，frame / active 两种模式交给解码器的
detection event 流应逐字节相同。
"""

import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel


@pytest.mark.parametrize("name,kwargs", [("steane", {}), ("five_qubit", {})])
@pytest.mark.parametrize("rounds", [2, 4])
def test_frame_and_active_produce_identical_event_streams(name, kwargs, rounds):
    """两种模式交给解码器的 detection event 流必须逐字节相同。

    这正是「active 模式需扣除已施加修正的综合征贡献」那处细节的守卫测试。
    """
    code = get_code(name, **kwargs)
    common = dict(errors=PauliErrorModel(p_data=0.08, channel="depolarizing"),
                  rounds=rounds, shots=8, seed=17)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.detection_events, rb.detection_events)


@pytest.mark.parametrize("name,kwargs", [("steane", {}), ("five_qubit", {})])
def test_frame_and_active_agree_on_verdicts(name, kwargs):
    code = get_code(name, **kwargs)
    common = dict(errors=PauliErrorModel(p_data=0.08, channel="depolarizing"),
                  rounds=3, shots=16, seed=23)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    assert a.verdict_counts == b.verdict_counts
    assert [r.verdict for r in a.records] == [r.verdict for r in b.records]


def test_active_mode_rejects_frame_only_decoder():
    class FrameOnlyDecoder:
        name = "frame_only"
        window, commit_lag = 1, 0

        def reset(self, layout): self._c = -1
        def update(self, t, ev):
            self._c = t
            return DecodeStep(frame_flips=None, corrections=None, committed_through=t, cost=1.0)
        def flush(self): return DecodeStep(committed_through=self._c)
        def cost_of(self, t, ev): return 1.0

    code = get_code("steane")
    with pytest.raises(ValueError, match="active"):
        run(code, errors=PauliErrorModel(), decoder=FrameOnlyDecoder(),
            rounds=2, shots=1, correction_mode="active", seed=0)


def test_unknown_correction_mode_raises():
    code = get_code("steane")
    with pytest.raises(ValueError, match="correction_mode"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
            rounds=1, shots=1, correction_mode="teleport", seed=0)

import numpy as np
import pytest

from aicir.qec.code import gf2_to_pauli, pauli_to_gf2
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep, resolve_decoder
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.schedules import BareAncillaSchedule, build_layout


def _layout(code, rounds=1):
    return build_layout(code, BareAncillaSchedule(), rounds)


@pytest.mark.parametrize("name,kwargs", [
    ("five_qubit", {}), ("steane", {}), ("shor", {}), ("surface", {"d": 3}),
])
def test_lookup_corrects_every_weight_one_error(name, kwargs):
    """穷举：距离 3 的码必须纠正所有权重 1 错误。"""
    code = get_code(name, **kwargs)
    dec = LookupDecoder(code)
    dec.reset(_layout(code))
    for q in range(code.n):
        for p in "XYZ":
            err = pauli_to_gf2("I" * q + p + "I" * (code.n - q - 1), code.n)
            syn = code.syndrome(err)
            correction = dec.correction_for_syndrome(syn)
            residual = (err ^ correction) % 2
            assert code.verdict(residual) == "corrected", f"{name} 未纠正 {gf2_to_pauli(err)}"


def test_repetition_needs_explicit_t_because_distance_is_one():
    """重复码 distance()==1 → 默认 t 会算成 0，必须抛错而不是静默建无用的表。"""
    code = get_code("repetition", d=3, basis="Z")
    with pytest.raises(ValueError, match="t"):
        LookupDecoder(code)
    dec = LookupDecoder(code, t=1, error_basis="X")
    dec.reset(_layout(code))
    for q in range(3):
        err = pauli_to_gf2("I" * q + "X" + "I" * (2 - q), 3)
        residual = (err ^ dec.correction_for_syndrome(code.syndrome(err))) % 2
        assert code.verdict(residual) == "corrected"


def test_zero_syndrome_yields_identity_correction():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code))
    correction = dec.correction_for_syndrome(np.zeros(code.m, dtype=np.uint8))
    assert not correction.any()


def test_update_commits_every_round_immediately():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=3))
    assert (dec.window, dec.commit_lag) == (1, 0)
    for t in range(3):
        step = dec.update(t, np.zeros(code.m, dtype=np.uint8))
        assert isinstance(step, DecodeStep)
        assert step.committed_through == t


def test_flush_leaves_nothing_pending():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=2))
    dec.update(0, np.zeros(code.m, dtype=np.uint8))
    dec.update(1, np.zeros(code.m, dtype=np.uint8))
    assert dec.flush().committed_through == 1


def test_cost_is_reported_and_nonnegative():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=1))
    step = dec.update(0, np.zeros(code.m, dtype=np.uint8))
    assert step.cost >= 0.0
    assert dec.cost_of(0, np.zeros(code.m, dtype=np.uint8)) >= 0.0


def test_resolve_decoder_accepts_instance_and_rejects_unknown_name():
    code = get_code("steane")
    dec = LookupDecoder(code)
    assert resolve_decoder(dec) is dec
    with pytest.raises(KeyError):
        resolve_decoder("no_such_decoder")

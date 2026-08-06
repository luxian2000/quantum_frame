import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel


def test_noiseless_run_never_reports_a_logical_error():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=8, seed=0)
    assert result.logical_error_rate == 0.0
    assert result.verdict_counts == {"corrected": 8}


@pytest.mark.parametrize("name,kwargs", [
    ("five_qubit", {}), ("steane", {}), ("surface", {"d": 3}),
])
def test_low_rate_noise_after_preparation_is_corrected(name, kwargs):
    """轮 0 制备、轮 1 注入低速率噪声 → 权重 1 错误全可纠，逻辑错误率为 0。

    rounds=2 而非 1：轮 0 是投影式制备不注入错误（见 Global Constraints），
    rounds=1 只做制备，不构成纠错检验。
    """
    code = get_code(name, **kwargs)
    result = run(code, errors=PauliErrorModel(p_data=0.02, channel="depolarizing"),
                 decoder=LookupDecoder(code), rounds=2, shots=64, seed=3)
    assert result.logical_error_rate == 0.0


def test_round_zero_injects_no_errors():
    """轮 0 是制备轮 —— 注入的错误事件不得出现在轮 0。"""
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.5, p_measure=0.5),
                 decoder=LookupDecoder(code), rounds=3, shots=4, seed=1)
    for rec in result.records:
        assert all(e.round_index >= 1 for e in rec.injected_errors)


def test_result_reports_stderr_and_config():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=2, shots=32, seed=1)
    assert result.shots == 32 and result.rounds == 2
    assert result.code_name == "steane" and result.decoder_name == "lookup"
    assert result.logical_error_rate_stderr >= 0.0
    assert isinstance(result.summary(), str)


def test_records_capture_syndromes_and_detection_events():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.1), decoder=LookupDecoder(code),
                 rounds=3, shots=4, seed=2)
    rec = result.records[0]
    assert rec.raw_syndromes.shape == (3, code.m)
    assert rec.detection_events.shape == (3, code.m)
    assert len(rec.decode_steps) == 3
    assert rec.verdict in ("corrected", "logical_x", "logical_y", "logical_z")


def test_keep_records_caps_memory_but_aggregates_cover_all_shots():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=2, shots=40, seed=5, keep_records=3)
    assert len(result.records) == 3
    assert sum(result.verdict_counts.values()) == 40


def test_runner_rejects_missing_decoder():
    code = get_code("steane")
    with pytest.raises(ValueError, match="decoder"):
        run(code, errors=PauliErrorModel(), decoder=None, rounds=2, shots=1)


def test_timing_fields_are_none_without_a_timing_model():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=2, shots=4, seed=0)
    assert result.max_backlog is None
    assert result.mean_commit_latency is None
    assert result.budget_violations is None
    assert result.records[0].commit_latency is None


def test_runner_rejects_bad_rounds():
    code = get_code("steane")
    with pytest.raises(ValueError, match="rounds"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code), rounds=0, shots=1)


def test_runner_rejects_rounds_one_because_it_injects_nothing():
    """rounds=1 只有制备轮，恒不注入错误 → 逻辑错误率恒为 0.0 且与 errors 无关。

    这是静默乐观失效（一个看起来极好、实则毫无意义的数字），必须在入口拒绝，
    而不是让调用方拿到 0.0 之后自己去怀疑。
    """
    code = get_code("steane")
    with pytest.raises(ValueError, match="rounds"):
        run(code, errors=PauliErrorModel(p_data=0.5), decoder=LookupDecoder(code),
            rounds=1, shots=4)


def test_invalid_correction_mode_is_reported_even_when_rounds_also_invalid():
    """枚举先于数值区间校验：非法 correction_mode 不应被 rounds 的报错掩盖。"""
    code = get_code("steane")
    with pytest.raises(ValueError, match="correction_mode"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
            rounds=1, shots=1, correction_mode="teleport")

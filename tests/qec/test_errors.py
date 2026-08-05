import numpy as np
import pytest

from aicir.qec.errors import ErrorEvent, PauliErrorModel


def test_zero_probability_yields_no_events():
    model = PauliErrorModel(p_data=0.0, p_measure=0.0)
    rng = np.random.default_rng(0)
    assert model.sample_round(0, n_data=5, n_ancilla=4, rng=rng) == []


def test_certain_probability_hits_every_qubit():
    model = PauliErrorModel(p_data=1.0, p_measure=1.0, channel="bit_flip")
    rng = np.random.default_rng(0)
    events = model.sample_round(2, n_data=3, n_ancilla=2, rng=rng)
    data = [e for e in events if e.source == "data"]
    meas = [e for e in events if e.source == "measurement"]
    assert len(data) == 3 and len(meas) == 2
    assert {e.qubit for e in data} == {0, 1, 2}
    assert {e.qubit for e in meas} == {0, 1}
    assert all(e.round_index == 2 for e in events)


def test_bit_flip_channel_only_emits_x():
    model = PauliErrorModel(p_data=1.0, channel="bit_flip")
    rng = np.random.default_rng(1)
    events = model.sample_round(0, n_data=8, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"X"}


def test_phase_flip_channel_only_emits_z():
    model = PauliErrorModel(p_data=1.0, channel="phase_flip")
    rng = np.random.default_rng(1)
    events = model.sample_round(0, n_data=8, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"Z"}


def test_depolarizing_channel_emits_all_three():
    model = PauliErrorModel(p_data=1.0, channel="depolarizing")
    rng = np.random.default_rng(3)
    events = model.sample_round(0, n_data=300, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"X", "Y", "Z"}


def test_sampling_is_reproducible_under_seed():
    model = PauliErrorModel(p_data=0.3, p_measure=0.2)
    a = model.sample_round(0, 9, 8, np.random.default_rng(7))
    b = model.sample_round(0, 9, 8, np.random.default_rng(7))
    assert a == b


def test_rate_is_approximately_p():
    model = PauliErrorModel(p_data=0.1, channel="bit_flip")
    rng = np.random.default_rng(11)
    hits = sum(len(model.sample_round(t, 100, 0, rng)) for t in range(100))
    assert 800 < hits < 1200          # 10000 次伯努利试验，p=0.1


def test_rejects_bad_probability_and_channel():
    with pytest.raises(ValueError, match="概率"):
        PauliErrorModel(p_data=1.5)
    with pytest.raises(ValueError, match="channel"):
        PauliErrorModel(channel="no_such_channel")


def test_event_partition_helpers():
    model = PauliErrorModel(p_data=1.0, p_measure=1.0)
    events = model.sample_round(0, 2, 2, np.random.default_rng(0))
    assert len(model.data_events(events)) == 2
    assert len(model.measurement_events(events)) == 2
    assert isinstance(events[0], ErrorEvent)

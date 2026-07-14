import numpy as np
import pytest
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot, measure, reset


def run(cir, **kw):
    return Measure(NumpyBackend()).run(cir, **kw)


def test_shots_none_no_terminal_measurement_state_equals_final():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=None)
    assert np.allclose(r.final_state, r.state, atol=1e-6)
    with pytest.raises(ValueError):
        r.output(-1)


def test_shots0_alias_none():
    cir = Circuit(hadamard(0), n_qubits=1)
    r = run(cir, shots=0)
    assert np.allclose(r.final_state, r.state, atol=1e-6)


def test_shots_m_terminal_shapes_and_density_state():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=64)
    assert r.output(-1).shape == (64, 2)
    # 无噪声共享纯态前态：聚合 state 保持向量形态；末端测量后的 final 为真混合态
    assert np.asarray(r.state).shape == (4,)
    assert r.final_state_kind == "density_matrix"
    counts = r.counts(-1)
    assert set(counts) <= {"00", "11"}


def test_incircuit_measure_collapses_and_output_indexed():
    cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1]), n_qubits=2)  # op2 = measure
    r = run(cir, shots=16, measure_qubits=None)
    assert r.output(2).shape == (16, 1)
    assert set(np.unique(r.output(2))) <= {1}


def test_terminal_order_preserved_in_output():
    import aicir
    cir = Circuit(aicir.pauli_x(1), n_qubits=2)
    r = run(cir, shots=1, measure_qubits=[1, 0])
    assert r.output(-1).tolist() == [[-1, 1]]


def test_measure_qubits_none_disables_terminal_in_shot_mode():
    cir = Circuit(hadamard(0), n_qubits=1)
    r = run(cir, shots=8, measure_qubits=None)
    assert r.terminal_qubits is None
    with pytest.raises(ValueError):
        r.output(-1)


def test_measure_qubits_empty_reads_all_in_shot_mode():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=8, measure_qubits=[])
    assert r.terminal_qubits == [0, 1]
    assert r.output(-1).shape == (8, 2)


def test_exact_mode_ignores_measure_qubits_no_error():
    cir = Circuit(hadamard(0), n_qubits=1)
    for mq in (None, [], [0]):
        r = run(cir, shots=None, measure_qubits=mq)
        assert r.terminal_qubits is None
        with pytest.raises(ValueError):
            r.output(-1)


def test_invalid_shots():
    cir = Circuit(hadamard(0), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=-1)
    with pytest.raises(ValueError):
        run(cir, shots=1.5)
    with pytest.raises(ValueError):
        run(cir, shots=False)   # bool 非法（False == 0 不应被当作 exact）
    with pytest.raises(ValueError):
        run(cir, shots=True)


def test_seed_reproducible():
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    a = run(cir, shots=32, seed=7, measure_qubits=None).output(1)
    b = run(cir, shots=32, seed=7, measure_qubits=None).output(1)
    assert np.array_equal(a, b)


def test_duplicate_id_raises():
    cir = Circuit(measure(0, id="m"), measure(0, id="m"), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=4, measure_qubits=None)


def test_sm_shot_not_implemented():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    with pytest.raises(NotImplementedError):
        run(cir, shots=4, snap=[0], sm="shot")


def test_shotsampler_defaults_to_full_readout():
    from aicir.primitives.sampler import ShotSampler
    # ShotSampler.run 默认 measure_qubits=()，即读出全部比特
    # SampleResult.counts 是 dict 属性，键格式为 "|bitstring>"
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    res = ShotSampler(shots=128).run(cir)
    counts = res.counts  # SampleResult.counts 是属性，非方法
    assert sum(counts.values()) == 128
    assert set(counts) <= {"|00>", "|11>"}

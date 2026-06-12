"""
机制一（Measure.run 独立测量）的 shots/final_state 语义测试。

约定（README §4.1）：
- result.state 始终是测量前的完整末态；
- shots=None/0 不测量，final_state 与 state 相同；
- shots=1 做单次投影测量：全比特读出时 final_state 为坍缩后的基态，
  子集读出时 final_state 为其余比特的坍缩纯态、output 为 Z⊗...⊗Z 关联结果（±1）；
- shots>1 时 final_state 为对被测比特求偏迹后的约化密度矩阵（无剩余比特则为 None）。
"""

import numpy as np
import pytest

from aicir import Circuit, Measure, cnot, hadamard, pauli_x
from aicir.channel.backends import NumpyBackend

BELL = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


@pytest.fixture
def m():
    return Measure(NumpyBackend())


def bell_circuit():
    return Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)


def test_state_is_pre_measurement_final_state(m):
    # 无论是否采样，result.state 都是演化后的（测量前）完整末态
    for shots in (None, 1, 7):
        result = m.run(bell_circuit(), shots=shots)
        np.testing.assert_allclose(result.state.reshape(-1), BELL, atol=1e-6)


def test_default_shots_is_one(m):
    result = m.run(bell_circuit())
    assert result.shots == 1
    assert sum(result.counts.values()) == 1


def test_shots_none_or_zero_skips_measurement(m):
    for shots in (None, 0):
        result = m.run(bell_circuit(), shots=shots, measure_qubits=[0])
        assert result.counts is None
        assert result.output is None
        np.testing.assert_allclose(
            result.final_state.reshape(-1), result.state.reshape(-1), atol=1e-6
        )


def test_single_shot_full_readout_collapses_all_qubits(m):
    result = m.run(bell_circuit(), shots=1)

    # Bell 态只会测得 00 或 11，Z⊗Z 本征值恒为 +1
    assert result.output == 1
    label = next(iter(result.counts))
    assert label in {"|00>", "|11>"}
    assert result.counts == {label: 1}

    final = result.final_state.reshape(-1)
    idx = int(label.strip("|>"), 2)
    assert abs(final[idx]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(np.delete(final, idx)) == pytest.approx(0.0, abs=1e-6)
    # state 不坍缩
    np.testing.assert_allclose(result.state.reshape(-1), BELL, atol=1e-6)


def test_single_shot_full_readout_odd_parity(m):
    # |10>：全比特单次测量，Z⊗Z 本征值为 -1
    result = m.run(Circuit(pauli_x(0), n_qubits=2), shots=1)

    assert result.output == -1
    assert result.counts == {"|10>": 1}


def test_single_shot_subset_is_correlated_z_measurement(m):
    # |10>：测 qubit0 得 Z=-1，qubit1 坍缩在 |0>
    result = m.run(Circuit(pauli_x(0), n_qubits=2), shots=1, measure_qubits=[0])

    assert result.output == -1
    np.testing.assert_allclose(result.final_state.reshape(-1), [1.0, 0.0], atol=1e-6)
    assert result.metadata["final_state_qubits"] == [1]


def test_single_shot_subset_parity_on_ghz(m):
    # GHZ 态测 qubit0、qubit1：结果 00 或 11，Z0⊗Z1 恒为 +1
    ghz = Circuit(hadamard(0), cnot(1, [0]), cnot(2, [0]), n_qubits=3)
    result = m.run(ghz, shots=1, measure_qubits=[0, 1])

    assert result.output == 1
    outcome = next(iter(result.counts)).strip("|>")
    expected = [1.0, 0.0] if outcome == "00" else [0.0, 1.0]
    np.testing.assert_allclose(result.final_state.reshape(-1), expected, atol=1e-6)


def test_multi_shot_subset_returns_reduced_density_matrix(m):
    result = m.run(bell_circuit(), shots=100, measure_qubits=[0])

    assert result.output is None
    assert sum(result.counts.values()) == 100
    assert result.final_state.shape == (2, 2)
    np.testing.assert_allclose(result.final_state, np.eye(2) / 2.0, atol=1e-6)
    assert result.metadata["final_state_kind"] == "density_matrix"


def test_multi_shot_full_readout_has_no_remaining_qubits(m):
    result = m.run(bell_circuit(), shots=100)

    assert result.final_state is None
    np.testing.assert_allclose(result.state.reshape(-1), BELL, atol=1e-6)


def test_return_state_false_drops_states_but_keeps_output(m):
    result = m.run(bell_circuit(), shots=1, return_state=False)

    assert result.state is None
    assert result.final_state is None
    assert result.output is not None


def test_density_matrix_path_mirrors_semantics(m):
    # shots=None：不测量，final_state 即完整末态（flatten 密度矩阵）
    result = m.run_density_matrix(bell_circuit(), shots=None)
    np.testing.assert_allclose(result.final_state, result.state, atol=1e-6)
    rho = result.state.reshape(4, 4)
    np.testing.assert_allclose(rho, np.outer(BELL, BELL.conj()), atol=1e-6)

    # shots=1 子集：output 为关联结果，final_state 为其余比特坍缩态（flatten）
    result = m.run_density_matrix(
        Circuit(pauli_x(0), n_qubits=2), shots=1, measure_qubits=[0]
    )
    assert result.output == -1
    np.testing.assert_allclose(
        result.final_state.reshape(2, 2), [[1.0, 0.0], [0.0, 0.0]], atol=1e-6
    )

    # shots>1 子集：对被测比特求偏迹后的约化密度矩阵（flatten）
    result = m.run_density_matrix(bell_circuit(), shots=50, measure_qubits=[0])
    np.testing.assert_allclose(result.final_state.reshape(2, 2), np.eye(2) / 2.0, atol=1e-6)

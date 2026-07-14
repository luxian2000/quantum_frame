"""
tests/measure/test_measure_run_semantics.py

统一测量模型的 shots/state/output/counts 语义测试。

已迁移至新 API（Measure.run 单入口）。

已删除（测试已移除功能）：
- test_density_matrix_path_mirrors_semantics
    → run_density_matrix 已删除；密度矩阵路径通过 initial_density_matrix 参数实现，
      参见 test_unified_run.py 及 test_measure.py::test_reset_with_density_matrix_initial_state

新模型语义约定（见 README §4 与 aicir/measure/measure.py）：
- shots ∈ {None, 0}  → exact 模式：单条精确轨迹，不做末端测量，
                        output(-1) 抛 ValueError，counts 抛 RuntimeError；
                        state == final_state（均为态矢）
- shots = 1          → 单轨迹：state 为测量前态矢，final_state 为坍缩后态矢；
                        output(-1).shape == (1, n_measured)（±1 本征值）；
                        counts(-1) 为 {"bitstring": 1} 裸比特串键（无 |> 符号）
- shots > 1          → M 条轨迹 avg 聚合：state 与 final_state 为密度矩阵；
                        output(-1).shape == (M, n_measured)；
                        counts(-1) 为 {"bitstring": N, ...} 裸比特串键
- measure_qubits     → None=不测；空(默认)=全测；[list]=子集；
                        exact 模式下忽略 measure_qubits（不报错）
- result.output/counts 均为方法，target = -1（末端）或操作下标 / id 字符串
"""

import numpy as np
import pytest

from aicir import Circuit, Measure, cnot, hadamard, pauli_x
from aicir.backends import NumpyBackend

BELL = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


@pytest.fixture
def m():
    return Measure(NumpyBackend())


def bell_circuit():
    return Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)


def test_exact_mode_state_equals_final_state(m):
    # shots=None/0：exact 模式，state 与 final_state 均为演化后态矢且相等
    for shots in (None, 0):
        result = m.run(bell_circuit(), shots=shots)
        np.testing.assert_allclose(result.state.array, BELL, atol=1e-6)
        np.testing.assert_allclose(
            result.final_state.array, result.state.array, atol=1e-6
        )


def test_exact_mode_no_terminal_measurement(m):
    # shots=None：output(-1) 抛 ValueError；counts(-1) 抛 RuntimeError
    result = m.run(bell_circuit(), shots=None)
    with pytest.raises(ValueError):
        result.output(-1)
    with pytest.raises(RuntimeError):
        result.counts(-1)


def test_default_shots_is_one(m):
    result = m.run(bell_circuit())
    assert result.shots == 1
    assert sum(result.counts(-1).values()) == 1


def test_exact_mode_ignores_measure_qubits(m):
    # exact 模式忽略 measure_qubits（None/[]/[list] 均不测、不报错）
    for shots in (None, 0):
        for mq in (None, [], [0]):
            result = m.run(bell_circuit(), shots=shots, measure_qubits=mq)
            assert result.terminal_qubits is None
            with pytest.raises(ValueError):
                result.output(-1)


def test_single_shot_state_is_pre_measurement_state_vector(m):
    # shots=1：state 为测量前纯态（态矢，1D 或 2D 列向量）
    result = m.run(bell_circuit(), shots=1)
    np.testing.assert_allclose(result.state.array, BELL, atol=1e-6)


def test_single_shot_full_readout_collapses_to_basis_state(m):
    # shots=1 全比特读出：final_state 应坍缩到某一基态
    result = m.run(bell_circuit(), shots=1)

    # Bell 态只能测得 "00" 或 "11"
    counts = result.counts(-1)
    assert set(counts.keys()) <= {"00", "11"}
    assert sum(counts.values()) == 1

    # output(-1) 为 (1, 2) 的 ±1 矩阵；ZZ 乘积恒为 +1
    out = result.output(-1)
    assert out.shape == (1, 2)
    assert int(out[0, 0]) * int(out[0, 1]) == 1

    # state 不坍缩
    np.testing.assert_allclose(result.state.array, BELL, atol=1e-6)


def test_single_shot_full_readout_odd_parity(m):
    # |10>：全比特单次测量，output 为 [[-1, 1]]（qubit0=1 → -1；qubit1=0 → +1）
    result = m.run(Circuit(pauli_x(0), n_qubits=2), shots=1)

    counts = result.counts(-1)
    assert "10" in counts
    out = result.output(-1)
    assert out.shape == (1, 2)
    assert int(out[0, 0]) == -1
    assert int(out[0, 1]) == 1


def test_single_shot_subset_measure_qubits(m):
    # |10>：仅测 qubit0，output(-1) 应为 [[-1]]（Z 本征值）
    result = m.run(Circuit(pauli_x(0), n_qubits=2), shots=1, measure_qubits=[0])

    out = result.output(-1)
    assert out.shape == (1, 1)
    assert int(out[0, 0]) == -1
    assert result.terminal_qubits == [0]


def test_single_shot_ghz_subset_output(m):
    # GHZ 态测 qubit0、qubit1：结果 "00" 或 "11"，ZZ 乘积恒为 +1
    ghz = Circuit(hadamard(0), cnot(1, [0]), cnot(2, [0]), n_qubits=3)
    result = m.run(ghz, shots=1, measure_qubits=[0, 1])

    out = result.output(-1)
    assert out.shape == (1, 2)
    assert int(out[0, 0]) * int(out[0, 1]) == 1

    counts = result.counts(-1)
    assert set(counts.keys()) <= {"00", "11"}


def test_multi_shot_output_shape(m):
    # shots>1：output(-1).shape == (M, n_qubits)；state 为密度矩阵
    result = m.run(bell_circuit(), shots=100)

    out = result.output(-1)
    assert out.shape == (100, 2)

    counts = result.counts(-1)
    assert set(counts.keys()) <= {"00", "11"}
    assert sum(counts.values()) == 100

    # shots>1 无噪声共享纯态前态：state 保持向量形态（avg(|ψ><ψ|)==|ψ><ψ|）
    assert np.asarray(result.state).shape == (4,)
    assert not result.state.is_density


def test_multi_shot_subset_output_and_density(m):
    # shots>1 子集读出：output(-1).shape == (M, 1)；final_state 为密度矩阵
    result = m.run(bell_circuit(), shots=100, measure_qubits=[0])

    out = result.output(-1)
    assert out.shape == (100, 1)

    assert result.final_state_kind == "density_matrix"
    assert np.asarray(result.final_state).shape == (4, 4)


def test_return_state_false_drops_states(m):
    result = m.run(bell_circuit(), shots=1, return_state=False)

    assert result.state is None
    assert result.final_state is None
    # output(-1) 仍可用
    assert result.output(-1) is not None


def test_snap_records_intermediate_full_states(m):
    # snap=[0, 1]：操作下标 0（H 门后）与 1（CNOT 后）的完整态快照
    result = m.run(bell_circuit(), shots=None, snap=[0, 1])

    after_h = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)
    np.testing.assert_allclose(result.snap(0).array, after_h, atol=1e-6)
    np.testing.assert_allclose(result.snap(1).array, BELL, atol=1e-6)
    # 未记录下标返回 None
    assert result.snap(2) is None
    # snapshot_states 中有且只有 0 和 1 两个快照
    assert set(result.snapshot_states.keys()) == {0, 1}


def test_no_randomness_path_batch_samples_terminal(monkeypatch, m):
    # 无噪声、无线路内随机源的 shots>1 路径：末端读出应一次性批量采样，
    # 不再逐 shot 调用 terminal_z_measure（O(shots·n·2^n) → O(2^n + shots)）
    from aicir.measure import projector as projector_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("无中途随机源路径不应逐 shot 调用 terminal_z_measure")

    monkeypatch.setattr(projector_mod, "terminal_z_measure", _boom)

    result = m.run(bell_circuit(), shots=50, seed=7)

    out = result.output(-1)
    assert out.shape == (50, 2)
    counts = result.counts(-1)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 50
    # 坍缩末态仍按读出结果正确构造
    assert result.final_state_kind == "density_matrix"
    diag = np.real(np.diag(np.asarray(result.final_state)))
    np.testing.assert_allclose(
        [diag[0], diag[3]],
        [counts.get("00", 0) / 50, counts.get("11", 0) / 50],
        atol=1e-6,
    )


def test_batch_terminal_sampling_matches_born_rule(m):
    # 批量采样与 Born 规则分布一致：ry(0.8) 后 p(|0>) = cos^2(0.4) ≈ 0.8477
    from aicir.core.circuit import ry

    shots = 4000
    result = m.run(Circuit(ry(0.8, 0), n_qubits=1), shots=shots, seed=42, return_state=False)
    counts = result.counts(-1)
    freq0 = counts.get("0", 0) / shots
    assert abs(freq0 - np.cos(0.4) ** 2) < 0.02


def test_return_probabilities_false_skips_probability_array(monkeypatch, m):
    # 概率数组按需生成：return_probabilities=False 时不计算/不返回 2^n 概率数组
    from aicir.core.state import State as _State
    from aicir.measure import aggregate as agg_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("return_probabilities=False 不应计算全谱概率数组")

    # exact 模式：概率来自 State.probabilities
    monkeypatch.setattr(_State, "probabilities", _boom)
    r = m.run(bell_circuit(), shots=None, return_probabilities=False)
    assert r.probabilities is None
    np.testing.assert_allclose(r.state.array, BELL, atol=1e-6)
    with pytest.raises(ValueError):
        r.most_probable()
    monkeypatch.undo()

    # shots>1 轻量路径：概率来自逐轨迹 _traj_probs
    monkeypatch.setattr(agg_mod, "_traj_probs", _boom)
    r2 = m.run(bell_circuit(), shots=32, seed=9, return_state=False,
               return_probabilities=False)
    assert r2.probabilities is None
    counts = r2.counts(-1)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 32

import numpy as np
import pytest

from aicir.core.state import State
from aicir.backends.numpy_backend import NumpyBackend
from aicir.measure.trajectory import TrajectoryResult
from aicir.measure.aggregate import aggregate_avg


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_aggregate_two_shots_outputs_stacked_and_state_is_density():
    t0 = TrajectoryResult(pre=_sv([1, 0]), post=_sv([1, 0]), incircuit={1: 1}, terminal=[1])
    t1 = TrajectoryResult(pre=_sv([0, 1]), post=_sv([0, 1]), incircuit={1: -1}, terminal=[-1])
    agg = aggregate_avg([t0, t1], n_qubits=1, measurement_specs=[], terminal_qubits=[0])
    assert agg["state"].shape == (2, 2)
    assert np.allclose(agg["state"], np.array([[0.5, 0], [0, 0.5]]), atol=1e-6)
    assert agg["incircuit_outputs"][1].shape == (2, 1)
    assert agg["terminal_output"].shape == (2, 1)
    assert agg["incircuit_counts"][1] == {1: 1, -1: 1}


def test_aggregate_include_states_false_skips_density(monkeypatch):
    # include_states=False：不构造 2^n×2^n 密度矩阵，state/final_state 为 None，
    # probabilities 为逐轨迹概率的平均（与 diag(平均密度矩阵) 数学等价）
    from aicir.measure import aggregate as agg_mod

    def _boom(_state):
        raise AssertionError("include_states=False 不应触发密度矩阵构造")

    monkeypatch.setattr(agg_mod, "_as_density", _boom)

    t0 = TrajectoryResult(pre=_sv([1, 0]), post=_sv([1, 0]), incircuit={1: 1}, terminal=[1])
    t1 = TrajectoryResult(pre=_sv([0, 1]), post=_sv([0, 1]), incircuit={1: -1}, terminal=[-1])
    agg = aggregate_avg([t0, t1], n_qubits=1, measurement_specs=[], terminal_qubits=[0],
                        include_states=False)

    assert agg["state"] is None
    assert agg["final_state"] is None
    np.testing.assert_allclose(agg["probabilities"], [0.5, 0.5], atol=1e-6)
    assert agg["terminal_output"].shape == (2, 1)
    assert agg["incircuit_counts"][1] == {1: 1, -1: 1}


def test_multi_shot_return_state_false_skips_density(monkeypatch):
    # Measure.run(shots>1, return_state=False)：整条链路不应构造密度矩阵
    from aicir.measure import aggregate as agg_mod
    from aicir.core.circuit import Circuit, hadamard, cnot
    from aicir.measure.measure import Measure

    def _boom(_state):
        raise AssertionError("return_state=False 的多 shot 路径不应构造密度矩阵")

    monkeypatch.setattr(agg_mod, "_as_density", _boom)

    bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    result = Measure(NumpyBackend()).run(bell, shots=50, seed=7, return_state=False)

    assert result.state is None
    assert result.final_state is None
    np.testing.assert_allclose(result.probabilities, [0.5, 0, 0, 0.5], atol=1e-6)
    counts = result.counts(-1)
    assert set(counts.keys()) <= {"00", "11"}
    assert sum(counts.values()) == 50


def test_multi_shot_observables_still_work_without_state(monkeypatch):
    # observables 需要聚合态：即使 return_state=False 也应正确计算期望值
    from aicir.core.circuit import Circuit, hadamard, cnot
    from aicir.measure.measure import Measure

    bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    zz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
    result = Measure(NumpyBackend()).run(bell, shots=20, seed=3, return_state=False,
                                         observables={"ZZ": zz})

    assert result.state is None
    assert abs(result.expectation_values["ZZ"] - 1.0) < 1e-6

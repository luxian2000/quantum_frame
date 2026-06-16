"""Sampler/Estimator primitives 测试：统一执行入口包装现有 Measure 与 PauliEstimator。

第一片范围（NEXT.md 第 4 节）：

- ``BaseSampler``/``BaseEstimator`` 接口 + 最小 ``SampleResult``/``EstimateResult``。
- ``ShotSampler``：有限 shots 采样，包装 ``Measure``。
- ``StatevectorEstimator``：精确期望（态向量路径）。
- ``ShotEstimator``：有限 shots 能量估计，包装 ``PauliEstimator``。

约定：本片接收**已绑定参数**的电路；单个电路入参返回单个结果，
序列入参返回结果列表；单个可观测量可广播到多个电路。
"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, NumpyBackend, cx, hadamard, measure, pauli_x
from aicir.primitives import (
    BaseEstimator,
    BaseSampler,
    EstimateResult,
    SampleResult,
    ShotEstimator,
    ShotSampler,
    StatevectorEstimator,
)


def _bell():
    return Circuit(hadamard(0), cx(1, [0]), n_qubits=2)


# ---------------------------------------------------------------------------
# ShotSampler
# ---------------------------------------------------------------------------


def test_shot_sampler_samples_bell_state():
    result = ShotSampler(NumpyBackend(), shots=2048).run(_bell())
    assert isinstance(result, SampleResult)
    assert isinstance(BaseSampler.__abstractmethods__, frozenset)
    assert result.shots == 2048
    assert set(result.counts) <= {"|00>", "|11>"}
    assert sum(result.counts.values()) == 2048
    assert result.measured_qubits == (0, 1)
    assert abs(result.probs["|00>"] - 0.5) < 0.1


def test_shot_sampler_respects_embedded_measure_gates():
    cir = Circuit(hadamard(0), cx(1, [0]), measure(1), n_qubits=2)
    result = ShotSampler(shots=512).run(cir)
    assert result.measured_qubits == (1,)
    assert set(result.counts) <= {"|0>", "|1>"}


def test_shot_sampler_list_input_returns_list():
    results = ShotSampler(shots=64).run([_bell(), _bell()])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, SampleResult) for r in results)


# ---------------------------------------------------------------------------
# StatevectorEstimator（精确）
# ---------------------------------------------------------------------------


def test_statevector_estimator_exact_eigenstate():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    result = StatevectorEstimator().run(Circuit(pauli_x(0), n_qubits=1), ham)
    assert isinstance(result, EstimateResult)
    assert result.value == pytest.approx(-1.0)
    assert result.shots is None


def test_statevector_estimator_superposition():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    result = StatevectorEstimator().run(Circuit(hadamard(0), n_qubits=1), ham)
    assert result.value == pytest.approx(0.0, abs=1e-6)


def test_statevector_estimator_broadcasts_single_observable():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 0.5)])
    circuits = [Circuit(n_qubits=1), Circuit(pauli_x(0), n_qubits=1)]
    results = StatevectorEstimator().run(circuits, ham)
    assert [r.value for r in results] == pytest.approx([0.5, -0.5])


def test_statevector_estimator_pairs_observable_list():
    h1 = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    h2 = Hamiltonian(n_qubits=1, terms=[("Z", -2.0)])
    circuits = [Circuit(n_qubits=1), Circuit(n_qubits=1)]
    results = StatevectorEstimator().run(circuits, [h1, h2])
    assert [r.value for r in results] == pytest.approx([1.0, -2.0])


# ---------------------------------------------------------------------------
# ShotEstimator（有限 shots）
# ---------------------------------------------------------------------------


def test_shot_estimator_eigenstate_is_exact():
    # |1> 是 Z 的本征态，任何 shots 下期望都精确为 -1
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    estimator = ShotEstimator(NumpyBackend(), shots=256)
    assert isinstance(estimator, BaseEstimator)
    result = estimator.run(Circuit(pauli_x(0), n_qubits=1), ham)
    assert isinstance(result, EstimateResult)
    assert result.value == pytest.approx(-1.0)
    assert result.shots == 256
    assert result.variance == pytest.approx(0.0)
    assert result.term_results  # 携带逐项明细


def test_shot_estimator_plugs_into_vqe_contract():
    """ShotEstimator 暴露 estimate(circuit, hamiltonian)，可直接作 BasicVQE 的 energy_estimator。"""
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    raw = ShotEstimator(shots=128).estimate(Circuit(pauli_x(0), n_qubits=1), ham)
    assert raw.energy == pytest.approx(-1.0)

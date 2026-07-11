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

from aicir import (
    BitFlipChannel,
    Circuit,
    Hamiltonian,
    NoiseModel,
    NumpyBackend,
    Parameter,
    cx,
    hadamard,
    measure,
    pauli_x,
    ry,
)
from aicir.primitives import (
    BackendEstimator,
    BackendSampler,
    BaseEstimator,
    BaseSampler,
    EstimateResult,
    NoisyEstimator,
    NoisySampler,
    SampleResult,
    ShotEstimator,
    ShotSampler,
    StatevectorEstimator,
    StatevectorSampler,
)


def _bell():
    return Circuit(hadamard(0), cx(1, [0]), n_qubits=2)


def _bitflip(p):
    return NoiseModel().add_channel(BitFlipChannel(0, p), after_gates=["ry"])


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


# ---------------------------------------------------------------------------
# StatevectorSampler（精确概率，无散粒噪声）
# ---------------------------------------------------------------------------


def test_statevector_sampler_exact_probs():
    result = StatevectorSampler().run(_bell())
    assert isinstance(result, SampleResult)
    assert result.shots is None
    assert result.counts == {}  # 精确路径无采样计数
    assert result.probs["|00>"] == pytest.approx(0.5)
    assert result.probs["|11>"] == pytest.approx(0.5)


def test_statevector_sampler_rejects_shots():
    with pytest.raises(ValueError):
        StatevectorSampler().run(_bell(), shots=100)


# ---------------------------------------------------------------------------
# NoisySampler / NoisyEstimator（噪声 / 密度矩阵路径）
# ---------------------------------------------------------------------------


def test_noisy_estimator_exact_density_matrix():
    # ry(0.3) 后 bit-flip p=0.25：<Z> → (1-2p)cos = 0.5·cos(0.3)，确定值
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    est = NoisyEstimator(noise_model=_bitflip(0.25))
    result = est.run(Circuit(ry(0.3, 0), n_qubits=1), ham)
    assert isinstance(result, EstimateResult)
    assert result.value == pytest.approx(0.5 * np.cos(0.3))
    assert result.shots is None


def test_noisy_estimator_plugs_into_vqe_contract():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    raw = NoisyEstimator(noise_model=_bitflip(0.25)).estimate(Circuit(ry(0.3, 0), n_qubits=1), ham)
    assert raw.energy == pytest.approx(0.5 * np.cos(0.3))


def test_noisy_estimator_drives_vqe_energy():
    """加性集成：BasicVQE 直接以 NoisyEstimator 作 energy_estimator，无需改 VQE。"""
    from aicir.vqc import BasicVQE

    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    ansatz = Circuit(ry(Parameter("t"), 0), n_qubits=1)
    vqe = BasicVQE(ham, ansatz=ansatz, energy_estimator=NoisyEstimator(noise_model=_bitflip(0.25)))
    assert vqe.energy(np.array([0.3])) == pytest.approx(0.5 * np.cos(0.3))


def test_noisy_sampler_counts_sum_to_shots():
    result = NoisySampler(noise_model=_bitflip(0.5), shots=256).run(Circuit(ry(0.0, 0), n_qubits=1))
    assert isinstance(result, SampleResult)
    assert result.shots == 256
    assert sum(result.counts.values()) == 256


# ---------------------------------------------------------------------------
# BackendSampler / BackendEstimator（注入式扩展点）
# ---------------------------------------------------------------------------


def test_backend_sampler_delegates_to_runner():
    calls = []

    def runner(circuit, *, shots):
        calls.append((circuit, shots))
        return {"|0>": shots}

    result = BackendSampler(runner, shots=10).run(_bell())
    assert isinstance(result, SampleResult)
    assert result.counts == {"|0>": 10}
    assert result.shots == 10
    assert len(calls) == 1


def test_backend_estimator_delegates_to_runner():
    def runner(circuit, observable, *, shots):
        return -0.5

    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    result = BackendEstimator(runner, shots=32).run(Circuit(pauli_x(0), n_qubits=1), ham)
    assert isinstance(result, EstimateResult)
    assert result.value == pytest.approx(-0.5)
    assert result.shots == 32


def test_backend_estimator_broadcasts_and_lists():
    def runner(circuit, observable, *, shots):
        return 1.0

    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    results = BackendEstimator(runner).run([Circuit(n_qubits=1), Circuit(n_qubits=1)], ham)
    assert [r.value for r in results] == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# parameter_values= 延迟绑定
# ---------------------------------------------------------------------------


def test_statevector_estimator_binds_parameter_values():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    template = Circuit(ry(Parameter("t"), 0), n_qubits=1)
    # t=pi → |1>，<Z> = -1
    result = StatevectorEstimator().run(template, ham, parameter_values=[np.pi])
    assert result.value == pytest.approx(-1.0)


def test_estimator_binds_per_circuit_parameter_values():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    template = Circuit(ry(Parameter("t"), 0), n_qubits=1)
    results = StatevectorEstimator().run([template, template], ham, parameter_values=[[0.0], [np.pi]])
    assert [r.value for r in results] == pytest.approx([1.0, -1.0])


def test_sampler_binds_parameter_values():
    template = Circuit(ry(Parameter("t"), 0), n_qubits=1)
    result = StatevectorSampler().run(template, parameter_values=[np.pi])
    # ry(pi)|0> = |1> → P(|1>) = 1
    assert result.probs["|1>"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# estimate() 委托 run()：数值一致性（Phase 1 item 3，_EnergyResult/PauliEstimateResult 已弃用包装）
# ---------------------------------------------------------------------------


def test_statevector_estimator_estimate_matches_run_value():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    circuit = Circuit(pauli_x(0), n_qubits=1)
    est = StatevectorEstimator(NumpyBackend())

    assert est.estimate(circuit, ham).energy == pytest.approx(est.run(circuit, ham).value)


def test_noisy_estimator_estimate_matches_run_value():
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    circuit = Circuit(ry(0.3, 0), n_qubits=1)
    est = NoisyEstimator(noise_model=_bitflip(0.25))

    assert est.estimate(circuit, ham).energy == pytest.approx(est.run(circuit, ham).value)


def test_shot_estimator_estimate_matches_run_value_and_returns_pauli_estimate_result():
    from aicir import PauliEstimateResult

    # |1> 是 Z 的本征态，方差为 0：即便 estimate()/run() 各自独立采样，数值仍精确一致
    ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    circuit = Circuit(pauli_x(0), n_qubits=1)
    est = ShotEstimator(NumpyBackend(), shots=256)

    estimated = est.estimate(circuit, ham)
    run_result = est.run(circuit, ham)
    assert isinstance(estimated, PauliEstimateResult)
    assert estimated.energy == pytest.approx(run_result.value)
    assert estimated.shots == run_result.shots
    assert estimated.variance == pytest.approx(run_result.variance)

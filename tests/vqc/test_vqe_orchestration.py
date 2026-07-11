from types import SimpleNamespace

import numpy as np

from aicir import BitFlipChannel, Circuit, Hamiltonian, NoiseModel, NumpyBackend, Parameter, PauliEstimator, ry
from aicir.optimizer import GD
from aicir.vqc import BasicVQE, run_vqe


def _z_hamiltonian_object():
    return Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])


def _single_ry_template():
    return Circuit(ry(Parameter("theta"), 0), n_qubits=1)


def test_vqe_accepts_circuit_parameter_hamiltonian_backend_and_shots():
    solver = BasicVQE(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        shots=64,
    )

    assert solver.n_params == 1
    assert solver.energy(np.array([0.0])) == np.float32(1.0)
    assert solver._last_measurement.counts is not None

    grad = solver.parameter_shift_gradient(np.array([0.5]))
    assert np.allclose(grad, [-np.sin(0.5)], atol=1e-6)


def test_vqe_optimizer_orchestrates_circuit_and_returns_measurement_result():
    solver = BasicVQE(_z_hamiltonian_object(), ansatz=_single_ry_template(), backend=NumpyBackend())
    optimizer = GD(max_iters=80, learning_rate=0.15, gradient_method="psr")

    result = solver.run(init_params=np.array([0.1]), optimizer=optimizer)

    assert result.energy < -0.99
    assert result.optimizer_result is not None
    assert result.circuit is not None
    assert result.circuit.parameters == ()
    assert result.measurement_result is not None
    assert result.measurement_result.final_state is not None
    assert result.metadata["mode"] == "circuit"


def test_vqe_accepts_optimizer_result_with_best_fields_only():
    class BestOnlyOptimizer:
        def minimize(self, fn, init_params, *, callback=None):
            params = np.array([np.pi])
            value = fn(params)
            return SimpleNamespace(best_x=params, best_fun=value, history=[{"fun": value}])

    solver = BasicVQE(_z_hamiltonian_object(), ansatz=_single_ry_template(), backend=NumpyBackend())

    result = solver.run(init_params=np.array([0.1]), optimizer=BestOnlyOptimizer())

    assert result.energy == np.float32(-1.0)
    assert result.parameters.shape == (1,)
    assert result.energy_history == [-1.0]


def test_vqe_accepts_callable_ansatz_with_explicit_n_params():
    def build(params):
        return Circuit(ry(params[0], 0), n_qubits=1)

    solver = BasicVQE(_z_hamiltonian_object(), ansatz=build, n_params=1, backend=NumpyBackend())

    assert solver.energy(np.array([np.pi])) == np.float32(-1.0)
    bound = solver.bind_ansatz(np.array([0.25]))
    assert bound.parameters == ()
    assert bound.gates == [ry(0.25, 0)]


def test_vqe_density_matrix_noise_path_uses_measure_noise_model():
    noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=1.0), after_gates=["ry"])
    solver = BasicVQE(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        noise_model=noise,
    )

    # Without noise, theta=0 leaves |0> with <Z>=+1. A full bit flip after RY
    # moves it to |1>, so the noisy density-matrix path gives <Z>=-1.
    assert solver.energy(np.array([0.0])) == np.float32(-1.0)
    assert solver._last_measurement.metadata["state_mode"] == "density_matrix"
    assert solver._last_measurement.metadata["noise_model"] == "NoiseModel"


def test_vqe_accepts_pauli_estimator_for_circuit_energy():
    solver = BasicVQE(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        energy_estimator=PauliEstimator(NumpyBackend(), shots=32),
    )

    assert solver.energy(np.array([0.0])) == np.float32(1.0)
    assert solver._last_estimator_result is not None
    assert solver._last_estimator_result.shots == 32
    assert solver._last_estimator_result.metadata["estimator"] == "PauliEstimator"
    assert solver._last_measurement is None

    result = solver.run(init_params=np.array([0.0]), max_iters=1, lr=0.1)
    assert result.energy == np.float32(1.0)
    assert result.estimator_result is not None
    assert result.measurement_result is not None
    assert result.metadata["energy_estimator"] == "PauliEstimator"


def test_run_vqe_convenience_accepts_generic_ansatz_and_optimizer():
    result = run_vqe(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        optimizer=GD(max_iters=80, learning_rate=0.15, gradient_method="psr"),
        init_params=np.array([0.1]),
    )

    assert result.energy < -0.99
    assert result.optimizer_result is not None


def test_run_vqe_convenience_forwards_energy_estimator():
    result = run_vqe(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        energy_estimator=PauliEstimator(NumpyBackend(), shots=16),
        init_params=np.array([0.0]),
        max_iters=1,
    )

    assert result.energy == np.float32(1.0)
    assert result.estimator_result is not None
    assert result.estimator_result.shots == 16
    assert result.metadata["energy_estimator"] == "PauliEstimator"


def test_run_vqe_convenience_forwards_measurement_configuration():
    initial_density_matrix = np.diag([0.0, 1.0]).astype(np.complex64)

    result = run_vqe(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        init_params=np.array([0.0]),
        max_iters=1,
        use_density_matrix=True,
        initial_density_matrix=initial_density_matrix,
        observable_name="h",
        shots=32,
    )

    assert result.energy == np.float32(-1.0)
    assert result.measurement_result.expectation_values["h"] == np.float32(-1.0)
    assert result.measurement_result.metadata["state_mode"] == "density_matrix"
    assert result.measurement_result.shots == 32


def test_legacy_dense_vqe_path_still_runs():
    result = BasicVQE(np.diag([1.0, -1.0]), depth=1, seed=1).run(max_iters=20, lr=0.2)

    assert result.parameters.shape == (1, 1)
    assert len(result.energy_history) == 20
    assert result.metadata["mode"] == "legacy_dense"


def test_non_exact_energy_estimator_requires_circuit_ansatz():
    try:
        BasicVQE(np.diag([1.0, -1.0]), energy_estimator=PauliEstimator(NumpyBackend(), shots=16))
    except ValueError as exc:
        assert "requires a Circuit" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_vqe_default_exact_energy_uses_statevector_estimator():
    # §4 收尾：默认精确（|0> 起点、无 shots/噪声）能量经 StatevectorEstimator primitive。
    from aicir.primitives import StatevectorEstimator

    solver = BasicVQE(_z_hamiltonian_object(), ansatz=_single_ry_template(), backend=NumpyBackend())
    assert solver.energy(np.array([0.0])) == np.float32(1.0)
    assert isinstance(solver._default_estimator(), StatevectorEstimator)
    assert solver._last_measurement is None
    assert solver._last_estimator_result is not None
    # 数值与解析一致：<Z> = cos(theta)
    assert np.isclose(solver.energy(np.array([0.7])), np.cos(0.7), atol=1e-7)


class _RunOnlyEstimator:
    """仅暴露 run()（无 estimate()），验证 BasicVQE 的 energy_estimator 优先走 run()
    的 duck-typed 注入路径（phase-1 item 3）。"""

    def __init__(self, backend=None):
        from aicir.primitives import StatevectorEstimator

        self._inner = StatevectorEstimator(backend)

    def run(self, circuits, observables, *, shots=None, parameter_values=None):
        return self._inner.run(circuits, observables, shots=shots, parameter_values=parameter_values)


def test_vqe_injected_estimator_with_only_run_method_works():
    estimator = _RunOnlyEstimator(NumpyBackend())
    assert not hasattr(estimator, "estimate")

    solver = BasicVQE(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        energy_estimator=estimator,
    )

    assert solver.energy(np.array([0.0])) == np.float32(1.0)
    assert np.isclose(solver.energy(np.array([0.7])), np.cos(0.7), atol=1e-7)
    assert solver._last_estimator_result is not None


def test_vqe_injected_estimator_with_only_estimate_method_still_works():
    # PauliEstimator（aicir.measure.estimator）只暴露 estimate()，无 run()：
    # 验证 BasicVQE 保留旧 estimate() 退回路径（duck-typed 外部注入向后兼容）。
    estimator = PauliEstimator(NumpyBackend(), shots=32)
    assert not hasattr(estimator, "run")

    solver = BasicVQE(
        _z_hamiltonian_object(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        energy_estimator=estimator,
    )

    assert solver.energy(np.array([0.0])) == np.float32(1.0)

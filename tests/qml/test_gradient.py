import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, Parameter, State, ry
from aicir.qml import multipsr, psr, spsr


def test_psr_matches_analytic_gradient_for_vector_params():
    params = np.array([0.3, -0.4])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    grad = psr(objective, params)

    assert np.allclose(grad, [-np.sin(params[0]), np.cos(params[1])])


def test_psr_preserves_parameter_shape():
    params = np.array([[0.1, 0.2], [0.3, 0.4]])

    def objective(theta):
        return np.sum(np.cos(theta))

    grad = psr(objective, params)

    assert grad.shape == params.shape
    assert np.allclose(grad, -np.sin(params))


def test_psr_supports_scalar_param():
    theta = np.array(0.7)

    def objective(value):
        return np.cos(value)

    grad = psr(objective, theta)

    assert grad.shape == ()
    assert np.allclose(grad, -np.sin(theta))


def test_psr_with_parameterized_circuit_template():
    theta = Parameter("theta")
    template = Circuit(ry(theta, 0), n_qubits=1)
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0])

    def objective(values):
        circuit = template.bind_parameters({"theta": values[0]})
        state = State.zero_state(1, backend).evolve(circuit.unitary()).to_numpy().reshape(-1)
        return np.real(np.vdot(state, z @ state))

    value = np.array([0.5])
    grad = psr(objective, value)

    assert np.allclose(grad, [-np.sin(value[0])])


def test_psr_rejects_vector_valued_objective():
    def objective(theta):
        return np.array([theta[0], theta[0]])

    with pytest.raises(ValueError, match="scalar"):
        psr(objective, np.array([0.1]))


def test_spsr_matches_psr_when_all_parameters_are_sampled():
    params = np.array([0.1, 0.2, 0.3])

    def objective(theta):
        return np.sum(np.cos(theta))

    grad = spsr(objective, params, n_samples=params.size, rng=123)

    assert np.allclose(grad, psr(objective, params))


def test_spsr_returns_unbiased_scaled_coordinate_estimate():
    params = np.array([0.1, 0.2, 0.3])

    def objective(theta):
        return np.sum(np.cos(theta))

    sampled = np.random.default_rng(7).choice(params.size, size=1, replace=False)
    grad = spsr(objective, params, n_samples=1, rng=7)
    expected = np.zeros_like(params)
    expected[sampled[0]] = params.size * (-np.sin(params[sampled[0]]))

    assert np.allclose(grad, expected)


def test_spsr_rejects_too_many_samples_without_replacement():
    with pytest.raises(ValueError, match="n_samples"):
        spsr(lambda theta: theta[0], np.array([0.1]), n_samples=2)


def test_multipsr_single_index_matches_psr_coordinate():
    params = np.array([0.4, -0.2])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    assert np.allclose(multipsr(objective, params, parameter_indices=1), psr(objective, params)[1])


def test_multipsr_computes_mixed_partial_derivative():
    params = np.array([0.4, -0.2])

    def objective(theta):
        return np.cos(theta[0]) * np.sin(theta[1])

    mixed = multipsr(objective, params, parameter_indices=[0, 1])

    assert np.allclose(mixed, -np.sin(params[0]) * np.cos(params[1]))


def test_multipsr_supports_tuple_indices_for_shaped_params():
    params = np.array([[0.4, 0.1], [-0.2, 0.3]])

    def objective(theta):
        return np.cos(theta[0, 0]) * np.sin(theta[1, 0])

    mixed = multipsr(objective, params, parameter_indices=[(0, 0), (1, 0)])

    assert np.allclose(mixed, -np.sin(params[0, 0]) * np.cos(params[1, 0]))


def test_multipsr_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicates"):
        multipsr(lambda theta: np.sum(theta), np.array([0.1, 0.2]), parameter_indices=[0, 0])

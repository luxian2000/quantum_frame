import numpy as np
import pytest

from nexq import Circuit, NumpyBackend, Parameter, State, ry
from nexq.qml import psr


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

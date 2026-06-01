import math

import numpy as np
import pytest

from nexq import Circuit, Parameter, crz, rx, ry, u3


def test_parameter_validates_name():
    assert str(Parameter(" theta ")) == "theta"
    with pytest.raises(ValueError):
        Parameter(" ")
    with pytest.raises(TypeError):
        Parameter(1)


def test_circuit_collects_parameters_in_first_use_order():
    theta = Parameter("theta")
    phi = Parameter("phi")

    circuit = Circuit(
        rx(theta, 0),
        crz(phi, target_qubit=1, control_qubits=[0]),
        ry(theta, 1),
        n_qubits=2,
    )

    assert circuit.parameters == (theta, phi)


def test_bind_parameters_returns_bound_copy_without_mutating_template():
    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lambda")
    template = Circuit(
        rx(theta, 0),
        crz(phi, target_qubit=1, control_qubits=[0]),
        u3(theta, 0.25, lam, target_qubit=1),
        n_qubits=2,
    )

    bound = template.bind_parameters({"theta": 0.1, phi: 0.2, "lambda": 0.3})

    assert template.gates[0]["parameter"] == theta
    assert bound.gates[0]["parameter"] == 0.1
    assert bound.gates[1]["parameter"] == 0.2
    assert bound.gates[2]["parameter"] == [0.1, 0.25, 0.3]
    assert bound.parameters == ()

    expected = Circuit(
        rx(0.1, 0),
        crz(0.2, target_qubit=1, control_qubits=[0]),
        u3(0.1, 0.25, 0.3, target_qubit=1),
        n_qubits=2,
    )
    assert np.allclose(bound.unitary(), expected.unitary())


def test_bind_parameters_accepts_sequence_in_parameter_order():
    theta = Parameter("theta")
    phi = Parameter("phi")
    template = Circuit(rx(theta, 0), ry(phi, 1), n_qubits=2)

    bound = template.bind_parameters([math.pi / 2, math.pi])

    assert bound.gates == [rx(math.pi / 2, 0), ry(math.pi, 1)]


def test_bind_parameters_can_update_in_place_or_partially():
    theta = Parameter("theta")
    phi = Parameter("phi")
    template = Circuit(rx(theta, 0), ry(phi, 1), n_qubits=2)

    partial = template.bind_parameters({"theta": 0.4}, allow_partial=True)
    assert partial.gates[0]["parameter"] == 0.4
    assert partial.gates[1]["parameter"] == phi
    assert partial.parameters == (phi,)

    returned = template.bind_parameters({"theta": 0.4, "phi": 0.5}, inplace=True)
    assert returned is template
    assert template.parameters == ()
    assert template.gates == [rx(0.4, 0), ry(0.5, 1)]


def test_unbound_parameter_circuit_cannot_be_evaluated():
    circuit = Circuit(rx(Parameter("theta"), 0), n_qubits=1)

    with pytest.raises(ValueError, match="unbound parameter"):
        circuit.unitary()


def test_bind_parameters_rejects_missing_unknown_and_wrong_length():
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = Circuit(rx(theta, 0), ry(phi, 1), n_qubits=2)

    with pytest.raises(ValueError, match="Missing parameter"):
        circuit.bind_parameters({"theta": 0.1})
    with pytest.raises(ValueError, match="Unknown parameter"):
        circuit.bind_parameters({"theta": 0.1, "phi": 0.2, "extra": 0.3})
    with pytest.raises(ValueError, match="Expected 2 parameter"):
        circuit.bind_parameters([0.1])

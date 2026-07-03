import numpy as np
import pytest

from aicir import NumpyBackend
from aicir.ansatze import entangling_edges, hea, hea_parameter_count, hardware_efficient_ansatz


def test_default_hea_returns_symbolic_circuit_with_expected_layout():
    circuit = hardware_efficient_ansatz(3, layers=2)

    assert circuit.n_qubits == 3
    assert len(circuit.parameters) == hea_parameter_count(3, layers=2)
    assert [parameter.name for parameter in circuit.parameters] == [f"theta_{index}" for index in range(18)]

    cx_gates = [gate for gate in circuit.gates if gate["type"] == "cx"]
    assert len(cx_gates) == 4
    assert cx_gates[0]["control_qubits"] == [0]
    assert cx_gates[0]["target_qubit"] == 1


def test_hea_binds_parameters_and_builds_unitary():
    count = hea_parameter_count(2, layers=1, rotation_gates=("ry", "rz"), final_rotation_gates=("ry",))
    circuit = hea(2, layers=1, final_rotation_gates=("ry",))

    bound = circuit.bind_parameters(np.linspace(0.1, 0.7, count))

    assert bound.parameters == ()
    assert bound.unitary(backend=NumpyBackend()).shape == (4, 4)


def test_hea_supports_ring_all_to_all_and_custom_edges():
    assert entangling_edges(4, "ring") == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert entangling_edges(3, "all_to_all") == [(0, 1), (0, 2), (1, 2)]

    circuit = hardware_efficient_ansatz(
        3,
        layers=1,
        rotation_gates="ry",
        entangler="rzz",
        topology=[(0, 2)],
        final_rotation_layer=False,
    )

    assert len(circuit.parameters) == 4
    assert circuit.gates[-1]["type"] == "rzz"
    assert circuit.gates[-1]["qubit_1"] == 0
    assert circuit.gates[-1]["qubit_2"] == 2

    rxx_circuit = hardware_efficient_ansatz(
        3,
        layers=1,
        rotation_gates="ry",
        entangler="rxx",
        topology=[(0, 2)],
        final_rotation_layer=False,
    )

    assert len(rxx_circuit.parameters) == 4
    assert rxx_circuit.gates[-1]["type"] == "rxx"
    assert rxx_circuit.gates[-1]["qubit_1"] == 0
    assert rxx_circuit.gates[-1]["qubit_2"] == 2


def test_hea_accepts_numeric_parameters_directly():
    count = hea_parameter_count(2, layers=1, rotation_gates="u3", entangler="cz", final_rotation_layer=False)
    circuit = hardware_efficient_ansatz(
        2,
        layers=1,
        rotation_gates="u3",
        entangler="cz",
        final_rotation_layer=False,
        parameters=np.arange(count) / 10.0,
    )

    assert circuit.parameters == ()
    assert circuit.gates[0]["type"] == "u3"
    assert circuit.gates[0]["parameter"] == [0.0, 0.1, 0.2]
    assert circuit.gates[-1]["type"] == "cz"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_qubits": 0}, "n_qubits"),
        ({"n_qubits": 2, "layers": -1}, "layers"),
        ({"n_qubits": 2, "rotation_gates": "h"}, "Unsupported rotation_gates"),
        ({"n_qubits": 2, "entangler": "iswap"}, "Unsupported entangler"),
        ({"n_qubits": 2, "topology": [(0, 2)]}, "out of range"),
        ({"n_qubits": 2, "parameters": [0.1]}, "Expected at least"),
    ],
)
def test_hea_validates_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        hardware_efficient_ansatz(**kwargs)

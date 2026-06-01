import math

import pytest

from nexq.core.circuit import Circuit
from nexq.universal import qft, qft_circuit


def test_qft_builds_default_gate_sequence():
    gates = qft(3)

    assert gates == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "crz", "target_qubit": 0, "control_qubits": [1], "parameter": math.pi / 2, "control_states": [1]},
        {"type": "crz", "target_qubit": 0, "control_qubits": [2], "parameter": math.pi / 4, "control_states": [1]},
        {"type": "hadamard", "target_qubit": 1},
        {"type": "crz", "target_qubit": 1, "control_qubits": [2], "parameter": math.pi / 2, "control_states": [1]},
        {"type": "hadamard", "target_qubit": 2},
        {"type": "swap", "qubit_1": 0, "qubit_2": 2},
    ]


def test_qft_supports_nonzero_start_qubit():
    gates = qft(2, start_qubit=3)

    assert gates == [
        {"type": "hadamard", "target_qubit": 3},
        {"type": "crz", "target_qubit": 3, "control_qubits": [4], "parameter": math.pi / 2, "control_states": [1]},
        {"type": "hadamard", "target_qubit": 4},
        {"type": "swap", "qubit_1": 3, "qubit_2": 4},
    ]


def test_qft_circuit_wraps_gate_sequence():
    circuit = qft_circuit(2, start_qubit=1)

    assert isinstance(circuit, Circuit)
    assert circuit.gates == qft(2, start_qubit=1)
    assert circuit.n_qubits == 3


@pytest.mark.parametrize(
    ("n_qubits", "start_qubit"),
    [
        (0, 0),
        (-1, 0),
        (2, -1),
    ],
)
def test_qft_rejects_invalid_qubit_ranges(n_qubits, start_qubit):
    with pytest.raises(ValueError):
        qft(n_qubits, start_qubit=start_qubit)


def test_qft_rejects_non_integer_qubit_ranges():
    with pytest.raises(TypeError):
        qft(2.5)

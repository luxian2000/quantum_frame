import numpy as np

from aicir import Operation as TopLevelOperation
from aicir.core.circuit import Circuit, Parameter
from aicir.ir import Operation, normalize_gate


def test_operation_is_available_from_top_level_package():
    assert TopLevelOperation is Operation


def test_operation_round_trips_existing_gate_dict_with_metadata():
    gate = {
        "type": "rzz",
        "qubit_1": 0,
        "qubit_2": 2,
        "parameter": 0.5,
        "label": "zz-block",
    }

    op = Operation.from_dict(gate)

    assert op.name == "rzz"
    assert op.qubits == (0, 2)
    assert op.params == (0.5,)
    assert op.metadata == {"label": "zz-block"}
    assert op.to_dict() == gate


def test_operation_can_build_circuit_without_changing_gate_dict_surface():
    theta = Parameter("theta")
    circuit = Circuit(
        Operation("rx", qubits=(0,), params=(theta,)),
        Operation("cx", qubits=(1,), controls=(0,)),
        n_qubits=2,
    )

    assert circuit.gates == [
        {"type": "rx", "target_qubit": 0, "parameter": theta},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
    ]
    assert circuit.parameters == (theta,)

    bound = circuit.bind_parameters({"theta": np.pi})

    assert bound.gates[0]["parameter"] == np.pi
    assert bound.unitary().shape == (4, 4)


def test_circuit_append_and_extend_accept_operation_and_dict_inputs():
    circuit = Circuit(Operation("hadamard", qubits=(0,)), n_qubits=2)

    circuit.append(Operation("pauli_x", qubits=(1,)))
    circuit.extend(
        {"type": "pauli_x", "target_qubit": 1},
        Operation("rz", qubits=(0,), params=(0.25,)),
    )

    assert circuit.gates == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 1},
        {"type": "pauli_x", "target_qubit": 1},
        {"type": "rz", "target_qubit": 0, "parameter": 0.25},
    ]


def test_normalize_gate_accepts_operation_and_copies_dict():
    gate = {"type": "pauli_z", "target_qubit": 0}

    normalized_dict = normalize_gate(gate)
    normalized_operation = normalize_gate(Operation("pauli_z", qubits=(0,)))

    assert normalized_dict == gate
    assert normalized_dict is not gate
    assert normalized_operation == gate

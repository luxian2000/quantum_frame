import numpy as np

from aicir import (
    CircuitIR,
    Measurement,
    Observable,
    Operation as TopLevelOperation,
)
from aicir.channel.backends import NumpyBackend
from aicir.channel.operators import Hamiltonian
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


def test_measurement_round_trips_existing_measure_gate_and_builds_circuit():
    measurement = Measurement.from_dict(
        {
            "type": "measure",
            "qubits": [0, 2],
            "return_type": "sample",
            "classical_bits": [0, 1],
            "label": "readout",
        }
    )

    assert measurement.measurement_type == "measure"
    assert measurement.qubits == (0, 2)
    assert measurement.return_type == "sample"
    assert measurement.classical_bits == (0, 1)
    assert measurement.metadata == {"label": "readout"}
    assert measurement.to_dict() == {
        "type": "measure",
        "qubits": [0, 2],
        "return_type": "sample",
        "classical_bits": [0, 1],
        "label": "readout",
    }

    circuit = Circuit(Operation("hadamard", qubits=(0,)), measurement, n_qubits=3)

    assert circuit.gates == [
        {"type": "hadamard", "target_qubit": 0},
        {
            "type": "measure",
            "qubits": [0, 2],
            "return_type": "sample",
            "classical_bits": [0, 1],
            "label": "readout",
        },
    ]


def test_observable_wraps_pauli_hamiltonian_and_dense_matrix():
    backend = NumpyBackend()

    pauli = Observable.pauli("ZZ", coefficient=-0.5, n_qubits=2, name="zz")

    assert pauli.kind == "pauli"
    assert pauli.name == "zz"
    assert pauli.n_qubits == 2
    np.testing.assert_allclose(
        backend.to_numpy(pauli.to_matrix(backend)),
        backend.to_numpy(Hamiltonian([("ZZ", -0.5)]).to_matrix(backend)),
    )

    hamiltonian = Hamiltonian([("ZI", 1.0), ("IZ", 0.25)])
    h_obs = Observable.from_object(hamiltonian, name="cost")

    assert h_obs.kind == "hamiltonian"
    assert h_obs.name == "cost"
    assert h_obs.n_qubits == 2
    assert h_obs.to_operator() is hamiltonian

    z_matrix = np.diag([1.0, -1.0]).astype(np.complex64)
    matrix_obs = Observable.matrix(z_matrix, name="z")

    assert matrix_obs.kind == "matrix"
    assert matrix_obs.n_qubits == 1
    np.testing.assert_allclose(backend.to_numpy(matrix_obs.to_matrix(backend)), z_matrix)


def test_circuit_ir_round_trips_existing_circuit_surface():
    circuit = Circuit(
        Operation("rx", qubits=(0,), params=(0.25,)),
        Operation("cx", qubits=(1,), controls=(0,)),
        Measurement((0,), return_type="counts"),
        n_qubits=2,
    )

    ir = CircuitIR.from_circuit(
        circuit,
        classical_bits=(0,),
        metadata={"name": "example"},
    )

    assert ir.n_qubits == 2
    assert ir.classical_bits == (0,)
    assert ir.metadata == {"name": "example"}
    assert isinstance(ir.operations[0], Operation)
    assert isinstance(ir.operations[2], Measurement)
    assert ir.to_gate_dicts() == circuit.gates

    restored = ir.to_circuit()

    assert restored.n_qubits == circuit.n_qubits
    assert restored.gates == circuit.gates

    from_payload = CircuitIR.from_dict(ir.to_dict())

    assert from_payload == ir

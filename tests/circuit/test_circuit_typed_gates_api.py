from aicir import (
    Circuit,
    ClassicalRegister,
    Measurement,
    Operation,
    Parameter,
    cnot,
    hadamard,
    if_,
    measure,
    pauli_x,
    rx,
)
from aicir.core.io.json_io import circuit_from_json, circuit_to_json
from aicir.ir import ControlFlow


def test_circuit_gates_returns_typed_instructions():
    reg = ClassicalRegister(1, "c")
    branch = Circuit(pauli_x(1), n_qubits=2)
    circuit = Circuit(
        hadamard(0),
        cnot(1, [0]),
        measure(0, creg=reg),
        if_(reg[0] == 1, branch),
        n_qubits=2,
    )

    assert [type(gate) for gate in circuit.gates] == [
        Operation,
        Operation,
        Measurement,
        ControlFlow,
    ]
    assert [gate["type"] for gate in circuit.gates] == ["hadamard", "cx", "measure", "if"]


def test_legacy_gates_returns_detached_dicts():
    circuit = Circuit(hadamard(0), rx(0.25, 0), n_qubits=1)

    legacy = circuit.legacy_gates
    assert legacy == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "rx", "target_qubit": 0, "parameter": 0.25},
    ]

    legacy[0]["type"] = "mutated"
    assert circuit.legacy_gates[0]["type"] == "hadamard"
    assert isinstance(circuit.gates[0], Operation)


def test_append_extend_and_bind_keep_typed_gates():
    circuit = Circuit(hadamard(0), n_qubits=2)
    circuit.append(rx(0.1, 0)).extend(cnot(1, [0]))

    assert [gate.name for gate in circuit.gates] == ["hadamard", "rx", "cx"]

    bound = Circuit(rx(Parameter("theta"), 0), n_qubits=1).bind_parameters({"theta": 0.5})
    assert isinstance(bound.gates[0], Operation)
    assert bound.legacy_gates[0]["parameter"] == 0.5


def test_json_roundtrip_preserves_legacy_wire_format_with_typed_gates():
    circuit = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    rebuilt = circuit_from_json(circuit_to_json(circuit))

    assert [gate.name for gate in rebuilt.gates] == ["hadamard", "cx"]
    assert rebuilt.legacy_gates == circuit.legacy_gates

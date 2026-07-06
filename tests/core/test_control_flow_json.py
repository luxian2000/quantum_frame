from aicir import Circuit, Measure, NumpyBackend, hadamard, if_, pauli_x, while_
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister
from aicir.core.io.json_io import circuit_from_json, circuit_to_json


def _circ():
    reg = ClassicalRegister(1, "c")
    return Circuit(
        hadamard(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2),
            else_body=Circuit(hadamard(1), n_qubits=2)),
        while_(reg[0] == 1, Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=2), max_iterations=4),
        n_qubits=2,
    )


def test_json_roundtrip_structure():
    c = _circ()
    back = circuit_from_json(circuit_to_json(c))
    assert back.n_qubits == 2
    assert back.gates[2]["type"] == "if"
    assert back.gates[2]["else_body"][0]["type"] == "hadamard"
    assert back.gates[3]["type"] == "while" and back.gates[3]["max_iterations"] == 4
    assert back.gates[3]["condition"]["target"]["register"] == "c"


def test_json_roundtrip_execution_equivalence():
    c = _circ()
    back = circuit_from_json(circuit_to_json(c))
    reg = ClassicalRegister(1, "c")
    r1 = Measure(NumpyBackend()).run(c, shots=200, seed=5).classical_counts(reg)
    r2 = Measure(NumpyBackend()).run(back, shots=200, seed=5).classical_counts(reg)
    assert r1 == r2

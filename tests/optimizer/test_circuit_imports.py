from aicir.core.circuit import Circuit
from aicir.optimizer import optimize_basic
from aicir.optimizer.circuit import optimize_basic as optimize_basic_circuit


def test_circuit_optimizer_package_and_module_imports_match():
    assert optimize_basic is optimize_basic_circuit

    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 0},
        n_qubits=1,
    )

    optimized = optimize_basic_circuit(circuit)

    assert optimized.gates == []

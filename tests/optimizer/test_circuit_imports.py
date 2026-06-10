from aicir.core.circuit import Circuit
from aicir.optimizer import optimize_basic, optimize_circuit
from aicir.optimizer.circuit import optimize_basic as optimize_basic_circuit
from aicir.optimizer.circuit import optimize_circuit as optimize_circuit_module


def test_circuit_optimizer_package_and_module_imports_match():
    assert optimize_basic is optimize_basic_circuit
    assert optimize_circuit is optimize_circuit_module

    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 0},
        n_qubits=1,
    )

    optimized = optimize_basic_circuit(circuit)
    optimized_direct = optimize_circuit(circuit)

    assert optimized.gates == []
    assert optimized_direct.gates == []

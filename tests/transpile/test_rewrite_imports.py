from aicir.core.circuit import Circuit
from aicir.transpile import optimize_basic, optimize_circuit
from aicir.transpile.rewrite import optimize_basic as optimize_basic_module
from aicir.transpile.rewrite import optimize_circuit as optimize_circuit_module


def test_transpile_package_and_module_imports_match():
    assert optimize_basic is optimize_basic_module
    assert optimize_circuit is optimize_circuit_module

    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 0},
        n_qubits=1,
    )

    optimized = optimize_basic(circuit)
    optimized_direct = optimize_circuit(circuit)

    assert optimized.gates == []
    assert optimized_direct.gates == []

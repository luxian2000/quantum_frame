import numpy as np

from aicir.core.circuit import Circuit
from aicir.optimizer import optimize_circuit
from aicir.transpile import (
    CancelInversePass,
    CommuteSingleQubitPass,
    MergeRotationsPass,
    PassManager,
    default_optimization_pipeline,
)


def test_pass_manager_runs_named_passes_to_fixed_point():
    circuit = Circuit(
        {"type": "hadamard", "target_qubit": 0},
        {"type": "hadamard", "target_qubit": 0},
        {"type": "rx", "target_qubit": 1, "parameter": 0.1},
        {"type": "rx", "target_qubit": 1, "parameter": 0.2},
        n_qubits=2,
    )

    pm = PassManager(["cancel_inverse", "merge_rotations"], fixed_point=True)
    out = pm.run(circuit)

    assert out.gates == [{"type": "rx", "target_qubit": 1, "parameter": 0.30000000000000004}]
    assert out.n_qubits == circuit.n_qubits


def test_default_optimization_pipeline_matches_legacy_optimize_circuit():
    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 1},
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 1},
        {"type": "rz", "target_qubit": 0, "parameter": 0.1},
        {"type": "cx", "control_qubits": [0], "control_states": [1], "target_qubit": 1},
        {"type": "rz", "target_qubit": 0, "parameter": 0.2},
        n_qubits=2,
    )

    via_pipeline = default_optimization_pipeline().run(circuit)
    via_legacy = optimize_circuit(circuit)

    assert via_pipeline.gates == via_legacy.gates
    assert len(via_pipeline.gates) == 3
    assert via_pipeline.gates[0] == {"type": "pauli_x", "target_qubit": 0}
    assert via_pipeline.gates[1]["type"] == "rz"
    assert np.isclose(via_pipeline.gates[1]["parameter"], 0.3)
    assert via_pipeline.gates[2] == {
        "type": "cx",
        "control_qubits": [0],
        "control_states": [1],
        "target_qubit": 1,
    }


def test_transpile_passes_are_composable_objects():
    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 1},
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 1},
        n_qubits=2,
    )

    pm = PassManager(
        [
            CancelInversePass(),
            MergeRotationsPass(),
            CommuteSingleQubitPass(max_reorder_hops=8),
        ],
        fixed_point=True,
    )
    out = pm.run(circuit)

    assert out.gates == [{"type": "pauli_x", "target_qubit": 0}]

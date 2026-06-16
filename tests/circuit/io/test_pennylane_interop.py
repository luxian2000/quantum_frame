import math

import pytest

qml = pytest.importorskip("pennylane")

from aicir import Circuit, cnot, crz, hadamard, rx, rxx, rzz, swap, toffoli, u3
from aicir.core.io.pennylane_io import circuit_from_pennylane, circuit_to_pennylane


def _pennylane_gate_rows(script):
    return [
        (
            op.name,
            op.wires.tolist(),
            [float(param) for param in op.parameters],
        )
        for op in script.operations
    ]


def test_circuit_to_pennylane_preserves_supported_gate_order_and_wires():
    cir = Circuit(
        {"type": "identity", "n_qubits": 3},
        hadamard(0),
        cnot(1, [0]),
        rx(math.pi / 3, 1),
        crz(math.pi / 4, 2, [1]),
        rzz(math.pi / 5, 0, 2),
        rxx(math.pi / 6, 1, 2),
        swap(0, 1),
        u3(math.pi / 7, math.pi / 8, math.pi / 9, 2),
        toffoli(2, [0, 1]),
        n_qubits=3,
    )

    script = circuit_to_pennylane(cir)

    assert script.wires.tolist() == [0, 1, 2]
    assert _pennylane_gate_rows(script) == [
        ("Identity", [0, 1, 2], []),
        ("Hadamard", [0], []),
        ("CNOT", [0, 1], []),
        ("RX", [1], [math.pi / 3]),
        ("CRZ", [1, 2], [math.pi / 4]),
        ("IsingZZ", [0, 2], [math.pi / 5]),
        ("IsingXX", [1, 2], [math.pi / 6]),
        ("SWAP", [0, 1], []),
        ("U3", [2], [math.pi / 7, math.pi / 8, math.pi / 9]),
        ("Toffoli", [0, 1, 2], []),
    ]


def test_circuit_from_pennylane_preserves_supported_gate_order_and_wires():
    script = qml.tape.QuantumScript(
        [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(math.pi / 3, wires=1),
            qml.CRZ(math.pi / 4, wires=[1, 2]),
            qml.IsingZZ(math.pi / 5, wires=[0, 2]),
            qml.IsingXX(math.pi / 6, wires=[1, 2]),
            qml.SWAP(wires=[0, 1]),
            qml.U3(math.pi / 7, math.pi / 8, math.pi / 9, wires=2),
            qml.Toffoli(wires=[0, 1, 2]),
        ],
        [],
    )

    cir = circuit_from_pennylane(script)

    assert cir.n_qubits == 3
    assert cir.gates == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        {"type": "rx", "target_qubit": 1, "parameter": math.pi / 3},
        {"type": "crz", "target_qubit": 2, "control_qubits": [1], "control_states": [1], "parameter": math.pi / 4},
        {"type": "rzz", "qubit_1": 0, "qubit_2": 2, "parameter": math.pi / 5},
        {"type": "rxx", "qubit_1": 1, "qubit_2": 2, "parameter": math.pi / 6},
        {"type": "swap", "qubit_1": 0, "qubit_2": 1},
        {"type": "u3", "target_qubit": 2, "parameter": [math.pi / 7, math.pi / 8, math.pi / 9]},
        {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1]},
    ]


def test_top_level_exports_pennylane_interop_helpers():
    import aicir
    from aicir.core.io import circuit_from_pennylane as core_from_pennylane
    from aicir.core.io import circuit_to_pennylane as core_to_pennylane
    from aicir.core.io import from_pennylane as core_from_pennylane_alias
    from aicir.core.io import to_pennylane as core_to_pennylane_alias

    assert aicir.circuit_to_pennylane is core_to_pennylane
    assert aicir.circuit_from_pennylane is core_from_pennylane
    assert aicir.to_pennylane is core_to_pennylane_alias
    assert aicir.from_pennylane is core_from_pennylane_alias
    assert core_to_pennylane_alias is core_to_pennylane
    assert core_from_pennylane_alias is core_from_pennylane

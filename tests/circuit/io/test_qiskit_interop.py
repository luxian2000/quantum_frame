import math

import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit

from aicir import Circuit, cnot, crz, hadamard, measure, rx, rxx, rzz, swap, toffoli, u3
from aicir.core.io.qiskit_io import circuit_from_qiskit, circuit_to_qiskit


def _qiskit_gate_rows(qc):
    return [
        (
            item.operation.name,
            [qc.find_bit(qubit).index for qubit in item.qubits],
            [float(param) for param in item.operation.params],
        )
        for item in qc.data
    ]


def test_circuit_to_qiskit_preserves_supported_gate_order_and_qubits():
    cir = Circuit(
        hadamard(0),
        cnot(1, [0]),
        rx(math.pi / 3, 1),
        crz(math.pi / 4, 2, [1]),
        rzz(math.pi / 5, 0, 2),
        rxx(math.pi / 6, 1, 2),
        swap(0, 1),
        u3(math.pi / 7, math.pi / 8, math.pi / 9, 2),
        toffoli(2, [0, 1]),
        measure(0, 2),
        n_qubits=3,
    )

    qc = circuit_to_qiskit(cir)

    assert qc.num_qubits == 3
    assert qc.num_clbits == 2
    assert _qiskit_gate_rows(qc) == [
        ("h", [0], []),
        ("cx", [0, 1], []),
        ("rx", [1], [math.pi / 3]),
        ("crz", [1, 2], [math.pi / 4]),
        ("rzz", [0, 2], [math.pi / 5]),
        ("rxx", [1, 2], [math.pi / 6]),
        ("swap", [0, 1], []),
        ("u", [2], [math.pi / 7, math.pi / 8, math.pi / 9]),
        ("ccx", [0, 1, 2], []),
        ("measure", [0], []),
        ("measure", [2], []),
    ]


def test_circuit_from_qiskit_preserves_supported_gate_order_and_qubits():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(math.pi / 3, 1)
    qc.crz(math.pi / 4, 1, 2)
    qc.rzz(math.pi / 5, 0, 2)
    qc.rxx(math.pi / 6, 1, 2)
    qc.swap(0, 1)
    qc.u(math.pi / 7, math.pi / 8, math.pi / 9, 2)
    qc.ccx(0, 1, 2)
    qc.measure([0, 2], [0, 1])

    cir = circuit_from_qiskit(qc)

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
        {"type": "measure", "qubits": [0]},
        {"type": "measure", "qubits": [2]},
    ]


def test_top_level_exports_qiskit_interop_helpers():
    import aicir
    from aicir.core.io import circuit_from_qiskit as core_from_qiskit
    from aicir.core.io import circuit_to_qiskit as core_to_qiskit
    from aicir.core.io import from_qiskit as core_from_qiskit_alias
    from aicir.core.io import to_qiskit as core_to_qiskit_alias

    assert aicir.circuit_to_qiskit is core_to_qiskit
    assert aicir.circuit_from_qiskit is core_from_qiskit
    assert aicir.to_qiskit is core_to_qiskit_alias
    assert aicir.from_qiskit is core_from_qiskit_alias
    assert core_to_qiskit_alias is core_to_qiskit
    assert core_from_qiskit_alias is core_from_qiskit

import math

import pytest

wuyue = pytest.importorskip("wuyue")
from wuyue.circuit.circuit import QuantumCircuit
from wuyue.element.gate import CX, H, IsingZZ, MEASURE, RX, SWAP, TOFFOLI, U3
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.register.quantumregister import QuantumRegister

from aicir import Circuit, cnot, hadamard, measure, rx, rzz, swap, toffoli, u3
from aicir.core.io.wuyue_io import circuit_from_wuyue, circuit_to_wuyue


def _wuyue_gate_rows(qc):
    qc.apply_circuit()
    return [
        (
            gate.name,
            list(gate.target),
            None if gate.ctrl is None else list(gate.ctrl),
            [] if getattr(gate, "paras", None) is None else [float(param) for param in gate.paras],
            None if not hasattr(gate, "cbit") else list(gate.cbit),
        )
        for gate in qc.gates
    ]


def test_circuit_to_wuyue_preserves_supported_gate_order_and_qubits():
    cir = Circuit(
        hadamard(0),
        cnot(1, [0]),
        rx(math.pi / 3, 1),
        rzz(math.pi / 5, 0, 2),
        swap(0, 1),
        u3(math.pi / 7, math.pi / 8, math.pi / 9, 2),
        toffoli(2, [0, 1]),
        measure(0, 2),
        n_qubits=3,
    )

    qc = circuit_to_wuyue(cir)

    assert qc.qubits == 3
    assert qc.cbits == 2
    assert _wuyue_gate_rows(qc) == [
        ("H", [0], None, [], None),
        ("CX", [1], [0], [], None),
        ("RX", [1], None, [math.pi / 3], None),
        ("IZZ", [0, 2], None, [math.pi / 5], None),
        ("SWAP", [0, 1], None, [], None),
        ("U3", [2], None, [math.pi / 7, math.pi / 8, math.pi / 9], None),
        ("TOFFOLI", [2], [0, 1], [], None),
        ("MEASURE", [0], None, [], [0]),
        ("MEASURE", [2], None, [], [1]),
    ]


def test_circuit_from_wuyue_preserves_supported_gate_order_and_qubits():
    qreg = QuantumRegister(3)
    creg = ClassicalRegister(2)
    qc = QuantumCircuit(qreg, creg)
    qc.add(H, qreg[0])
    qc.add(CX, qreg[1], control=qreg[0])
    qc.add(RX, qreg[1], paras=[math.pi / 3])
    qc.add(IsingZZ, qreg[[0, 2]], paras=[math.pi / 5])
    qc.add(SWAP, qreg[[0, 1]])
    qc.add(U3, qreg[2], paras=[math.pi / 7, math.pi / 8, math.pi / 9])
    qc.add(TOFFOLI, qreg[2], control=qreg[[0, 1]])
    qc.add(MEASURE, qreg[[0, 2]], cbit=creg[[0, 1]])

    cir = circuit_from_wuyue(qc)

    assert cir.n_qubits == 3
    assert cir.gates == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        {"type": "rx", "target_qubit": 1, "parameter": math.pi / 3},
        {"type": "rzz", "qubit_1": 0, "qubit_2": 2, "parameter": math.pi / 5},
        {"type": "swap", "qubit_1": 0, "qubit_2": 1},
        {"type": "u3", "target_qubit": 2, "parameter": [math.pi / 7, math.pi / 8, math.pi / 9]},
        {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1]},
        {"type": "measure", "qubits": [0]},
        {"type": "measure", "qubits": [2]},
    ]


def test_circuit_to_wuyue_rejects_non_native_gate():
    cir = Circuit(
        {"type": "rxx", "parameter": math.pi / 6, "qubit_1": 0, "qubit_2": 1},
        n_qubits=2,
    )

    with pytest.raises(ValueError, match="暂不支持门类型"):
        circuit_to_wuyue(cir)


def test_top_level_exports_wuyue_interop_helpers():
    import aicir
    from aicir.core.io import circuit_from_wuyue as core_from_wuyue
    from aicir.core.io import circuit_to_wuyue as core_to_wuyue
    from aicir.core.io import from_wuyue as core_from_wuyue_alias
    from aicir.core.io import to_wuyue as core_to_wuyue_alias

    assert aicir.circuit_to_wuyue is core_to_wuyue
    assert aicir.circuit_from_wuyue is core_from_wuyue
    assert aicir.to_wuyue is core_to_wuyue_alias
    assert aicir.from_wuyue is core_from_wuyue_alias
    assert core_to_wuyue_alias is core_to_wuyue
    assert core_from_wuyue_alias is core_from_wuyue

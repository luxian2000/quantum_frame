"""QASM 门名单一来源测试：导出名来自 GateSpec 注册表，别名写法导出不变。"""

from aicir.core.circuit import Circuit
from aicir.core.io.qasm import circuit_from_qasm, circuit_to_qasm
from aicir.gates import get_gate_spec


def test_alias_gate_types_export_to_same_qasm_as_canonical():
    """别名写法（X/cnot/ccnot）与规范名导出结果一致（回归保护）。"""
    aliased = Circuit(
        {"type": "X", "target_qubit": 0},
        {"type": "cnot", "target_qubit": 1, "control_qubits": [0]},
        {"type": "ccnot", "target_qubit": 2, "control_qubits": [0, 1]},
        n_qubits=3,
    )
    canonical = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0]},
        {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1]},
        n_qubits=3,
    )
    text = circuit_to_qasm(aliased, version="2.0")
    assert text == circuit_to_qasm(canonical, version="2.0")
    assert "x q[0];" in text
    assert "cx q[0],q[1];" in text
    assert "ccx q[0],q[1],q[2];" in text


def test_export_tables_agree_with_registry_qasm_names():
    """qasm.py 的导出名与注册表 qasm_name 字段一致（单一来源耦合保护）。"""
    from aicir.core.io import qasm as qasm_module

    for table_name in (
        "_SINGLE_NO_PARAM_EXPORT",
        "_PARAM_EXPORT",
        "_DOUBLE_EXPORT",
        "_THREE_EXPORT",
    ):
        table = getattr(qasm_module, table_name)
        for gate_name, qasm_name in table.items():
            spec = get_gate_spec(gate_name)
            assert spec is not None, f"{table_name}: {gate_name} 未注册"
            assert spec.qasm_name == qasm_name, f"{table_name}: {gate_name}"


def test_qasm_roundtrip_preserves_canonical_names():
    cir = Circuit(
        {"type": "hadamard", "target_qubit": 0},
        {"type": "rz", "target_qubit": 1, "parameter": 0.5},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0]},
        n_qubits=2,
    )
    back = circuit_from_qasm(circuit_to_qasm(cir, version="2.0"))
    assert [g["type"] for g in back.gates] == ["hadamard", "rz", "cx"]

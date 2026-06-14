"""
tests/circuit/io/test_measure_io.py

验证 measure 门在 JSON 和 QASM 导出中的行为：
1. JSON 往返保留 basis 与 id 字段
2. QASM 对不可表达的 measure（联合/非Z基/带id）抛出 NotImplementedError
3. 普通单比特 Z 测量（无 id）可正常导出为标准 QASM
"""

import pytest

from aicir import Circuit, hadamard
from aicir.core.circuit import measure
from aicir.core.io.json_io import circuit_from_json, circuit_to_json
from aicir.core.io.qasm import circuit_to_qasm


# ──────────────────────────────────────────────────────────────────────────────
# JSON 往返测试
# ──────────────────────────────────────────────────────────────────────────────


def test_json_round_trip_preserves_basis_and_id():
    """JSON 序列化/反序列化后 basis 与 id 字段保持不变。"""
    cir = Circuit(hadamard(0), measure([0, 1], basis="X", id="m0"), n_qubits=2)
    back = circuit_from_json(circuit_to_json(cir))
    g = back.gates[-1]
    assert g["basis"] == "X"
    assert g["id"] == "m0"


def test_json_round_trip_preserves_default_z_basis():
    """默认 Z 基测量在 JSON 往返后 basis 字段为 'Z'（或未设置时从 to_dict 推断）。"""
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    back = circuit_from_json(circuit_to_json(cir))
    g = back.gates[-1]
    # Z 基为默认值，to_dict 不写出 basis 字段；id 默认为 None 也不写出
    assert g.get("basis", "Z") == "Z"
    assert g.get("id") is None


def test_json_round_trip_multi_qubit_basis_y():
    """多比特 Y 基测量的 qubits/basis/id 在 JSON 往返中完整保留。"""
    cir = Circuit(measure([0, 1, 2], basis="Y", id="joint_y"), n_qubits=3)
    back = circuit_from_json(circuit_to_json(cir))
    g = back.gates[-1]
    assert g["type"] == "measure"
    assert g["qubits"] == [0, 1, 2]
    assert g["basis"] == "Y"
    assert g["id"] == "joint_y"


# ──────────────────────────────────────────────────────────────────────────────
# QASM 导出不可表达测试
# ──────────────────────────────────────────────────────────────────────────────


def test_qasm_raises_on_joint_measure():
    """联合多比特 measure 无法导出为标准 QASM，应抛出 NotImplementedError。"""
    cir = Circuit(hadamard(0), measure([0, 1]), n_qubits=2)
    with pytest.raises(NotImplementedError):
        circuit_to_qasm(cir)


def test_qasm_raises_on_non_z_basis():
    """非 Z 基 measure 无法导出为标准 QASM，应抛出 NotImplementedError。"""
    cir = Circuit(hadamard(0), measure(0, basis="X"), n_qubits=1)
    with pytest.raises(NotImplementedError):
        circuit_to_qasm(cir)


def test_qasm_raises_on_measure_with_id():
    """带 id 的 measure 无法导出为标准 QASM，应抛出 NotImplementedError。"""
    cir = Circuit(hadamard(0), measure(0, id="snap1"), n_qubits=1)
    with pytest.raises(NotImplementedError):
        circuit_to_qasm(cir)


def test_qasm_raises_on_joint_non_z_with_id():
    """联合 + 非Z基 + id 的组合也应抛出 NotImplementedError。"""
    cir = Circuit(measure([0, 1], basis="X", id="m0"), n_qubits=2)
    with pytest.raises(NotImplementedError):
        circuit_to_qasm(cir)


# ──────────────────────────────────────────────────────────────────────────────
# QASM 导出正常路径测试
# ──────────────────────────────────────────────────────────────────────────────


def test_qasm_plain_single_z_measure_exports_qasm2():
    """普通单比特 Z 测量（无 id）在 QASM 2.0 中可正常导出，且包含 measure 关键字。"""
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    s = circuit_to_qasm(cir, version="2.0")
    assert "measure" in s.lower()


def test_qasm_plain_single_z_measure_exports_qasm3():
    """普通单比特 Z 测量（无 id）在 QASM 3.0 中可正常导出，且包含 measure 关键字。"""
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    s = circuit_to_qasm(cir, version="3.0")
    assert "measure" in s.lower()


def test_qasm_multiple_plain_z_measures():
    """多个单比特 Z 测量（各自独立，无 id）均可正常导出。"""
    cir = Circuit(hadamard(0), hadamard(1), measure(0), measure(1), n_qubits=2)
    s = circuit_to_qasm(cir, version="2.0")
    assert s.lower().count("measure") >= 2

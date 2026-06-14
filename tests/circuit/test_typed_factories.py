"""门工厂返回类型化 IR（Operation/Measurement）且保持旧字典读取兼容的测试。

约定：工厂函数签名与参数顺序完全不变（``rz(theta, qubit)`` 等），
但返回值由裸字典升级为类型化 IR 对象；对象支持只读的旧字典键访问，
喂给 ``Circuit`` 后内部存储仍为与旧版完全一致的字典。
"""

import pytest

from aicir.core.circuit import (
    Circuit,
    cnot,
    cry,
    cx,
    cz,
    hadamard,
    measure,
    pauli_x,
    reset,
    rx,
    rz,
    rxx,
    rzz,
    s_gate,
    swap,
    u2,
    u3,
)
from aicir.ir import Measurement, Operation


# ---------------------------------------------------------------------------
# 工厂返回类型化 IR
# ---------------------------------------------------------------------------


def test_single_qubit_factory_returns_operation():
    gate = rz(0.5, 1)
    assert isinstance(gate, Operation)
    assert gate.name == "rz"
    assert gate.qubits == (1,)
    assert gate.params == (0.5,)


def test_controlled_factory_returns_operation_with_controls():
    gate = cry(0.3, 0, [1, 3, 2])
    assert isinstance(gate, Operation)
    assert gate.name == "cry"
    assert gate.qubits == (0,)
    assert gate.params == (0.3,)
    assert gate.controls == (1, 3, 2)
    assert gate.control_states == (1, 1, 1)


def test_pair_qubit_factory_returns_operation():
    gate = rxx(0.25, 2, 3)
    assert isinstance(gate, Operation)
    assert gate.qubits == (2, 3)
    assert gate.params == (0.25,)


def test_cnot_alias_still_builds_cx_operation():
    gate = cnot(1, [0, 3])
    assert isinstance(gate, Operation)
    assert gate.name == "cx"
    assert gate.controls == (0, 3)


def test_measure_factory_returns_measurement():
    marker = measure([1, 3])
    assert isinstance(marker, Measurement)
    assert marker.qubits == (1, 3)

    iterable_form = measure([0, 1])
    assert isinstance(iterable_form, Measurement)
    assert iterable_form.qubits == (0, 1)

    empty = measure()
    assert isinstance(empty, Measurement)
    assert empty.qubits == ()


def test_reset_factory_returns_measurement_marker():
    marker = reset([1, 3])
    assert isinstance(marker, Measurement)
    assert marker.measurement_type == "reset"
    assert marker.qubits == (1, 3)

    iterable_form = reset([0, 1])
    assert isinstance(iterable_form, Measurement)
    assert iterable_form.qubits == (0, 1)

    empty = reset()
    assert isinstance(empty, Measurement)
    assert empty.measurement_type == "reset"
    assert empty.qubits == ()


# ---------------------------------------------------------------------------
# 旧字典只读访问兼容
# ---------------------------------------------------------------------------


def test_operation_supports_legacy_dict_reads():
    gate = rz(0.5, 1)
    assert gate["type"] == "rz"
    assert gate["target_qubit"] == 1
    assert gate["parameter"] == 0.5
    assert gate.get("parameter") == 0.5
    assert gate.get("control_qubits") is None
    assert gate.get("control_qubits", []) == []
    assert "type" in gate
    assert "control_qubits" not in gate
    with pytest.raises(KeyError):
        gate["missing_key"]


def test_operation_legacy_reads_match_old_factory_dicts():
    assert dict(rz(0.5, 1)) == {"type": "rz", "target_qubit": 1, "parameter": 0.5}
    assert dict(hadamard(0)) == {"type": "hadamard", "target_qubit": 0}
    assert dict(cx(1, [0, 3])) == {
        "type": "cx",
        "target_qubit": 1,
        "control_qubits": [0, 3],
        "control_states": [1, 1],
    }
    assert dict(swap(0, 3)) == {"type": "swap", "qubit_1": 0, "qubit_2": 3}
    assert dict(rzz(0.7, 0, 3)) == {
        "type": "rzz",
        "qubit_1": 0,
        "qubit_2": 3,
        "parameter": 0.7,
    }
    assert dict(u3(0.1, 0.2, 0.3, 2)) == {
        "type": "u3",
        "target_qubit": 2,
        "parameter": [0.1, 0.2, 0.3],
    }
    assert dict(u2(0.4, 0.5, 3)) == {
        "type": "u2",
        "target_qubit": 3,
        "parameter": [0.4, 0.5],
    }


def test_operation_supports_mapping_protocol_helpers():
    gate = cx(1, [0])
    keys = set(gate.keys())
    assert keys == {"type", "target_qubit", "control_qubits", "control_states"}
    assert set(iter(gate)) == keys
    assert len(gate) == 4
    assert dict(gate.items()) == dict(gate)
    assert {**gate} == dict(gate)


def test_measurement_supports_legacy_dict_reads():
    marker = measure([1, 3])
    assert marker["type"] == "measure"
    assert marker["qubits"] == [1, 3]
    assert marker.get("qubits") == [1, 3]
    assert "type" in marker
    assert dict(marker) == {"type": "measure", "qubits": [1, 3]}


def test_reset_supports_legacy_dict_reads():
    marker = reset([1, 3])
    assert marker["type"] == "reset"
    assert marker["qubits"] == [1, 3]
    assert marker.get("qubits") == [1, 3]
    assert "type" in marker
    assert dict(marker) == {"type": "reset", "qubits": [1, 3]}


def test_operation_compares_equal_to_legacy_dict_form():
    gate = rz(0.5, 1)
    legacy = {"type": "rz", "target_qubit": 1, "parameter": 0.5}
    assert gate == legacy
    assert legacy == gate
    assert gate != {"type": "rz", "target_qubit": 0, "parameter": 0.5}
    assert rz(0.5, 1) == rz(0.5, 1)
    assert rz(0.5, 1) != rx(0.5, 1)


def test_measurement_compares_equal_to_legacy_dict_form():
    assert measure([1, 3]) == {"type": "measure", "qubits": [1, 3]}
    assert {"type": "measure", "qubits": [1, 3]} == measure([1, 3])
    assert measure([1, 3]) == measure([1, 3])
    assert measure([1, 3]) != measure([1, 2])


def test_reset_compares_equal_to_legacy_dict_form():
    assert reset([1, 3]) == {"type": "reset", "qubits": [1, 3]}
    assert {"type": "reset", "qubits": [1, 3]} == reset([1, 3])
    assert reset([1, 3]) == reset([1, 3])
    assert reset([1, 3]) != reset([1, 2])


def test_operation_is_immutable_via_item_assignment():
    gate = rz(0.5, 1)
    with pytest.raises(TypeError):
        gate["parameter"] = 1.0


# ---------------------------------------------------------------------------
# Circuit 内部存储与旧版字典完全一致
# ---------------------------------------------------------------------------


def test_circuit_stores_same_dicts_as_before():
    cir = Circuit(
        hadamard(0),
        rz(0.5, 1),
        cx(1, [0]),
        pauli_x(3),
        s_gate(2),
        rx(0.1, 2),
        cz(2, [3]),
        measure([1, 3]),
        reset([1, 3]),
        n_qubits=4,
    )
    assert cir.gates == [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "rz", "target_qubit": 1, "parameter": 0.5},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        {"type": "pauli_x", "target_qubit": 3},
        {"type": "s_gate", "target_qubit": 2},
        {"type": "rx", "target_qubit": 2, "parameter": 0.1},
        {"type": "cz", "target_qubit": 2, "control_qubits": [3], "control_states": [1]},
        {"type": "measure", "qubits": [1, 3]},
        {"type": "reset", "qubits": [1, 3]},
    ]
    assert all(type(gate) is dict for gate in cir.gates)

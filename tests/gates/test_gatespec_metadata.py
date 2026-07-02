"""GateSpec 元数据扩充测试（NEXT.md §7 第三片 / QML todo 2.3）。

为 GateSpec 增加 ``generator``（Pauli 生成元，用于解析参数移位自省）与
``decomposition``（分解到更基础门集的规则），并由 ``aicir.gates`` 提供
查询 helper 供 ``transpile``/``qml`` 消费，替代散落的硬编码门集。
"""

import numpy as np
import pytest

from aicir.gates import (
    GateSpec,
    gate_decomposition,
    gate_generator,
    get_gate_spec,
    parametric_pauli_gates,
    register_gate,
    unregister_gate,
)


# --- generator 字段 ---

def test_rotation_gates_carry_pauli_generator():
    assert gate_generator("rx") == "X"
    assert gate_generator("ry") == "Y"
    assert gate_generator("rz") == "Z"
    assert gate_generator("rzz") == "ZZ"
    assert gate_generator("rxx") == "XX"


def test_controlled_rotation_generator_is_target_pauli():
    assert gate_generator("crx") == "X"
    assert gate_generator("cry") == "Y"
    assert gate_generator("crz") == "Z"


def test_non_parametric_gates_have_no_generator():
    assert gate_generator("hadamard") is None
    assert gate_generator("cx") is None
    assert gate_generator("swap") is None
    assert gate_generator("definitely_not_a_gate") is None


def test_parametric_pauli_gates_set():
    gates = parametric_pauli_gates()
    assert {"rx", "ry", "rz", "crx", "cry", "crz", "rzz", "rxx"} <= gates
    assert "hadamard" not in gates
    assert "cx" not in gates


def test_generator_alias_resolves():
    # 别名经规范名解析到同一 generator。
    assert gate_generator("X") is None  # pauli_x 不是参数化旋转
    spec = get_gate_spec("rx")
    assert spec.generator == "X"


# --- decomposition 字段 ---

def test_swap_decomposition_is_three_cx():
    deco = gate_decomposition("swap")
    assert deco is not None
    gates = deco((0, 1), (), None, None)
    assert [g["type"] for g in gates] == ["cx", "cx", "cx"]
    # 中间 cx 控制方向相反。
    assert gates[0]["control_qubits"] == [1]
    assert gates[1]["control_qubits"] == [0]


def test_cz_decomposition_is_h_cx_h():
    deco = gate_decomposition("cz")
    gates = deco((0,), (1,), (1,), None)
    assert [g["type"] for g in gates] == ["hadamard", "cx", "hadamard"]


def test_decomposition_rejects_multi_control():
    deco = gate_decomposition("cz")
    assert deco((0,), (1, 2), (1, 1), None) is None


def test_non_decomposable_gate_returns_none():
    assert gate_decomposition("rx") is None
    assert gate_decomposition("cx") is None


# --- 自定义门注册扩展字段 ---

def test_custom_gate_registers_generator_for_psr_autodetect():
    spec = GateSpec(name="my_rot", num_qubits=1, num_params=1, generator="Z")
    register_gate(spec)
    try:
        assert gate_generator("my_rot") == "Z"
        assert "my_rot" in parametric_pauli_gates()
    finally:
        unregister_gate("my_rot")

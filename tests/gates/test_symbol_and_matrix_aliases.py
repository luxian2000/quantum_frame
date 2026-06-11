"""§7 第二片测试：GateSpec 增加 symbol 字段，矩阵/绘图路径经注册表统一别名。

- ``GateSpec.symbol``：绘图显示符号的单一来源。
- ``gate_to_matrix``/绘图对别名 type（X/cnot/ccnot/I）与规范名结果完全一致。
- 注册自定义门时可携带 symbol，ASCII 绘图直接使用。
"""

import numpy as np
import pytest

from aicir import Circuit, NumpyBackend
from aicir.core.circuit import _circuit_to_ascii
from aicir.core.gates import gate_to_matrix
from aicir.gates import GateSpec, get_gate_spec, register_gate
from aicir.gates.registry import unregister_gate


# ---------------------------------------------------------------------------
# GateSpec.symbol 字段
# ---------------------------------------------------------------------------


def test_standard_gate_specs_carry_display_symbols():
    assert get_gate_spec("pauli_x").symbol == "X"
    assert get_gate_spec("hadamard").symbol == "H"
    assert get_gate_spec("rx").symbol == "Rx"
    assert get_gate_spec("u3").symbol == "U3"
    assert get_gate_spec("identity").symbol == "I"
    assert get_gate_spec("unitary").symbol == "U"
    # 受控门的 symbol 是目标位显示符号
    assert get_gate_spec("cx").symbol == "X"
    assert get_gate_spec("crz").symbol == "Rz"
    assert get_gate_spec("toffoli").symbol == "X"
    # 特殊绘制的门不带 symbol
    assert get_gate_spec("swap").symbol is None
    assert get_gate_spec("measure").symbol is None


# ---------------------------------------------------------------------------
# 别名与规范名等价：矩阵路径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias_gate, canonical_gate",
    [
        ({"type": "X", "target_qubit": 0}, {"type": "pauli_x", "target_qubit": 0}),
        ({"type": "H", "target_qubit": 1}, {"type": "hadamard", "target_qubit": 1}),
        (
            {"type": "cnot", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
            {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        ),
        (
            {"type": "ccnot", "target_qubit": 2, "control_qubits": [0, 1]},
            {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1]},
        ),
    ],
)
def test_alias_and_canonical_matrices_match(alias_gate, canonical_gate):
    cir_qubits = 3
    np.testing.assert_allclose(
        gate_to_matrix(alias_gate, cir_qubits=cir_qubits),
        gate_to_matrix(canonical_gate, cir_qubits=cir_qubits),
    )
    backend = NumpyBackend()
    np.testing.assert_allclose(
        backend.to_numpy(gate_to_matrix(alias_gate, cir_qubits=cir_qubits, backend=backend)),
        backend.to_numpy(gate_to_matrix(canonical_gate, cir_qubits=cir_qubits, backend=backend)),
    )


def test_identity_alias_matrix_matches():
    np.testing.assert_allclose(
        gate_to_matrix({"type": "I", "n_qubits": 2}, cir_qubits=2),
        gate_to_matrix({"type": "identity", "n_qubits": 2}, cir_qubits=2),
    )


# ---------------------------------------------------------------------------
# 别名与规范名等价：ASCII 绘图
# ---------------------------------------------------------------------------


def test_alias_circuit_renders_same_ascii_as_canonical():
    alias_cir = Circuit(
        {"type": "X", "target_qubit": 0},
        {"type": "cnot", "target_qubit": 1, "control_qubits": [0]},
        n_qubits=2,
    )
    canonical_cir = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0]},
        n_qubits=2,
    )
    assert _circuit_to_ascii(alias_cir) == _circuit_to_ascii(canonical_cir)


def test_custom_registered_symbol_is_used_in_ascii():
    register_gate(GateSpec(name="my_blk", num_qubits=1, num_params=0, symbol="Bk"))
    try:
        cir = Circuit({"type": "my_blk", "target_qubit": 0}, n_qubits=1)
        assert "Bk" in _circuit_to_ascii(cir)
    finally:
        unregister_gate("my_blk")

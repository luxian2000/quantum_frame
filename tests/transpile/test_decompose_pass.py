"""DecomposePass 测试：高级门分解到目标门集，且保持幺正等价。"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, cy, cz, hadamard, rz, swap
from aicir.devices import Target
from aicir.ir import circuit_gate_dicts
from aicir.transpile import DecomposePass, PassManager


def _equiv(a: Circuit, b: Circuit) -> bool:
    return np.allclose(np.asarray(a.unitary()), np.asarray(b.unitary()), atol=1e-5)


def test_swap_cz_cy_decompose_to_cx_and_preserve_unitary():
    cir = Circuit(swap(0, 1), cz(1, [0]), cy(2, [0]), n_qubits=3)
    out = DecomposePass(basis_gates=("cx",)).run(cir)
    types = {gate["type"] for gate in circuit_gate_dicts(out)}
    assert types <= {"cx", "hadamard", "rz"}
    assert "swap" not in types and "cz" not in types and "cy" not in types
    assert _equiv(cir, out)


def test_custom_gate_decomposition_is_registry_driven():
    # 注册自定义门并携带 decomposition -> DecomposePass 无需改动即可识别（§7）。
    from aicir.gates import GateSpec, register_gate, unregister_gate

    def _my_swap_rule(qubits, controls, control_states, params):
        if len(qubits) != 2 or controls:
            return None
        a, b = int(qubits[0]), int(qubits[1])
        return [
            {"type": "cx", "target_qubit": a, "control_qubits": [b], "control_states": [1]},
            {"type": "cx", "target_qubit": b, "control_qubits": [a], "control_states": [1]},
            {"type": "cx", "target_qubit": a, "control_qubits": [b], "control_states": [1]},
        ]

    register_gate(GateSpec("my_swap", 2, 0, decomposition=_my_swap_rule))
    try:
        cir = Circuit({"type": "my_swap", "qubit_1": 0, "qubit_2": 1}, n_qubits=2)
        out = DecomposePass(basis_gates=("cx",)).run(cir)
        assert [g["type"] for g in circuit_gate_dicts(out)] == ["cx", "cx", "cx"]
    finally:
        unregister_gate("my_swap")


def test_gates_in_basis_are_left_untouched():
    cir = Circuit(swap(0, 1), hadamard(0), rz(0.3, 1), n_qubits=2)
    out = DecomposePass(basis_gates=("swap", "hadamard", "rz", "cx")).run(cir)
    # swap 在门集内 -> 不分解
    assert [g["type"] for g in circuit_gate_dicts(out)] == ["swap", "hadamard", "rz"]


def test_target_supplies_basis_gates():
    target = Target(n_qubits=2, basis_gates=("cx", "hadamard", "rz"))
    cir = Circuit(cz(1, [0]), n_qubits=2)
    out = DecomposePass(target=target).run(cir)
    assert "cz" not in {g["type"] for g in circuit_gate_dicts(out)}
    assert _equiv(cir, out)


def test_unsupported_two_qubit_gate_raises_unless_skipped():
    cir = Circuit(cx(1, [0]), n_qubits=2)
    # cx 不在门集且无规则 -> 默认报错
    with pytest.raises(ValueError):
        DecomposePass(basis_gates=("rz",)).run(cir)
    # skip_unsupported 时原样保留
    out = DecomposePass(basis_gates=("rz",), skip_unsupported=True).run(cir)
    assert [g["type"] for g in circuit_gate_dicts(out)] == ["cx"]


def test_runs_inside_passmanager_by_name():
    cir = Circuit(swap(0, 1), n_qubits=2)
    out = PassManager(["decompose"]).run(cir)
    assert all(g["type"] == "cx" for g in circuit_gate_dicts(out))
    assert _equiv(cir, out)

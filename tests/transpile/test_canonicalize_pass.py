"""CanonicalizePass 实质规范化测试：别名门名重写为规范名，其余字段不动。"""

from aicir.core.circuit import Circuit
from aicir.transpile import CanonicalizePass, PassManager


def test_alias_gate_types_are_rewritten_to_canonical_names():
    cir = Circuit(
        {"type": "X", "target_qubit": 0},
        {"type": "H", "target_qubit": 1},
        {"type": "cnot", "target_qubit": 1, "control_qubits": [0]},
        {"type": "ccnot", "target_qubit": 2, "control_qubits": [0, 1]},
        n_qubits=3,
    )
    out = CanonicalizePass().run(cir)
    assert [gate["type"] for gate in out.gates] == ["pauli_x", "hadamard", "cx", "toffoli"]
    # 其余字段保持不变
    assert out.gates[2]["control_qubits"] == [0]
    assert out.gates[3]["control_qubits"] == [0, 1]
    assert out.n_qubits == 3


def test_canonical_and_unknown_names_pass_through_unchanged():
    cir = Circuit(
        {"type": "rz", "target_qubit": 0, "parameter": 0.5},
        {"type": "my_custom_block", "qubits": [0, 1], "parameter": 0.1},
        n_qubits=2,
    )
    out = CanonicalizePass().run(cir)
    assert out.gates[0]["type"] == "rz"
    assert out.gates[0]["parameter"] == 0.5
    assert out.gates[1]["type"] == "my_custom_block"


def test_canonicalize_runs_inside_passmanager_by_name():
    cir = Circuit({"type": "Y", "target_qubit": 0}, n_qubits=1)
    out = PassManager(["canonicalize"]).run(cir)
    assert out.gates[0]["type"] == "pauli_y"

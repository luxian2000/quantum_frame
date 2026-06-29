"""粒子数守恒激发门 + shift_rule 元数据测试。"""

from aicir.gates import get_gate_spec, gate_shift_rule


def test_shift_rule_defaults_to_none():
    assert get_gate_spec("rx").shift_rule is None
    assert get_gate_spec("rzz").shift_rule is None


def test_gate_shift_rule_helper_reads_registry():
    # 未注册门返回 None；标准门 None
    assert gate_shift_rule("rx") is None
    assert gate_shift_rule("not_a_gate") is None


import math
import numpy as np
from aicir import single_excitation, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.core.circuit import Circuit


def _num_op(n):
    # 粒子数算符 N = sum_i (I - Z_i)/2，对角线为各基态的 popcount
    return np.diag([bin(i).count("1") for i in range(1 << n)]).astype(complex)


def test_single_excitation_matrix_is_real_givens():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 0.7}, cir_qubits=2, backend=NumpyBackend())
    c, s = math.cos(0.35), math.sin(0.35)
    expected = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=complex)
    assert np.allclose(m, expected)


def test_single_excitation_is_unitary():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 1.3}, cir_qubits=2, backend=NumpyBackend())
    assert np.allclose(m.conj().T @ m, np.eye(4))


def test_single_excitation_conserves_particle_number():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 0.9}, cir_qubits=2, backend=NumpyBackend())
    N = _num_op(2)
    assert np.allclose(m.conj().T @ N @ m, N)  # [G, N] = 0


def test_givens_alias_and_factory():
    from aicir.gates import canonical_gate_name
    assert canonical_gate_name("givens") == "single_excitation"
    op = single_excitation(0.5, qubit_1=1, qubit_2=2)
    d = op.to_dict()
    assert d["type"] == "single_excitation" and d["qubit_1"] == 1 and d["qubit_2"] == 2

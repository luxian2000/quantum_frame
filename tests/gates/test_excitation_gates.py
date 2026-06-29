"""粒子数守恒激发门 + shift_rule 元数据测试。"""

import math
import numpy as np
import pytest

from aicir.gates import get_gate_spec, gate_shift_rule
from aicir import single_excitation, double_excitation, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.core.circuit import Circuit, pauli_x


def test_shift_rule_defaults_to_none():
    assert get_gate_spec("rx").shift_rule is None
    assert get_gate_spec("rzz").shift_rule is None


def test_gate_shift_rule_helper_reads_registry():
    # 未注册门返回 None；标准门 None
    assert gate_shift_rule("rx") is None
    assert gate_shift_rule("not_a_gate") is None


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


def test_double_excitation_matrix_couples_0011_and_1100():
    m = gate_to_matrix({"type": "double_excitation", "qubits": [0, 1, 2, 3],
                        "parameter": 0.8}, cir_qubits=4, backend=NumpyBackend())
    c, s = math.cos(0.4), math.sin(0.4)
    expected = np.eye(16, dtype=complex)
    expected[3, 3] = c
    expected[3, 12] = -s
    expected[12, 3] = s
    expected[12, 12] = c
    assert np.allclose(m, expected)


def test_double_excitation_is_unitary_and_conserves_N():
    m = gate_to_matrix({"type": "double_excitation", "qubits": [0, 1, 2, 3],
                        "parameter": 1.1}, cir_qubits=4, backend=NumpyBackend())
    assert np.allclose(m.conj().T @ m, np.eye(16))
    N = _num_op(4)
    assert np.allclose(m.conj().T @ N @ m, N)


def test_double_excitation_factory_serializes_qubits_list():
    d = double_excitation(0.3, 0, 1, 2, 3).to_dict()
    assert d["type"] == "double_excitation" and d["qubits"] == [0, 1, 2, 3]


def _energy_grad_autograd_vs_fd(circuit_fn, theta0, n_qubits, prep_qubits):
    """比较自动微分梯度与有限差分梯度。

    prep_qubits: 在目标门前施加 X 门的量子比特列表，使初态非平凡。
    """
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import GPUBackend
    from aicir.core.gates import apply_gate_to_state
    backend = GPUBackend(device="cpu")

    def energy(theta_t):
        circuit = circuit_fn(theta_t)
        state = backend.zeros_state(n_qubits)
        # 初态制备：对指定量子比特施加 X 门
        for q in prep_qubits:
            state = apply_gate_to_state(pauli_x(q).to_dict(), state, n_qubits, backend)
        for g in circuit.gates:
            state = apply_gate_to_state(g, state, n_qubits, backend)
        # H = Z on qubit 0
        z0 = np.kron(np.diag([1.0, -1.0]), np.eye(1 << (n_qubits - 1)))
        H = backend.cast(z0.astype(np.complex64))
        return backend.expectation_sv(state, H)

    theta_t = torch.tensor(float(theta0), requires_grad=True)
    e = energy(theta_t)
    e.real.backward()
    grad_ad = float(theta_t.grad)

    eps = 1e-4
    ep = float(energy(torch.tensor(theta0 + eps)).real.detach())
    em = float(energy(torch.tensor(theta0 - eps)).real.detach())
    grad_fd = (ep - em) / (2 * eps)

    # 梯度非零：确保测试不因平凡态而空过
    assert abs(grad_ad) > 1e-3, f"autograd 梯度接近零 ({grad_ad:.6f})，初态可能未激活门"
    # 自动微分与有限差分一致
    assert abs(grad_ad - grad_fd) < 1e-3, f"AD={grad_ad:.6f} FD={grad_fd:.6f} 差异过大"


def test_single_excitation_autograd_grad_matches_fd():
    # 初态 |01>（qubit 1 置 1），使 single_excitation 门实际演化态
    _energy_grad_autograd_vs_fd(
        lambda th: Circuit(single_excitation(th, 0, 1), n_qubits=2),
        theta0=0.6,
        n_qubits=2,
        prep_qubits=[1],
    )


def test_double_excitation_autograd_grad_matches_fd():
    # 初态 |0011>（qubit 2, 3 置 1），使 double_excitation 门实际演化态
    _energy_grad_autograd_vs_fd(
        lambda th: Circuit(double_excitation(th, 0, 1, 2, 3), n_qubits=4),
        theta0=0.7,
        n_qubits=4,
        prep_qubits=[2, 3],
    )

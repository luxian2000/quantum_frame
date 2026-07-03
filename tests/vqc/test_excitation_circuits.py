import numpy as np
import pytest

from aicir import Circuit, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.ansatze._excitation import (
    double_excitation_ops,
    fswap_ops,
    single_excitation_ops,
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron(paulis):
    out = np.array([[1]], dtype=complex)
    for p in paulis:
        out = np.kron(out, _PAULI[p])
    return out


def _circuit_unitary(ops, n_qubits):
    circ = Circuit(*ops, n_qubits=n_qubits)
    return np.asarray(NumpyBackend().to_numpy(circ.unitary()))


def _single_excitation_generator(n, p, q):
    # JW: T_pq = a_p^dagger a_q - a_q^dagger a_p = (i/4)(X_p Z...Z Y_q - Y_p Z...Z X_q)，
    # interior 为 Z 串。T_pq 是反厄米的（T_pq^dagger = -T_pq），exp(theta*T_pq) 才是酉矩阵。
    # 数值核对：expm(theta*(i/4)*(XY-YX)) 与既有 single_excitation gate（c=cos(theta/2)，
    # s=sin(theta/2)）在相邻（无 Z 串）情形下逐元素相等（见 task-3-report.md）。
    def string(a_gate, b_gate):
        labels = ["I"] * n
        labels[p], labels[q] = a_gate, b_gate
        for k in range(p + 1, q):
            labels[k] = "Z"
        return _kron(labels)

    return 0.25j * (string("X", "Y") - string("Y", "X"))


def test_fswap_is_swap_times_cz():
    theta = 0.0  # 无关，fswap 无参
    got = _circuit_unitary(fswap_ops(0, 1), 2)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    assert np.allclose(got, swap @ cz, atol=1e-9)


def test_adjacent_single_excitation_matches_existing_gate():
    theta = 0.7
    ops = single_excitation_ops(theta, 0, 1)
    got = _circuit_unitary(ops, 2)
    ref = np.asarray(
        NumpyBackend().to_numpy(
            gate_to_matrix(
                {"type": "single_excitation", "qubit_1": 0, "qubit_2": 1, "parameter": theta},
                2,
                NumpyBackend(),
            )
        )
    )
    assert np.allclose(got, ref, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.1, -0.8])
def test_nonadjacent_single_excitation_matches_expm_generator(theta):
    from scipy.linalg import expm

    n, p, q = 4, 0, 3
    got = _circuit_unitary(single_excitation_ops(theta, p, q), n)
    gen = _single_excitation_generator(n, p, q)
    ref = expm(theta * gen)
    # 允许全局相位差
    phase = np.vdot(ref.reshape(-1), got.reshape(-1))
    phase /= abs(phase)
    assert np.allclose(got, phase * ref, atol=1e-8)


def _ladder_op(n, k, dagger=False):
    """JW fermionic ladder operator a_k（或 a_k^dagger）在 n qubit 空间的矩阵。

    约定与 ``_single_excitation_generator`` 一致：qubit 0 是 Kron 乘积中最左（最高位）
    的因子；mode k 左侧（index < k）的 qubit 贡献 Z 串，右侧贡献 I，自身贡献
    |0><1|（湮灭，把占据 |1> 打到空 |0>）或其共轭转置（产生）。
    """
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|, 湮灭 a_k
    op = sigma_minus.conj().T if dagger else sigma_minus
    out = np.array([[1]], dtype=complex)
    for j in range(n):
        if j < k:
            out = np.kron(out, _PAULI["Z"])
        elif j == k:
            out = np.kron(out, op)
        else:
            out = np.kron(out, _PAULI["I"])
    return out


def _double_excitation_generator(n, p, q, r, s):
    # T = (1/2)(a_p^dagger a_q^dagger a_r a_s - h.c.)（反厄米，exp(theta*T) 为酉矩阵）。
    # 1/2 因子与 _single_excitation_generator 的 0.25j 前因子同一约定：既有
    # single/double_excitation gate 都用 cos(theta/2)/sin(theta/2)（半角）参数化，
    # 生成元里预先烤入这个 1/2，好让 oracle 侧直接用 expm(theta*T)（不额外对 theta
    # 做 /2 的算术），theta 原样对照生产代码的输入。
    ap_dag = _ladder_op(n, p, dagger=True)
    aq_dag = _ladder_op(n, q, dagger=True)
    ar = _ladder_op(n, r, dagger=False)
    as_ = _ladder_op(n, s, dagger=False)
    term = ap_dag @ aq_dag @ ar @ as_
    return 0.5 * (term - term.conj().T)


@pytest.mark.parametrize(
    "n,p,q,r,s",
    [
        (4, 0, 1, 2, 3),  # 相邻（沿用旧配置，保底不回归）
        (6, 0, 1, 3, 5),  # 非相邻，真正跑 fSWAP 网络
        (8, 0, 2, 3, 7),  # 非相邻，跨度更大
        (4, 0, 2, 1, 3),  # 创生/湮灭对在数值上交错（p<r<q<s）——UCCSD 集成回归用例：
        # H2（4 qubit JW）占据 {1,3}、未占据 {0,2} 就是这种交错分布，若误按数值
        # 排序会拆错创生/湮灭配对，导致该激发在占据 {1,3} 的态上完全不起作用。
        (6, 0, 3, 1, 5),  # 交错 + 非相邻的组合
    ],
)
@pytest.mark.parametrize("theta", [0.3, -0.7])
def test_nonadjacent_double_excitation_matches_jw_generator(n, p, q, r, s, theta):
    from scipy.linalg import expm

    got = _circuit_unitary(double_excitation_ops(theta, p, q, r, s), n)
    gen = _double_excitation_generator(n, p, q, r, s)

    ok = False
    for sign in (1.0, -1.0):
        ref = expm(sign * theta * gen)
        phase = np.vdot(ref.reshape(-1), got.reshape(-1))
        phase /= abs(phase)
        if np.allclose(got, phase * ref, atol=1e-8):
            ok = True
            break
    assert ok, f"double_excitation_ops({theta}, {p}, {q}, {r}, {s}) 与 JW 生成元 oracle 不符（两种 theta 符号均不匹配）"

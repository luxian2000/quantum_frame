import numpy as np
import pytest

from aicir import Circuit, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.vqc.ansatz._excitation import (
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


def test_nonadjacent_double_excitation_matches_qiskit_ucc():
    pytest.importorskip("qiskit_nature")
    from qiskit.quantum_info import Operator
    from qiskit_nature.second_q.circuit.library import UCC
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    # 2 空间轨道、2 电子 → 4 qubit JW；取其唯一 double 激发做对照
    ucc = UCC(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        excitations="d",
        qubit_mapper=JordanWignerMapper(),
    )
    theta = 0.37
    # Qiskit UCC 的旋转角约定是 cos(theta)/sin(theta)（无 1/2 因子），而 aicir 既有
    # double_excitation gate 是 cos(theta/2)/sin(theta/2)；为让同一个 theta 原样喂给
    # 生产代码（不做任何算术），这里只在构造 oracle 侧用 theta/2 绑定 UCC 参数——
    # 这是 oracle 的角度换算，不是对 double_excitation_ops 输入参数的算术。
    bound = ucc.assign_parameters([theta / 2])
    ref = Operator(bound).data

    # aicir 对应的 double 激发 orbital 索引（与 UCC 的 JW 约定对齐）
    ops = double_excitation_ops(theta, 0, 1, 2, 3)
    got = _circuit_unitary(ops, 4)
    phase = np.vdot(ref.reshape(-1), got.reshape(-1))
    phase /= abs(phase)
    assert np.allclose(got, phase * ref, atol=1e-7)

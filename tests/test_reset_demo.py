"""tests/test_reset_demo.py

验证 reset_demo 在新统一测量模型下的行为：
- reset 是无需前置 measure 的直接信道
- 对纠缠比特施加 reset 产生混合态（密度矩阵）
- 各阶段快照与理论期望严格吻合

电路（3 量子比特，无 measure 门）：
    op 0: H(0)
    op 1: CNOT(1, [0])
    op 2: CNOT(2, [1])
    op 3: reset(1)
    op 4: CNOT(1, [2])

理论期望快照：
    snap(0): (|000> + |100>) / sqrt(2)     纯态向量，flat index 0,4
    snap(1): (|000> + |110>) / sqrt(2)     纯态向量，flat index 0,6
    snap(2): (|000> + |111>) / sqrt(2)     纯态向量，flat index 0,7（GHZ 态）
    snap(3): (|000><000| + |101><101|) / 2 密度矩阵，对角元 [0]=0.5, [5]=0.5
    snap(4): (|000><000| + |111><111|) / 2 密度矩阵，对角元 [0]=0.5, [7]=0.5

推导：
  - reset(1) 作用于 GHZ 态：纯态升级为密度矩阵
    ρ_reset = (|000><000| + |101><101|) / 2
    对角线：diag[0]=0.5（|000>），diag[5]=0.5（|101>=5）
  - CNOT(1,[2])（控制 q2，目标 q1）作用于 ρ_reset：
    |000> → |000>（q2=0，不翻转），|101> → |111>（q2=1，翻转 q1）
    ρ_final = (|000><000| + |111><111|) / 2
    对角线：diag[0]=0.5，diag[7]=0.5（|111>=7）
"""

import numpy as np
import pytest

from demos.reset_demo import run_demo


@pytest.fixture(scope="module")
def report():
    return run_demo(verbose=False)


# ---------------------------------------------------------------------------
# 基本结构检查
# ---------------------------------------------------------------------------

def test_circuit_n_qubits(report):
    assert report["circuit"].n_qubits == 3


def test_reset_verified_flag(report):
    assert report["reset_verified"] is True


# ---------------------------------------------------------------------------
# 纯态阶段（snap 0-2）
# ---------------------------------------------------------------------------

def test_snap_after_hadamard(report):
    """snap(0): (|000> + |100>) / sqrt(2)，flat index 0=000, 4=100。"""
    expected = _sv([0, 4], [1.0 / np.sqrt(2)] * 2)
    np.testing.assert_allclose(
        np.asarray(report["snap_after_hadamard"]).reshape(-1),
        expected,
        atol=1e-6,
    )


def test_snap_after_cnot_10(report):
    """snap(1): (|000> + |110>) / sqrt(2)，flat index 0=000, 6=110。"""
    expected = _sv([0, 6], [1.0 / np.sqrt(2)] * 2)
    np.testing.assert_allclose(
        np.asarray(report["snap_after_cnot_10"]).reshape(-1),
        expected,
        atol=1e-6,
    )


def test_snap_after_cnot_21(report):
    """snap(2): GHZ (|000> + |111>) / sqrt(2)，flat index 0=000, 7=111。"""
    expected = _sv([0, 7], [1.0 / np.sqrt(2)] * 2)
    np.testing.assert_allclose(
        np.asarray(report["snap_after_cnot_21"]).reshape(-1),
        expected,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 密度矩阵阶段（snap 3-4）
# ---------------------------------------------------------------------------

def test_snap_after_reset_is_density_matrix(report):
    """reset(1) 作用于纠缠态后，快照必须是密度矩阵（shape (8,8)）。"""
    assert report["is_dm_after_reset"] is True


def test_snap_after_reset_correct_probabilities(report):
    """snap(3) 对角线：|000>(index 0) 和 |101>(index 5) 各占 0.5。"""
    rho = np.asarray(report["snap_after_reset"]).reshape(8, 8)
    diag = np.real(np.diag(rho))
    expected_diag = np.zeros(8, dtype=np.float64)
    expected_diag[0] = 0.5   # |000>
    expected_diag[5] = 0.5   # |101> = 5
    np.testing.assert_allclose(diag, expected_diag, atol=1e-6)


def test_snap_after_reset_full_density_matrix(report):
    """snap(3) 完整密度矩阵：(|000><000| + |101><101|) / 2。"""
    rho_actual = np.asarray(report["snap_after_reset"]).reshape(8, 8)
    rho_expected = _dm([_sv([0], [1.0]), _sv([5], [1.0])])
    np.testing.assert_allclose(rho_actual, rho_expected, atol=1e-6)


def test_snap_after_cnot_12_is_density_matrix(report):
    """CNOT(1,[2]) 作用于混合态后，快照仍是密度矩阵（shape (8,8)）。"""
    assert report["is_dm_after_cnot12"] is True


def test_snap_after_cnot_12_correct_probabilities(report):
    """snap(4) 对角线：|000>(index 0) 和 |111>(index 7) 各占 0.5。"""
    rho = np.asarray(report["snap_after_cnot_12"]).reshape(8, 8)
    diag = np.real(np.diag(rho))
    expected_diag = np.zeros(8, dtype=np.float64)
    expected_diag[0] = 0.5   # |000>
    expected_diag[7] = 0.5   # |111> = 7
    np.testing.assert_allclose(diag, expected_diag, atol=1e-6)


def test_snap_after_cnot_12_full_density_matrix(report):
    """snap(4) 完整密度矩阵：(|000><000| + |111><111|) / 2。"""
    rho_actual = np.asarray(report["snap_after_cnot_12"]).reshape(8, 8)
    rho_expected = _dm([_sv([0], [1.0]), _sv([7], [1.0])])
    np.testing.assert_allclose(rho_actual, rho_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _sv(indices, values):
    """构造 8 维复数态向量（纯态）。"""
    state = np.zeros(8, dtype=np.complex128)
    for index, value in zip(indices, values):
        state[index] = value
    return state


def _dm(vecs):
    """构造等权混合密度矩阵 ρ = Σ_k |ψ_k><ψ_k| / K（8x8）。"""
    rho = np.zeros((8, 8), dtype=np.complex128)
    for v in vecs:
        v = v.reshape(-1, 1)
        rho += v @ v.conj().T
    return rho / len(vecs)

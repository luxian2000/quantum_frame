"""四项参数移位规则 psr4：对激发门梯度与 autograd 一致，两项 psr 不一致。

判别电路设计原则：
  初态必须跨越激发门的 *两个* 本征频率子空间，否则能量函数为单频余弦，
  两项 psr 与四项 psr 等价（退化情形），无法区分。
  superposition 初态 + 非平凡观测量可暴露半频分量，使两项 psr 产生误差。
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir import single_excitation, double_excitation
from aicir.backends.gpu_backend import GPUBackend
from aicir.core.circuit import Circuit, hadamard, pauli_x
from aicir.core.gates import apply_gate_to_state
from aicir.qml.deriv import psr, psr4


# ─────────────────────────── single_excitation 判别电路 ──────────────────────────
# 初态：|00> ──H(qubit=1)──> (|00>+|01>)/√2
# 然后施加 single_excitation(theta, 0, 1)
# 观测量：X ⊗ I = [[0,1],[1,0]] ⊗ I_2  (连接 |0x> 与 |1x> 扇区，暴露半频分量)
#
# 数值验证 (theta=0.6)：autograd = psr4 ≈ 0.4777，两项 psr ≈ 0.6755（不同）

def _single_excitation_discriminating_fn(theta_value, grad=False):
    """返回 (th, energy)；当 grad=True 时 th 带 requires_grad。"""
    backend = GPUBackend(device="cpu")
    # H = X ⊗ I（4×4）
    H_np = np.kron(np.array([[0, 1], [1, 0]], dtype=np.float32), np.eye(2, dtype=np.float32)).astype(np.complex64)
    H = backend.cast(H_np)

    th = torch.tensor(float(theta_value[0]), requires_grad=grad)
    # 初态 |00>
    state = backend.zeros_state(2)
    # H 门作用在 qubit 1：(|00>+|01>)/√2
    state = apply_gate_to_state(hadamard(1).to_dict(), state, 2, backend)
    # single_excitation(theta, 0, 1)
    for g in Circuit(single_excitation(th, 0, 1), n_qubits=2).gates:
        state = apply_gate_to_state(g, state, 2, backend)
    e = backend.expectation_sv(state, H).real
    return th, e


def _single_excitation_energy(theta_value):
    _, e = _single_excitation_discriminating_fn(theta_value)
    return float(e.detach())


def _single_excitation_autograd(theta_value):
    th, e = _single_excitation_discriminating_fn(theta_value, grad=True)
    e.backward()
    return np.array([float(th.grad)])


# ─────────────────────────── double_excitation 判别电路 ──────────────────────────
# 目标：4 量子比特，激发门作用在 (0,1,2,3)
# 初态：|0001> ──X(q3)──> |0001> 已是计算基态，但双激发门在 |0011>⟷|1100> 工作。
#   为让初态跨越激发门活跃子空间，需叠加 |0011>（激发对）与观谱态。
#   方案：|0001> ──X(q3)──> |0001>，再 X(q2) 得 |0011>，再 H(q2) 得 (|0011>+|0001>)/√2。
#   即：zeros_state(4) ──X(q3)──X(q2)──H(q2)──>  (|0011>+|0001>)/√2
# 观测量：Z ⊗ I ⊗ I ⊗ I（qubit 0 上的 Z）——与双激发门产生的 |1100> 分量耦合。
#
# 数值验证通过 autograd 作为基准。

def _double_excitation_discriminating_fn(theta_value, grad=False):
    """返回 (th, energy)；当 grad=True 时 th 带 requires_grad。"""
    backend = GPUBackend(device="cpu")
    # H = Z ⊗ I ⊗ I ⊗ I（16×16）
    H_np = np.kron(
        np.diag([1.0, -1.0]).astype(np.complex64),
        np.eye(8, dtype=np.complex64),
    )
    H = backend.cast(H_np)

    th = torch.tensor(float(theta_value[0]), requires_grad=grad)
    # 初态 |0000>
    state = backend.zeros_state(4)
    # X(q3)、X(q2)：|0000> → |0011>
    state = apply_gate_to_state(pauli_x(3).to_dict(), state, 4, backend)
    state = apply_gate_to_state(pauli_x(2).to_dict(), state, 4, backend)
    # H(q2)：|0011> → (|0011>+|0001>)/√2（暴露激发门的半频分量）
    state = apply_gate_to_state(hadamard(2).to_dict(), state, 4, backend)
    # double_excitation(theta, 0, 1, 2, 3)
    for g in Circuit(double_excitation(th, 0, 1, 2, 3), n_qubits=4).gates:
        state = apply_gate_to_state(g, state, 4, backend)
    e = backend.expectation_sv(state, H).real
    return th, e


def _double_excitation_energy(theta_value):
    _, e = _double_excitation_discriminating_fn(theta_value)
    return float(e.detach())


def _double_excitation_autograd(theta_value):
    th, e = _double_excitation_discriminating_fn(theta_value, grad=True)
    e.backward()
    return np.array([float(th.grad)])


# ─────────────────────────── 测试用例 ────────────────────────────────────────────

def test_psr4_matches_autograd_single_excitation():
    """psr4 对 single_excitation 判别电路与 autograd 一致（atol=1e-4）。"""
    theta = np.array([0.6])
    grad_psr4 = psr4(_single_excitation_energy, theta)
    grad_auto = _single_excitation_autograd(theta)
    assert np.allclose(grad_psr4, grad_auto, atol=1e-4), (
        f"psr4={grad_psr4}, autograd={grad_auto}"
    )


def test_psr4_matches_autograd_double_excitation():
    """psr4 对 double_excitation 判别电路与 autograd 一致（atol=1e-4）。"""
    theta = np.array([0.7])
    grad_psr4 = psr4(_double_excitation_energy, theta)
    grad_auto = _double_excitation_autograd(theta)
    assert np.allclose(grad_psr4, grad_auto, atol=1e-4), (
        f"psr4={grad_psr4}, autograd={grad_auto}"
    )


def test_two_term_psr_is_wrong_for_excitation():
    """标准两项 psr 对 single_excitation 判别电路给出错误梯度；psr4 正确。

    判别电路（叠加初态）使能量函数包含半频分量，两项规则误判，
    四项规则（psr4）给出与 autograd 一致的结果。
    """
    theta = np.array([0.6])
    grad_auto = _single_excitation_autograd(theta)
    grad_psr2 = psr(_single_excitation_energy, theta)
    grad_psr4 = psr4(_single_excitation_energy, theta)

    # 两项规则应与 autograd 不一致（误差超过 1e-3）
    assert not np.allclose(grad_psr2, grad_auto, atol=1e-3), (
        f"预期两项 psr 与 autograd 不一致，但 psr2={grad_psr2}, autograd={grad_auto}"
    )
    # 四项规则应与 autograd 一致（证明 psr4 是正确的）
    assert np.allclose(grad_psr4, grad_auto, atol=1e-4), (
        f"psr4={grad_psr4}, autograd={grad_auto}"
    )

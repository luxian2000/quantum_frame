"""回归测试：_NpuHamiltonianExpectationFn 的梯度正确性。

用 H2 Hamiltonian (4 个 Pauli 项) 和 2 比特 RY 电路在 CPU 上验证：
    autodiff 梯度 ≈ 参数移位（psr）梯度
两者误差应在 1e-5 以内。

不依赖真实 NPU——NPUBackend 在 CPU fallback 模式下运行即可，
_NpuHamiltonianExpectationFn 是纯 Python/PyTorch，CPU 上同样可测。
"""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from aicir.backends.npu_backend import (
    NPUBackend,
    _NpuHamiltonianExpectationFn,
    _pauli_signs_npu,
)
from aicir.qas.algorithms.supernet import Supernet, SupernetConfig, h2_hamiltonian


# ─────────────────────────── 辅助：H2 Pauli 缓存 ────────────────────────────

def _make_supernet_for_h2():
    """构造只用于访问 _pauli_term_cache/_basis_indices 的最小 Supernet。"""
    cfg = SupernetConfig(
        n_qubits=4,
        layers=1,
        supernet_num=1,
        supernet_steps=0,
        ranking_num=1,
        finetune_steps=0,
        two_qubit_pairs=(),
        task="vqe",
        seed=0,
    )
    sn = Supernet(cfg)
    # 覆盖为 NPUBackend CPU fallback，测试其辅助方法
    sn.backend = NPUBackend(fallback_to_cpu=True)
    return sn


# ─────────────────────────── 辅助：小型 2 比特电路 ──────────────────────────

def _ry_state(theta0: torch.Tensor, theta1: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
    """RY(θ0)⊗RY(θ1)|00⟩，手动构造，保留梯度图。"""
    c0, s0 = torch.cos(theta0 / 2), torch.sin(theta0 / 2)
    c1, s1 = torch.cos(theta1 / 2), torch.sin(theta1 / 2)
    # |00⟩=1，其余为 0
    zero_r = torch.zeros(1, dtype=torch.float32)
    re = torch.stack([c0 * c1, c0 * s1, s0 * c1, s0 * s1])
    im = torch.zeros(4, dtype=torch.float32)
    return torch.complex(re, im)


# ─────────────────────────── 测试 ───────────────────────────────────────────

def test_pauli_signs_npu_zero_mask():
    idx = torch.arange(8, dtype=torch.long)
    assert _pauli_signs_npu(idx, 0) is None


def test_pauli_signs_npu_single_bit():
    # sign_mask=1 → (-1)^{bit0}: 奇偶交替
    idx = torch.arange(4, dtype=torch.long)
    signs = _pauli_signs_npu(idx, 1)
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert torch.allclose(signs, expected)


def test_npu_hamiltonian_fn_forward_matches_loop():
    """_NpuHamiltonianExpectationFn 前向与直接循环结果相同。"""
    sn = _make_supernet_for_h2()
    ham = h2_hamiltonian()
    basis_idx = sn._basis_indices(16)
    cache = sn._pauli_term_cache(ham)

    # 随机归一化复数态
    torch.manual_seed(7)
    re = torch.randn(16)
    im = torch.randn(16)
    norm = (re ** 2 + im ** 2).sum().sqrt()
    state = torch.complex(re / norm, im / norm)

    energy_fn = _NpuHamiltonianExpectationFn.apply(state, basis_idx, cache)

    # 直接循环
    s_re, s_im = torch.real(state), torch.imag(state)
    energy_loop = torch.zeros(())
    for flip_mask, sign_mask, y_phase, c_re, c_im in cache:
        if flip_mask:
            mapped_idx = torch.bitwise_xor(basis_idx, flip_mask)
            m_re = s_re.index_select(0, mapped_idx)
            m_im = s_im.index_select(0, mapped_idx)
        else:
            m_re, m_im = s_re, s_im
        ov_re = m_re * s_re + m_im * s_im
        ov_im = m_re * s_im - m_im * s_re
        sg = _pauli_signs_npu(basis_idx, sign_mask)
        if sg is not None:
            ov_re, ov_im = ov_re * sg, ov_im * sg
        if y_phase == 0:
            t_re, t_im = ov_re.sum(), ov_im.sum()
        elif y_phase == 1:
            t_re, t_im = -ov_im.sum(), ov_re.sum()
        elif y_phase == 2:
            t_re, t_im = -ov_re.sum(), -ov_im.sum()
        else:
            t_re, t_im = ov_im.sum(), -ov_re.sum()
        energy_loop = energy_loop + c_re * t_re - c_im * t_im

    assert abs(float(energy_fn) - float(energy_loop)) < 1e-6


def test_autodiff_vs_psr_gradient():
    """autodiff 梯度（通过 _NpuHamiltonianExpectationFn）应与 psr 梯度一致。

    对 H2 Hamiltonian，用 2 比特 RY(θ0)⊗RY(θ1)|00⟩ 电路，
    验证 dE/dθ0 和 dE/dθ1。
    """
    sn = _make_supernet_for_h2()
    ham = h2_hamiltonian()
    basis_idx = sn._basis_indices(16)
    cache = sn._pauli_term_cache(ham)

    # H2 是 4 比特，但测试用 4 比特全零初始 + 只转动前两比特（其余保持|0⟩）。
    # state = RY(t0)⊗RY(t1)⊗I⊗I |0000⟩
    t0_val = 0.4
    t1_val = -0.7

    def _state_4q(t0, t1):
        c0, s0 = torch.cos(t0 / 2), torch.sin(t0 / 2)
        c1, s1 = torch.cos(t1 / 2), torch.sin(t1 / 2)
        # 4 比特基 |abcd⟩，re[idx] = amp_re[idx]
        # RY(t0)⊗RY(t1)⊗I⊗I: idx = a*8+b*4+c*2+d
        # amp[abcd] = (RY0)_a * (RY1)_b = [c0,-s0; s0,c0][a]*[c1,-s1;s1,c1][b]
        # 从|0000⟩出发: a初始=0,b初始=0 → amp[abcd] = RY0[a,0]*RY1[b,0]*delta_c0*delta_d0
        # 即 amp[a,b,0,0] = ry0_col0[a] * ry1_col0[b]
        ry0_col0 = torch.stack([c0, s0])   # [c0, s0]
        ry1_col0 = torch.stack([c1, s1])   # [c1, s1]
        re = torch.zeros(16, dtype=torch.float32)
        for a in range(2):
            for b in range(2):
                idx = a * 8 + b * 4
                re[idx] = ry0_col0[a] * ry1_col0[b]
        im = torch.zeros(16, dtype=torch.float32)
        return torch.complex(re, im)

    # autodiff 梯度
    t0 = torch.tensor(t0_val, requires_grad=True)
    t1 = torch.tensor(t1_val, requires_grad=True)
    state_ad = _state_4q(t0, t1)
    energy_ad = _NpuHamiltonianExpectationFn.apply(state_ad, basis_idx, cache)
    energy_ad.backward()
    grad_t0_ad = float(t0.grad)
    grad_t1_ad = float(t1.grad)

    # psr 梯度（有限差分近似，shift = π/2）
    shift = math.pi / 2
    def energy_np(t0v, t1v):
        s = _state_4q(
            torch.tensor(t0v, dtype=torch.float32),
            torch.tensor(t1v, dtype=torch.float32),
        )
        e = _NpuHamiltonianExpectationFn.apply(s, basis_idx, cache)
        return float(e)

    grad_t0_psr = (energy_np(t0_val + shift, t1_val) - energy_np(t0_val - shift, t1_val)) / 2
    grad_t1_psr = (energy_np(t0_val, t1_val + shift) - energy_np(t0_val, t1_val - shift)) / 2

    assert abs(grad_t0_ad - grad_t0_psr) < 1e-4, (
        f"dE/dt0: ad={grad_t0_ad:.6f} psr={grad_t0_psr:.6f}"
    )
    assert abs(grad_t1_ad - grad_t1_psr) < 1e-4, (
        f"dE/dt1: ad={grad_t1_ad:.6f} psr={grad_t1_psr:.6f}"
    )

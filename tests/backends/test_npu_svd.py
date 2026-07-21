import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.backends.npu_backend import _real_embedding_svd, is_npu_available


def _recon(u, s, vh):
    return u @ torch.diag(s.to(u.dtype)) @ vh


def test_real_embedding_svd_reconstructs_cpu():
    torch.manual_seed(0)
    for shape in [(4, 4), (8, 4), (4, 8), (16, 16)]:
        a = torch.randn(*shape, dtype=torch.complex64)
        u, s, vh = _real_embedding_svd(a)
        p = min(shape)
        assert u.shape == (shape[0], p) and s.shape == (p,) and vh.shape == (p, shape[1])
        # 重建 A ≈ U diag(S) Vh（规范无关）
        assert torch.allclose(_recon(u, s, vh), a, atol=1e-4)
        # 奇异值与 CPU 参考一致，降序
        ref = torch.linalg.svdvals(a.to(torch.complex128)).to(torch.float32)
        assert torch.allclose(s, ref, atol=1e-4)
        assert torch.all(torch.diff(s) <= 1e-5)


def test_real_embedding_svd_differentiable_cpu():
    torch.manual_seed(1)
    a = torch.randn(6, 4, dtype=torch.complex64, requires_grad=True)
    _, s, _ = _real_embedding_svd(a)
    s.sum().backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad.abs()).all())
    # 交叉核对：奇异值和对输入的梯度应与 CPU 原生复数 SVD 一致
    a2 = a.detach().clone().requires_grad_(True)
    torch.linalg.svdvals(a2).sum().backward()
    assert torch.allclose(a.grad, a2.grad, atol=1e-3)


def test_real_embedding_svd_degenerate_cpu():
    torch.manual_seed(2)

    def _rand_unitary(k):
        q, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.complex128))
        return q

    u0, v0 = _rand_unitary(4), _rand_unitary(4)
    s0 = torch.tensor([3.0, 3.0, 2.0, 1.0], dtype=torch.float64)  # 重复奇异值
    a = (u0 @ torch.diag(s0).to(torch.complex128) @ v0.conj().T).to(torch.complex64)
    u, s, vh = _real_embedding_svd(a)
    # 简并下仍须 A ≈ U diag(S) Vh（旧 stride-2 方案在此误差 ~0.386）
    assert torch.allclose(u @ torch.diag(s.to(u.dtype)) @ vh, a, atol=1e-3)
    ref = torch.linalg.svdvals(a.to(torch.complex128)).to(torch.float32)
    assert torch.allclose(s, ref, atol=1e-3)
    # U 列正交归一
    assert torch.allclose(u.conj().T @ u, torch.eye(4, dtype=u.dtype), atol=1e-3)


def test_real_embedding_svd_rank_deficient_64_by_64():
    """QRC reduced density matrices are 64x64 and commonly rank deficient."""
    torch.manual_seed(9)
    factors = torch.randn(64, 8, dtype=torch.complex64)
    matrix = factors @ factors.conj().T

    u, s, vh = _real_embedding_svd(matrix)

    assert u.shape == (64, 64)
    assert s.shape == (64,)
    assert vh.shape == (64, 64)
    assert torch.allclose(_recon(u, s, vh), matrix, atol=2e-3, rtol=2e-4)


@pytest.mark.skipif(not is_npu_available(), reason="需要真实 NPU")
def test_npu_svd_on_device():
    from aicir.backends.npu_backend import NPUBackend

    torch.ones(1).npu()  # 预热，规避 ACL 初始化毒化
    bk = NPUBackend()
    re = torch.randn(8, 4, device="npu:0")
    im = torch.randn(8, 4, device="npu:0")
    a = torch.complex(re, im).detach().requires_grad_(True)
    u, s, vh = bk.svd(a)
    recon = u @ torch.diag(s.to(u.dtype)) @ vh
    assert float((recon - a).abs().max().cpu()) < 1e-3
    s.sum().backward()
    assert bool(torch.isfinite(a.grad.abs()).all())

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

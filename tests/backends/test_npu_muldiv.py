import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.backends.npu_backend import is_npu_available


@pytest.mark.skipif(not is_npu_available(), reason="需要真实 NPU")
def test_npu_mul_div_vs_cpu_and_grad():
    from aicir.backends.npu_backend import NPUBackend

    torch.ones(1).npu()
    bk = NPUBackend()
    a_np = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    b_np = (np.random.randn(2, 1) + 1j * np.random.randn(2, 1)).astype(np.complex64)  # 广播
    a = bk.cast(a_np).detach().requires_grad_(True)
    b = bk.cast(b_np)
    m = bk.mul(a, b)
    assert np.allclose(bk.to_numpy(m), a_np * b_np, atol=1e-4)
    d = bk.div(a, bk.cast(b_np + 1.0))
    assert np.allclose(bk.to_numpy(d), a_np / (b_np + 1.0), atol=1e-4)
    m.abs().sum().backward()
    assert bool(torch.isfinite(a.grad.abs()).all())

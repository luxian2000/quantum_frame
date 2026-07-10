import numpy as np
import pytest

from aicir.backends import NumpyBackend


def test_numpy_mul_div():
    bk = NumpyBackend()
    a = bk.cast(np.array([[1 + 2j, 3]], dtype=np.complex64))
    b = bk.cast(np.array([[2 + 0j, 1j]], dtype=np.complex64))
    assert np.allclose(bk.to_numpy(bk.mul(a, b)), bk.to_numpy(a) * bk.to_numpy(b))
    assert np.allclose(bk.to_numpy(bk.div(a, b)), bk.to_numpy(a) / bk.to_numpy(b))


def test_numpy_mul_broadcast():
    bk = NumpyBackend()
    vh = bk.cast(np.arange(6, dtype=np.complex64).reshape(2, 3))
    s = bk.cast(np.array([[2 + 0j], [3 + 0j]], dtype=np.complex64))  # (2,1) 广播
    assert np.allclose(bk.to_numpy(bk.mul(vh, s)), bk.to_numpy(vh) * bk.to_numpy(s))


def test_gpu_mul_div_grad():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    a = bk.cast(np.array([[1 + 1j, 2 - 1j]], dtype=np.complex64)).requires_grad_(True)
    b = bk.cast(np.array([[2 + 0j, 1 + 1j]], dtype=np.complex64))
    (bk.mul(a, b).abs().sum() + bk.div(a, b).abs().sum()).backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad.abs()).all())

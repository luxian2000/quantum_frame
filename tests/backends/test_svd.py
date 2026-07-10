import numpy as np
import pytest

from aicir.backends import NumpyBackend


def test_numpy_svd_reconstructs():
    bk = NumpyBackend()
    m = bk.cast(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.complex64))
    u, s, vh = bk.svd(m)
    u, s, vh = bk.to_numpy(u), bk.to_numpy(s), bk.to_numpy(vh)
    recon = u @ np.diag(s) @ vh
    assert np.allclose(recon, bk.to_numpy(m), atol=1e-5)
    assert u.shape == (3, 2) and s.shape == (2,) and vh.shape == (2, 2)
    assert np.all(np.diff(s.real) <= 1e-6)  # 降序


def test_gpu_svd_reconstructs():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    m = bk.cast(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64))
    u, s, vh = bk.svd(m)
    recon = torch.matmul(torch.matmul(u, torch.diag(s).to(u.dtype)), vh)
    assert torch.allclose(recon, m, atol=1e-4)


def test_npu_svd_cpu_device_falls_back_to_parent():
    """NPUBackend.svd() on a non-npu device (here cpu) skips real-embedding and
    falls back to GPUBackend.svd (torch.linalg.svd directly) — real-embedding only
    triggers when _is_npu_complex(matrix) is True (device.type == 'npu')."""
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    bk = NPUBackend(device="cpu")
    m = bk.cast(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    u, s, vh = bk.svd(m)
    recon = torch.matmul(torch.matmul(u, torch.diag(s).to(u.dtype)), vh)
    assert torch.allclose(recon, m, atol=1e-4)

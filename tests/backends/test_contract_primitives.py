import numpy as np
from aicir.backends import NumpyBackend


def test_numpy_tensordot_matches_numpy():
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_numpy_transpose_reshape_conj():
    bk = NumpyBackend()
    a = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4)
    ca = bk.cast(a)
    assert np.allclose(bk.to_numpy(bk.transpose(ca, (2, 0, 1))), np.transpose(a, (2, 0, 1)), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.reshape(ca, (6, 4))), a.reshape(6, 4), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.conj(ca)), np.conj(a), atol=1e-5)


def test_tensordot_via_matmul_matches_numpy():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_tensordot_via_matmul_outer_product():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.array([1.0, 2.0], dtype=np.complex64)
    b = np.array([3.0, 4.0, 5.0], dtype=np.complex64)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([], []))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([], [])), atol=1e-5)


def test_gpu_tensordot_matches_numpy():
    import pytest
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-4)

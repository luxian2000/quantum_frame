import numpy as np
import pytest

from aicir.backends import NumpyBackend


def test_numpy_take_drops_axis():
    bk = NumpyBackend()
    a = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4)
    out = bk.to_numpy(bk.take(bk.cast(a), 1, 2))
    assert np.allclose(out, a[:, 2, :], atol=1e-5)
    assert out.shape == (2, 4)


def test_numpy_add():
    bk = NumpyBackend()
    a = np.array([1 + 2j, 3 + 4j])
    b = np.array([5 + 6j, 7 + 8j])
    assert np.allclose(bk.to_numpy(bk.add(bk.cast(a), bk.cast(b))), a + b, atol=1e-5)


def test_gpu_take_add():
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    a = (np.arange(8) + 1j * np.arange(8)).reshape(2, 2, 2)
    out = bk.to_numpy(bk.take(bk.cast(a), 0, 1))
    assert np.allclose(out, a[1], atol=1e-4)
    s = bk.to_numpy(bk.add(bk.cast(a), bk.cast(a)))
    assert np.allclose(s, 2 * a, atol=1e-4)


def test_base_take_add_not_implemented():
    from aicir.backends.base import Backend
    class Dummy(Backend):
        pass
    # Backend 抽象方法较多，直接实例化不可行时改为检查基类方法体抛 NotImplementedError
    import aicir.backends.base as base_mod
    import inspect
    src = inspect.getsource(base_mod.Backend.take)
    assert "NotImplementedError" in src
    src = inspect.getsource(base_mod.Backend.add)
    assert "NotImplementedError" in src

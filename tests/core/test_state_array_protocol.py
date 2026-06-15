"""State 支持 numpy 数组协议（__array__），便于与裸数组互操作。"""

import numpy as np

from aicir import NumpyBackend, State


def test_asarray_vector_shape():
    s = State.from_array([1.0, 0.0, 0.0, 0.0], backend=NumpyBackend())  # 2 qubits
    arr = np.asarray(s)
    assert arr.shape == (4,)
    np.testing.assert_allclose(arr, [1, 0, 0, 0], atol=1e-6)


def test_asarray_density_shape():
    rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex64)
    s = State.from_matrix(rho)
    arr = np.asarray(s)
    assert arr.shape == (4, 4)
    np.testing.assert_allclose(arr, rho, atol=1e-6)


def test_allclose_between_states():
    a = State.from_array([1.0, 0.0], backend=NumpyBackend())
    b = State.from_array([1.0, 0.0], backend=NumpyBackend())
    assert np.allclose(a, b)


def test_asarray_dtype_arg():
    s = State.from_array([1.0, 0.0], backend=NumpyBackend())
    arr = np.asarray(s, dtype=np.complex128)
    assert arr.dtype == np.complex128


def test_asarray_returns_copy_not_view():
    s = State.from_array([1.0, 0.0], backend=NumpyBackend())
    arr = np.asarray(s)
    arr[0] = 0.0
    # 修改返回数组不应污染原 State
    np.testing.assert_allclose(s.array, [1.0, 0.0], atol=1e-6)

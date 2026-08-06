"""NumpyBackend 局部门应用的 strided in-place 快路径。

背景：`apply_statevector_local` 原先物化 int64 索引数组做 gather/scatter。那是
**昇腾的约束**——`aclnnComplex` 限 8 维，NPU 上无法把复数态 reshape 成 `(2,)*n`，
只能走扁平位置换 gather。但 NumPy 没有这个限制，CPU 后端不该替 NPU 交税。

快路径把态视作 `(2^a, 2, 2^(b-a-1), 2, 2^(n-1-b))` 的**跨步视图**（秩恒为
`2*len(axes)+1`，与比特数无关，故不触碰 numpy 的维度上限），直接按分块读写，
不物化任何索引数组。

本文件以**原 gather 实现为正确性 oracle**：两条路径必须逐位相同。
"""

import numpy as np
import pytest

from aicir.backends.numpy_backend import NumpyBackend, _apply_local_gather


def _random_state(n_qubits, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return vec.astype(dtype).reshape(dim, 1)


def _random_unitary(dim, dtype, seed=0):
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    return q.astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5, 8, 11])
def test_single_qubit_matches_gather_oracle_on_every_axis(n_qubits, dtype):
    backend = NumpyBackend(dtype=dtype)
    state = _random_state(n_qubits, dtype, seed=n_qubits)
    matrix = _random_unitary(2, dtype, seed=n_qubits)

    for axis in range(n_qubits):
        fast = backend.apply_statevector_local(state, matrix, (axis,), n_qubits)
        oracle = _apply_local_gather(state, matrix, (axis,), n_qubits, dtype)
        np.testing.assert_allclose(
            np.asarray(fast).reshape(-1),
            np.asarray(oracle).reshape(-1),
            rtol=1e-6 if dtype is np.complex64 else 1e-12,
            atol=1e-6 if dtype is np.complex64 else 1e-12,
            err_msg=f"n={n_qubits} axis={axis} dtype={dtype}",
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("n_qubits", [2, 3, 5, 8])
def test_two_qubit_matches_gather_oracle_including_reversed_axis_order(n_qubits, dtype):
    """必须覆盖 axes=(b,a) 且 b>a 的情形——局部矩阵的比特序跟随**给定顺序**，
    不是排序后的顺序。只测 a<b 会漏掉这个反序 bug。"""

    backend = NumpyBackend(dtype=dtype)
    state = _random_state(n_qubits, dtype, seed=n_qubits + 100)
    matrix = _random_unitary(4, dtype, seed=n_qubits + 100)

    for a in range(n_qubits):
        for b in range(n_qubits):
            if a == b:
                continue
            fast = backend.apply_statevector_local(state, matrix, (a, b), n_qubits)
            oracle = _apply_local_gather(state, matrix, (a, b), n_qubits, dtype)
            np.testing.assert_allclose(
                np.asarray(fast).reshape(-1),
                np.asarray(oracle).reshape(-1),
                rtol=1e-6 if dtype is np.complex64 else 1e-12,
                atol=1e-6 if dtype is np.complex64 else 1e-12,
                err_msg=f"n={n_qubits} axes=({a},{b}) dtype={dtype}",
            )


class TestContractPreserved:
    """快路径不得改变既有契约。"""

    def test_input_state_is_not_mutated(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(6, np.complex128, seed=3)
        before = np.array(state, copy=True)
        backend.apply_statevector_local(state, _random_unitary(2, np.complex128), (2,), 6)
        np.testing.assert_array_equal(np.asarray(state), before)

    def test_returns_column_vector_shape(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(5, np.complex128)
        out = backend.apply_statevector_local(state, _random_unitary(2, np.complex128), (0,), 5)
        assert np.asarray(out).shape == (1 << 5, 1)

    def test_output_dtype_follows_backend(self):
        for dtype in (np.complex64, np.complex128):
            backend = NumpyBackend(dtype=dtype)
            state = _random_state(4, dtype)
            out = backend.apply_statevector_local(state, _random_unitary(2, dtype), (1,), 4)
            assert np.asarray(out).dtype == dtype

    def test_more_than_two_axes_still_returns_none(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(4, np.complex128)
        assert backend.apply_statevector_local(state, np.eye(8), (0, 1, 2), 4) is None

    def test_empty_axes_returns_state_unchanged(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(4, np.complex128)
        assert backend.apply_statevector_local(state, np.eye(2), (), 4) is state

    def test_duplicate_axes_raise(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(4, np.complex128)
        with pytest.raises(ValueError):
            backend.apply_statevector_local(state, np.eye(4), (1, 1), 4)

    def test_out_of_range_axis_raises(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(4, np.complex128)
        with pytest.raises(ValueError):
            backend.apply_statevector_local(state, np.eye(2), (4,), 4)

    def test_matrix_shape_mismatch_raises(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(4, np.complex128)
        with pytest.raises(ValueError):
            backend.apply_statevector_local(state, np.eye(4), (0,), 4)


class TestHighQubitCountAvoidsDimensionLimit:
    """秩恒为 2*len(axes)+1，不随比特数增长，故不会撞 numpy 的维度上限。"""

    def test_works_well_beyond_numpy_max_dims(self):
        n_qubits = 24  # (2,)*24 的 reshape 对 NPU 是禁区，这里必须照常工作
        backend = NumpyBackend(dtype=np.complex64)
        dim = 1 << n_qubits
        state = np.zeros((dim, 1), dtype=np.complex64)
        state[0, 0] = 1.0
        hadamard = (np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2))
        out = backend.apply_statevector_local(state, hadamard, (0,), n_qubits)
        flat = np.asarray(out).reshape(-1)
        assert flat[0] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
        assert flat[dim // 2] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


class TestNPUKeepsGather:
    """NPU 必须继续走 gather——昇腾的 8 维上限没有消失。"""

    def test_npu_backend_does_not_inherit_numpy_fast_path(self):
        pytest.importorskip("torch")
        from aicir.backends.npu_backend import NPUBackend

        assert not issubclass(NPUBackend, NumpyBackend)

    def test_gather_oracle_remains_available(self):
        # 它既是 NPU 的策略参照，也是本文件的 oracle，不能被顺手删掉。
        assert callable(_apply_local_gather)

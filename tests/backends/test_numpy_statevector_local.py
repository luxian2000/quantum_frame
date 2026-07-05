import numpy as np

from aicir.backends.numpy_backend import NumpyBackend


def _random_state(n_qubits: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec = (vec / np.linalg.norm(vec)).astype(np.complex64)
    return vec.reshape(-1, 1)


def _random_unitary(dim: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = phases / np.where(np.abs(phases) == 0, 1.0, np.abs(phases))
    return (q * phases).astype(np.complex64)


def _reference_apply(state: np.ndarray, local: np.ndarray, axes: tuple[int, ...], n_qubits: int) -> np.ndarray:
    axes = tuple(int(axis) for axis in axes)
    dim_local = 1 << len(axes)
    out = np.empty_like(state.reshape(-1))
    flat = state.reshape(-1)
    for basis in range(1 << n_qubits):
        bits = [(basis >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_col = 0
        for axis in axes:
            local_col = (local_col << 1) | bits[axis]
        value = 0.0 + 0.0j
        for local_row in range(dim_local):
            row_bits = bits.copy()
            for pos, axis in enumerate(axes):
                row_bits[axis] = (local_row >> (len(axes) - 1 - pos)) & 1
            row_index = 0
            for bit in row_bits:
                row_index = (row_index << 1) | bit
            value += local[local_col, local_row] * flat[row_index]
        out[basis] = value
    return out.reshape(-1, 1)


def test_apply_statevector_local_one_qubit_matches_reference_with_small_chunks():
    backend = NumpyBackend()
    backend._statevector_chunk_size = 5
    state = _random_state(5)
    local = _random_unitary(2)

    actual = backend.apply_statevector_local(state, local, axes=(2,), n_qubits=5)
    expected = _reference_apply(state, local, axes=(2,), n_qubits=5)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (32, 1)


def test_apply_statevector_local_two_qubit_matches_reference_non_adjacent_axes():
    backend = NumpyBackend()
    backend._statevector_chunk_size = 7
    state = _random_state(5, seed=23)
    local = _random_unitary(4, seed=29)

    actual = backend.apply_statevector_local(state, local, axes=(3, 0), n_qubits=5)
    expected = _reference_apply(state, local, axes=(3, 0), n_qubits=5)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (32, 1)


def test_apply_statevector_local_returns_none_for_three_qubit_gate():
    backend = NumpyBackend()
    state = _random_state(4)
    local = np.eye(8, dtype=np.complex64)

    assert backend.apply_statevector_local(state, local, axes=(0, 1, 2), n_qubits=4) is None

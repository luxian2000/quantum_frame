import numpy as np
from aicir.core.state import State
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.measure import projector


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_basis_change_x_maps_plus_to_zero():
    plus = _sv([1, 1])
    plus = State(plus.backend.cast((plus.to_numpy() / np.sqrt(2))), 1, plus.backend)
    out = projector.pauli_basis_change(plus, [0], "X", inverse=False)
    v = out.to_numpy().reshape(-1)
    assert np.allclose(v, [1, 0], atol=1e-6)


def test_basis_change_round_trip_identity():
    psi = _sv([0.5, 0.5, 0.5, 0.5])
    fwd = projector.pauli_basis_change(psi, [0, 1], "Y", inverse=False)
    back = projector.pauli_basis_change(fwd, [0, 1], "Y", inverse=True)
    assert np.allclose(back.to_numpy().reshape(-1), psi.to_numpy().reshape(-1), atol=1e-6)


def test_basis_change_z_is_noop():
    psi = _sv([0.6, 0.8])
    out = projector.pauli_basis_change(psi, [0], "Z", inverse=False)
    assert np.allclose(out.to_numpy().reshape(-1), [0.6, 0.8], atol=1e-6)

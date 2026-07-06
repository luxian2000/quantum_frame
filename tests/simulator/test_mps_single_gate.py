import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState

_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)


def test_apply_x_flips_qubit0():
    bk = NumpyBackend()
    mps = MPSState.zero_state(2, bk)
    mps._apply_one_site(bk.cast(_X), 0)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0  # qubit0=MSB -> |10>
    assert np.allclose(sv, expected, atol=1e-6)


def test_apply_h_superposition():
    bk = NumpyBackend()
    mps = MPSState.zero_state(1, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    assert np.allclose(sv, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6)

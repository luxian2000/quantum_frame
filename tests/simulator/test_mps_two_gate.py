import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState

_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
)


def test_bell_state_exact():
    bk = NumpyBackend()
    mps = MPSState.zero_state(2, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    op4 = bk.reshape(bk.cast(_CNOT), (2, 2, 2, 2))
    mps._apply_two_site(op4, 0, truncate=True)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(4, dtype=complex)
    expected[0b00] = expected[0b11] = 1 / np.sqrt(2)
    assert np.allclose(sv, expected, atol=1e-6)
    assert mps.truncation_error < 1e-9  # 无截断


def test_center_move_preserves_state():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    op4 = bk.reshape(bk.cast(_CNOT), (2, 2, 2, 2))
    mps._apply_two_site(op4, 0, truncate=True)  # 纠缠 0-1
    before = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    mps._ensure_center(2)
    mps._ensure_center(0)
    after = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    assert np.allclose(before, after, atol=1e-6)

import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState

_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
)


def test_cnot_non_adjacent_exact():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    mps.apply_two_qubit(bk.cast(_CNOT), [0, 2])  # 非相邻 control=0,target=2
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0b000] = expected[0b101] = 1 / np.sqrt(2)  # |000>+|101>
    assert np.allclose(sv, expected, atol=1e-6)


def test_reversed_axes_control_gt_target():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 2)
    mps.apply_two_qubit(bk.cast(_CNOT), [2, 0])  # control=2 (MSB), target=0
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0b000] = expected[0b101] = 1 / np.sqrt(2)
    assert np.allclose(sv, expected, atol=1e-6)


def test_swap_bookkeeping_restored_state():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._swap_adjacent(0)
    assert mps.logical_at[0] == 1 and mps.logical_at[1] == 0
    assert mps.site_of[0] == 1 and mps.site_of[1] == 0

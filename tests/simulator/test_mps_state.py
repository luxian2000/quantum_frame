import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState


def test_zero_state_statevector():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-6)


def test_zero_state_shapes_and_bookkeeping():
    bk = NumpyBackend()
    mps = MPSState.zero_state(4, bk)
    assert mps.n_qubits == 4
    assert len(mps.tensors) == 4
    for t in mps.tensors:
        assert np.asarray(bk.to_numpy(t)).shape == (1, 2, 1)
    assert mps.logical_at == [0, 1, 2, 3]
    assert mps.site_of == [0, 1, 2, 3]
    assert mps.truncation_error == 0.0

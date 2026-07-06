import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit, measure
from aicir import hadamard, cnot, rx
from aicir.measure import Measure
from aicir.noise import NoiseModel


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    return c


def test_mps_matches_statevector_method():
    bk = NumpyBackend()
    c = _circ(9)
    sv = Measure(bk).run(c, shots=None, return_state=True, method="statevector")
    mp = Measure(bk).run(c, shots=None, return_state=True, method="mps")
    a = np.asarray(sv.state.to_numpy()).reshape(-1)
    b = np.asarray(mp.state.to_numpy()).reshape(-1)
    assert np.allclose(a, b, atol=1e-5)


def test_mps_rejects_noise():
    bk = NumpyBackend()
    c = _circ(1)
    c.noise_model = NoiseModel()
    with pytest.raises(ValueError):
        Measure(bk).run(c, method="mps")


def test_mps_rejects_embedded_measure():
    bk = NumpyBackend()
    c = Circuit(n_qubits=2)
    c.append(measure(0))
    with pytest.raises(ValueError):
        Measure(bk).run(c, method="mps")


def test_mps_rejects_nonempty_snap():
    bk = NumpyBackend()
    c = _circ(2)
    with pytest.raises(ValueError):
        Measure(bk).run(c, snap=[0], method="mps")


def test_mps_rejects_initial_state():
    bk = NumpyBackend()
    c = _circ(3)
    psi = bk.zeros_state(4)
    with pytest.raises(ValueError):
        Measure(bk).run(c, initial_state=psi, method="mps")

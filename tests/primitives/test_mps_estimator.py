import numpy as np

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import cnot, rx, rz
from aicir.core.operators import Hamiltonian
from aicir.primitives import StatevectorEstimator, MPSEstimator


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    for q in range(4):
        c.append(rz(q, float(rng.uniform(0, np.pi))))
    return c


_H = Hamiltonian([("ZZII", -1.0), ("IZZI", -1.0), ("IIZZ", -1.0), ("XIII", 0.5)])


def test_run_matches_statevector():
    bk = NumpyBackend()
    c = _circ(4)
    got = MPSEstimator(backend=bk).run(c, _H)
    ref = StatevectorEstimator(bk).run(c, _H).value
    assert abs(got.value - ref) < 1e-5
    assert got.metadata["method"] == "mps"
    assert "truncation_error" in got.metadata


def test_estimate_energy_contract():
    bk = NumpyBackend()
    c = _circ(6)
    res = MPSEstimator(backend=bk).estimate(c, _H)
    ref = StatevectorEstimator(bk).run(c, _H).value
    assert abs(res.energy - ref) < 1e-5


def test_shots_rejected():
    import pytest

    bk = NumpyBackend()
    c = _circ(1)
    with pytest.raises(ValueError):
        MPSEstimator(backend=bk).run(c, _H, shots=100)


def test_gradient_matches_statevector():
    bk = NumpyBackend()
    from aicir import Parameter

    theta = Parameter("t")
    c = Circuit(n_qubits=2)
    c.append(rx(theta, 0))
    c.append(cnot(1, [0]))
    H = Hamiltonian([("ZI", 1.0)])
    x = np.array([0.7])
    g_mps = MPSEstimator(backend=bk).gradient(c, H, parameter_values=x, method="psr").gradient
    g_ref = StatevectorEstimator(bk).gradient(c, H, parameter_values=x, method="psr").gradient
    assert np.allclose(g_mps, g_ref, atol=1e-5)

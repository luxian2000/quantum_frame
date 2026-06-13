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


def _bell():
    b = NumpyBackend()
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return State(b.cast(v.reshape(-1, 1)), 2, b)


def test_joint_zz_on_bell_is_deterministic_plus_and_keeps_entanglement():
    psi = _bell()
    rng = np.random.default_rng(0)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "Z", rng)
    assert lam == 1  # Bell 态 Z⊗Z 恒 +1
    v = out.to_numpy().reshape(-1)
    assert np.allclose(np.abs(v), [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], atol=1e-6)


def test_joint_measure_single_qubit_not_in_pa_eigenstate():
    b = NumpyBackend()
    v = np.array([1, 1, 1, -1], dtype=complex) / 2
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(1)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "X", rng)
    assert lam in (1, -1)
    red = out.partial_trace([0]).to_numpy().reshape(2, 2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    exp_x = np.real(np.trace(red @ X))
    assert abs(exp_x) < 1 - 1e-6


def test_joint_probs_match_born():
    b = NumpyBackend()
    v = np.array([np.sqrt(0.3), 0, 0, np.sqrt(0.7)], dtype=complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    p_plus, p_minus = projector.joint_parity_probs(psi, [0, 1], "Z")
    assert np.isclose(p_plus, 1.0, atol=1e-6)
    assert np.isclose(p_minus, 0.0, atol=1e-6)

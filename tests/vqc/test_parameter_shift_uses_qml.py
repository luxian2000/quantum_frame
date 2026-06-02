import numpy as np

import aicir.vqc.SSVQE as ssvqe_module
import aicir.vqc.VQD as vqd_module
import aicir.vqc.VQE as vqe_module
from aicir.vqc import BasicSSVQE, BasicVQD, BasicVQE


def _z_hamiltonian():
    return np.diag([1.0, -1.0])


def test_vqe_parameter_shift_gradient_uses_qml_psr(monkeypatch):
    calls = []

    def fake_psr(fn, params):
        calls.append((fn, params.copy()))
        return np.full_like(params, 0.25)

    monkeypatch.setattr(vqe_module, "psr", fake_psr)
    solver = BasicVQE(_z_hamiltonian(), depth=1)

    grad = solver.parameter_shift_gradient(np.array([[0.1]]))

    assert len(calls) == 1
    assert calls[0][0] == solver.energy
    assert np.array_equal(calls[0][1], np.array([[0.1]]))
    assert np.array_equal(grad, np.array([[0.25]]))


def test_ssvqe_parameter_shift_gradient_uses_qml_psr(monkeypatch):
    calls = []

    def fake_psr(fn, params):
        calls.append((fn, params.copy()))
        return np.full_like(params, -0.5)

    monkeypatch.setattr(ssvqe_module, "psr", fake_psr)
    solver = BasicSSVQE(_z_hamiltonian(), n_states=1, depth=1)

    grad = solver.parameter_shift_gradient(np.array([[0.2]]))

    assert len(calls) == 1
    assert calls[0][0] == solver.cost
    assert np.array_equal(calls[0][1], np.array([[0.2]]))
    assert np.array_equal(grad, np.array([[-0.5]]))


def test_vqd_parameter_shift_gradient_uses_qml_psr(monkeypatch):
    calls = []

    def fake_psr(fn, params):
        calls.append((fn, params.copy()))
        return np.full_like(params, 0.75)

    monkeypatch.setattr(vqd_module, "psr", fake_psr)
    solver = BasicVQD(_z_hamiltonian(), n_states=1, depth=1)

    grad = solver.parameter_shift_gradient(np.array([[0.3]]), prev_states=[], level=0)

    assert len(calls) == 1
    assert np.array_equal(calls[0][1], np.array([[0.3]]))
    assert np.array_equal(grad, np.array([[0.75]]))
    assert calls[0][0](np.array([[0.3]])) == solver.objective(np.array([[0.3]]), [], 0)

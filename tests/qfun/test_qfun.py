"""qfun（PennyLane 风格量子函数）测试（NEXT.md §5）。"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun

H_Z = Hamiltonian([("Z", 1.0)])


def _make_cost(**kw):
    @qfun(observable=H_Z, **kw)
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return c

    return cost


def test_call_returns_expectation():
    cost = _make_cost(differential="psr")
    assert np.isclose(cost(0.0), 1.0)
    assert np.isclose(cost(np.pi / 2), 0.0, atol=1e-7)
    assert np.isclose(cost(0.3), np.cos(0.3))


def test_grad_matches_analytic():
    cost = _make_cost(differential="psr")
    g = cost.grad(0.3)
    assert np.isscalar(g) or np.ndim(g) == 0
    assert np.isclose(g, -np.sin(0.3))


def test_grad_auto_differential():
    cost = _make_cost(differential="auto")
    assert np.isclose(cost.grad(0.3), -np.sin(0.3))


def test_body_must_return_circuit():
    @qfun(observable=H_Z)
    def bad(theta):
        return theta

    with pytest.raises(TypeError):
        bad(0.1)


def test_observable_required():
    with pytest.raises(ValueError):
        @qfun()
        def cost(theta):
            return Circuit(n_qubits=1)


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        qfun(observable=H_Z, device="quantum_foo")(lambda t: Circuit(n_qubits=1))

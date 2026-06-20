"""BasicVQE 接受外部 qfun 作为代价函数（NEXT.md §5）。"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun
from aicir.vqc import BasicVQE


def _make_cost():
    H = Hamiltonian([("Z", 1.0)])  # min <Z> = -1 at theta = pi

    @qfun(observable=H, differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta[0], 0))
        return c

    return cost


def test_vqe_with_external_qfun_minimizes():
    vqe = BasicVQE(cost=_make_cost(), n_params=1)
    res = vqe.run(max_iters=200, lr=0.3, init_params=np.array([0.1]))
    assert np.isclose(res.energy, -1.0, atol=1e-2)
    assert np.isclose(float(res.parameters.reshape(-1)[0]) % (2 * np.pi), np.pi, atol=1e-1)


def test_vqe_with_external_qfun_uses_qfun_gradient():
    vqe = BasicVQE(cost=_make_cost(), n_params=1)
    g = vqe.parameter_shift_gradient(np.array([0.3]))
    assert g.shape == (1,)
    assert np.allclose(g, [-np.sin(0.3)], atol=1e-7)


def test_vqe_cost_requires_n_params():
    with pytest.raises(ValueError):
        BasicVQE(cost=_make_cost())


def test_vqe_rejects_multi_observable_cost():
    H = Hamiltonian([("Z", 1.0)])
    X = Hamiltonian([("X", 1.0)])

    @qfun(observable=[H, X], differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta[0], 0))
        return c

    with pytest.raises(ValueError):
        BasicVQE(cost=cost, n_params=1)

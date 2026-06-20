"""BasicQAOA 接受外部 qfun 作为代价函数（NEXT.md §5）。"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun
from aicir.vqc import BasicQAOA


def _make_cost():
    H = Hamiltonian([("Z", 1.0)])  # min <Z> = -1 at theta = pi

    @qfun(observable=H, differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta[0], 0))  # 仅用 gamma；beta 不影响（退化代价）
        return c

    return cost


def test_qaoa_energy_delegates_to_qfun():
    qaoa = BasicQAOA(cost=_make_cost(), p=1)  # n_params = 2p = 2
    assert qaoa.n_params == 2
    assert np.isclose(qaoa.energy(np.array([0.3, 0.0])), np.cos(0.3))


def test_qaoa_with_external_qfun_minimizes():
    qaoa = BasicQAOA(cost=_make_cost(), p=1)
    res = qaoa.run(max_iters=200, lr=0.3, init_params=np.array([0.1, 0.0]))
    assert np.isclose(res.energy, -1.0, atol=1e-2)


def test_qaoa_cost_defaults_p_to_one():
    qaoa = BasicQAOA(cost=_make_cost())  # p 默认 1 → n_params = 2
    assert qaoa.p == 1
    assert qaoa.n_params == 2


def test_qaoa_rejects_multi_observable_cost():
    H = Hamiltonian([("Z", 1.0)])
    X = Hamiltonian([("X", 1.0)])

    @qfun(observable=[H, X], differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta[0], 0))
        return c

    with pytest.raises(ValueError):
        BasicQAOA(cost=cost, p=1)

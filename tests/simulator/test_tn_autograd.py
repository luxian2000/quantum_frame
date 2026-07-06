import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir import Circuit, Hamiltonian, ry
from aicir.backends import GPUBackend
from aicir.qml import psr
from aicir.simulator import tn_expectation


def test_tn_expectation_value_matches_reference():
    bk = GPUBackend(device="cpu")
    c = Circuit(ry(0.6, 0), n_qubits=1)
    val = float(tn_expectation(c, Hamiltonian([("Z", 1.0)]), backend=bk))
    assert np.isclose(val, np.cos(0.6), atol=1e-5)


def test_tn_expectation_is_differentiable():
    bk = GPUBackend(device="cpu")
    theta = torch.tensor(0.6, dtype=torch.float32, requires_grad=True)

    def build(t):
        return Circuit(ry(t, 0), n_qubits=1)

    energy = tn_expectation(build(theta), Hamiltonian([("Z", 1.0)]), backend=bk)
    energy.backward()
    # 解析梯度 -sin(theta)，交叉核对 psr
    ref = float(psr(lambda p: float(tn_expectation(build(float(p[0])), Hamiltonian([("Z", 1.0)]), backend=bk)),
                    np.array([0.6]))[0])
    assert np.isclose(float(theta.grad), -np.sin(0.6), atol=1e-4)
    assert np.isclose(float(theta.grad), ref, atol=1e-4)

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir import Circuit, Hamiltonian, ry
from aicir.qml import QLayer, qfun


def _cost(device="numpy", observable=None):
    if observable is None:
        observable = Hamiltonian([("Z", 1.0)])  # <Z> = cos(theta)

    @qfun(device=device, differential="psr", observable=observable)
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta[0], 0))
        return c

    return cost


def test_forward_matches_qfun_value():
    cost = _cost()
    layer = QLayer(cost, n_weights=1, init=np.array([0.4]))
    out = layer()
    assert out.shape == ()
    assert np.isclose(float(out.detach()), np.cos(0.4))


def test_backward_matches_analytic_gradient():
    cost = _cost()
    layer = QLayer(cost, n_weights=1, init=np.array([0.4]))
    out = layer()
    out.backward()
    # d<Z>/dtheta = -sin(theta)
    assert np.allclose(layer.weights.grad.numpy(), [-np.sin(0.4)])


def test_backward_matches_qfun_grad_vector():
    obs = Hamiltonian([("ZI", 1.0)])  # <Z on q0> = cos(theta[0])

    @qfun(observable=obs)
    def cost(theta):
        c = Circuit(n_qubits=2)
        c.append(ry(theta[0], 0))
        c.append(ry(theta[1], 1))
        return c

    w = np.array([0.3, 0.7])
    layer = QLayer(cost, n_weights=2, init=w)
    layer().backward()
    assert np.allclose(layer.weights.grad.numpy(), cost.grad(w))


def test_gradients_flow_into_preceding_classical_layer():
    cost = _cost()
    linear = torch.nn.Linear(2, 1)
    layer = QLayer(cost, n_weights=0)  # all params come from inputs
    x = torch.tensor([1.0, 2.0])
    out = layer(linear(x))
    out.backward()
    assert linear.weight.grad is not None
    assert torch.any(linear.weight.grad != 0)


def test_multi_observable_output_and_grad_shape():
    obs = [Hamiltonian([("Z", 1.0)]), Hamiltonian([("X", 1.0)])]
    cost = _cost(observable=obs)
    layer = QLayer(cost, n_weights=1, init=np.array([0.3]))
    out = layer()
    assert out.shape == (2,)
    assert np.allclose(out.detach().numpy(), [np.cos(0.3), np.sin(0.3)])
    out.sum().backward()
    # d/dtheta [cos, sin] summed = -sin + cos
    assert np.allclose(layer.weights.grad.numpy(), [-np.sin(0.3) + np.cos(0.3)])


def test_batched_inputs_shape_and_grad():
    cost = _cost()
    layer = QLayer(cost, n_weights=0)
    x = torch.tensor([[0.2], [0.5], [0.9]], requires_grad=True)
    out = layer(x)
    assert out.shape == (3,)
    assert np.allclose(out.detach().numpy(), np.cos([0.2, 0.5, 0.9]))
    out.sum().backward()
    assert np.allclose(x.grad.numpy().reshape(-1), -np.sin([0.2, 0.5, 0.9]))


def test_gpu_backend_bridge():
    cost = _cost(device="gpu")
    layer = QLayer(cost, n_weights=1, init=np.array([0.4]))
    out = layer()
    out.backward()
    assert np.allclose(layer.weights.grad.numpy(), [-np.sin(0.4)])


def test_optimizer_step_reduces_loss():
    cost = _cost()  # minimize <Z> = cos(theta) -> theta -> pi
    layer = QLayer(cost, n_weights=1, init=np.array([0.3]))
    opt = torch.optim.SGD(layer.parameters(), lr=0.3)
    first = float(layer().detach())
    for _ in range(20):
        opt.zero_grad()
        loss = layer()
        loss.backward()
        opt.step()
    assert float(layer().detach()) < first

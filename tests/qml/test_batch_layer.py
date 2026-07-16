"""BatchLayer：模板线路批量量子层（BatchSV 前向 + 原生 autograd 反向）。

正确性基准：逐样本 bind_parameters → 单态路径演化 → 逐比特 <Z_q>。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir import Circuit, Parameter, cx, hadamard, rx, ry, rzz
from aicir.backends.gpu_backend import GPUBackend
from aicir.core.state import State
from aicir.qml import BatchLayer


def _template(n, n_inputs, n_weights):
    """rx 数据编码 + ry/rzz 权重层 + cx 纠缠的固定模板。"""
    xs = [Parameter(f"x{i}") for i in range(n_inputs)]
    ws = [Parameter(f"w{i}") for i in range(n_weights)]
    gates = [rx(xs[i], i % n) for i in range(n_inputs)]
    gates += [hadamard(0)]
    wi = 0
    for q in range(n):
        gates.append(ry(ws[wi], q)); wi += 1
    gates.append(rzz(ws[wi], 0, n - 1)); wi += 1
    gates.append(cx(1, [0]))
    assert wi == n_weights
    return Circuit(*gates, n_qubits=n)


def _reference_z(circuit, param_rows, backend):
    """逐样本绑定参数走单态路径，返回 (batch, n) 的 <Z_q>。"""
    n = circuit.n_qubits
    idx = np.arange(1 << n)
    zsign = np.stack([1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1) for q in range(n)], axis=0)
    out = []
    for row in param_rows:
        bound = circuit.bind_parameters(np.asarray(row, dtype=float))
        state = State.zero_state(n, backend).evolve(bound.unitary(backend=backend))
        probs = np.abs(np.asarray(state.to_numpy()).reshape(-1)) ** 2
        out.append(zsign @ probs)
    return np.asarray(out)


def test_forward_matches_single_state_reference():
    n, n_inputs, n_weights, batch = 3, 3, 4, 5
    circuit = _template(n, n_inputs, n_weights)
    backend = GPUBackend()
    layer = BatchLayer(circuit, n_inputs, backend=backend, init=np.linspace(0.1, 0.9, n_weights))

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(batch, n_inputs))
    got = layer(torch.as_tensor(x)).detach().cpu().numpy()

    w = layer.weights.detach().cpu().numpy()
    ref = _reference_z(circuit, [np.concatenate([row, w]) for row in x], backend)
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_backward_matches_finite_difference():
    n, n_inputs, n_weights, batch = 2, 2, 3, 4
    circuit = _template(n, n_inputs, n_weights)
    layer = BatchLayer(circuit, n_inputs, backend=GPUBackend(),
                       init=np.array([0.3, -0.4, 0.7]))
    x = torch.as_tensor(np.random.default_rng(1).uniform(-1, 1, size=(batch, n_inputs)))

    loss = layer(x).sum()
    loss.backward()
    grad = layer.weights.grad.detach().cpu().numpy().copy()

    eps = 1e-4
    fd = np.zeros(n_weights)
    base_w = layer.weights.detach().cpu().numpy().copy()
    for p in range(n_weights):
        for sign, slot in ((1, 0), (-1, 1)):
            with torch.no_grad():
                layer.weights[p] = float(base_w[p] + sign * eps)
            val = float(layer(x).sum())
            fd[p] += val if slot == 0 else -val
        fd[p] /= 2 * eps
        with torch.no_grad():
            layer.weights[p] = float(base_w[p])
    np.testing.assert_allclose(grad, fd, atol=1e-2)


def test_input_gradient_flows_to_classical_layer():
    # 混合网络：Linear → BatchLayer，梯度须回流到 Linear 权重
    n, n_inputs, n_weights = 2, 2, 3
    circuit = _template(n, n_inputs, n_weights)
    layer = BatchLayer(circuit, n_inputs, backend=GPUBackend())
    linear = torch.nn.Linear(3, n_inputs)
    x = torch.randn(4, 3)
    loss = layer(linear(x)).sum()
    loss.backward()
    assert linear.weight.grad is not None
    assert torch.any(linear.weight.grad != 0)


def test_training_reduces_loss():
    # 玩具回归：学习使 <Z_0> 逼近目标常数
    n, n_inputs, n_weights = 2, 2, 3
    circuit = _template(n, n_inputs, n_weights)
    torch.manual_seed(0)
    layer = BatchLayer(circuit, n_inputs, backend=GPUBackend())
    opt = torch.optim.Adam(layer.parameters(), lr=0.1)
    x = torch.as_tensor(np.random.default_rng(2).uniform(-1, 1, size=(8, n_inputs)))
    target = torch.full((8,), 0.5, dtype=torch.float32)

    def loss_fn():
        return ((layer(x)[:, 0] - target) ** 2).mean()

    first = float(loss_fn())
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert float(loss_fn()) < first


def test_rejects_unsupported_parametrized_gate():
    from aicir.core.circuit import u3
    theta = Parameter("t")
    circ = Circuit(u3(theta, 0.1, 0.2, 0), n_qubits=1)
    with pytest.raises(ValueError):
        BatchLayer(circ, 0, backend=GPUBackend())

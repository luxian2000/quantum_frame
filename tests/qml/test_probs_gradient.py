"""QFun.grad 对 probs 返回的 Jacobian（参数移位，(D, P) 形状）。

解析基准：单 ry(θ) 的 probs = [cos²(θ/2), sin²(θ/2)]，
d/dθ = [-½sinθ, ½sinθ]。
"""

import numpy as np
import pytest

from aicir import Circuit, ry, rx, cx
from aicir.core.circuit import Parameter
from aicir.qml import qfun, probs


def test_probs_grad_scalar_param_matches_analytic():
    @qfun(device="numpy")
    def circ(theta):
        c = Circuit(ry(theta[0], 0), n_qubits=1)
        return probs(c)

    for t in (0.3, 1.1, -0.7):
        jac = np.asarray(circ.grad(np.array([t])), dtype=float)
        assert jac.shape == (2, 1)
        expected = np.array([[-0.5 * np.sin(t)], [0.5 * np.sin(t)]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)


def test_probs_grad_vector_param_matches_finite_difference():
    @qfun(device="numpy")
    def circ(p):
        c = Circuit(ry(p[0], 0), rx(p[1], 1), cx(1, [0]), n_qubits=2)
        return probs(c)

    x = np.array([0.4, -0.9])
    jac = np.asarray(circ.grad(x), dtype=float)
    assert jac.shape == (4, 2)

    # eps 需大于 complex64 态精度噪声（~1e-7）：太小则 FD 陷入 float32 灾难性抵消
    eps = 1e-3
    fd = np.zeros((4, 2))
    for j in range(2):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        fd[:, j] = (np.asarray(circ(xp)) - np.asarray(circ(xm))) / (2 * eps)
    np.testing.assert_allclose(jac, fd, atol=1e-4)


def test_probs_grad_marginalized_wires_shape():
    @qfun(device="numpy")
    def circ(p):
        c = Circuit(ry(p[0], 0), rx(p[1], 1), cx(1, [0]), n_qubits=2)
        return probs(c, wires=[0])  # 边缘化到 1 个 wire → D=2

    x = np.array([0.5, 0.2])
    jac = np.asarray(circ.grad(x), dtype=float)
    assert jac.shape == (2, 2)
    # 概率梯度逐参数求和为 0（概率归一化守恒）
    np.testing.assert_allclose(jac.sum(axis=0), np.zeros(2), atol=1e-6)


def test_probs_grad_columns_sum_to_zero():
    # 全寄存器概率梯度对基态求和恒为 0（Σp_i = 1 对 θ 求导）
    @qfun(device="numpy")
    def circ(p):
        c = Circuit(ry(p[0], 0), ry(p[1], 1), cx(1, [0]), n_qubits=2)
        return probs(c)

    jac = np.asarray(circ.grad(np.array([0.7, -0.3])), dtype=float)
    np.testing.assert_allclose(jac.sum(axis=0), np.zeros(2), atol=1e-6)


def test_sample_grad_still_raises():
    from aicir.qml import sample

    @qfun(device="numpy", shots=100)
    def circ(theta):
        c = Circuit(ry(theta[0], 0), n_qubits=1)
        return sample(c)

    with pytest.raises(ValueError):
        circ.grad(np.array([0.3]))


def test_qlayer_probs_backward_flows():
    torch = pytest.importorskip("torch")
    from aicir.qml import QLayer

    @qfun(device="numpy")
    def circ(p):
        c = Circuit(ry(p[0], 0), rx(p[1], 1), cx(1, [0]), n_qubits=2)
        return probs(c, wires=[0])

    layer = QLayer(circ, n_weights=2, init=np.array([0.3, 0.5]))
    out = layer()          # (2,) 概率向量
    assert out.shape == (2,)
    loss = (out ** 2).sum()
    loss.backward()
    assert layer.weights.grad is not None
    assert torch.all(torch.isfinite(layer.weights.grad))

"""qfun（PennyLane 风格量子函数）测试（NEXT.md §5）。"""

import numpy as np
import pytest

from aicir import BitFlipChannel, Circuit, Hamiltonian, NoiseModel, ry
from aicir.qml import qfun

H_Z = Hamiltonian([("Z", 1.0)])
H_X = Hamiltonian([("X", 1.0)])


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


def test_observable_required_at_call():
    # observable= 不再于装饰期强制；返回裸 Circuit 且无 observable 时调用期报错。
    @qfun()
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return c

    with pytest.raises(ValueError):
        cost(0.1)


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        qfun(observable=H_Z, device="quantum_foo")(lambda t: Circuit(n_qubits=1))


# --- 多参数（单数组参 vector）---

def _make_two_param(**kw):
    # <ZI> = <Z on q0> = cos(theta0)，与 theta1 无关
    H_ZI = Hamiltonian([("ZI", 1.0)])

    @qfun(observable=H_ZI, **kw)
    def cost(theta):
        c = Circuit(n_qubits=2)
        c.append(ry(theta[0], 0))
        c.append(ry(theta[1], 1))
        return c

    return cost


def test_vector_param_call():
    cost = _make_two_param(differential="psr")
    assert np.isclose(cost(np.array([0.3, 0.7])), np.cos(0.3))


def test_vector_param_grad_shape_and_value():
    cost = _make_two_param(differential="psr")
    g = cost.grad(np.array([0.3, 0.7]))
    assert g.shape == (2,)
    assert np.allclose(g, [-np.sin(0.3), 0.0], atol=1e-7)


# --- 多测量（observable=list → 数组 / Jacobian）---

def _make_multi_obs(**kw):
    @qfun(observable=[H_Z, H_X], **kw)
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return c

    return cost


def test_multi_observable_returns_array():
    cost = _make_multi_obs(differential="psr")
    v = cost(0.3)
    assert np.shape(v) == (2,)
    assert np.allclose(v, [np.cos(0.3), np.sin(0.3)], atol=1e-7)


def test_multi_observable_scalar_param_jacobian():
    cost = _make_multi_obs(differential="psr")
    J = cost.grad(0.3)
    # 标量参 + 2 观测量 → (n_obs,)
    assert J.shape == (2,)
    assert np.allclose(J, [-np.sin(0.3), np.cos(0.3)], atol=1e-7)


def test_multi_observable_vector_param_jacobian():
    H_ZI = Hamiltonian([("ZI", 1.0)])
    H_IZ = Hamiltonian([("IZ", 1.0)])

    @qfun(observable=[H_ZI, H_IZ], differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=2)
        c.append(ry(theta[0], 0))
        c.append(ry(theta[1], 1))
        return c

    J = cost.grad(np.array([0.3, 0.7]))
    # 向量参 (2) + 2 观测量 → (n_obs, n_param) = (2, 2)
    assert J.shape == (2, 2)
    expected = np.array([[-np.sin(0.3), 0.0], [0.0, -np.sin(0.7)]])
    assert np.allclose(J, expected, atol=1e-7)


# --- 噪声路径便捷封装（noise_model=）---

def _bitflip_model(p):
    # ry 之后对 qubit0 施加 bit-flip：<Z> → (1-2p)<Z>
    return NoiseModel().add_channel(BitFlipChannel(0, p), after_gates=["ry"])


def _make_noisy_cost(p, **kw):
    @qfun(observable=H_Z, noise_model=_bitflip_model(p), **kw)
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return c

    return cost


def test_noise_model_scales_expectation():
    cost = _make_noisy_cost(0.25, differential="psr")
    assert np.isclose(cost(0.3), 0.5 * np.cos(0.3), atol=1e-7)


def test_noise_model_gradient_psr():
    cost = _make_noisy_cost(0.25, differential="psr")
    # f(θ)=(1-2p)cosθ 线性于无噪期望，PSR 仍解析精确
    assert np.isclose(cost.grad(0.3), -0.5 * np.sin(0.3), atol=1e-7)


def test_noise_model_auto_differential_runs():
    cost = _make_noisy_cost(0.25, differential="auto")
    # auto 在 noisy=True 下选取合适方法，仍给出有限梯度
    g = cost.grad(0.3)
    assert np.isfinite(g)


# ── §5 测量返回构造器：expval / probs / sample ──────────────────────────


def test_expval_body_return_matches_decorator():
    from aicir.qml import expval

    @qfun(differential="psr")
    def cost(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return expval(c, H_Z)

    assert np.isclose(cost(0.3), np.cos(0.3))
    assert np.isclose(cost.grad(0.3), -np.sin(0.3))


def test_probs_body_return_distribution():
    from aicir.qml import probs

    @qfun()
    def circ(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return probs(c)

    p = circ(np.pi / 2)  # |+y> 测量 Z 基 → [0.5, 0.5]
    assert p.shape == (2,)
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, [0.5, 0.5], atol=1e-7)


def test_probs_wires_marginalizes():
    from aicir import cx, pauli_x
    from aicir.qml import probs

    @qfun()
    def circ(_):
        c = Circuit(n_qubits=2)
        c.append(pauli_x(0))  # |10>
        c.append(cx(target_qubit=1, control_qubits=[0]))  # |11>
        return probs(c, wires=[1])

    p = circ(0.0)
    assert p.shape == (2,)
    assert np.allclose(p, [0.0, 1.0], atol=1e-7)


def test_sample_body_return_counts():
    from aicir import pauli_x
    from aicir.qml import sample

    @qfun(shots=128)
    def circ(_):
        c = Circuit(n_qubits=1)
        c.append(pauli_x(0))
        return sample(c)

    counts = circ(0.0)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 128


def test_sample_requires_shots():
    from aicir.qml import sample

    @qfun()
    def circ(_):
        return sample(Circuit(n_qubits=1))

    with pytest.raises(ValueError):
        circ(0.0)


def test_grad_rejects_probs():
    from aicir.qml import probs

    @qfun()
    def circ(theta):
        c = Circuit(n_qubits=1)
        c.append(ry(theta, 0))
        return probs(c)

    with pytest.raises(ValueError):
        circ.grad(0.3)

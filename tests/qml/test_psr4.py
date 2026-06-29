"""四项参数移位规则 psr4：对激发门梯度与 autograd 一致，两项 psr 不一致。"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir import single_excitation, double_excitation
from aicir.backends.gpu_backend import GPUBackend
from aicir.core.circuit import Circuit
from aicir.core.gates import apply_gate_to_state
from aicir.qml.deriv import psr, psr4


def _energy_fn(gate_factory, n_qubits, init_x):
    backend = GPUBackend(device="cpu")
    z0 = np.kron(np.diag([1.0, -1.0]), np.eye(1 << (n_qubits - 1))).astype(np.complex64)
    H = backend.cast(z0)

    def run(theta_value, grad=False):
        th = torch.tensor(float(theta_value[0]), requires_grad=grad)
        state = backend.zeros_state(n_qubits)
        for q in init_x:  # 预置 X 翻转，使梯度非平凡
            from aicir.core.circuit import pauli_x
            state = apply_gate_to_state(pauli_x(q).to_dict(), state, n_qubits, backend)
        for g in Circuit(gate_factory(th), n_qubits=n_qubits).gates:
            state = apply_gate_to_state(g, state, n_qubits, backend)
        e = backend.expectation_sv(state, H).real
        return th, e

    def fn(theta_value):
        _, e = run(theta_value)
        return float(e.detach())

    def autograd_grad(theta_value):
        th, e = run(theta_value, grad=True)
        e.backward()
        return np.array([float(th.grad)])

    return fn, autograd_grad


def test_psr4_matches_autograd_single_excitation():
    fn, autograd_grad = _energy_fn(lambda th: single_excitation(th, 0, 1), 2, init_x=(1,))
    theta = np.array([0.6])
    assert np.allclose(psr4(fn, theta), autograd_grad(theta), atol=1e-4)


def test_psr4_matches_autograd_double_excitation():
    fn, autograd_grad = _energy_fn(lambda th: double_excitation(th, 0, 1, 2, 3), 4, init_x=(2, 3))
    theta = np.array([0.7])
    assert np.allclose(psr4(fn, theta), autograd_grad(theta), atol=1e-4)


def test_two_term_psr_is_wrong_for_excitation():
    fn, autograd_grad = _energy_fn(lambda th: single_excitation(th, 0, 1), 2, init_x=(1,))
    theta = np.array([0.6])
    # 标准两项规则对激发门不正确，应与 autograd 不一致
    assert not np.allclose(psr(fn, theta), autograd_grad(theta), atol=1e-3)

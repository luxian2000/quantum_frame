"""门矩阵分发的 torch autograd 保持（Approach A）。

Approach A 把矩阵构造改为经注册表 ``GateSpec.matrix`` 分发。本测试断言两条路径
（``gate_to_matrix`` 全矩阵 / ``apply_gate_to_state`` 局部）对参数门的梯度仍与有限
差分一致，即计算图未被破坏。跨 torch / npu 后端（可用才跑）。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir.core.circuit import (
    crx,
    double_excitation,
    rx,
    ry,
    rz,
    single_excitation,
)
from aicir.core.gates import apply_gate_to_state, gate_to_matrix

N_QUBITS = 4


def _backends():
    from aicir.backends.gpu_backend import GPUBackend

    out = [("torch", GPUBackend(device="cpu"))]
    try:
        from aicir.backends.npu_backend import NPUBackend

        out.append(("npu", NPUBackend()))
    except Exception:
        pass
    return out


BACKENDS = _backends()
BACKEND_IDS = [b[0] for b in BACKENDS]

GATES = [
    ("rx", lambda th: rx(th, 0), 0.37),
    ("ry", lambda th: ry(th, 1), -0.71),
    ("rz", lambda th: rz(th, 2), 1.13),
    ("crx", lambda th: crx(th, 1, [0]), 0.44),
    ("single_excitation", lambda th: single_excitation(th, 0, 1), 0.42),
    ("double_excitation", lambda th: double_excitation(th, 0, 1, 2, 3), 0.63),
]


def _fixtures(backend):
    rng = np.random.default_rng(7)
    dim = 1 << N_QUBITS
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec = (vec / np.linalg.norm(vec)).reshape(-1, 1)
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    obs = a + a.conj().T  # Hermitian
    return backend.cast(vec), backend.cast(obs)


def _expectation(theta_t, gate_fn, backend, psi, obs, path):
    gate = gate_fn(theta_t)
    if path == "matrix":
        unitary = gate_to_matrix(gate, N_QUBITS, backend=backend)
        psip = backend.apply_unitary(psi, unitary)
    else:
        psip = apply_gate_to_state(gate, psi, N_QUBITS, backend)
    psip = psip.reshape(-1, 1)
    return backend.expectation_sv(psip, obs).reshape(())


def test_expectation_helper_uses_backend_expectation_sv_for_backend_compatibility():
    from aicir.backends.gpu_backend import GPUBackend

    class RecordingBackend(GPUBackend):
        def __init__(self):
            super().__init__(device="cpu")
            self.expectation_calls = 0

        def expectation_sv(self, state, operator):
            self.expectation_calls += 1
            return super().expectation_sv(state, operator)

    backend = RecordingBackend()
    psi, obs = _fixtures(backend)

    _expectation(torch.tensor(0.37), lambda th: rx(th, 0), backend, psi, obs, "local")

    assert backend.expectation_calls == 1


@pytest.mark.parametrize("path", ["matrix", "local"])
@pytest.mark.parametrize("gate_id,gate_fn,theta0", GATES, ids=[g[0] for g in GATES])
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_gradient_matches_finite_difference(path, gate_id, gate_fn, theta0, backend):
    name, be = backend
    psi, obs = _fixtures(be)

    theta = torch.tensor(float(theta0), dtype=torch.float32, requires_grad=True)
    value = _expectation(theta, gate_fn, be, psi, obs, path)
    value.backward()
    grad_autograd = float(theta.grad)

    eps = 1e-2
    with torch.no_grad():
        plus = float(_expectation(torch.tensor(theta0 + eps), gate_fn, be, psi, obs, path))
        minus = float(_expectation(torch.tensor(theta0 - eps), gate_fn, be, psi, obs, path))
    grad_fd = (plus - minus) / (2 * eps)

    assert grad_autograd == pytest.approx(grad_fd, abs=2e-2), f"{name}/{gate_id}/{path}"

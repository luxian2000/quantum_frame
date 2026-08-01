"""Stinespring channels remain physical and differentiate paired-real leaves."""

from __future__ import annotations

import numpy as np
import torch

from aicir import PauliString
from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._channels import _householder_isometry, _stinespring_kraus
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._parameters import StinespringParam
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)


def _parameter(*, requires_grad=True):
    real = torch.tensor(
        [[0.2, -0.4, 0.1, 0.3], [0.5, 0.7, -0.2, 0.6], [-0.1, 0.2, 0.8, -0.3], [0.4, -0.5, 0.9, 0.1]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    imag = torch.tensor(
        [[0.1, 0.3, -0.6, 0.2], [-0.7, 0.4, 0.5, -0.1], [0.6, -0.2, 0.3, 0.8], [0.2, 0.1, -0.4, 0.7]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    return StinespringParam(2, 2, 2, real, imag)


def _one_density(backend):
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "matrix", layout)
    pair = _Pair(torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32), torch.zeros((2, 2), dtype=torch.float32))
    return DistState.from_pair(pair, spec=spec, backend=backend)


def _complex(pair):
    return pair.real.detach().numpy() + 1j * pair.imag.detach().numpy()


def _float64_kraus(real, imag):
    """Independent complex128 Householder oracle; never calls production code."""

    current = np.eye(4, 2, dtype=np.complex128)
    for vector in (real + 1j * imag):
        denominator = np.vdot(vector, vector).real + 1e-12
        current = current - 2.0 * np.outer(vector, vector.conj() @ current) / denominator
    return tuple(current[index * 2:(index + 1) * 2] for index in range(2))


def test_householder_stinespring_isometry_and_kraus_completeness():
    parameter = _parameter(requires_grad=False)
    isometry = _complex(_householder_isometry(parameter))
    kraus = [_complex(value) for value in _stinespring_kraus(parameter)]
    np.testing.assert_allclose(isometry.conj().T @ isometry, np.eye(2), atol=1e-5)
    np.testing.assert_allclose(sum(value.conj().T @ value for value in kraus), np.eye(2), atol=1e-5)


def test_stinespring_channel_is_tp_psd_and_raw_real_imag_gradients_match_float64(monkeypatch):
    backend = _backend(monkeypatch)
    parameter = _parameter()
    evolved = _PairMatrixKernel(backend).apply_channel(_one_density(backend), parameter, instruction_index=53)
    actual = _complex(evolved._pair)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-5)
    assert np.linalg.eigvalsh(actual).min() >= -1e-5
    assert abs(np.trace(actual) - 1.0) <= 1e-5

    value = _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("Z", n_qubits=1))
    value.backward()
    assert parameter.real.grad is not None and parameter.imag.grad is not None
    assert torch.isfinite(parameter.real.grad).all() and torch.isfinite(parameter.imag.grad).all()

    epsilon = 1e-4
    def reference(real_entry, imag_entry):
        probe = _parameter(requires_grad=False)
        real = probe.real.detach().numpy().astype(np.float64); imag = probe.imag.detach().numpy().astype(np.float64)
        real[0, 0], imag[0, 0] = real_entry, imag_entry
        matrices = _float64_kraus(real, imag)
        rho = np.diag([0.0, 1.0]).astype(np.complex128)
        out = sum(matrix @ rho @ matrix.conj().T for matrix in matrices)
        return float(np.trace(out @ np.diag([1.0, -1.0])).real)

    initial_real, initial_imag = float(parameter.real.detach()[0, 0]), float(parameter.imag.detach()[0, 0])
    real_fd = (reference(initial_real + epsilon, initial_imag) - reference(initial_real - epsilon, initial_imag)) / (2 * epsilon)
    imag_fd = (reference(initial_real, initial_imag + epsilon) - reference(initial_real, initial_imag - epsilon)) / (2 * epsilon)
    assert abs(float(parameter.real.grad[0, 0]) - real_fd) <= 1e-4
    assert abs(float(parameter.imag.grad[0, 0]) - imag_fd) <= 1e-4

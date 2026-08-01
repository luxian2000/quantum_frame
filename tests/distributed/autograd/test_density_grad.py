"""Paired-real density-matrix backward contracts on one CPU rank."""

from __future__ import annotations

import numpy as np
import torch

from aicir import PauliString
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, parameter_shift_gradient
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._parameters import DensityParam, PureStateParam
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState
from aicir.ir import Observable


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)


def _state(pair, backend, layout, n_qubits, kind):
    spec = _ShardSpec.build(n_qubits, 1, 0, kind, layout)
    return DistState.from_pair(pair, spec=spec, backend=backend), spec


def _complex(pair):
    return pair.real.detach().numpy() + 1j * pair.imag.detach().numpy()


def _ry(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def test_pair_density_unitary_preserves_physicality_and_matches_float64(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    factor = np.array([[1.0 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.7 - 0.4j]])
    density = factor @ factor.conj().T
    density /= np.trace(density)
    state, _ = _state(
        _Pair(torch.tensor(density.real, dtype=torch.float32), torch.tensor(density.imag, dtype=torch.float32)),
        backend, layout, 1, "matrix",
    )
    theta = torch.tensor(0.37, dtype=torch.float32, requires_grad=True)
    plan = _GatePlanner(backend, layout, 1, execution_context=_AutogradExecutionContext()).plan(ry(theta, 0), 0)

    evolved = _PairMatrixKernel(backend).apply_unitary(state, plan, operation_index=0)
    actual = _complex(evolved._pair)
    expected = _ry(0.37) @ density @ _ry(0.37).conj().T

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-5)
    assert np.linalg.eigvalsh(actual).min() >= -1e-5
    assert abs(np.trace(actual) - 1.0) <= 1e-5

    _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("Z", n_qubits=1)).backward()
    finite_difference = (
        np.trace(_ry(0.37 + 1e-6) @ density @ _ry(0.37 + 1e-6).conj().T @ np.diag([1.0, -1.0])).real
        - np.trace(_ry(0.37 - 1e-6) @ density @ _ry(0.37 - 1e-6).conj().T @ np.diag([1.0, -1.0])).real
    ) / (2e-6)
    assert abs(float(theta.grad) - finite_difference) <= 1e-4
    shifted = parameter_shift_gradient(
        lambda values: float(np.trace(
            _ry(float(values[0])) @ density @ _ry(float(values[0])).conj().T
            @ np.diag([1.0, -1.0])
        ).real),
        np.array([0.37], dtype=np.float64),
    )[0]
    assert abs(float(theta.grad) - shifted) <= 1e-4


def test_density_param_factor_and_promoted_pure_state_are_differentiable(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    factor_real = torch.tensor([[1.0, -0.2], [0.3, 0.5]], dtype=torch.float32, requires_grad=True)
    factor_imag = torch.tensor([[0.1, 0.4], [-0.2, 0.6]], dtype=torch.float32, requires_grad=True)
    parameter = DensityParam(factor_real, factor_imag)
    state, spec = _state(parameter.density_pair(), backend, layout, 1, "matrix")
    value = _PairReducer(backend).expectation(state._pair, spec, PauliString("Z", n_qubits=1))
    value.backward()

    eps = 1e-4
    def reference(value):
        real = factor_real.detach().numpy().astype(np.float64).copy(); real[0, 0] = value
        imag = factor_imag.detach().numpy().astype(np.float64)
        factor = real + 1j * imag; rho = factor @ factor.conj().T; rho /= np.trace(rho)
        return float(np.trace(rho @ np.diag([1.0, -1.0])).real)
    finite_difference = (reference(float(factor_real.detach()[0, 0]) + eps) - reference(float(factor_real.detach()[0, 0]) - eps)) / (2 * eps)
    assert abs(float(factor_real.grad[0, 0]) - finite_difference) <= 1e-4
    assert factor_imag.grad is not None

    pure_real = torch.tensor([[1.0], [1.0]], dtype=torch.float32, requires_grad=True)
    pure_imag = torch.tensor([[0.0], [0.0]], dtype=torch.float32, requires_grad=True)
    vector, _ = _state(PureStateParam(pure_real, pure_imag).normalized_pair(), backend, layout, 1, "vector")
    promoted = _PairMatrixKernel(backend).promote_vector(vector)
    probabilities = _PairReducer(backend).probabilities(promoted._pair, promoted.spec)
    probabilities[0].backward()
    torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.5]))
    assert pure_real.grad is not None


def test_density_reducers_honor_nonidentity_layout_for_probabilities_pauli_and_dense(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=0)
    raw_real = torch.tensor([[1.0], [0.2], [-0.4], [0.3]], dtype=torch.float32, requires_grad=True)
    raw_imag = torch.tensor([[0.1], [0.3], [0.2], [-0.5]], dtype=torch.float32, requires_grad=True)
    vector, _ = _state(PureStateParam(raw_real, raw_imag).normalized_pair(), backend, layout, 2, "vector")
    density = _PairMatrixKernel(backend).promote_vector(vector)
    reducer = _PairReducer(backend)
    probabilities = reducer.probabilities(density._pair, density.spec)
    pauli = reducer.expectation(density._pair, density.spec, PauliString("ZI", n_qubits=2))
    dense = reducer.expectation(
        density._pair, density.spec,
        Observable("matrix", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64), metadata={"qubits": (0,)}),
    )
    (probabilities[0] + pauli + dense).backward()

    assert torch.isfinite(raw_real.grad).all()
    assert torch.isfinite(raw_imag.grad).all()
    np.testing.assert_allclose(float(pauli.detach()), float(dense.detach()), atol=1e-5)
    assert abs(float(probabilities.sum().detach()) - 1.0) <= 1e-5

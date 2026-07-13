"""量子 Fisher 信息矩阵（QFIM）估计：``qfim``/``qfim_diag``/``qfim_blocks``。

``qfim``/``qfim_diag`` 对 torch/NPU 态输入内部走 torch 路径（设备驻留的中心差分 +
内积），但最终都 ``.detach().cpu().numpy()`` 转回 NumPy——与 ``qng``/``rotosolve``
族「torch 入参→同设备 torch 出参」不同，这三个函数的返回值契约一律是 NumPy
（详见包 ``__init__.py`` 顶部的 array-in/array-out 契约说明）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._coerce import (
    _as_state_vector,
    _as_torch_state_vector,
    _contains_torch_tensor,
    _flat_to_index,
    _normalize_qng_blocks,
    _state_tensor_and_backend,
    _torch_or_none,
)


def _qfim_from_state_fd(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Estimate the pure-state Quantum Fisher Information Matrix.

    Uses the Fubini-Study metric

        F_ij = 4 Re[<∂iψ|∂jψ> - <∂iψ|ψ><ψ|∂jψ>]

    with central finite differences for the state derivatives.
    """
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("metric_eps must be a positive number")

    base_state = _as_state_vector(state_fn(theta), label="state_fn(params)")
    derivatives = np.zeros((theta.size, base_state.size), dtype=np.complex128)

    for flat_index in range(theta.size):
        index = _flat_to_index(theta, flat_index)
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps_value
        minus[index] -= eps_value

        state_plus = _as_state_vector(
            state_fn(plus),
            label="state_fn(params + metric_eps)",
        )
        state_minus = _as_state_vector(
            state_fn(minus),
            label="state_fn(params - metric_eps)",
        )
        if state_plus.shape != base_state.shape or state_minus.shape != base_state.shape:
            raise ValueError("state_fn must return state vectors with a consistent shape")
        derivatives[flat_index] = (state_plus - state_minus) / (2.0 * eps_value)

    qfim = np.zeros((theta.size, theta.size), dtype=float)
    for i in range(theta.size):
        for j in range(i, theta.size):
            overlap = np.vdot(derivatives[i], derivatives[j])
            projection = np.vdot(derivatives[i], base_state) * np.vdot(base_state, derivatives[j])
            value = 4.0 * float(np.real(overlap - projection))
            qfim[i, j] = value
            qfim[j, i] = value

    # Remove finite-difference asymmetry and tiny negative numerical noise.
    qfim = 0.5 * (qfim + qfim.T)
    qfim[np.isclose(qfim, 0.0, atol=1e-12)] = 0.0
    return qfim


def _qfim_from_derivatives(base_state: np.ndarray, derivatives: np.ndarray) -> np.ndarray:
    qfim = np.zeros((derivatives.shape[0], derivatives.shape[0]), dtype=float)
    for i in range(derivatives.shape[0]):
        for j in range(i, derivatives.shape[0]):
            overlap = np.vdot(derivatives[i], derivatives[j])
            projection = np.vdot(derivatives[i], base_state) * np.vdot(base_state, derivatives[j])
            value = 4.0 * float(np.real(overlap - projection))
            qfim[i, j] = value
            qfim[j, i] = value

    qfim = 0.5 * (qfim + qfim.T)
    qfim[np.isclose(qfim, 0.0, atol=1e-12)] = 0.0
    return qfim


def _state_derivatives_fd(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    block: list[tuple[int, ...]],
    *,
    base_state: np.ndarray,
    eps: float,
) -> np.ndarray:
    derivatives = np.zeros((len(block), base_state.size), dtype=np.complex128)
    for derivative_index, index in enumerate(block):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps
        minus[index] -= eps

        state_plus = _as_state_vector(
            state_fn(plus),
            label="state_fn(params + metric_eps)",
        )
        state_minus = _as_state_vector(
            state_fn(minus),
            label="state_fn(params - metric_eps)",
        )
        if state_plus.shape != base_state.shape or state_minus.shape != base_state.shape:
            raise ValueError("state_fn must return state vectors with a consistent shape")
        derivatives[derivative_index] = (state_plus - state_minus) / (2.0 * eps)
    return derivatives


def _qfim_blocks_from_state_fd(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    blocks: list[list[tuple[int, ...]]],
    *,
    eps: float,
) -> list[np.ndarray]:
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("metric_eps must be a positive number")

    base_state = _as_state_vector(state_fn(theta), label="state_fn(params)")
    qfim_blocks = []
    for block in blocks:
        derivatives = _state_derivatives_fd(
            state_fn,
            theta,
            block,
            base_state=base_state,
            eps=eps_value,
        )
        qfim_blocks.append(_qfim_from_derivatives(base_state, derivatives))
    return qfim_blocks


def _qfim_diag_from_state_fd(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    blocks = [[_flat_to_index(theta, flat_index)] for flat_index in range(theta.size)]
    qfim_blocks = _qfim_blocks_from_state_fd(state_fn, theta, blocks, eps=eps)
    return np.array([block[0, 0] for block in qfim_blocks], dtype=float)


def qfim(
    state_fn: Callable[[np.ndarray], Any],
    params: Any,
    *,
    metric_eps: float = 1e-3,
    backend: Any = None,
) -> np.ndarray:
    """估计纯态 ansatz 的量子 Fisher 信息矩阵。"""
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    sample = state_fn(theta)
    sample_tensor, sample_backend = _state_tensor_and_backend(sample, backend)
    if _contains_torch_tensor(sample_tensor):
        eps_value = float(metric_eps)
        if not eps_value > 0.0:
            raise ValueError("metric_eps must be a positive number")
        base_state, resolved_backend = _as_torch_state_vector(
            sample,
            label="state_fn(params)",
            backend=sample_backend,
        )
        block = list(np.ndindex(theta.shape))
        derivatives, resolved_backend = _state_derivatives_fd_torch(
            state_fn,
            theta,
            block,
            base_state=base_state,
            eps=eps_value,
            backend=resolved_backend,
        )
        metric = _qfim_from_derivatives_torch(base_state, derivatives, resolved_backend)
        return metric.detach().cpu().numpy().astype(float)

    return _qfim_from_state_fd(state_fn, theta, eps=metric_eps)


metric_tensor = qfim


def qfim_diag(
    state_fn: Callable[[np.ndarray], Any],
    params: Any,
    *,
    metric_eps: float = 1e-3,
    backend: Any = None,
) -> np.ndarray:
    """估计 QFIM 对角线。"""
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")
    sample = state_fn(theta)
    sample_tensor, sample_backend = _state_tensor_and_backend(sample, backend)
    if _contains_torch_tensor(sample_tensor):
        diag, _ = _qfim_diag_from_state_fd_torch(
            state_fn,
            theta,
            eps=metric_eps,
            backend=sample_backend,
        )
        return diag.detach().cpu().numpy().astype(float)
    return _qfim_diag_from_state_fd(state_fn, theta, eps=metric_eps)


def qfim_blocks(
    state_fn: Callable[[np.ndarray], Any],
    params: Any,
    *,
    blocks: Any = None,
    block_size: int = 1,
    metric_eps: float = 1e-3,
) -> list[np.ndarray]:
    """按参数块估计 QFIM 子矩阵。"""
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")
    normalized_blocks = _normalize_qng_blocks(theta, blocks, block_size)
    return _qfim_blocks_from_state_fd(state_fn, theta, normalized_blocks, eps=metric_eps)


def _qfim_diag_from_state_fd_torch(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    eps: float,
    backend: Any = None,
):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU QFIM path requires PyTorch")

    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("metric_eps must be a positive number")

    base_state, resolved_backend = _as_torch_state_vector(
        state_fn(theta),
        label="state_fn(params)",
        backend=backend,
    )
    values = []
    for flat_index in range(theta.size):
        index = _flat_to_index(theta, flat_index)
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps_value
        minus[index] -= eps_value

        state_plus, resolved_backend = _as_torch_state_vector(
            state_fn(plus),
            label="state_fn(params + metric_eps)",
            backend=resolved_backend,
        )
        state_minus, resolved_backend = _as_torch_state_vector(
            state_fn(minus),
            label="state_fn(params - metric_eps)",
            backend=resolved_backend,
        )
        if state_plus.shape != base_state.shape or state_minus.shape != base_state.shape:
            raise ValueError("state_fn must return state vectors with a consistent shape")

        derivative = (state_plus - state_minus) / (2.0 * eps_value)
        overlap = _torch_inner_product(derivative, derivative, resolved_backend)
        projection = (
            _torch_inner_product(derivative, base_state, resolved_backend)
            * _torch_inner_product(base_state, derivative, resolved_backend)
        )
        values.append(4.0 * torch.real(overlap - projection))

    return torch.stack(values), resolved_backend


def _torch_inner_product(bra, ket, backend: Any = None):
    torch = _torch_or_none()
    if backend is not None and hasattr(backend, "inner_product"):
        return backend.inner_product(bra.reshape(-1, 1), ket.reshape(-1, 1))
    return torch.sum(torch.conj(bra.reshape(-1)) * ket.reshape(-1))


def _qfim_from_derivatives_torch(base_state: Any, derivatives: list[Any], backend: Any = None):
    torch = _torch_or_none()
    rows = []
    for left_derivative in derivatives:
        row = []
        for right_derivative in derivatives:
            overlap = _torch_inner_product(left_derivative, right_derivative, backend)
            projection = (
                _torch_inner_product(left_derivative, base_state, backend)
                * _torch_inner_product(base_state, right_derivative, backend)
            )
            row.append(4.0 * torch.real(overlap - projection))
        rows.append(torch.stack(row))
    metric = torch.stack(rows)
    return 0.5 * (metric + metric.transpose(-2, -1))


def _state_derivatives_fd_torch(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    block: list[tuple[int, ...]],
    *,
    base_state: Any,
    eps: float,
    backend: Any = None,
) -> tuple[list[Any], Any]:
    derivatives = []
    resolved_backend = backend
    for index in block:
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps
        minus[index] -= eps

        state_plus, resolved_backend = _as_torch_state_vector(
            state_fn(plus),
            label="state_fn(params + metric_eps)",
            backend=resolved_backend,
        )
        state_minus, resolved_backend = _as_torch_state_vector(
            state_fn(minus),
            label="state_fn(params - metric_eps)",
            backend=resolved_backend,
        )
        if state_plus.shape != base_state.shape or state_minus.shape != base_state.shape:
            raise ValueError("state_fn must return state vectors with a consistent shape")
        derivatives.append((state_plus - state_minus) / (2.0 * eps))
    return derivatives, resolved_backend

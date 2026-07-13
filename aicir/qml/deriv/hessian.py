"""标量目标函数的 Hessian 矩阵：psr 对角线自检 + fd 降级路径。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._coerce import _as_scalar
from .fn_gradient import mpsr


def _psr_second_at_index(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    index: tuple[int, ...],
    shift: float,
    coefficient: float,
) -> float:
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += 2.0 * shift
    minus[index] -= 2.0 * shift
    center = _as_scalar(fn(theta), label="fn(params)")
    forward = _as_scalar(fn(plus), label="fn(params + 2*shift)")
    backward = _as_scalar(fn(minus), label="fn(params - 2*shift)")
    return float((coefficient ** 2) * (forward - 2.0 * center + backward))


def _fd_second_at_indices(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    left: tuple[int, ...],
    right: tuple[int, ...],
    eps: float,
) -> float:
    if left == right:
        plus = theta.copy()
        minus = theta.copy()
        plus[left] += eps
        minus[left] -= eps
        center = _as_scalar(fn(theta), label="fn(params)")
        forward = _as_scalar(fn(plus), label="fn(params + eps)")
        backward = _as_scalar(fn(minus), label="fn(params - eps)")
        return float((forward - 2.0 * center + backward) / (eps ** 2))

    pp = theta.copy()
    pm = theta.copy()
    mp = theta.copy()
    mm = theta.copy()
    pp[left] += eps
    pp[right] += eps
    pm[left] += eps
    pm[right] -= eps
    mp[left] -= eps
    mp[right] += eps
    mm[left] -= eps
    mm[right] -= eps
    return float((
        _as_scalar(fn(pp), label="fn(params + eps_i + eps_j)")
        - _as_scalar(fn(pm), label="fn(params + eps_i - eps_j)")
        - _as_scalar(fn(mp), label="fn(params - eps_i + eps_j)")
        + _as_scalar(fn(mm), label="fn(params - eps_i - eps_j)")
    ) / (4.0 * eps ** 2))


def hessian(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    method: str = "psr",
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    eps: float = 1e-3,
) -> np.ndarray:
    """计算标量目标函数的完整 Hessian 矩阵。"""
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    method_norm = str(method).strip().lower()
    if method_norm not in {"psr", "fd"}:
        raise ValueError("method must be 'psr' or 'fd'")
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    indices = list(np.ndindex(theta.shape))
    psr_diag_cache: dict[tuple[int, ...], float] = {}
    fd_diag_cache: dict[tuple[int, ...], float] = {}
    if method_norm == "psr":
        for index in indices:
            shifted_value = _psr_second_at_index(fn, theta, index, float(shift), float(coefficient))
            fd_value = _fd_second_at_indices(fn, theta, index, index, eps_value)
            psr_diag_cache[index] = shifted_value
            fd_diag_cache[index] = fd_value
            if not np.isclose(shifted_value, fd_value, rtol=1e-3, atol=1e-5):
                method_norm = "fd"
                break

    matrix = np.zeros((theta.size, theta.size), dtype=float)
    for row, left in enumerate(indices):
        for col in range(row, theta.size):
            right = indices[col]
            if method_norm == "psr":
                if left == right:
                    value = psr_diag_cache.get(left)
                    if value is None:
                        value = _psr_second_at_index(fn, theta, left, float(shift), float(coefficient))
                else:
                    value = mpsr(
                        fn,
                        theta,
                        parameter_indices=[left, right],
                        shift=shift,
                        coefficient=coefficient,
                    )
            else:
                if left == right and left in fd_diag_cache:
                    value = fd_diag_cache[left]
                else:
                    value = _fd_second_at_indices(fn, theta, left, right, eps_value)
            matrix[row, col] = value
            matrix[col, row] = value
    return 0.5 * (matrix + matrix.T)

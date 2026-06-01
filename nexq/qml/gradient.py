"""Gradient utilities for quantum machine learning workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _as_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must return a scalar value")
    return float(array)


def psr(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
) -> np.ndarray:
    """Compute a gradient using the parameter-shift rule.

    Args:
        fn: Scalar-valued function that accepts the full parameter array.
        params: Current parameter value(s). Scalars and arbitrary-shaped arrays
            are supported.
        shift: Positive/negative shift applied to each parameter.
        coefficient: Multiplicative factor for the shifted difference. The
            default ``0.5`` is the standard rule for Pauli-rotation generators.

    Returns:
        A NumPy array with the same shape as ``params``.
    """
    theta = np.asarray(params, dtype=float)
    shift_value = float(shift)
    coeff = float(coefficient)
    grad = np.zeros_like(theta, dtype=float)

    for index in np.ndindex(theta.shape):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += shift_value
        minus[index] -= shift_value
        forward = _as_scalar(fn(plus), label="fn(params + shift)")
        backward = _as_scalar(fn(minus), label="fn(params - shift)")
        grad[index] = coeff * (forward - backward)

    return grad


__all__ = ["psr"]

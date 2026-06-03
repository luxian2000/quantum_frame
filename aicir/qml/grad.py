"""Gradient utilities for quantum machine learning workflows."""

from __future__ import annotations

from collections.abc import Callable
import itertools
import numbers
from typing import Any

import numpy as np


def _as_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must return a scalar value")
    return float(array)


def _shifted_difference(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    index: tuple[int, ...],
    shift: float,
    coefficient: float,
) -> float:
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += shift
    minus[index] -= shift
    forward = _as_scalar(fn(plus), label="fn(params + shift)")
    backward = _as_scalar(fn(minus), label="fn(params - shift)")
    return coefficient * (forward - backward)


def _flat_to_index(theta: np.ndarray, flat_index: int) -> tuple[int, ...]:
    if flat_index < 0 or flat_index >= theta.size:
        raise IndexError(f"Parameter index {flat_index} is out of bounds for {theta.size} parameter(s)")
    return np.unravel_index(flat_index, theta.shape)


def _is_multi_index(value: Any, ndim: int) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == ndim
        and all(isinstance(item, numbers.Integral) for item in value)
    )


def _normalize_parameter_indices(params: np.ndarray, parameter_indices: Any) -> list[tuple[int, ...]]:
    if params.size == 0:
        raise ValueError("params must contain at least one parameter")
    if parameter_indices is None:
        return list(np.ndindex(params.shape))
    if params.shape == ():
        if parameter_indices in (0, (), [0], [()]):
            return [()]
        raise IndexError("Scalar params only support parameter index 0 or ()")
    if isinstance(parameter_indices, numbers.Integral):
        return [_flat_to_index(params, int(parameter_indices))]
    if _is_multi_index(parameter_indices, params.ndim):
        return [tuple(int(item) for item in parameter_indices)]

    indices = []
    for item in parameter_indices:
        if isinstance(item, numbers.Integral):
            indices.append(_flat_to_index(params, int(item)))
        elif _is_multi_index(item, params.ndim):
            index = tuple(int(axis_index) for axis_index in item)
            for axis_index, axis_size in zip(index, params.shape):
                if axis_index < 0 or axis_index >= axis_size:
                    raise IndexError(f"Parameter index {index} is out of bounds for shape {params.shape}")
            indices.append(index)
        else:
            raise TypeError("parameter_indices must contain flat integer indices or tuple indices")

    if not indices:
        raise ValueError("parameter_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("parameter_indices must not contain duplicates")
    return indices


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
        grad[index] = _shifted_difference(fn, theta, index, shift_value, coeff)

    return grad


def spsr(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    n_samples: int = 1,
    rng: Any = None,
    replace: bool = False,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    unbiased: bool = True,
) -> np.ndarray:
    """Estimate a parameter-shift gradient by sampling parameter coordinates.

    This stochastic parameter-shift rule evaluates the ordinary two-point shift
    rule only on sampled coordinates. With ``unbiased=True`` the sampled entries
    are scaled so the expectation equals the full ``psr`` gradient.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    sample_count = int(n_samples)
    if sample_count <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not replace and sample_count > theta.size:
        raise ValueError("n_samples cannot exceed the number of parameters when replace=False")

    generator = np.random.default_rng(rng)
    flat_indices = np.atleast_1d(generator.choice(theta.size, size=sample_count, replace=replace))
    scale = theta.size / sample_count if unbiased else 1.0

    shift_value = float(shift)
    coeff = float(coefficient)
    grad = np.zeros_like(theta, dtype=float)
    for flat_index in flat_indices:
        index = _flat_to_index(theta, int(flat_index))
        grad[index] += scale * _shifted_difference(fn, theta, index, shift_value, coeff)
    return grad


def multipsr(
    fn: Callable[[np.ndarray], float],
    params: Any,
    parameter_indices: Any = None,
    *,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
) -> float:
    """Compute a multi-parameter mixed derivative by parameter shifts.

    ``parameter_indices`` selects the coordinates of the mixed derivative. Flat
    integer indices and tuple indices are both supported. If omitted, all
    parameters are used, which requires ``2 ** params.size`` function calls.
    """
    theta = np.asarray(params, dtype=float)
    indices = _normalize_parameter_indices(theta, parameter_indices)
    shift_value = float(shift)
    coeff = float(coefficient)

    total = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=len(indices)):
        shifted = theta.copy()
        sign_product = 1.0
        for index, sign in zip(indices, signs):
            shifted[index] += sign * shift_value
            sign_product *= sign
        total += sign_product * _as_scalar(fn(shifted), label="fn(multivariate shifted params)")
    return float((coeff ** len(indices)) * total)


__all__ = ["psr", "spsr", "multipsr"]

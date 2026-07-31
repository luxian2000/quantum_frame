"""Explicit numerical gradient oracles for distributed validation.

These functions intentionally operate outside :class:`DistSimulator`.  The
distributed simulator remains a forward-only API; callers use these oracles to
compare independently evaluated scalar and vector objectives.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from aicir.qml.deriv import psr


def parameter_shift_gradient(
    objective: Callable[[np.ndarray], Any],
    parameters: Any,
    *,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
) -> np.ndarray:
    """Return a scalar objective gradient using the parameter-shift rule."""

    values = np.asarray(parameters, dtype=np.float64)
    return psr(
        lambda point: float(np.asarray(objective(point)).reshape(())),
        values,
        shift=shift,
        coefficient=coefficient,
    )


def parameter_shift_jacobian(
    objective: Callable[[np.ndarray], Any],
    parameters: Any,
    *,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
) -> np.ndarray:
    """Return a vector objective Jacobian without invoking autograd."""

    values = np.asarray(parameters, dtype=np.float64)
    baseline = np.asarray(objective(values), dtype=np.float64)
    jacobian = np.empty(baseline.shape + values.shape, dtype=np.float64)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += shift
        minus[index] -= shift
        jacobian[(Ellipsis,) + index] = coefficient * (
            np.asarray(objective(plus), dtype=np.float64)
            - np.asarray(objective(minus), dtype=np.float64)
        )
    return jacobian


def finite_difference_gradient(
    objective: Callable[[np.ndarray], Any],
    parameters: Any,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return a central finite-difference gradient for a scalar objective."""

    values = np.asarray(parameters, dtype=np.float64)
    gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (
            float(objective(plus)) - float(objective(minus))
        ) / (2.0 * epsilon)
    return gradient

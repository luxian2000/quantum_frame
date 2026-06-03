"""Gradient-free optimization utilities for quantum machine learning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numbers

import numpy as np


def _torch_or_none():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _is_torch_family_backend(backend: Any) -> bool:
    return backend is not None and type(backend).__name__ in {"TorchBackend", "NPUBackend"}


def _real_torch_dtype_from_backend(backend: Any, torch):
    complex_dtype = getattr(backend, "_dtype", torch.complex64)
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def _as_numpy_scalar(value: Any, *, label: str) -> float:
    torch = _torch_or_none()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must return a scalar value")
        if value.device.type != "cpu":
            raise TypeError(
                f"{label} returned a device tensor; pass backend=... or tensor params "
                "so rotosolve can keep all scalar computations on the device."
            )
        value = value.detach().reshape(())
        if torch.is_complex(value):
            value = torch.real(value)
        return float(value.numpy())

    if hasattr(value, "to_numpy") and callable(value.to_numpy):
        value = value.to_numpy()
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must return a scalar value")
    if np.iscomplexobj(array):
        array = array.real
    return float(array)


def _as_torch_scalar(value: Any, *, label: str, backend: Any = None, reference: Any = None):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU rotosolve path requires PyTorch")

    if not isinstance(value, torch.Tensor):
        if isinstance(reference, torch.Tensor):
            dtype = reference.dtype
            device = reference.device
        else:
            dtype = _real_torch_dtype_from_backend(backend, torch)
            device = getattr(backend, "_device", None)
        value = torch.as_tensor(value, dtype=dtype, device=device)
    if value.numel() != 1:
        raise ValueError(f"{label} must return a scalar value")
    value = value.reshape(())
    if torch.is_complex(value):
        value = torch.real(value)
    return value


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


def _validate_shift(shift: float) -> tuple[float, float, float]:
    shift_value = float(shift)
    sin_shift = float(np.sin(shift_value))
    one_minus_cos_shift = float(1.0 - np.cos(shift_value))
    if abs(sin_shift) <= 1e-15 or abs(one_minus_cos_shift) <= 1e-15:
        raise ValueError("shift must not be an integer multiple of pi")
    return shift_value, sin_shift, one_minus_cos_shift


def _rotosolve_delta_numpy(
    base_value: float,
    plus_value: float,
    minus_value: float,
    *,
    sin_shift: float,
    one_minus_cos_shift: float,
    atol: float,
) -> float:
    cos_coeff = (base_value - 0.5 * (plus_value + minus_value)) / one_minus_cos_shift
    sin_coeff = (plus_value - minus_value) / (2.0 * sin_shift)
    if cos_coeff * cos_coeff + sin_coeff * sin_coeff <= atol * atol:
        return 0.0
    return float(-np.pi / 2.0 - np.arctan2(cos_coeff, sin_coeff))


def _add_at_torch(theta: Any, index: tuple[int, ...], value: Any) -> Any:
    if theta.shape == ():
        return theta + value
    updated = theta.clone()
    updated[index] = updated[index] + value
    return updated


def _rotosolve_delta_torch(
    base_value: Any,
    plus_value: Any,
    minus_value: Any,
    *,
    sin_shift: float,
    one_minus_cos_shift: float,
    atol: float,
):
    torch = _torch_or_none()
    cos_coeff = (base_value - 0.5 * (plus_value + minus_value)) / one_minus_cos_shift
    sin_coeff = (plus_value - minus_value) / (2.0 * sin_shift)
    delta = -torch.pi / 2.0 - torch.atan2(cos_coeff, sin_coeff)
    amplitude_sq = cos_coeff * cos_coeff + sin_coeff * sin_coeff
    threshold = torch.as_tensor(atol * atol, dtype=delta.dtype, device=delta.device)
    return torch.where(amplitude_sq <= threshold, torch.zeros_like(delta), delta)


def _rotosolve_numpy(
    fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    indices: list[tuple[int, ...]],
    *,
    n_sweeps: int,
    shift: float,
    sin_shift: float,
    one_minus_cos_shift: float,
    atol: float,
    return_value: bool,
) -> np.ndarray | tuple[np.ndarray, float]:
    params = theta.copy()
    for _ in range(n_sweeps):
        for index in indices:
            base_value = _as_numpy_scalar(fn(params), label="fn(params)")

            plus = params.copy()
            minus = params.copy()
            plus[index] += shift
            minus[index] -= shift
            plus_value = _as_numpy_scalar(fn(plus), label="fn(params + shift)")
            minus_value = _as_numpy_scalar(fn(minus), label="fn(params - shift)")

            params[index] += _rotosolve_delta_numpy(
                base_value,
                plus_value,
                minus_value,
                sin_shift=sin_shift,
                one_minus_cos_shift=one_minus_cos_shift,
                atol=atol,
            )

    if return_value:
        return params, _as_numpy_scalar(fn(params), label="fn(params)")
    return params


def _rotosolve_torch(
    fn: Callable[[Any], Any],
    params: Any,
    indices: list[tuple[int, ...]],
    *,
    n_sweeps: int,
    shift: float,
    sin_shift: float,
    one_minus_cos_shift: float,
    atol: float,
    backend: Any,
    return_value: bool,
):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU rotosolve path requires PyTorch")

    if isinstance(params, torch.Tensor):
        theta = params.detach().clone()
        if backend is not None and hasattr(backend, "_device"):
            theta = theta.to(device=backend._device)
    else:
        dtype = _real_torch_dtype_from_backend(backend, torch)
        device = getattr(backend, "_device", None)
        theta = torch.tensor(np.asarray(params, dtype=float), dtype=dtype, device=device)

    for _ in range(n_sweeps):
        for index in indices:
            base_value = _as_torch_scalar(fn(theta), label="fn(params)", backend=backend, reference=theta)
            plus = _add_at_torch(theta, index, shift)
            minus = _add_at_torch(theta, index, -shift)
            plus_value = _as_torch_scalar(fn(plus), label="fn(params + shift)", backend=backend, reference=theta)
            minus_value = _as_torch_scalar(fn(minus), label="fn(params - shift)", backend=backend, reference=theta)

            delta = _rotosolve_delta_torch(
                base_value,
                plus_value,
                minus_value,
                sin_shift=sin_shift,
                one_minus_cos_shift=one_minus_cos_shift,
                atol=atol,
            )
            theta = _add_at_torch(theta, index, delta)

    if return_value:
        return theta, _as_torch_scalar(fn(theta), label="fn(params)", backend=backend, reference=theta)
    return theta


def rotosolve(
    fn: Callable[[Any], Any],
    params: Any,
    *,
    n_sweeps: int = 1,
    parameter_indices: Any = None,
    shift: float = np.pi / 2.0,
    atol: float = 1e-12,
    backend: Any = None,
    return_value: bool = False,
) -> Any:
    """Minimize sinusoidal ansatz coordinates with the ROTOSOLVE rule.

    For a coordinate whose loss has the form ``a * sin(theta_k + phi) + c``,
    ROTOSOLVE evaluates the objective at ``theta_k`` and ``theta_k +/- shift``
    and analytically sets ``theta_k`` to the minimum of that trigonometric
    curve. No derivative is computed.

    Args:
        fn: Scalar objective that accepts the full parameter array/tensor.
        params: Current parameter value(s). Scalars and arbitrary-shaped arrays
            are supported.
        n_sweeps: Number of coordinate sweeps.
        parameter_indices: Optional subset of coordinates to update. Supports
            flat integer indices and tuple indices.
        shift: Offset for the two shifted evaluations. The default ``pi/2`` is
            the standard ROTOSOLVE stencil for frequency-one rotations.
        atol: If the fitted sinusoid amplitude is below this tolerance, the
            coordinate is treated as flat and left unchanged.
        backend: Torch/NPU backend. When provided, or when ``params`` is a
            torch tensor, all scalar objective values and trigonometric updates
            stay as torch tensors on the backend device.
        return_value: If ``True``, also return the final objective value.

    Returns:
        Optimized parameters with the same shape as ``params``. NumPy inputs
        return a NumPy array; Torch/NPU inputs or Torch/NPU backends return a
        torch tensor. If ``return_value`` is ``True``, returns
        ``(optimized_params, final_value)``.
    """
    if _is_torch_tensor(params):
        shape = tuple(params.shape)
        if params.numel() == 0:
            raise ValueError("params must contain at least one parameter")
        index_ref = np.empty(shape, dtype=float)
    else:
        index_ref = np.asarray(params, dtype=float)
        if index_ref.size == 0:
            raise ValueError("params must contain at least one parameter")

    sweep_count = int(n_sweeps)
    if sweep_count <= 0:
        raise ValueError("n_sweeps must be a positive integer")
    atol_value = float(atol)
    if atol_value < 0.0:
        raise ValueError("atol must be non-negative")

    shift_value, sin_shift, one_minus_cos_shift = _validate_shift(shift)
    indices = _normalize_parameter_indices(index_ref, parameter_indices)
    use_torch_path = _is_torch_tensor(params) or _is_torch_family_backend(backend)

    if use_torch_path:
        return _rotosolve_torch(
            fn,
            params,
            indices,
            n_sweeps=sweep_count,
            shift=shift_value,
            sin_shift=sin_shift,
            one_minus_cos_shift=one_minus_cos_shift,
            atol=atol_value,
            backend=backend,
            return_value=return_value,
        )

    return _rotosolve_numpy(
        fn,
        index_ref,
        indices,
        n_sweeps=sweep_count,
        shift=shift_value,
        sin_shift=sin_shift,
        one_minus_cos_shift=one_minus_cos_shift,
        atol=atol_value,
        return_value=return_value,
    )


__all__ = ["rotosolve"]

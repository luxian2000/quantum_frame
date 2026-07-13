"""跨子模块共享的返回值/参数归一化 helper。

``psr``/``fd``/``hessian`` 等 (fn_gradient.py, hessian.py) 与 ``qfim``/``qng`` 族
(qfim.py, qng.py) 都需要把目标函数返回值、参数索引、backend 原生张量归一化为
统一形状；``rotosolve.py`` 也复用其中的参数索引与 torch 探测 helper。放在这里
避免 fn_gradient/qfim/qng/rotosolve 之间互相 import 造成循环依赖。
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np


def _as_scalar(value: Any, *, label: str) -> float:
    # Accept backend-native tensors in addition to numpy arrays and Python
    # scalars, so every gradient rule works regardless of which backend
    # produced the objective value. In particular a NumpyBackend/GPUBackend/
    # NPUBackend objective may return a tensor that is autograd-tracked, lives
    # on an accelerator device (NPU/CUDA), and/or is complex. Detaching and
    # moving to host first avoids numpy conversion errors on such tensors.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must return a scalar value")
    if np.iscomplexobj(array):
        # Objectives (expectation values) are real; drop negligible imaginary
        # noise rather than failing in float().
        array = array.real
    return float(array)


def _as_state_vector(value: Any, *, label: str) -> np.ndarray:
    """Convert a backend-native pure state to a normalized 1-D NumPy vector."""
    if hasattr(value, "to_numpy") and callable(value.to_numpy):
        value = value.to_numpy()
    else:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()

    raw = np.asarray(value, dtype=np.complex128)
    if raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError(f"{label} must return a pure state vector, not a density matrix/operator")

    array = raw.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{label} must return a non-empty state vector")
    norm = float(np.linalg.norm(array))
    if not norm > 0.0:
        raise ValueError(f"{label} must return a state vector with non-zero norm")
    return array / norm


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


def _index_to_flat(theta: np.ndarray, index: tuple[int, ...]) -> int:
    if theta.shape == ():
        if index == ():
            return 0
        raise IndexError("Scalar params only support parameter index ()")
    return int(np.ravel_multi_index(index, theta.shape))


def _normalize_qng_blocks(
    params: np.ndarray,
    blocks: Any,
    block_size: int,
) -> list[list[tuple[int, ...]]]:
    if params.size == 0:
        raise ValueError("params must contain at least one parameter")

    if blocks is None:
        size = int(block_size)
        if size <= 0:
            raise ValueError("block_size must be a positive integer")
        flat_blocks = [
            range(start, min(start + size, params.size))
            for start in range(0, params.size, size)
        ]
        return [[_flat_to_index(params, flat_index) for flat_index in block] for block in flat_blocks]

    if isinstance(blocks, numbers.Integral) or _is_multi_index(blocks, params.ndim):
        raw_blocks = [blocks]
    else:
        raw_blocks = list(blocks)

    if not raw_blocks:
        raise ValueError("blocks must not be empty")

    normalized = []
    seen = set()
    for block in raw_blocks:
        indices = _normalize_parameter_indices(params, block)
        normalized.append(indices)
        for index in indices:
            if index in seen:
                raise ValueError("blocks must not contain duplicate parameter indices")
            seen.add(index)

    expected = set(np.ndindex(params.shape))
    if seen != expected:
        missing = sorted(_index_to_flat(params, index) for index in expected - seen)
        extra = sorted(_index_to_flat(params, index) for index in seen - expected)
        details = []
        if missing:
            details.append(f"missing flat indices {missing}")
        if extra:
            details.append(f"extra flat indices {extra}")
        raise ValueError("blocks must cover every parameter exactly once" + (": " + ", ".join(details) if details else ""))
    return normalized


def _torch_or_none():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _real_torch_dtype_from_backend(backend: Any, torch):
    complex_dtype = getattr(backend, "_dtype", torch.complex64)
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def _state_tensor_and_backend(value: Any, backend: Any = None) -> tuple[Any, Any]:
    if all(hasattr(value, attr) for attr in ("data", "backend", "n_qubits")):
        return value.data, backend if backend is not None else value.backend
    return value, backend


def _as_torch_state_vector(value: Any, *, label: str, backend: Any = None):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU QFIM path requires PyTorch")

    raw, resolved_backend = _state_tensor_and_backend(value, backend)
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"{label} must return a Torch/NPU tensor or a State backed by Torch/NPU")
    if raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError(f"{label} must return a pure state vector, not a density matrix/operator")

    if resolved_backend is not None and hasattr(resolved_backend, "_device"):
        raw = raw.to(device=resolved_backend._device)
    flat = raw.reshape(-1)
    if resolved_backend is not None and hasattr(resolved_backend, "abs_sq"):
        norm_sq = resolved_backend.abs_sq(flat).sum()
    elif torch.is_complex(flat):
        norm_sq = (torch.real(flat) ** 2 + torch.imag(flat) ** 2).sum()
    else:
        norm_sq = (flat ** 2).sum()
    norm = torch.sqrt(torch.real(norm_sq))
    return flat / norm, resolved_backend


def _contains_torch_tensor(value: Any) -> bool:
    if _is_torch_tensor(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_torch_tensor(item) for item in value)
    return False

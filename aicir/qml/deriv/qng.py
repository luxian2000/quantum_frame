"""量子自然梯度（QNG）族：``qng``/``bdqng``/``kqng``/``dqng``。

对 NPU-family backend（或 Torch/NPU 张量输入）全程走 torch 私有变体，保持梯度/
QFIM/Kronecker 因子在设备上，不做 ``.cpu()``/``to_numpy()`` 转换；NumPy 路径与
torch 路径是刻意保留的双实现（NPU complex64 kernel 缺口，见仓库 CLAUDE.md），
不做合并。返回值契约见包 ``__init__.py``：torch 入参/backend → 同设备 torch
出参，否则一律 NumPy。
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any

import numpy as np

from ._coerce import (
    _as_torch_state_vector,
    _contains_torch_tensor,
    _flat_to_index,
    _index_to_flat,
    _is_torch_tensor,
    _normalize_qng_blocks,
    _real_torch_dtype_from_backend,
    _torch_or_none,
)
from .fn_gradient import (
    _auto_torch_device,
    _fd_torch,
    _psr_torch,
    _spsa_torch,
    auto,
    fd,
    psr,
    spsa,
)
from .qfim import (
    _qfim_blocks_from_state_fd,
    _qfim_diag_from_state_fd,
    _qfim_diag_from_state_fd_torch,
    _qfim_from_derivatives_torch,
    _qfim_from_state_fd,
    _state_derivatives_fd_torch,
)


def _validate_qfim(qfim: Any, parameter_count: int) -> np.ndarray:
    matrix = np.asarray(qfim, dtype=float)
    expected = (parameter_count, parameter_count)
    if matrix.shape != expected:
        raise ValueError(f"qfim must have shape {expected}, got {matrix.shape}")
    return 0.5 * (matrix + matrix.T)


def _is_npu_family_backend(backend: Any) -> bool:
    if backend is None:
        return False
    device = getattr(backend, "_device", None)
    device_type = getattr(device, "type", None)
    return device_type == "npu" or type(backend).__name__ == "NPUBackend"


def _torch_real_tensor(value: Any, *, backend: Any, shape: tuple[int, ...], reference: Any = None):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU gradient path requires PyTorch")

    if backend is None and isinstance(reference, torch.Tensor):
        dtype = reference.dtype if not torch.is_complex(reference) else torch.float64 if reference.dtype == torch.complex128 else torch.float32
        device = reference.device
    else:
        dtype = _real_torch_dtype_from_backend(backend, torch)
        device = getattr(backend, "_device", None)
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=dtype, device=device)
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tuple(tensor.shape) != shape:
        try:
            tensor = tensor.reshape(shape)
        except RuntimeError as exc:
            raise ValueError(f"tensor must have shape {shape}, got {tuple(tensor.shape)}") from exc
    return tensor


def _solve_damped_system(metric: np.ndarray, gradient: np.ndarray, damping: float) -> np.ndarray:
    damping_value = float(damping)
    if damping_value < 0.0:
        raise ValueError("damping must be non-negative")

    regularized = metric + damping_value * np.eye(metric.shape[0], dtype=float)
    try:
        return np.linalg.solve(regularized, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized) @ gradient


def _validate_qfim_diag(qfim_diag: Any, parameter_count: int) -> np.ndarray:
    diag = np.asarray(qfim_diag, dtype=float).reshape(-1)
    if diag.shape != (parameter_count,):
        raise ValueError(f"qfim_diag must have shape {(parameter_count,)}, got {diag.shape}")
    return diag


def _ordinary_gradient_for_qng(
    fn: Callable[[Any], Any] | None,
    theta: np.ndarray,
    *,
    method: str,
    shift: float,
    coefficient: float,
    backend: Any,
    gradient_kwargs: dict[str, Any],
) -> np.ndarray:
    if fn is None:
        raise ValueError("fn is required when grad is not provided")

    method_norm = str(method).strip().lower()
    kwargs = dict(gradient_kwargs)
    if method_norm == "psr":
        kwargs.setdefault("shift", shift)
        kwargs.setdefault("coefficient", coefficient)
        return psr(fn, theta, **kwargs)
    if method_norm == "fd":
        return fd(fn, theta, **kwargs)
    if method_norm == "spsa":
        return spsa(fn, theta, **kwargs)
    if method_norm == "auto":
        kwargs.setdefault("backend", backend)
        return auto(fn, theta, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', 'spsa', or 'auto'")


def _ordinary_gradient_for_dqng_torch(
    fn: Callable[[Any], Any] | None,
    theta: np.ndarray,
    *,
    method: str,
    shift: float,
    coefficient: float,
    backend: Any,
    gradient_kwargs: dict[str, Any],
):
    if fn is None:
        raise ValueError("fn is required when grad is not provided")

    method_norm = str(method).strip().lower()
    kwargs = dict(gradient_kwargs)
    if method_norm == "psr":
        return _psr_torch(
            fn,
            theta,
            shift=float(kwargs.pop("shift", shift)),
            coefficient=float(kwargs.pop("coefficient", coefficient)),
            backend=backend,
        )
    if method_norm == "fd":
        return _fd_torch(
            fn,
            theta,
            eps=float(kwargs.pop("eps", 1e-3)),
            mode=kwargs.pop("mode", "central"),
            backend=backend,
        )
    if method_norm == "spsa":
        return _spsa_torch(
            fn,
            theta,
            eps=float(kwargs.pop("eps", 1e-3)),
            n_samples=int(kwargs.pop("n_samples", 1)),
            rng=kwargs.pop("rng", None),
            perturbations=kwargs.pop("perturbations", None),
            backend=backend,
        )
    if method_norm == "auto":
        kwargs.setdefault("backend", backend)
        return _auto_torch_device(fn, theta, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', 'spsa', or 'auto'")


def _normalize_kfac_factor_shapes(factor_shapes: Any, block_count: int) -> list[tuple[int, int]] | None:
    if factor_shapes is None:
        return None
    if (
        isinstance(factor_shapes, (list, tuple))
        and len(factor_shapes) == 2
        and all(isinstance(item, numbers.Integral) for item in factor_shapes)
    ):
        shapes = [tuple(int(item) for item in factor_shapes)]
    else:
        shapes = [tuple(int(item) for item in shape) for shape in factor_shapes]
    if len(shapes) != block_count:
        raise ValueError(f"factor_shapes must contain {block_count} shape(s), got {len(shapes)}")
    for shape in shapes:
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("each factor shape must be a pair of positive integers")
    return shapes


def _normalize_kfac_blocks(
    params: np.ndarray,
    blocks: Any,
    factor_shapes: Any,
    block_size: int | None,
) -> tuple[list[list[tuple[int, ...]]], list[tuple[int, int]]]:
    if params.size == 0:
        raise ValueError("params must contain at least one parameter")

    if blocks is None:
        preliminary_shapes = None
        if factor_shapes is not None:
            if (
                isinstance(factor_shapes, (list, tuple))
                and len(factor_shapes) == 2
                and all(isinstance(item, numbers.Integral) for item in factor_shapes)
            ):
                preliminary_shapes = [tuple(int(item) for item in factor_shapes)]
            else:
                preliminary_shapes = [tuple(int(item) for item in shape) for shape in factor_shapes]
            products = [shape[0] * shape[1] for shape in preliminary_shapes]
            if sum(products) != params.size:
                raise ValueError("factor_shapes products must sum to the number of parameters")
            normalized_blocks = []
            start = 0
            for size in products:
                normalized_blocks.append([
                    _flat_to_index(params, flat_index)
                    for flat_index in range(start, start + size)
                ])
                start += size
            return normalized_blocks, _normalize_kfac_factor_shapes(preliminary_shapes, len(normalized_blocks))

        if block_size is not None:
            normalized_blocks = _normalize_qng_blocks(params, None, int(block_size))
            return normalized_blocks, [(len(block), 1) for block in normalized_blocks]

        if params.shape != () and params.ndim == 2:
            normalized_blocks = [list(np.ndindex(params.shape))]
            return normalized_blocks, [tuple(int(dim) for dim in params.shape)]

        normalized_blocks = [list(np.ndindex(params.shape))]
        return normalized_blocks, [(params.size, 1)]

    normalized_blocks = _normalize_qng_blocks(params, blocks, block_size or 1)
    shapes = _normalize_kfac_factor_shapes(factor_shapes, len(normalized_blocks))
    if shapes is None:
        shapes = [(len(block), 1) for block in normalized_blocks]
    for block, shape in zip(normalized_blocks, shapes):
        if shape[0] * shape[1] != len(block):
            raise ValueError(
                f"factor shape {shape} is incompatible with block of size {len(block)}"
            )
    return normalized_blocks, shapes


def _kfac_factors_from_qfim_block(metric: np.ndarray, factor_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = factor_shape
    expected = rows * cols
    if metric.shape != (expected, expected):
        raise ValueError(f"qfim block must have shape {(expected, expected)}, got {metric.shape}")

    block = np.asarray(metric, dtype=float).reshape(rows, cols, rows, cols)
    left = np.zeros((rows, rows), dtype=float)
    right = np.zeros((cols, cols), dtype=float)
    for col in range(cols):
        left += block[:, col, :, col]
    for row in range(rows):
        right += block[row, :, row, :]

    trace = float(np.trace(metric))
    scale = float(np.sqrt(abs(trace))) if abs(trace) > 1e-15 else 1.0
    left = 0.5 * (left / scale + (left / scale).T)
    right = 0.5 * (right / scale + (right / scale).T)
    return left, right


def _qfim_kfac_factors_from_state_fd(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    blocks: list[list[tuple[int, ...]]],
    factor_shapes: list[tuple[int, int]],
    *,
    eps: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    qfim_blocks = _qfim_blocks_from_state_fd(state_fn, theta, blocks, eps=eps)
    return [
        _kfac_factors_from_qfim_block(metric, shape)
        for metric, shape in zip(qfim_blocks, factor_shapes)
    ]


def _validate_kfac_factors(
    factors: Any,
    factor_shapes: list[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = list(factors)
    if len(pairs) != len(factor_shapes):
        raise ValueError(f"kfac_factors must contain {len(factor_shapes)} pair(s), got {len(pairs)}")

    normalized = []
    for pair, (rows, cols) in zip(pairs, factor_shapes):
        if len(pair) != 2:
            raise ValueError("each KFAC factor entry must be a pair (left, right)")
        left = np.asarray(pair[0], dtype=float)
        right = np.asarray(pair[1], dtype=float)
        if left.shape != (rows, rows):
            raise ValueError(f"left KFAC factor must have shape {(rows, rows)}, got {left.shape}")
        if right.shape != (cols, cols):
            raise ValueError(f"right KFAC factor must have shape {(cols, cols)}, got {right.shape}")
        normalized.append((0.5 * (left + left.T), 0.5 * (right + right.T)))
    return normalized


def _solve_linear_matrix(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def _solve_kfac_factor_block(
    left: np.ndarray,
    right: np.ndarray,
    gradient: np.ndarray,
    factor_shape: tuple[int, int],
    damping: float,
) -> np.ndarray:
    damping_value = float(damping)
    if damping_value < 0.0:
        raise ValueError("damping must be non-negative")
    factor_damping = float(np.sqrt(damping_value))
    left_reg = left + factor_damping * np.eye(left.shape[0], dtype=float)
    right_reg = right + factor_damping * np.eye(right.shape[0], dtype=float)

    grad_matrix = np.asarray(gradient, dtype=float).reshape(factor_shape)
    tmp = _solve_linear_matrix(left_reg, grad_matrix)
    solved = _solve_linear_matrix(right_reg, tmp.T).T
    return solved.reshape(-1)


def _validate_qfim_blocks(qfim_blocks: Any, blocks: list[list[tuple[int, ...]]]) -> list[np.ndarray]:
    matrices = list(qfim_blocks)
    if len(matrices) != len(blocks):
        raise ValueError(f"qfim_blocks must contain {len(blocks)} block(s), got {len(matrices)}")

    normalized = []
    for matrix, block in zip(matrices, blocks):
        array = np.asarray(matrix, dtype=float)
        expected = (len(block), len(block))
        if array.shape != expected:
            raise ValueError(f"qfim block must have shape {expected}, got {array.shape}")
        normalized.append(0.5 * (array + array.T))
    return normalized


def _kfac_factors_from_qfim_block_torch(metric: Any, factor_shape: tuple[int, int]):
    torch = _torch_or_none()
    rows, cols = factor_shape
    expected = rows * cols
    if tuple(metric.shape) != (expected, expected):
        raise ValueError(f"qfim block must have shape {(expected, expected)}, got {tuple(metric.shape)}")

    block = metric.reshape(rows, cols, rows, cols)
    left = sum(block[:, col, :, col] for col in range(cols))
    right = sum(block[row, :, row, :] for row in range(rows))
    trace = torch.real(torch.trace(metric))
    eps = torch.finfo(trace.dtype).eps
    scale = torch.sqrt(torch.clamp(torch.abs(trace), min=eps))
    left = left / scale
    right = right / scale
    left = 0.5 * (left + left.transpose(-2, -1))
    right = 0.5 * (right + right.transpose(-2, -1))
    return left, right


def _qfim_kfac_factors_from_state_fd_torch(
    state_fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    blocks: list[list[tuple[int, ...]]],
    factor_shapes: list[tuple[int, int]],
    *,
    eps: float,
    backend: Any = None,
):
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("metric_eps must be a positive number")

    base_state, resolved_backend = _as_torch_state_vector(
        state_fn(theta),
        label="state_fn(params)",
        backend=backend,
    )
    factors = []
    for block, factor_shape in zip(blocks, factor_shapes):
        derivatives, resolved_backend = _state_derivatives_fd_torch(
            state_fn,
            theta,
            block,
            base_state=base_state,
            eps=eps_value,
            backend=resolved_backend,
        )
        metric = _qfim_from_derivatives_torch(base_state, derivatives, resolved_backend)
        factors.append(_kfac_factors_from_qfim_block_torch(metric, factor_shape))
    return factors, resolved_backend


def _validate_kfac_factors_torch(
    factors: Any,
    factor_shapes: list[tuple[int, int]],
    *,
    backend: Any = None,
    reference: Any = None,
) -> list[tuple[Any, Any]]:
    pairs = list(factors)
    if len(pairs) != len(factor_shapes):
        raise ValueError(f"kfac_factors must contain {len(factor_shapes)} pair(s), got {len(pairs)}")

    normalized = []
    for pair, (rows, cols) in zip(pairs, factor_shapes):
        if len(pair) != 2:
            raise ValueError("each KFAC factor entry must be a pair (left, right)")
        left = _torch_real_tensor(pair[0], backend=backend, shape=(rows, rows), reference=reference)
        right = _torch_real_tensor(pair[1], backend=backend, shape=(cols, cols), reference=left)
        normalized.append((
            0.5 * (left + left.transpose(-2, -1)),
            0.5 * (right + right.transpose(-2, -1)),
        ))
    return normalized


def _solve_kfac_factor_block_torch(
    left: Any,
    right: Any,
    gradient: Any,
    factor_shape: tuple[int, int],
    damping: float,
):
    torch = _torch_or_none()
    damping_value = float(damping)
    if damping_value < 0.0:
        raise ValueError("damping must be non-negative")
    factor_damping = torch.sqrt(torch.as_tensor(damping_value, dtype=left.dtype, device=left.device))
    left_reg = left + factor_damping * torch.eye(left.shape[0], dtype=left.dtype, device=left.device)
    right_reg = right + factor_damping * torch.eye(right.shape[0], dtype=right.dtype, device=right.device)

    grad_matrix = gradient.reshape(factor_shape)
    tmp = torch.linalg.solve(left_reg, grad_matrix)
    solved = torch.linalg.solve(right_reg, tmp.transpose(-2, -1)).transpose(-2, -1)
    return solved.reshape(-1)


def qng(
    fn: Callable[[Any], Any] | None,
    state_fn: Callable[[np.ndarray], Any] | None,
    params: Any,
    *,
    grad: Any = None,
    qfim: Any = None,
    gradient_method: str = "psr",
    gradient_kwargs: dict[str, Any] | None = None,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    metric_eps: float = 1e-3,
    damping: float = 1e-6,
    backend: Any = None,
    return_gradient: bool = False,
    return_qfim: bool = False,
) -> np.ndarray | tuple[Any, ...]:
    """Compute a Quantum Natural Gradient (QNG) direction.

    QNG preconditions an ordinary objective gradient by the inverse Quantum
    Fisher Information Matrix (QFIM):

        natural_grad = (F + damping * I)^(-1) @ grad

    ``fn`` is the scalar objective and ``state_fn`` returns the pure ansatz
    state at the same parameters. By default the ordinary gradient is computed
    with :func:`psr`, while the QFIM is estimated from central finite
    differences of ``state_fn`` using the pure-state Fubini-Study metric.

    Backend compatibility:
        ``state_fn`` may return a NumPy array, a backend-native tensor
        (including Torch/NPU tensors), or an aicir ``State`` object. Device
        tensors are detached and moved to host only for the small QFIM solve,
        avoiding assumptions about NPU complex linear-solve support.

    Args:
        fn: Scalar objective ``fn(params)``. Required unless ``grad`` is given.
        state_fn: Pure-state ansatz function ``state_fn(params)``. Required
            unless ``qfim`` is given.
        params: Current parameter value(s). Scalars and arbitrary-shaped arrays
            are supported; the returned natural gradient has the same shape.
        grad: Optional precomputed ordinary gradient.
        qfim: Optional precomputed QFIM.
        gradient_method: Ordinary-gradient method when ``grad`` is omitted:
            ``"psr"`` (default), ``"fd"``, ``"spsa"``, or ``"auto"``.
        gradient_kwargs: Extra keyword arguments forwarded to the selected
            ordinary-gradient method.
        shift: Parameter-shift amount used by ``gradient_method="psr"``.
        coefficient: Parameter-shift coefficient used by
            ``gradient_method="psr"``.
        metric_eps: Central-difference step for QFIM state derivatives.
        damping: Non-negative Tikhonov damping added to QFIM before solving.
        backend: Torch-family backend used only by ``gradient_method="auto"``.
        return_gradient: If ``True``, include the ordinary gradient.
        return_qfim: If ``True``, include the QFIM.

    Returns:
        Natural gradient with the same shape as ``params``. Optional returns
        are appended as ``(natural_grad, gradient, qfim)`` according to the
        requested flags.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    if grad is None:
        gradient = _ordinary_gradient_for_qng(
            fn,
            theta,
            method=gradient_method,
            shift=float(shift),
            coefficient=float(coefficient),
            backend=backend,
            gradient_kwargs={} if gradient_kwargs is None else gradient_kwargs,
        )
    else:
        gradient = np.asarray(grad, dtype=float)

    if gradient.shape != theta.shape:
        try:
            gradient = gradient.reshape(theta.shape)
        except ValueError as exc:
            raise ValueError(
                f"grad must have shape {theta.shape}, got {gradient.shape}"
            ) from exc

    if qfim is None:
        if state_fn is None:
            raise ValueError("state_fn is required when qfim is not provided")
        metric = _qfim_from_state_fd(state_fn, theta, eps=metric_eps)
    else:
        metric = _validate_qfim(qfim, theta.size)

    flat_grad = gradient.reshape(-1)
    natural_flat = _solve_damped_system(metric, flat_grad, damping)

    natural = np.asarray(natural_flat, dtype=float).reshape(theta.shape)
    extras: list[Any] = []
    if return_gradient:
        extras.append(gradient)
    if return_qfim:
        extras.append(metric)
    if extras:
        return (natural, *extras)
    return natural


def bdqng(
    fn: Callable[[Any], Any] | None,
    state_fn: Callable[[np.ndarray], Any] | None,
    params: Any,
    *,
    blocks: Any = None,
    block_size: int = 1,
    grad: Any = None,
    qfim_blocks: Any = None,
    gradient_method: str = "psr",
    gradient_kwargs: dict[str, Any] | None = None,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    metric_eps: float = 1e-3,
    damping: float = 1e-6,
    backend: Any = None,
    return_gradient: bool = False,
    return_qfim_blocks: bool = False,
) -> np.ndarray | tuple[Any, ...]:
    """Compute a block-diagonal Quantum Natural Gradient direction.

    ``bdqng`` is the tractable block-diagonal approximation to :func:`qng`.
    Instead of building and inverting the full ``P x P`` QFIM, it partitions
    parameters into blocks and solves one natural-gradient system per block:

        natural_grad_B = (F_B + damping * I)^(-1) @ grad_B

    Cross-block QFIM entries are assumed to be zero. This reduces solve cost
    from one dense ``P x P`` system to multiple smaller systems and lets large
    ansatzes use QNG-style geometry with controllable block sizes.

    Args:
        fn: Scalar objective ``fn(params)``. Required unless ``grad`` is given.
        state_fn: Pure-state ansatz function ``state_fn(params)``. Required
            unless ``qfim_blocks`` is given.
        params: Current parameter value(s). Scalars and arbitrary-shaped arrays
            are supported; the returned natural gradient has the same shape.
        blocks: Optional explicit parameter blocks. Each block contains flat
            integer indices or tuple indices, e.g. ``[[0, 1], [2, 3]]``.
            Blocks must cover every parameter exactly once.
        block_size: Contiguous block size used when ``blocks`` is omitted.
            Defaults to ``1`` (diagonal QNG).
        grad: Optional precomputed ordinary gradient.
        qfim_blocks: Optional precomputed QFIM block matrices, one per block.
        gradient_method: Ordinary-gradient method when ``grad`` is omitted:
            ``"psr"`` (default), ``"fd"``, ``"spsa"``, or ``"auto"``.
        gradient_kwargs: Extra keyword arguments forwarded to the selected
            ordinary-gradient method.
        shift: Parameter-shift amount used by ``gradient_method="psr"``.
        coefficient: Parameter-shift coefficient used by
            ``gradient_method="psr"``.
        metric_eps: Central-difference step for block-QFIM state derivatives.
        damping: Non-negative Tikhonov damping added to each QFIM block before
            solving.
        backend: Torch-family backend used only by ``gradient_method="auto"``.
        return_gradient: If ``True``, include the ordinary gradient.
        return_qfim_blocks: If ``True``, include the list of QFIM blocks.

    Returns:
        Block-diagonal natural gradient with the same shape as ``params``.
        Optional returns are appended as ``(natural_grad, gradient,
        qfim_blocks)`` according to the requested flags.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    normalized_blocks = _normalize_qng_blocks(theta, blocks, block_size)

    if grad is None:
        gradient = _ordinary_gradient_for_qng(
            fn,
            theta,
            method=gradient_method,
            shift=float(shift),
            coefficient=float(coefficient),
            backend=backend,
            gradient_kwargs={} if gradient_kwargs is None else gradient_kwargs,
        )
    else:
        gradient = np.asarray(grad, dtype=float)

    if gradient.shape != theta.shape:
        try:
            gradient = gradient.reshape(theta.shape)
        except ValueError as exc:
            raise ValueError(
                f"grad must have shape {theta.shape}, got {gradient.shape}"
            ) from exc

    if qfim_blocks is None:
        if state_fn is None:
            raise ValueError("state_fn is required when qfim_blocks is not provided")
        metric_blocks = _qfim_blocks_from_state_fd(
            state_fn,
            theta,
            normalized_blocks,
            eps=metric_eps,
        )
    else:
        metric_blocks = _validate_qfim_blocks(qfim_blocks, normalized_blocks)

    flat_grad = gradient.reshape(-1)
    natural_flat = np.zeros(theta.size, dtype=float)
    for block, metric in zip(normalized_blocks, metric_blocks):
        flat_indices = [_index_to_flat(theta, index) for index in block]
        block_grad = flat_grad[flat_indices]
        natural_flat[flat_indices] = _solve_damped_system(metric, block_grad, damping)

    natural = natural_flat.reshape(theta.shape)
    extras: list[Any] = []
    if return_gradient:
        extras.append(gradient)
    if return_qfim_blocks:
        extras.append(metric_blocks)
    if extras:
        return (natural, *extras)
    return natural


def kqng(
    fn: Callable[[Any], Any] | None,
    state_fn: Callable[[np.ndarray], Any] | None,
    params: Any,
    *,
    blocks: Any = None,
    factor_shapes: Any = None,
    block_size: int | None = None,
    grad: Any = None,
    kfac_factors: Any = None,
    gradient_method: str = "psr",
    gradient_kwargs: dict[str, Any] | None = None,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    metric_eps: float = 1e-3,
    damping: float = 1e-6,
    backend: Any = None,
    return_gradient: bool = False,
    return_kfac_factors: bool = False,
) -> Any:
    """Compute a KFAC-style Quantum Natural Gradient direction.

    KFAC-style QNG approximates each selected QFIM block as a Kronecker
    product ``A ⊗ B``. A block gradient is reshaped to ``(A_dim, B_dim)`` and
    preconditioned by solving two smaller systems:

        A_damped @ X @ B_damped.T = grad_block

    For NPU-family backends (or Torch/NPU tensor factors/gradients), all
    ordinary-gradient, factor-estimation, and Kronecker-factor solves stay as
    Torch/NPU tensors. That path intentionally avoids ``.cpu()`` and
    ``to_numpy()``.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    normalized_blocks, normalized_shapes = _normalize_kfac_blocks(
        theta,
        blocks,
        factor_shapes,
        block_size,
    )
    kwargs = {} if gradient_kwargs is None else gradient_kwargs
    use_torch_path = (
        _is_npu_family_backend(backend)
        or _is_torch_tensor(grad)
        or _contains_torch_tensor(kfac_factors)
    )

    if use_torch_path:
        tensor_reference = (
            grad if _is_torch_tensor(grad)
            else kfac_factors[0][0] if _contains_torch_tensor(kfac_factors) else None
        )
        if grad is None:
            gradient = _ordinary_gradient_for_dqng_torch(
                fn,
                theta,
                method=gradient_method,
                shift=float(shift),
                coefficient=float(coefficient),
                backend=backend,
                gradient_kwargs=kwargs,
            )
        else:
            gradient = _torch_real_tensor(grad, backend=backend, shape=theta.shape, reference=tensor_reference)

        if kfac_factors is None:
            if state_fn is None:
                raise ValueError("state_fn is required when kfac_factors is not provided")
            factors, resolved_backend = _qfim_kfac_factors_from_state_fd_torch(
                state_fn,
                theta,
                normalized_blocks,
                normalized_shapes,
                eps=metric_eps,
                backend=backend,
            )
            if backend is None:
                backend = resolved_backend
        else:
            factors = _validate_kfac_factors_torch(
                kfac_factors,
                normalized_shapes,
                backend=backend,
                reference=gradient,
            )

        natural_flat = gradient.reshape(-1).new_zeros(theta.size)
        flat_grad = gradient.reshape(-1)
        for block, factor_shape, (left, right) in zip(normalized_blocks, normalized_shapes, factors):
            flat_indices = [_index_to_flat(theta, index) for index in block]
            index_tensor = _torch_or_none().as_tensor(flat_indices, dtype=_torch_or_none().long, device=gradient.device)
            block_grad = flat_grad.index_select(0, index_tensor)
            natural_flat.index_copy_(
                0,
                index_tensor,
                _solve_kfac_factor_block_torch(left, right, block_grad, factor_shape, damping),
            )

        natural = natural_flat.reshape(theta.shape)
        extras: list[Any] = []
        if return_gradient:
            extras.append(gradient)
        if return_kfac_factors:
            extras.append(factors)
        if extras:
            return (natural, *extras)
        return natural

    if grad is None:
        gradient = _ordinary_gradient_for_qng(
            fn,
            theta,
            method=gradient_method,
            shift=float(shift),
            coefficient=float(coefficient),
            backend=backend,
            gradient_kwargs=kwargs,
        )
    else:
        gradient = np.asarray(grad, dtype=float)

    if gradient.shape != theta.shape:
        try:
            gradient = gradient.reshape(theta.shape)
        except ValueError as exc:
            raise ValueError(
                f"grad must have shape {theta.shape}, got {gradient.shape}"
            ) from exc

    if kfac_factors is None:
        if state_fn is None:
            raise ValueError("state_fn is required when kfac_factors is not provided")
        factors = _qfim_kfac_factors_from_state_fd(
            state_fn,
            theta,
            normalized_blocks,
            normalized_shapes,
            eps=metric_eps,
        )
    else:
        factors = _validate_kfac_factors(kfac_factors, normalized_shapes)

    natural_flat = np.zeros(theta.size, dtype=float)
    flat_grad = gradient.reshape(-1)
    for block, factor_shape, (left, right) in zip(normalized_blocks, normalized_shapes, factors):
        flat_indices = [_index_to_flat(theta, index) for index in block]
        block_grad = flat_grad[flat_indices]
        natural_flat[flat_indices] = _solve_kfac_factor_block(
            left,
            right,
            block_grad,
            factor_shape,
            damping,
        )

    natural = natural_flat.reshape(theta.shape)
    extras: list[Any] = []
    if return_gradient:
        extras.append(gradient)
    if return_kfac_factors:
        extras.append(factors)
    if extras:
        return (natural, *extras)
    return natural


def dqng(
    fn: Callable[[Any], Any] | None,
    state_fn: Callable[[np.ndarray], Any] | None,
    params: Any,
    *,
    grad: Any = None,
    qfim_diag: Any = None,
    gradient_method: str = "psr",
    gradient_kwargs: dict[str, Any] | None = None,
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    metric_eps: float = 1e-3,
    damping: float = 1e-6,
    backend: Any = None,
    return_gradient: bool = False,
    return_qfim_diag: bool = False,
) -> Any:
    """Compute a diagonal Quantum Natural Gradient direction.

    Diagonal QNG is the cheapest QNG approximation. It keeps only the diagonal
    entries of the QFIM and preconditions each gradient coordinate
    independently:

        natural_grad_i = grad_i / (F_ii + damping)

    For NumPy ansatzes this returns NumPy arrays. For NPU-family backends (or
    when ``grad``/``qfim_diag`` are Torch tensors), all scalar gradients, state
    derivatives, QFIM diagonal entries, and the final division stay as
    Torch/NPU tensors. That path intentionally avoids ``.cpu()`` and
    ``to_numpy()`` so device-resident ansatz execution does not move gradient
    data back to host memory.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    use_torch_path = (
        _is_npu_family_backend(backend)
        or _is_torch_tensor(grad)
        or _is_torch_tensor(qfim_diag)
    )
    kwargs = {} if gradient_kwargs is None else gradient_kwargs

    if use_torch_path:
        tensor_reference = qfim_diag if _is_torch_tensor(qfim_diag) else grad if _is_torch_tensor(grad) else None
        if grad is None:
            gradient = _ordinary_gradient_for_dqng_torch(
                fn,
                theta,
                method=gradient_method,
                shift=float(shift),
                coefficient=float(coefficient),
                backend=backend,
                gradient_kwargs=kwargs,
            )
        else:
            gradient = _torch_real_tensor(grad, backend=backend, shape=theta.shape, reference=tensor_reference)

        if qfim_diag is None:
            if state_fn is None:
                raise ValueError("state_fn is required when qfim_diag is not provided")
            diag, resolved_backend = _qfim_diag_from_state_fd_torch(
                state_fn,
                theta,
                eps=metric_eps,
                backend=backend,
            )
            if backend is None:
                backend = resolved_backend
        else:
            diag = _torch_real_tensor(qfim_diag, backend=backend, shape=(theta.size,), reference=gradient)

        torch = _torch_or_none()
        damping_value = float(damping)
        if damping_value < 0.0:
            raise ValueError("damping must be non-negative")
        damping_tensor = torch.as_tensor(damping_value, dtype=diag.dtype, device=diag.device)

        natural_flat = gradient.reshape(-1) / (diag.reshape(-1) + damping_tensor)
        natural = natural_flat.reshape(theta.shape)
        extras: list[Any] = []
        if return_gradient:
            extras.append(gradient)
        if return_qfim_diag:
            extras.append(diag)
        if extras:
            return (natural, *extras)
        return natural

    if grad is None:
        gradient = _ordinary_gradient_for_qng(
            fn,
            theta,
            method=gradient_method,
            shift=float(shift),
            coefficient=float(coefficient),
            backend=backend,
            gradient_kwargs=kwargs,
        )
    else:
        gradient = np.asarray(grad, dtype=float)

    if gradient.shape != theta.shape:
        try:
            gradient = gradient.reshape(theta.shape)
        except ValueError as exc:
            raise ValueError(
                f"grad must have shape {theta.shape}, got {gradient.shape}"
            ) from exc

    if qfim_diag is None:
        if state_fn is None:
            raise ValueError("state_fn is required when qfim_diag is not provided")
        diag = _qfim_diag_from_state_fd(state_fn, theta, eps=metric_eps)
    else:
        diag = _validate_qfim_diag(qfim_diag, theta.size)

    damping_value = float(damping)
    if damping_value < 0.0:
        raise ValueError("damping must be non-negative")
    natural_flat = gradient.reshape(-1) / (diag.reshape(-1) + damping_value)
    natural = natural_flat.reshape(theta.shape)
    extras: list[Any] = []
    if return_gradient:
        extras.append(gradient)
    if return_qfim_diag:
        extras.append(diag)
    if extras:
        return (natural, *extras)
    return natural

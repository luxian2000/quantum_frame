"""标量目标函数梯度族：psr / psr4 / spsr / spsa / fd / auto / mpsr。

统一契约 ``(fn, params, **kw) -> ndarray``（``mpsr`` 例外，返回标量混合偏导，见
其 docstring）。每个方法都有一个仅供 NPU/Torch 设备驻留路径使用的 ``_*_torch``
私有变体（供 ``qng.py`` 的 ``_ordinary_gradient_for_dqng_torch`` 复用）——按仓库
CLAUDE.md 的约定，这些 numpy/torch 双实现是 NPU 支持所必需的，不做合并。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ._coerce import (
    _as_scalar,
    _flat_to_index,
    _normalize_parameter_indices,
    _real_torch_dtype_from_backend,
    _torch_or_none,
)


# ─────────────────────────── automatic differentiation ──────────────────────────


def _to_real_torch_scalar(value, *, label: str):
    import torch

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.numel() != 1:
        raise ValueError(f"{label} must return a scalar value")
    value = value.reshape(())
    if torch.is_complex(value):
        # Objectives (expectation values) are real-valued.
        value = value.real
    return value


def auto(
    fn: Callable[[Any], Any],
    params: Any,
    *,
    backend: Any = None,
) -> np.ndarray:
    """Compute an ansatz gradient by reverse-mode automatic differentiation.

    This is "backpropagation through the unitary": the objective is evaluated
    with PyTorch tensor parameters, and the gradient is obtained by
    backpropagating through every gate operation recorded on the autograd
    tape. It is exact (no shift/step-size error) and computes the full
    gradient in a single backward pass.

    Because it relies on PyTorch autograd, it requires a Torch-family backend
    (:class:`GPUBackend` or :class:`NPUBackend`) and an objective that stays
    inside the autograd graph: ``fn`` must build/simulate the circuit using the
    backend and return the expectation value as a differentiable tensor — it
    must **not** call ``float(...)``, ``.item()``, ``.detach()`` or
    ``to_numpy(...)`` on its result.

    Args:
        fn: Differentiable objective. Receives a ``torch.Tensor`` of parameters
            (same shape as ``params``, ``requires_grad=True``, placed on the
            backend device) and returns a scalar tensor.
        params: Initial parameter value(s); scalars and arbitrary-shaped arrays
            are supported. The returned gradient has the same shape.
        backend: Torch-family backend used to pick the autograd dtype/device so
            the parameters live where the simulation runs (important for NPU /
            CUDA). Defaults to a CPU ``GPUBackend``.

    Returns:
        A NumPy array with the same shape as ``params``.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch always present in tests
        raise ModuleNotFoundError(
            "auto (automatic differentiation) requires PyTorch; install torch or use psr/fd/spsa."
        ) from exc

    if backend is None:
        from aicir.backends.gpu_backend import GPUBackend

        backend = GPUBackend(device="cpu")

    complex_dtype = getattr(backend, "_dtype", torch.complex64)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    device = getattr(backend, "_device", None)

    base = np.asarray(params, dtype=float)
    theta = torch.tensor(base, dtype=real_dtype, device=device, requires_grad=True)

    value = fn(theta)
    scalar = _to_real_torch_scalar(value, label="fn(params)")

    if not scalar.requires_grad:
        raise ValueError(
            "auto objective is not connected to the autograd graph: ensure fn "
            "uses a Torch-family backend and returns a differentiable tensor "
            "without calling float()/.item()/.detach()/to_numpy()."
        )

    (grad,) = torch.autograd.grad(scalar, theta, allow_unused=True)
    if grad is None:
        grad = torch.zeros_like(theta)

    grad_np = grad.detach().cpu().numpy().astype(float)
    return grad_np.reshape(base.shape)


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


# 激发门四项参数移位规则系数（生成元谱 {-1,0,1}，PennyLane SingleExcitation/DoubleExcitation 配方）
# 移位 s1=π/2, s2=3π/2；系数 c1=(√2+1)/(4√2), c2=(√2−1)/(4√2)
_FOUR_TERM_C1 = (math.sqrt(2.0) + 1.0) / (4.0 * math.sqrt(2.0))
_FOUR_TERM_C2 = (math.sqrt(2.0) - 1.0) / (4.0 * math.sqrt(2.0))


def psr4(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    shifts: tuple[float, float] = (np.pi / 2.0, 3.0 * np.pi / 2.0),
    coefficients: tuple[float, float] | None = None,
) -> np.ndarray:
    """四项参数移位规则（激发门等 {-1,0,1} 谱生成元）。

    grad = c1[f(θ+s1) − f(θ−s1)] − c2[f(θ+s2) − f(θ−s2)]，逐参数计算。
    默认 (s1,s2)=(π/2, 3π/2)，(c1,c2)=((√2+1)/4√2, (√2−1)/4√2)。
    autograd 为权威校验：若与 autograd 不符，应据其修正系数。

    Args:
        fn: 接受完整参数数组的标量值函数。
        params: 当前参数值，支持标量与任意形状数组。
        shifts: 两个正移位量 (s1, s2)，默认 (π/2, 3π/2)。
        coefficients: 可选系数对 (c1, c2)，若为 None 则使用激发门默认值。

    Returns:
        与 params 形状相同的 NumPy 梯度数组。
    """
    s1, s2 = float(shifts[0]), float(shifts[1])
    c1, c2 = coefficients if coefficients is not None else (_FOUR_TERM_C1, _FOUR_TERM_C2)
    theta = np.asarray(params, dtype=float)
    grad = np.zeros_like(theta, dtype=float)
    for index in np.ndindex(theta.shape):
        base = theta.copy()

        def shifted(delta, _base=base, _index=index):
            p = _base.copy()
            p[_index] = _base[_index] + delta
            return _as_scalar(fn(p), label="fn(params + shift)")

        grad[index] = c1 * (shifted(s1) - shifted(-s1)) - c2 * (shifted(s2) - shifted(-s2))
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


def _spsa_perturbation_matrix(
    theta: np.ndarray,
    *,
    n_samples: int,
    rng: Any,
    perturbations: Any,
) -> np.ndarray:
    sample_count = int(n_samples)
    if sample_count <= 0:
        raise ValueError("n_samples must be a positive integer")

    if perturbations is None:
        generator = np.random.default_rng(rng)
        return generator.choice((-1.0, 1.0), size=(sample_count, theta.size)).astype(float)

    raw = np.asarray(perturbations, dtype=float)
    if raw.shape == theta.shape or raw.shape == (theta.size,):
        deltas = raw.reshape(1, theta.size)
    elif raw.ndim >= 1 and raw.shape[0] > 0 and raw.shape[1:] == theta.shape:
        deltas = raw.reshape(raw.shape[0], theta.size)
    elif raw.ndim == 2 and raw.shape[1] == theta.size:
        deltas = raw.reshape(raw.shape[0], theta.size)
    else:
        expected = f"{theta.shape}, {(theta.size,)}, (K, *{theta.shape}), or (K, {theta.size})"
        raise ValueError(f"perturbations must have shape {expected}")

    if deltas.shape[0] == 0:
        raise ValueError("perturbations must contain at least one sample")
    if np.any(deltas == 0.0):
        raise ValueError("perturbations must not contain zero entries")
    return deltas.astype(float, copy=False)


def spsa(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    eps: float = 1e-3,
    n_samples: int = 1,
    rng: Any = None,
    perturbations: Any = None,
) -> np.ndarray:
    """Estimate a gradient by simultaneous perturbation stochastic approximation.

    SPSA evaluates the objective at two points per random perturbation vector:

        g_hat = (f(theta + eps * delta) - f(theta - eps * delta)) / (2 eps) * delta^{-1}

    By default ``delta`` is sampled entrywise from the Rademacher distribution
    ``{-1, +1}``, so ``delta^{-1} == delta``. Multiple samples are averaged to
    reduce estimator variance. When ``perturbations`` is supplied, the number
    of samples is inferred from it and ``n_samples`` is used only for generated
    perturbations.
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    deltas = _spsa_perturbation_matrix(
        theta,
        n_samples=n_samples,
        rng=rng,
        perturbations=perturbations,
    )

    base = theta.reshape(-1)
    grad = np.zeros(theta.size, dtype=float)
    for delta in deltas:
        plus = (base + eps_value * delta).reshape(theta.shape)
        minus = (base - eps_value * delta).reshape(theta.shape)
        forward = _as_scalar(fn(plus), label="fn(params + eps * perturbation)")
        backward = _as_scalar(fn(minus), label="fn(params - eps * perturbation)")
        grad += ((forward - backward) / (2.0 * eps_value)) / delta

    return (grad / deltas.shape[0]).reshape(theta.shape)


def mpsr(
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


def _fd_at_index(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    index: tuple[int, ...],
    eps: float,
    mode: str,
    base_value: float | None,
) -> float:
    if mode == "central":
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += eps
        minus[index] -= eps
        forward = _as_scalar(fn(plus), label="fn(params + eps)")
        backward = _as_scalar(fn(minus), label="fn(params - eps)")
        return (forward - backward) / (2.0 * eps)
    if mode == "forward":
        plus = theta.copy()
        plus[index] += eps
        forward = _as_scalar(fn(plus), label="fn(params + eps)")
        return (forward - base_value) / eps
    # mode == "backward"
    minus = theta.copy()
    minus[index] -= eps
    backward = _as_scalar(fn(minus), label="fn(params - eps)")
    return (base_value - backward) / eps


def fd(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    eps: float = 1e-3,
    mode: str = "central",
) -> np.ndarray:
    """Compute an ansatz gradient by finite differences.

    Unlike the parameter-shift rule, finite differences make no assumption
    about the generator spectrum, so they apply to arbitrary differentiable
    objectives (at the cost of a truncation/round-off trade-off in ``eps``).

    Args:
        fn: Scalar-valued function that accepts the full parameter array.
        params: Current parameter value(s). Scalars and arbitrary-shaped arrays
            are supported; the returned gradient has the same shape.
        eps: Positive step size for the difference stencil. The default
            ``1e-3`` is tuned for aicir's default ``complex64`` (single
            precision) state simulation, where too small a step causes
            catastrophic cancellation. For ``float64``/``complex128``
            objectives a smaller step (e.g. ``1e-6``) is more accurate.
        mode: Difference scheme. ``"central"`` (default) is second-order
            accurate and uses ``2N`` evaluations; ``"forward"`` and
            ``"backward"`` are first-order and use ``N + 1`` evaluations
            (the unshifted value is reused across all coordinates).

    Returns:
        A NumPy array with the same shape as ``params``.
    """
    theta = np.asarray(params, dtype=float)
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"central", "forward", "backward"}:
        raise ValueError("mode must be 'central', 'forward', or 'backward'")
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    base_value: float | None = None
    if mode_norm in {"forward", "backward"}:
        base_value = _as_scalar(fn(theta), label="fn(params)")

    grad = np.zeros_like(theta, dtype=float)
    for index in np.ndindex(theta.shape):
        grad[index] = _fd_at_index(
            fn, theta, index, eps_value, mode_norm, base_value
        )
    return grad


# ───────────────────────── torch/NPU 设备驻留私有变体 ─────────────────────────
# 供 qng.py 的 _ordinary_gradient_for_dqng_torch 在 kqng/dqng 的 NPU 路径下复用，
# 全程保持 torch 张量在设备上，不做 .cpu()/to_numpy() 转换。与上面的 numpy 版本
# 是刻意保留的双实现（NPU complex64 kernel 缺口，见仓库 CLAUDE.md），不做合并。


def _as_torch_scalar(value: Any, *, label: str, backend: Any = None):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError("Torch/NPU gradient path requires PyTorch")

    if not isinstance(value, torch.Tensor):
        dtype = _real_torch_dtype_from_backend(backend, torch)
        device = getattr(backend, "_device", None)
        value = torch.as_tensor(value, dtype=dtype, device=device)
    if value.numel() != 1:
        raise ValueError(f"{label} must return a scalar value")
    value = value.reshape(())
    if torch.is_complex(value):
        value = torch.real(value)
    return value


def _torch_shifted_difference(
    fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    index: tuple[int, ...],
    shift: float,
    coefficient: float,
    backend: Any,
):
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += shift
    minus[index] -= shift
    forward = _as_torch_scalar(fn(plus), label="fn(params + shift)", backend=backend)
    backward = _as_torch_scalar(fn(minus), label="fn(params - shift)", backend=backend)
    return coefficient * (forward - backward)


def _psr_torch(
    fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    shift: float,
    coefficient: float,
    backend: Any,
):
    values = [
        _torch_shifted_difference(fn, theta, index, shift, coefficient, backend)
        for index in np.ndindex(theta.shape)
    ]
    return values[0].reshape(()) if theta.shape == () else _torch_or_none().stack(values).reshape(theta.shape)


def _fd_torch(
    fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    eps: float,
    mode: str,
    backend: Any,
):
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"central", "forward", "backward"}:
        raise ValueError("mode must be 'central', 'forward', or 'backward'")
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    base_value = None
    if mode_norm in {"forward", "backward"}:
        base_value = _as_torch_scalar(fn(theta), label="fn(params)", backend=backend)

    values = []
    for index in np.ndindex(theta.shape):
        if mode_norm == "central":
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += eps_value
            minus[index] -= eps_value
            forward = _as_torch_scalar(fn(plus), label="fn(params + eps)", backend=backend)
            backward = _as_torch_scalar(fn(minus), label="fn(params - eps)", backend=backend)
            values.append((forward - backward) / (2.0 * eps_value))
        elif mode_norm == "forward":
            plus = theta.copy()
            plus[index] += eps_value
            forward = _as_torch_scalar(fn(plus), label="fn(params + eps)", backend=backend)
            values.append((forward - base_value) / eps_value)
        else:
            minus = theta.copy()
            minus[index] -= eps_value
            backward = _as_torch_scalar(fn(minus), label="fn(params - eps)", backend=backend)
            values.append((base_value - backward) / eps_value)

    return values[0].reshape(()) if theta.shape == () else _torch_or_none().stack(values).reshape(theta.shape)


def _spsa_torch(
    fn: Callable[[np.ndarray], Any],
    theta: np.ndarray,
    *,
    eps: float,
    n_samples: int,
    rng: Any,
    perturbations: Any,
    backend: Any,
):
    torch = _torch_or_none()
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    deltas = _spsa_perturbation_matrix(
        theta,
        n_samples=n_samples,
        rng=rng,
        perturbations=perturbations,
    )

    base = theta.reshape(-1)
    estimates = []
    for delta in deltas:
        plus = (base + eps_value * delta).reshape(theta.shape)
        minus = (base - eps_value * delta).reshape(theta.shape)
        forward = _as_torch_scalar(
            fn(plus),
            label="fn(params + eps * perturbation)",
            backend=backend,
        )
        backward = _as_torch_scalar(
            fn(minus),
            label="fn(params - eps * perturbation)",
            backend=backend,
        )
        inverse_delta = torch.as_tensor(1.0 / delta, dtype=forward.dtype, device=forward.device)
        estimates.append(((forward - backward) / (2.0 * eps_value)) * inverse_delta)

    averaged = torch.stack(estimates).mean(dim=0)
    return averaged.reshape(theta.shape)


def _auto_torch_device(
    fn: Callable[[Any], Any],
    params: np.ndarray,
    *,
    backend: Any,
):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError(
            "auto (automatic differentiation) requires PyTorch; install torch or use psr/fd/spsa."
        )
    if backend is None:
        from aicir.backends.gpu_backend import GPUBackend

        backend = GPUBackend(device="cpu")

    real_dtype = _real_torch_dtype_from_backend(backend, torch)
    device = getattr(backend, "_device", None)
    theta = torch.tensor(params, dtype=real_dtype, device=device, requires_grad=True)
    scalar = _to_real_torch_scalar(fn(theta), label="fn(params)")
    if not scalar.requires_grad:
        raise ValueError(
            "auto objective is not connected to the autograd graph: ensure fn "
            "uses a Torch-family backend and returns a differentiable tensor "
            "without calling float()/.item()/.detach()/to_numpy()."
        )
    (gradient,) = torch.autograd.grad(scalar, theta, allow_unused=True)
    if gradient is None:
        gradient = torch.zeros_like(theta)
    return gradient.detach()

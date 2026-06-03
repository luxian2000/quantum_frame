"""Gradient utilities for quantum machine learning workflows."""

from __future__ import annotations

from collections.abc import Callable
import itertools
import numbers
from typing import Any

import numpy as np


def _as_scalar(value: Any, *, label: str) -> float:
    # Accept backend-native tensors in addition to numpy arrays and Python
    # scalars, so every gradient rule works regardless of which backend
    # produced the objective value. In particular a NumpyBackend/TorchBackend/
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
    (:class:`TorchBackend` or :class:`NPUBackend`) and an objective that stays
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
            CUDA). Defaults to a CPU ``TorchBackend``.

    Returns:
        A NumPy array with the same shape as ``params``.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch always present in tests
        raise ModuleNotFoundError(
            "auto (automatic differentiation) requires PyTorch; install torch or use psr/fd."
        ) from exc

    if backend is None:
        from aicir.channel.backends.torch_backend import TorchBackend

        backend = TorchBackend(device="cpu")

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


# ─────────────────────────── adjoint differentiation ───────────────────────────

_CDTYPE = np.complex64

# Single-parameter gates differentiated by the adjoint method, mapped to the
# (Hermitian) generator G in U = exp(-i θ G / 2).
_AD_PAULI_GENERATOR = {"rx": "X", "ry": "Y", "rz": "Z", "crx": "X", "cry": "Y", "crz": "Z"}
_AD_DIFFERENTIABLE = set(_AD_PAULI_GENERATOR) | {"rzz"}

_SWAP_LOCAL = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=_CDTYPE
)


def _pauli_local(label: str) -> np.ndarray:
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=_CDTYPE)
    if label == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=_CDTYPE)
    if label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=_CDTYPE)
    raise ValueError(f"unknown Pauli label {label!r}")


def _controlled_generator_local(pauli: np.ndarray, control_states) -> np.ndarray:
    """Generator |1…1><1…1|_controls ⊗ pauli as a (zeroed) local block matrix."""
    cs = [int(s) for s in control_states]
    dim = 1 << (len(cs) + 1)
    local = np.zeros((dim, dim), dtype=_CDTYPE)
    control_index = 0
    for state in cs:
        control_index = (control_index << 1) | state
    block = [(control_index << 1) | target for target in (0, 1)]
    local[np.ix_(block, block)] = pauli
    return local


def _ad_gate_local_matrix_and_axes(gate, backend):
    """Return ``(local_matrix, axes)`` for a gate's unitary (numpy local matrix).

    Mirrors the dispatch used by ``apply_gate_to_state`` so the forward and
    backward (dagger) passes are exact inverses by construction.
    """
    from aicir.core.gates import (
        _controlled_local_from_base,
        _normalized_control_data,
        _rzz,
        _single_qubit_base_for_gate,
        _unitary_parameter_matrix,
    )

    gate_type = gate["type"]
    if gate_type in ("identity", "I"):
        return None, None
    if gate_type == "swap":
        return _SWAP_LOCAL, [int(gate["qubit_1"]), int(gate["qubit_2"])]
    if gate_type == "rzz":
        return np.asarray(_rzz(gate["parameter"]), dtype=_CDTYPE), [
            int(gate["qubit_1"]),
            int(gate["qubit_2"]),
        ]
    if gate_type == "unitary":
        matrix = np.asarray(_unitary_parameter_matrix(gate.get("parameter"), backend=None), dtype=_CDTYPE)
        gate_qubits = int(round(np.log2(matrix.shape[0])))
        return matrix, list(range(gate_qubits))

    base = _single_qubit_base_for_gate(gate)
    if base is None:
        raise ValueError(f"adjoint differentiation does not support gate type {gate_type!r}")
    if "control_qubits" in gate:
        controls, control_states = _normalized_control_data(gate)
        local = _controlled_local_from_base(base, control_states)
        return np.asarray(local, dtype=_CDTYPE), controls + [int(gate["target_qubit"])]
    return np.asarray(base, dtype=_CDTYPE), [int(gate["target_qubit"])]


def _ad_generator_local_and_axes(gate):
    """Return ``(generator_matrix, axes)`` for a differentiable gate."""
    from aicir.core.gates import _normalized_control_data

    gate_type = gate["type"]
    if gate_type == "rzz":
        zz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(_CDTYPE)
        return zz, [int(gate["qubit_1"]), int(gate["qubit_2"])]
    pauli = _pauli_local(_AD_PAULI_GENERATOR[gate_type])
    if gate_type in ("crx", "cry", "crz"):
        controls, control_states = _normalized_control_data(gate)
        return _controlled_generator_local(pauli, control_states), controls + [int(gate["target_qubit"])]
    return pauli, [int(gate["target_qubit"])]


def _ad_apply(backend, matrix, axes, n_qubits, state):
    from aicir.core.gates import _apply_local_matrix_to_state

    return _apply_local_matrix_to_state(state, backend.cast(matrix), axes, n_qubits, backend)


def ad(circuit, observable, *, backend=None, return_value: bool = False):
    """Adjoint-differentiation gradient of ``<O>`` over an ansatz circuit.

    Reverse-mode differentiation specialised for noiseless state-vector
    simulation. It propagates a "lambda state" backwards through the circuit
    and reads each gradient off

        ∂<O>/∂θ_k = 2 Re[<λ_k| ∂U_k/∂θ_k |ψ_{k-1}>]  =  Im[<λ_k| G_k |ψ_k>],

    where ``U_k = exp(-i θ_k G_k / 2)``. The whole gradient costs one forward
    pass plus one backward pass — ``O(P)`` gate applications for ``P``
    parameters with only ``O(1)`` extra state storage — versus ``O(P^2)`` for
    the parameter-shift rule. State-vector simulators only (no noise / mixed
    states).

    Unlike :func:`psr`/:func:`fd`, which differentiate a black-box scalar
    function, ``ad`` is structure-aware: it differentiates each single-angle
    Pauli-rotation gate (``rx``, ``ry``, ``rz``, ``crx``, ``cry``, ``crz``,
    ``rzz``) in the circuit, returning one gradient per such gate in order of
    appearance. Other gates (``H``, ``cx``, ``u3``, ``unitary``, …) are applied
    but not differentiated (use :func:`psr` for those).

    Args:
        circuit: A fully-bound ``Circuit`` (no unbound ``Parameter`` symbols).
        observable: Hermitian operator ``O`` — a ``(2^n, 2^n)`` matrix
            (numpy/backend tensor) or any object exposing ``to_matrix(backend)``
            (e.g. ``Hamiltonian``).
        backend: Computation backend. Defaults to ``NumpyBackend``.
        return_value: If ``True``, also return the expectation value ``<O>``.

    Returns:
        ``np.ndarray`` of gradients (one per differentiable gate). If
        ``return_value`` is ``True``, returns ``(grad, expectation)``.
    """
    from aicir.channel.backends.numpy_backend import NumpyBackend

    bk = backend if backend is not None else NumpyBackend()

    unbound = getattr(circuit, "parameters", ())
    if unbound:
        names = ", ".join(parameter.name for parameter in unbound)
        raise ValueError(f"circuit has unbound parameter(s): {names}; call bind_parameters(...) first")

    n_qubits = int(circuit.n_qubits)
    gates = list(circuit.gates)

    if hasattr(observable, "to_matrix"):
        operator = observable.to_matrix(bk)
    else:
        operator = bk.cast(observable)

    # Forward pass: |ψ> = U_P ... U_1 |0>, caching each gate's local operator.
    psi = bk.zeros_state(n_qubits)
    cached = []
    for gate in gates:
        matrix, axes = _ad_gate_local_matrix_and_axes(gate, bk)
        cached.append((matrix, axes))
        if matrix is not None:
            psi = _ad_apply(bk, matrix, axes, n_qubits, psi)

    lam = bk.matmul(operator, psi)  # |λ_P> = O|ψ>
    expectation = float(np.real(np.asarray(bk.to_numpy(bk.inner_product(psi, lam)))))

    phi = psi  # walks backward as |ψ_k>
    grads_reversed: list[float] = []
    for gate, (matrix, axes) in zip(reversed(gates), reversed(cached)):
        if gate["type"] in _AD_DIFFERENTIABLE:
            gen, gen_axes = _ad_generator_local_and_axes(gate)
            g_phi = _ad_apply(bk, gen, gen_axes, n_qubits, phi)  # G_k |ψ_k>
            overlap = bk.to_numpy(bk.inner_product(lam, g_phi))  # <λ_k| G_k |ψ_k>
            grads_reversed.append(float(np.imag(np.asarray(overlap))))
        if matrix is not None:
            dagger = bk.dagger(bk.cast(matrix))
            phi = _ad_apply(bk, dagger, axes, n_qubits, phi)  # |ψ_{k-1}>
            lam = _ad_apply(bk, dagger, axes, n_qubits, lam)  # |λ_{k-1}>

    grad = np.array(list(reversed(grads_reversed)), dtype=float)
    if return_value:
        return grad, expectation
    return grad


# ─────────────────────────── quantum natural gradient ──────────────────────────


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


def _validate_qfim(qfim: Any, parameter_count: int) -> np.ndarray:
    matrix = np.asarray(qfim, dtype=float)
    expected = (parameter_count, parameter_count)
    if matrix.shape != expected:
        raise ValueError(f"qfim must have shape {expected}, got {matrix.shape}")
    return 0.5 * (matrix + matrix.T)


def _torch_or_none():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _is_npu_family_backend(backend: Any) -> bool:
    if backend is None:
        return False
    device = getattr(backend, "_device", None)
    device_type = getattr(device, "type", None)
    return device_type == "npu" or type(backend).__name__ == "NPUBackend"


def _real_torch_dtype_from_backend(backend: Any, torch):
    complex_dtype = getattr(backend, "_dtype", torch.complex64)
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def _state_tensor_and_backend(value: Any, backend: Any = None) -> tuple[Any, Any]:
    if all(hasattr(value, attr) for attr in ("data", "backend", "n_qubits")):
        return value.data, backend if backend is not None else value.backend
    return value, backend


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


def _torch_inner_product(bra, ket, backend: Any = None):
    torch = _torch_or_none()
    if backend is not None and hasattr(backend, "inner_product"):
        return backend.inner_product(bra.reshape(-1, 1), ket.reshape(-1, 1))
    return torch.sum(torch.conj(bra.reshape(-1)) * ket.reshape(-1))


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
    if method_norm == "auto":
        kwargs.setdefault("backend", backend)
        return auto(fn, theta, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', or 'auto'")


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


def _auto_torch_device(
    fn: Callable[[Any], Any],
    params: np.ndarray,
    *,
    backend: Any,
):
    torch = _torch_or_none()
    if torch is None:
        raise ModuleNotFoundError(
            "auto (automatic differentiation) requires PyTorch; install torch or use psr/fd."
        )
    if backend is None:
        from aicir.channel.backends.torch_backend import TorchBackend

        backend = TorchBackend(device="cpu")

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
    if method_norm == "auto":
        kwargs.setdefault("backend", backend)
        return _auto_torch_device(fn, theta, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', or 'auto'")


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
            ``"psr"`` (default), ``"fd"``, or ``"auto"``.
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
            ``"psr"`` (default), ``"fd"``, or ``"auto"``.
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


__all__ = ["auto", "psr", "spsr", "mpsr", "fd", "ad", "qng", "bdqng", "dqng"]

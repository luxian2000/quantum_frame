"""Shared behavioral contracts for the distributed simulator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.state import State


AUTOGRAD_ERROR = "DistSimulator 首期仅支持前向模拟，不支持自动微分"
PARAMETER_STRUCTURE_ERROR = "各 rank 的可训练参数结构不一致"
AUTOGRAD_ROUTE_MISMATCH_ERROR = "各 rank 的自动微分路由不一致"
AUTOGRAD_SAMPLING_ERROR = "自动微分模式不支持 sample 或 counts"
AUTOGRAD_COLLAPSE_ERROR = "自动微分模式不支持 collapse"
AUTOGRAD_DIRECT_COMPLEX_STATE_ERROR = (
    "原生 distributed autograd 不接受 requires_grad complex initial_state；"
    "请使用 PureStateParam(real, imag)"
)


def synchronize_autograd_failure(communicator) -> None:
    """Complete the control plane before raising an all-rank autograd error.

    The helper deliberately accepts the small communicator protocol used by
    CPU/Gloo tests as well as the production wrapper; it never starts a state
    or gradient-data collective.
    """

    barrier = getattr(communicator, "barrier", None)
    if callable(barrier):
        barrier()
        return
    dist = getattr(communicator, "_dist", None)
    if getattr(communicator, "world_size", 1) > 1 and dist is not None:
        dist.barrier(group=getattr(communicator, "group", None))


def contains_paired_real(value) -> bool:
    """Return whether a value belongs to the private paired-real engine."""

    from .state import DistState
    from .autograd._parameters import (
        DensityParam,
        PureStateParam,
        StinespringParam,
    )

    if isinstance(value, DistState):
        return getattr(value, "_pair", None) is not None
    if isinstance(value, (PureStateParam, DensityParam, StinespringParam)):
        return True
    if isinstance(value, Mapping):
        return any(contains_paired_real(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_paired_real(item) for item in value)
    return False


def contains_requires_grad(value) -> bool:
    """Return whether a supported nested value contains trainable data."""

    if getattr(value, "requires_grad", False):
        return True

    from .state import DistState

    if isinstance(value, DistState):
        pair = getattr(value, "_pair", None)
        if pair is not None:
            return (
                contains_requires_grad(pair.real)
                or contains_requires_grad(pair.imag)
            )
        return contains_requires_grad(value.local_data)
    from .autograd._parameters import (
        DensityParam,
        PureStateParam,
        StinespringParam,
    )

    if isinstance(value, (PureStateParam, DensityParam, StinespringParam)):
        return any(contains_requires_grad(item) for item in value.parameters())
    if isinstance(value, State):
        return contains_requires_grad(value.data)
    if isinstance(value, Mapping):
        return any(contains_requires_grad(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_requires_grad(item) for item in value)
    return False


def reject_requires_grad(value) -> None:
    """Reject trainable input for the forward-only distributed API."""

    if contains_requires_grad(value):
        raise ValueError(AUTOGRAD_ERROR)

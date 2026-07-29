"""Shared behavioral contracts for the distributed simulator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.state import State


AUTOGRAD_ERROR = "DistSimulator 首期仅支持前向模拟，不支持自动微分"


def contains_requires_grad(value) -> bool:
    """Return whether a supported nested value contains trainable data."""

    if getattr(value, "requires_grad", False):
        return True

    from .state import DistState

    if isinstance(value, DistState):
        return contains_requires_grad(value.local_data)
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

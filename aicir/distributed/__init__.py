"""Explicit multi-NPU state-sharding API."""

from .backend import DistNPUBackend
from .result import DistResult
from .simulator import DistSimulator
from .state import DistState

__all__ = [
    "DistNPUBackend",
    "DistState",
    "DistSimulator",
    "DistResult",
]

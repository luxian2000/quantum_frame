"""Explicit multi-NPU state-sharding API."""

from .backend import DistNPUBackend
from .autograd import DensityParam, PureStateParam, StinespringParam
from .grad import (
    finite_difference_gradient,
    parameter_shift_gradient,
    parameter_shift_jacobian,
)
from .result import DistResult
from .simulator import DistSimulator
from .state import DistState

__all__ = [
    "DistNPUBackend",
    "DistState",
    "DistSimulator",
    "DistResult",
    "parameter_shift_gradient",
    "parameter_shift_jacobian",
    "finite_difference_gradient",
]

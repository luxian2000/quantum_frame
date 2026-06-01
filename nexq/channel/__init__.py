from .backends.numpy_backend import NumpyBackend
from .backends.base import Backend
from .operators import PauliOp, PauliString, Hamiltonian
from .noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoiseChannel,
    NoiseModel,
    PhaseFlipChannel,
)
from ..measure.measure import Measure
from ..measure.result import Result

_OPTIONAL_BACKENDS: list[str] = []
try:
    from .backends.torch_backend import TorchBackend
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_BACKENDS.append("TorchBackend")

try:
    from .backends.npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_BACKENDS.extend(["NPUBackend", "NPURuntimeContext", "npu_runtime_context_from_env"])

__all__ = [
    "Backend",
    "NumpyBackend",
    "PauliOp",
    "PauliString",
    "Hamiltonian",
    "NoiseChannel",
    "NoiseModel",
    "DepolarizingChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "AmplitudeDampingChannel",
    "Measure",
    "Result",
] + _OPTIONAL_BACKENDS

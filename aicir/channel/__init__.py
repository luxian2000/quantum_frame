from .backends import Backend, NumpyBackend
from .operators import Hamiltonian, PauliOp, PauliString

__all__ = [
    "Backend",
    "NumpyBackend",
    "PauliOp",
    "PauliString",
    "Hamiltonian",
]

try:
    from .backends import GPUBackend, NPURuntimeContext, NPUBackend, TorchBackend, npu_runtime_context_from_env
except ImportError:
    pass
else:
    __all__.extend(
        [
            "GPUBackend",
            "TorchBackend",
            "NPUBackend",
            "NPURuntimeContext",
            "npu_runtime_context_from_env",
        ]
    )

try:
    from .noise import (
        AmplitudeDampingChannel,
        BitFlipChannel,
        DepolarizingChannel,
        NoiseChannel,
        NoiseModel,
        PhaseFlipChannel,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "NoiseChannel",
            "NoiseModel",
            "DepolarizingChannel",
            "BitFlipChannel",
            "PhaseFlipChannel",
            "AmplitudeDampingChannel",
        ]
    )

try:
    from ..measure.measure import Measure
    from ..measure.result import Result
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(["Measure", "Result"])

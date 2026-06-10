from .base import Backend
from .numpy_backend import NumpyBackend

__all__ = [
    "Backend",
    "NumpyBackend",
]

try:
    from .gpu_backend import GPUBackend, TorchBackend
    from .npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
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

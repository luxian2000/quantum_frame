from .base import Backend
from .gpu_backend import GPUBackend, TorchBackend
from .numpy_backend import NumpyBackend
from .npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env

__all__ = [
	"Backend",
	"GPUBackend",
	"TorchBackend",  # deprecated alias for GPUBackend
	"NumpyBackend",
	"NPUBackend",
	"NPURuntimeContext",
	"npu_runtime_context_from_env",
]

from .base import Backend
from .numpy_backend import NumpyBackend

_OPTIONAL_BACKENDS: list[str] = []
try:
	from .torch_backend import TorchBackend
except ModuleNotFoundError as exc:
	if exc.name != "torch":
		raise
else:
	_OPTIONAL_BACKENDS.append("TorchBackend")

try:
	from .npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
except ModuleNotFoundError as exc:
	if exc.name != "torch":
		raise
else:
	_OPTIONAL_BACKENDS.extend(["NPUBackend", "NPURuntimeContext", "npu_runtime_context_from_env"])

__all__ = [
	"Backend",
	"NumpyBackend",
] + _OPTIONAL_BACKENDS

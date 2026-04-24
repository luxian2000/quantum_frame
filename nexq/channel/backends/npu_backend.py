"""
nexq/channel/backends/npu_backend.py

Ascend NPU backend built on top of PyTorch + torch_npu.

Design goals:
- Reuse TorchBackend math kernels to keep behavior consistent.
- Prefer NPU automatically when available.
- Allow graceful CPU fallback for environments without torch_npu.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .torch_backend import TorchBackend

try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = hasattr(torch, "npu")
except ImportError:
    _HAS_TORCH_NPU = False


def is_npu_available() -> bool:
    """Return True if torch_npu is importable and runtime NPU is available."""
    return bool(_HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available())


@dataclass(frozen=True)
class NPURuntimeContext:
    """Runtime context inferred from distributed environment variables."""

    world_size: int
    rank: int
    local_rank: int
    distributed: bool


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def npu_runtime_context_from_env() -> NPURuntimeContext:
    """Parse WORLD_SIZE/RANK/LOCAL_RANK from env, with safe defaults for single-process runs."""
    world_size = max(1, _env_int("WORLD_SIZE", 1))
    rank = max(0, _env_int("RANK", 0))
    local_rank = max(0, _env_int("LOCAL_RANK", 0))
    return NPURuntimeContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=world_size > 1,
    )


class NPUBackend(TorchBackend):
    """NPU-first backend for Ascend devices, compatible with TorchBackend API."""

    def __init__(self, dtype=None, device=None, fallback_to_cpu: bool = True):
        """
        Args:
            dtype: torch complex dtype, default torch.complex64.
            device: target device. If None, auto-selects npu:0 when available.
            fallback_to_cpu: if True, fall back to cpu when NPU is unavailable.
        """
        resolved = self._resolve_device(device=device, fallback_to_cpu=fallback_to_cpu)

        if self._is_npu_device(resolved) and hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            # Keep behavior aligned with torch_npu best practice.
            torch.npu.set_device(resolved)

        super().__init__(dtype=dtype, device=resolved)
        self._requested_device = device
        self._fallback_to_cpu = bool(fallback_to_cpu)
        self._runtime_context = None

    @classmethod
    def from_distributed_env(cls, dtype=None, fallback_to_cpu: bool = True):
        """
        Create a backend instance from distributed env variables.

        Env variables used:
            WORLD_SIZE, RANK, LOCAL_RANK

        Behavior:
            - Preferred device is npu:{LOCAL_RANK}
            - Falls back to CPU when NPU is unavailable and fallback_to_cpu=True
        """
        ctx = npu_runtime_context_from_env()
        backend = cls(dtype=dtype, device=f"npu:{ctx.local_rank}", fallback_to_cpu=fallback_to_cpu)
        backend._runtime_context = ctx
        return backend

    @staticmethod
    def _is_npu_device(device) -> bool:
        try:
            return torch.device(device).type == "npu"
        except Exception:
            return str(device).startswith("npu")

    @staticmethod
    def _resolve_device(device=None, fallback_to_cpu: bool = True):
        if device is None:
            if is_npu_available():
                return torch.device("npu:0")
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError(
                "NPU is not available. Install torch_npu and configure Ascend runtime correctly."
            )

        if isinstance(device, str):
            dev_str = device.strip().lower()
            if dev_str.startswith("npu") and not is_npu_available():
                if fallback_to_cpu:
                    return torch.device("cpu")
                raise RuntimeError(
                    "Requested NPU device, but NPU is unavailable. "
                    "Install torch_npu and verify torch.npu.is_available()."
                )
            return torch.device(device)

        dev = device
        try:
            dev_type = torch.device(dev).type
        except Exception:
            dev_type = str(dev).split(":", 1)[0].lower()

        if dev_type == "npu" and not is_npu_available():
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError(
                "Requested NPU device, but NPU is unavailable. "
                "Install torch_npu and verify torch.npu.is_available()."
            )
        return dev

    @property
    def name(self) -> str:
        return (
            f"NPUBackend(dtype={self._dtype}, device={self._device}, "
            f"npu_available={is_npu_available()})"
        )

    @staticmethod
    def _is_complex_tensor(value) -> bool:
        return isinstance(value, torch.Tensor) and torch.is_complex(value)

    def _should_use_complex_matmul_workaround(self, a, b) -> bool:
        return (
            getattr(self._device, "type", None) == "npu"
            and self._is_complex_tensor(a)
            and self._is_complex_tensor(b)
        )

    @staticmethod
    def _complex_matmul_workaround(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_real, a_imag = torch.real(a), torch.imag(a)
        b_real, b_imag = torch.real(b), torch.imag(b)
        real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
        imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
        return torch.complex(real, imag)

    def matmul(self, a, b):
        if self._should_use_complex_matmul_workaround(a, b):
            return self._complex_matmul_workaround(a, b)
        return super().matmul(a, b)

    def apply_unitary(self, state, unitary):
        return self.matmul(unitary, state)

    @property
    def runtime_context(self):
        """Distributed runtime context if created via from_distributed_env, else None."""
        return self._runtime_context

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

    def _is_npu_complex(self, tensor) -> bool:
        """Return True when on NPU device and tensor is complex — triggers workaround path."""
        return getattr(self._device, "type", None) == "npu" and self._is_complex_tensor(tensor)

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

    def eye(self, dim: int):
        """NPU workaround: build complex identity from real eye when complex eye kernel is unsupported."""
        if getattr(self._device, "type", None) == "npu" and self._dtype in (torch.complex64, torch.complex128):
            real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
            real_eye = torch.eye(dim, dtype=real_dtype, device=self._device)
            imag_eye = torch.zeros_like(real_eye)
            return torch.complex(real_eye, imag_eye).to(dtype=self._dtype)
        return super().eye(dim)

    def zeros_state(self, n_qubits: int):
        """NPU workaround: avoid complex in-place write path when initializing |0...0>."""
        if getattr(self._device, "type", None) == "npu" and self._dtype in (torch.complex64, torch.complex128):
            dim = 1 << n_qubits
            real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
            head = torch.ones((1, 1), dtype=real_dtype, device=self._device)
            tail = torch.zeros((dim - 1, 1), dtype=real_dtype, device=self._device)
            real = torch.cat([head, tail], dim=0)
            imag = torch.zeros_like(real)
            return torch.complex(real, imag).to(dtype=self._dtype)
        return super().zeros_state(n_qubits)

    def apply_unitary(self, state, unitary):
        return self.matmul(unitary, state)

    def kron(self, a, b):
        """NPU workaround: Kronecker product via real/imag decomposition."""
        if self._is_npu_complex(a) and self._is_npu_complex(b):
            ar, ai = torch.real(a), torch.imag(a)
            br, bi = torch.real(b), torch.imag(b)
            real = torch.kron(ar, br) - torch.kron(ai, bi)
            imag = torch.kron(ar, bi) + torch.kron(ai, br)
            return torch.complex(real, imag)
        return super().kron(a, b)

    def dagger(self, matrix):
        """NPU workaround: conjugate transpose via real/imag split (avoids torch.conj on complex64)."""
        if self._is_npu_complex(matrix):
            t = torch.transpose(matrix, -2, -1).contiguous()
            return torch.complex(torch.real(t), -torch.imag(t))
        return super().dagger(matrix)

    def trace(self, matrix):
        """NPU workaround: trace via real/imag split (avoids torch.trace on complex64)."""
        if self._is_npu_complex(matrix):
            return torch.complex(
                torch.trace(torch.real(matrix)),
                torch.trace(torch.imag(matrix)),
            )
        return super().trace(matrix)

    def inner_product(self, bra, ket):
        """NPU workaround: inner product via real/imag dot products (avoids torch.dot on complex64)."""
        if self._is_npu_complex(bra) and self._is_npu_complex(ket):
            b = bra.reshape(-1)
            k = ket.reshape(-1)
            br, bi = torch.real(b), torch.imag(b)
            kr, ki = torch.real(k), torch.imag(k)
            real = torch.dot(br, kr) + torch.dot(bi, ki)
            imag = torch.dot(br, ki) - torch.dot(bi, kr)
            return torch.complex(real, imag)
        return super().inner_product(bra, ket)

    @staticmethod
    def _partial_trace_real(rho_real: torch.Tensor, keep: list, n_qubits: int) -> torch.Tensor:
        """Real-valued partial trace kernel; shared by the NPU complex workaround."""
        keep = sorted(set(int(k) for k in keep))
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if not trace_out:
            return rho_real.clone()
        reshaped = rho_real.reshape([2] * n_qubits + [2] * n_qubits)
        perm = (
            keep + trace_out
            + [k + n_qubits for k in keep]
            + [t + n_qubits for t in trace_out]
        )
        permuted = reshaped.permute(perm)
        d_keep = 1 << len(keep)
        d_trace = 1 << len(trace_out)
        permuted = permuted.reshape(d_keep, d_trace, d_keep, d_trace)
        return torch.einsum("abcb->ac", permuted)

    def partial_trace(self, rho, keep, n_qubits):
        """NPU workaround: partial trace via real/imag split (avoids torch.einsum on complex64)."""
        if self._is_npu_complex(rho):
            rho_r = self._partial_trace_real(torch.real(rho), keep, n_qubits)
            rho_i = self._partial_trace_real(torch.imag(rho), keep, n_qubits)
            return torch.complex(rho_r, rho_i)
        return super().partial_trace(rho, keep, n_qubits)

    def expectation_sv(self, state, operator):
        """NPU workaround: ⟨ψ|O|ψ⟩ via self.matmul/dagger (avoids @ operator on complex64)."""
        if self._is_npu_complex(state):
            s = state.reshape(-1, 1)
            op_s = self.matmul(operator, s)
            dag_s = self.dagger(s)
            val = self.matmul(dag_s, op_s)[0, 0]
            return torch.real(val)
        return super().expectation_sv(state, operator)

    def expectation_dm(self, rho, operator):
        """NPU workaround: Tr(ρO) via self.matmul/trace (avoids torch.trace/matmul on complex64)."""
        if self._is_npu_complex(rho):
            prod = self.matmul(rho, operator)
            val = self.trace(prod)
            return torch.real(val)
        return super().expectation_dm(rho, operator)

    def abs_sq(self, tensor):
        """NPU workaround: |z|² = real² + imag² (avoids aclnnAbs on complex64)."""
        if self._is_npu_complex(tensor):
            return torch.real(tensor) ** 2 + torch.imag(tensor) ** 2
        return super().abs_sq(tensor)

    def measure_probs(self, state):
        """NPU workaround: compute probabilities via real²+imag² instead of abs()."""
        if self._is_npu_complex(state):
            flat = state.reshape(-1)
            probs = torch.real(flat) ** 2 + torch.imag(flat) ** 2
            total = probs.sum()
            if total > 0:
                probs = probs / total
            return probs
        return super().measure_probs(state)

    @property
    def runtime_context(self):
        """Distributed runtime context if created via from_distributed_env, else None."""
        return self._runtime_context

"""Backend resolution helpers for QAS command-line runners.

These helpers keep CLI strings such as ``--backend npu --dtype complex64`` out
of the VQE/oracle logic and record the actual backend provenance in label rows.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...backends.base import Backend
from ...backends.numpy_backend import NumpyBackend

try:  # pragma: no cover - optional accelerator backend.
    from ...backends.npu_backend import NPUBackend
except Exception:  # pragma: no cover
    NPUBackend = None

try:  # pragma: no cover - optional torch backend.
    from ...backends.gpu_backend import TorchBackend
except Exception:  # pragma: no cover
    TorchBackend = None


def resolve_qas_backend(
    kind: Optional[str] = None,
    fallback_to_cpu: bool = True,
    dtype: Optional[str] = None,
) -> Backend:
    """Resolve numpy/torch/npu backend and dtype from CLI or environment settings."""

    import os

    backend_kind = (kind or os.environ.get("AICIR_QAS_BACKEND") or "numpy").strip().lower()
    dtype_name = (dtype or os.environ.get("AICIR_QAS_DTYPE") or "").strip().lower()
    numpy_dtype = None
    torch_dtype = None
    if dtype_name:
        if dtype_name in {"complex128", "c128", "float64"}:
            numpy_dtype = np.complex128
        elif dtype_name in {"complex64", "c64", "float32"}:
            numpy_dtype = np.complex64
        else:
            raise ValueError(f"Unsupported QAS dtype: {dtype_name!r}. Use complex64 or complex128.")
        if backend_kind in {"torch", "npu"}:
            try:
                import torch
            except Exception as exc:  # pragma: no cover - depends on optional torch install.
                raise RuntimeError(f"Backend {backend_kind!r} requires torch to honor dtype={dtype_name!r}") from exc
            torch_dtype = torch.complex128 if numpy_dtype == np.complex128 else torch.complex64
    if backend_kind in {"numpy", "cpu"}:
        return NumpyBackend(dtype=numpy_dtype)
    if backend_kind == "torch":
        if TorchBackend is None:
            raise RuntimeError("TorchBackend is unavailable; install torch or use AICIR_QAS_BACKEND=numpy.")
        device = os.environ.get("AICIR_QAS_TORCH_DEVICE") or "cpu"
        return TorchBackend(dtype=torch_dtype, device=device)
    if backend_kind == "npu":
        if NPUBackend is None:
            if fallback_to_cpu:
                return NumpyBackend(dtype=numpy_dtype)
            raise RuntimeError("NPUBackend is unavailable; install torch_npu or use AICIR_QAS_BACKEND=numpy.")
        return NPUBackend.from_distributed_env(dtype=torch_dtype, fallback_to_cpu=fallback_to_cpu)
    raise ValueError(f"Unsupported QAS backend: {backend_kind!r}. Use numpy, cpu, torch, or npu.")


def make_torch_backend(device: Any = None) -> Backend:
    """按 ``device`` 前缀在 ``GPUBackend``/``NPUBackend`` 之间选择 torch 后端。

    收敛 ``supernet.py``/``qdrats.py``/``dqas.py`` 里原本三份重复的
    ``_make_backend``：``device`` 字符串以 ``"npu"`` 开头（大小写不敏感）时选择
    ``NPUBackend``，否则一律使用 ``GPUBackend``（包含 ``device=None`` 的默认情形，
    与三份原实现行为逐字节一致）。这三个模块本身在导入时已硬依赖 torch，因此这里
    直接要求 torch 可用；``resolve_qas_backend`` 才是 numpy 默认、torch 可选的入口。
    """

    if str(device).lower().startswith("npu"):
        if NPUBackend is None:
            raise RuntimeError("NPUBackend is unavailable; install torch_npu or use a non-npu device.")
        return NPUBackend(device=device)
    if TorchBackend is None:
        raise RuntimeError("GPUBackend requires torch; install torch to use make_torch_backend.")
    return TorchBackend(device=device)


def backend_runtime_metadata(backend: Backend) -> dict[str, Any]:
    """Return actual backend provenance for result manifests."""

    dtype = getattr(backend, "_dtype", None)
    device = getattr(backend, "_device", None)
    return {
        "backend_name": getattr(backend, "name", type(backend).__name__),
        "backend_class": type(backend).__name__,
        "backend_dtype": str(dtype) if dtype is not None else None,
        "backend_device": str(device) if device is not None else None,
    }


__all__ = ["backend_runtime_metadata", "make_torch_backend", "resolve_qas_backend"]

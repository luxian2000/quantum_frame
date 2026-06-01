"""Shared helpers for visualization functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def to_numpy(data: Any) -> np.ndarray:
    """Convert nexq objects, backend tensors, or array-likes to numpy."""
    if hasattr(data, "to_numpy") and callable(data.to_numpy):
        return np.asarray(data.to_numpy())

    backend = getattr(data, "backend", None)
    raw = getattr(data, "data", None)
    if backend is not None and raw is not None and hasattr(backend, "to_numpy"):
        return np.asarray(backend.to_numpy(raw))

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional.
        torch = None
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def basis_labels(n_qubits: int, bit_order: str = "msb") -> list[str]:
    """Return computational basis labels such as ``|00>``."""
    order = bit_order.lower()
    if order not in {"msb", "lsb"}:
        raise ValueError("bit_order must be 'msb' or 'lsb'")

    labels = []
    for idx in range(1 << n_qubits):
        bits = f"{idx:0{n_qubits}b}"
        if order == "lsb":
            bits = bits[::-1]
        labels.append(f"|{bits}>")
    return labels


def infer_n_qubits_from_length(length: int) -> int:
    """Infer n_qubits from a vector length and validate it is a power of two."""
    if length <= 0:
        raise ValueError("state/probability length must be positive")
    n_qubits = int(round(np.log2(length)))
    if (1 << n_qubits) != length:
        raise ValueError("state/probability length must be a power of two")
    return n_qubits


def require_matplotlib():
    """Import matplotlib lazily so visual remains an optional dependency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "matplotlib is required for graphical visualizations. "
            "Install matplotlib or use text/statistical visual functions."
        ) from exc
    return plt


def normalize_part(part: str) -> str:
    normalized = str(part).lower()
    allowed = {"abs", "real", "imag", "phase"}
    if normalized not in allowed:
        raise ValueError(f"part must be one of {sorted(allowed)}")
    return normalized


def as_index_sequence(values: Sequence[int] | None, size: int) -> list[int]:
    if values is None:
        return list(range(size))
    return [int(value) for value in values]

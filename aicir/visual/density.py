"""Density-matrix visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import infer_n_qubits_from_length, normalize_part, require_matplotlib, to_numpy


def _density_matrix_array(rho: Any) -> np.ndarray:
    matrix = to_numpy(rho).squeeze()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density matrix must be a square 2D array")
    infer_n_qubits_from_length(matrix.shape[0])
    return matrix.astype(np.complex128, copy=False)


def _density_part(matrix: np.ndarray, part: str) -> np.ndarray:
    part = normalize_part(part)
    if part == "abs":
        return np.abs(matrix)
    if part == "real":
        return np.real(matrix)
    if part == "imag":
        return np.imag(matrix)
    return np.angle(matrix)


def plot_density_matrix(
    rho: Any,
    *,
    part: str = "abs",
    ax=None,
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str | None = None,
):
    """Plot a density matrix heatmap.

    ``part`` can be ``"abs"``, ``"real"``, ``"imag"``, or ``"phase"``.
    """
    plt = require_matplotlib()
    matrix = _density_matrix_array(rho)
    values = _density_part(matrix, part)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
    else:
        fig = ax.figure

    image = ax.imshow(values, cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Column basis index")
    ax.set_ylabel("Row basis index")
    ax.set_title(title or f"Density matrix ({normalize_part(part)})")
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_density_real_imag(rho: Any, *, cmap: str = "coolwarm", title: str | None = None):
    """Plot real and imaginary density-matrix heatmaps side by side."""
    plt = require_matplotlib()
    matrix = _density_matrix_array(rho)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))

    for ax, part in zip(axes, ("real", "imag")):
        values = _density_part(matrix, part)
        image = ax.imshow(values, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("Column basis index")
        ax.set_ylabel("Row basis index")
        ax.set_title(part)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

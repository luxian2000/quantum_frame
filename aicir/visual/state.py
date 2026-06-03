"""State-vector visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import basis_labels, infer_n_qubits_from_length, require_matplotlib, to_numpy


def _state_vector_array(state: Any) -> np.ndarray:
    if hasattr(state, "probabilities") and hasattr(state, "to_numpy"):
        array = to_numpy(state).reshape(-1)
    else:
        array = to_numpy(state).reshape(-1)
    infer_n_qubits_from_length(array.size)
    return array.astype(np.complex128, copy=False)


def _probability_array(state_or_probs: Any) -> np.ndarray:
    if hasattr(state_or_probs, "probabilities") and callable(state_or_probs.probabilities):
        probs = to_numpy(state_or_probs.probabilities()).reshape(-1)
    else:
        array = to_numpy(state_or_probs).reshape(-1)
        if np.iscomplexobj(array):
            probs = np.abs(array) ** 2
        else:
            probs = array.astype(float, copy=False)

    infer_n_qubits_from_length(probs.size)
    probs = np.real(probs).astype(float, copy=False)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    return probs / total if total > 0 else probs


def plot_state_probs(
    state_or_probs: Any,
    *,
    bit_order: str = "msb",
    threshold: float = 0.0,
    ax=None,
    title: str | None = None,
):
    """Plot computational-basis probabilities as a bar chart."""
    plt = require_matplotlib()
    probs = _probability_array(state_or_probs)
    n_qubits = infer_n_qubits_from_length(probs.size)
    labels = basis_labels(n_qubits, bit_order=bit_order)

    mask = probs >= float(threshold)
    x = np.arange(int(mask.sum()))
    shown_probs = probs[mask]
    shown_labels = [label for label, keep in zip(labels, mask) if keep]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, len(shown_labels) * 0.55), 3.5))
    else:
        fig = ax.figure

    ax.bar(x, shown_probs, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(shown_labels, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, min(1.0, max(1e-12, shown_probs.max(initial=0.0)) * 1.15))
    ax.set_title(title or "State probabilities")
    fig.tight_layout()
    return fig, ax


def plot_state_amplitudes(
    state: Any,
    *,
    bit_order: str = "msb",
    threshold: float = 0.0,
    ax=None,
    title: str | None = None,
):
    """Plot real part, imaginary part, and magnitude of state amplitudes."""
    plt = require_matplotlib()
    amplitudes = _state_vector_array(state)
    n_qubits = infer_n_qubits_from_length(amplitudes.size)
    labels = basis_labels(n_qubits, bit_order=bit_order)
    magnitudes = np.abs(amplitudes)
    mask = magnitudes >= float(threshold)

    shown = amplitudes[mask]
    shown_labels = [label for label, keep in zip(labels, mask) if keep]
    x = np.arange(len(shown_labels))
    width = 0.25

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, len(shown_labels) * 0.7), 3.8))
    else:
        fig = ax.figure

    ax.bar(x - width, np.real(shown), width=width, label="real", color="#4C78A8")
    ax.bar(x, np.imag(shown), width=width, label="imag", color="#F58518")
    ax.bar(x + width, np.abs(shown), width=width, label="abs", color="#54A24B")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(shown_labels, rotation=45, ha="right")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "State amplitudes")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_state_phase(
    state: Any,
    *,
    bit_order: str = "msb",
    threshold: float = 1e-12,
    ax=None,
    title: str | None = None,
):
    """Plot amplitude phases for basis states with non-negligible magnitude."""
    plt = require_matplotlib()
    amplitudes = _state_vector_array(state)
    n_qubits = infer_n_qubits_from_length(amplitudes.size)
    labels = basis_labels(n_qubits, bit_order=bit_order)
    mask = np.abs(amplitudes) >= float(threshold)
    phases = np.angle(amplitudes[mask])
    shown_labels = [label for label, keep in zip(labels, mask) if keep]
    x = np.arange(len(shown_labels))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, len(shown_labels) * 0.55), 3.5))
    else:
        fig = ax.figure

    ax.bar(x, phases, color="#B279A2")
    ax.set_xticks(x)
    ax.set_xticklabels(shown_labels, rotation=45, ha="right")
    ax.set_ylabel("Phase")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(title or "State phases")
    fig.tight_layout()
    return fig, ax

"""Visualization helpers for QAS search results and metric reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..metrics.hardware import native_depth_twoq_efficiency_details
from ..metrics.trainability import structure_proxy_details
from .circuit import circuit_to_text
from .utils import require_matplotlib


OBJECTIVE_GROUPS = ("expressibility", "trainability", "noise_robustness", "hardware_efficiency")


def _is_search_result(value: Any) -> bool:
    return hasattr(value, "scores") and hasattr(value, "candidates")


def _is_architecture_score(value: Any) -> bool:
    return hasattr(value, "architecture") and hasattr(value, "weighted_score") and hasattr(value, "groups")


def _is_architecture_spec(value: Any) -> bool:
    return hasattr(value, "circuit") and hasattr(value, "name") and hasattr(value, "n_gates")


def _iter_items(data: Any) -> list[Any]:
    if _is_search_result(data):
        return list(data.scores)
    if _is_architecture_score(data) or _is_architecture_spec(data) or isinstance(data, Mapping):
        return [data]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return list(data)
    raise TypeError("Expected SearchResult, ArchitectureScore, ArchitectureSpec, mapping, or a sequence of them")


def _score_to_row(score: Any) -> dict[str, Any]:
    row = dict(score.to_row()) if hasattr(score, "to_row") else {}
    if not row:
        architecture = score.architecture
        row = {
            "rank": getattr(score, "rank", None),
            "name": getattr(architecture, "name", ""),
            "n_qubits": getattr(architecture, "n_qubits", None),
            "n_gates": getattr(architecture, "n_gates", None),
            "n_parameters": getattr(architecture, "parameter_count", None),
            "two_qubit_gate_count": getattr(architecture, "two_qubit_gate_count", None),
            "weighted_score": getattr(score, "weighted_score", None),
        }
    for group_name, group in score.groups().items():
        row[group_name] = float(group.score)
        row[f"{group_name}_metric"] = group.active_metric
    return row


def _architecture_to_row(architecture: Any) -> dict[str, Any]:
    circuit = architecture.circuit
    trainability = structure_proxy_details(circuit)
    hardware = native_depth_twoq_efficiency_details(circuit)
    return {
        "rank": None,
        "name": architecture.name,
        "n_qubits": architecture.n_qubits,
        "n_gates": architecture.n_gates,
        "n_parameters": architecture.parameter_count,
        "two_qubit_gate_count": architecture.two_qubit_gate_count,
        "trainability": float(trainability["structure_proxy_score"]),
        "hardware_efficiency": float(hardware["native_depth_twoq_efficiency_score"]),
        "depth_proxy": float(hardware["depth_proxy"]),
        "native_gate_ratio": float(hardware["native_gate_ratio"]),
    }


def _mapping_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record)
    score = row.get("architecture_score")
    if score is not None and _is_architecture_score(score):
        row.update({f"score_{key}": value for key, value in _score_to_row(score).items()})
        row.setdefault("weighted_score", score.weighted_score)
    return row


def qas_scores_to_rows(data: Any) -> list[dict[str, Any]]:
    """Normalize QAS result-like inputs into flat plotting rows."""
    rows = []
    for item in _iter_items(data):
        if _is_architecture_score(item):
            rows.append(_score_to_row(item))
        elif _is_architecture_spec(item):
            rows.append(_architecture_to_row(item))
        elif isinstance(item, Mapping):
            rows.append(_mapping_to_row(item))
        else:
            raise TypeError(f"Unsupported QAS row type: {type(item)!r}")
    return rows


def _numeric_metric_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    ignored = {"rank", "episode", "step", "iteration", "name", "n_qubits"}
    for row in rows:
        for key, value in row.items():
            if key in ignored or key.endswith("_metric"):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in names:
                names.append(key)
    preferred = [
        "weighted_score",
        "reward",
        "expressibility",
        "trainability",
        "noise_robustness",
        "hardware_efficiency",
        "n_gates",
        "n_parameters",
        "two_qubit_gate_count",
    ]
    ordered = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _row_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = []
    for index, row in enumerate(rows):
        name = row.get("name")
        if not name:
            rank = row.get("rank")
            name = f"rank {rank}" if rank is not None else str(index)
        names.append(str(name))
    return names


def plot_search_history(
    history: Any,
    *,
    metrics: Sequence[str] | None = None,
    x: str | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot metric trajectories from QAS records, scores, or search results."""
    plt = require_matplotlib()
    rows = qas_scores_to_rows(history)
    if not rows:
        raise ValueError("history must contain at least one row")

    metric_names = list(metrics) if metrics is not None else _numeric_metric_names(rows)[:5]
    if not metric_names:
        raise ValueError("No numeric metrics found to plot")

    if x is not None:
        x_values = [row.get(x, idx) for idx, row in enumerate(rows)]
        x_label = x
    elif all(row.get("rank") is not None for row in rows):
        x_values = [row["rank"] for row in rows]
        x_label = "rank"
    else:
        x_values = list(range(len(rows)))
        x_label = "step"

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, len(rows) * 0.45), 4.0))
    else:
        fig = ax.figure

    for metric_name in metric_names:
        y_values = [row.get(metric_name, np.nan) for row in rows]
        ax.plot(x_values, y_values, marker="o", label=metric_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel("value")
    ax.set_title(title or "QAS metric history")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_architecture_metrics(
    item: Any,
    *,
    metrics: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot one architecture score/spec as a metric bar chart."""
    plt = require_matplotlib()
    row = qas_scores_to_rows(item)[0]
    metric_names = list(metrics) if metrics is not None else [
        name
        for name in ("weighted_score", *OBJECTIVE_GROUPS, "n_gates", "n_parameters", "two_qubit_gate_count")
        if name in row and isinstance(row[name], (int, float, np.integer, np.floating))
    ]
    if not metric_names:
        raise ValueError("No numeric metrics found to plot")

    values = [float(row[name]) for name in metric_names]
    x_values = np.arange(len(metric_names))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, len(metric_names) * 0.8), 3.8))
    else:
        fig = ax.figure

    ax.bar(x_values, values, color="#4C78A8")
    ax.set_xticks(x_values)
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.set_ylabel("value")
    ax.set_title(title or str(row.get("name", "Architecture metrics")))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def compare_architectures(
    data: Any,
    *,
    metrics: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
):
    """Compare multiple QAS candidates or scores with grouped bars."""
    plt = require_matplotlib()
    rows = qas_scores_to_rows(data)
    if len(rows) < 1:
        raise ValueError("data must contain at least one architecture")
    metric_names = list(metrics) if metrics is not None else [
        name
        for name in ("weighted_score", *OBJECTIVE_GROUPS, "n_gates", "n_parameters", "two_qubit_gate_count")
        if any(isinstance(row.get(name), (int, float, np.integer, np.floating)) for row in rows)
    ]
    if not metric_names:
        raise ValueError("No numeric metrics found to compare")

    names = _row_names(rows)
    x_values = np.arange(len(names))
    width = min(0.8 / len(metric_names), 0.22)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, len(names) * 0.8), 4.2))
    else:
        fig = ax.figure

    offsets = (np.arange(len(metric_names)) - (len(metric_names) - 1) / 2.0) * width
    for offset, metric_name in zip(offsets, metric_names):
        values = [float(row.get(metric_name, np.nan)) for row in rows]
        ax.bar(x_values + offset, values, width=width, label=metric_name)

    ax.set_xticks(x_values)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("value")
    ax.set_title(title or "QAS architecture comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_qas_summary(
    item: Any,
    *,
    metrics: Sequence[str] | None = None,
    title: str | None = None,
):
    """Render a compact summary with circuit text and metric bars."""
    plt = require_matplotlib()
    rows = qas_scores_to_rows(item)
    if len(rows) != 1:
        raise ValueError("plot_qas_summary expects exactly one ArchitectureScore or ArchitectureSpec")
    row = rows[0]
    source = _iter_items(item)[0]
    if not (_is_architecture_score(source) or _is_architecture_spec(source)):
        raise TypeError("plot_qas_summary expects an ArchitectureScore or ArchitectureSpec")
    architecture = source.architecture if _is_architecture_score(source) else source
    diagram = circuit_to_text(architecture.circuit)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes[0].axis("off")
    axes[0].text(
        0.0,
        1.0,
        diagram,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=axes[0].transAxes,
    )
    axes[0].set_title(str(row.get("name", "Circuit")))
    plot_architecture_metrics(row, metrics=metrics, ax=axes[1], title="Metrics")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "OBJECTIVE_GROUPS",
    "qas_scores_to_rows",
    "plot_search_history",
    "plot_architecture_metrics",
    "compare_architectures",
    "plot_qas_summary",
]

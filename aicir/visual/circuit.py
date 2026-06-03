"""Circuit visualization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..core.circuit import _circuit_to_ascii
from .utils import require_matplotlib


def circuit_to_text(circuit: Any) -> str:
    """Return an ASCII diagram for a ``Circuit``-like object."""
    if not hasattr(circuit, "gates") or not hasattr(circuit, "n_qubits"):
        raise TypeError("circuit_to_text expects an object with gates and n_qubits")
    return _circuit_to_ascii(circuit)


def gate_histogram(circuit: Any) -> dict[str, int]:
    """Count gates by their ``type`` field."""
    if not hasattr(circuit, "gates"):
        raise TypeError("gate_histogram expects an object with gates")
    counts = Counter(str(gate.get("type", "unknown")) for gate in circuit.gates)
    return dict(sorted(counts.items()))


def draw_circuit(circuit: Any, output: str = "text", **kwargs):
    """Draw a circuit using the requested output backend.

    Parameters
    ----------
    circuit:
        A aicir ``Circuit`` or compatible object.
    output:
        ``"text"`` returns an ASCII string. ``"mpl"`` returns a matplotlib
        ``(fig, ax)`` tuple containing the ASCII diagram in a monospace figure.
    """
    output = str(output).lower()
    if output == "text":
        return circuit_to_text(circuit)
    if output in {"mpl", "matplotlib"}:
        return circuit_to_mpl(circuit, **kwargs)
    raise ValueError("output must be 'text' or 'mpl'")


def circuit_to_mpl(circuit: Any, ax=None, fontsize: int = 11, title: str | None = None):
    """Render a circuit ASCII diagram into a matplotlib figure."""
    plt = require_matplotlib()
    diagram = circuit_to_text(circuit)

    if ax is None:
        line_count = max(1, diagram.count("\n") + 1)
        width = max(6.0, min(18.0, max(len(line) for line in diagram.splitlines()) * 0.12))
        height = max(1.8, line_count * 0.28)
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    ax.axis("off")
    if title:
        ax.set_title(title)
    ax.text(
        0.0,
        1.0,
        diagram,
        va="top",
        ha="left",
        family="monospace",
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    return fig, ax

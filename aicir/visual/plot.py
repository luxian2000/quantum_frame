"""Publication-style quantum-circuit figures.

Render a aicir :class:`~aicir.core.circuit.Circuit` (or any other circuit form
aicir defines) into a coloured, rounded-box circuit diagram in the style of the
common drag-and-drop quantum composers, and save it as a PNG.

The single public entry point is :func:`plot`. By default it writes the figure
next to the script that calls it, naming the file
``<executing-file>_<circuit-variable>.png`` (or ``<executing-file>_default_N.png``
when the circuit is not bound to a plain variable).
"""

from __future__ import annotations

import ast
import inspect
import linecache
import os
from pathlib import Path
from typing import Any

from ..core.circuit import (
    Circuit,
    _controlled_target_symbol,
    _format_angle_value,
    _single_gate_symbol,
)
from ..core.io.json_io import circuit_from_json, circuit_from_json_dict, load_circuit_json
from ..core.io.qasm import circuit_from_qasm, load_circuit_qasm
from .utils import require_matplotlib


# --- Style ----------------------------------------------------------------

# Per gate-family fill / edge colours, loosely matching the look of the popular
# graphical composers: blue Hadamards, amber Paulis / controls, red phase
# gates, green rotations and pink measurements.
_PALETTE: dict[str, tuple[str, str]] = {
    "hadamard": ("#CBDDF0", "#3B6FB5"),
    "pauli": ("#F7D9A8", "#E1922B"),
    "phase": ("#F6C2BE", "#D9534F"),
    "rotation": ("#CBE8C6", "#4C9A4A"),
    "unitary": ("#D8CCEC", "#7E57C2"),
    "measure": ("#F3C6DC", "#C2549B"),
    "default": ("#E2E2E2", "#888888"),
}

_FAMILY: dict[str, str] = {
    "hadamard": "hadamard",
    "H": "hadamard",
    "pauli_x": "pauli",
    "X": "pauli",
    "pauli_y": "pauli",
    "Y": "pauli",
    "pauli_z": "pauli",
    "Z": "pauli",
    "cx": "pauli",
    "cnot": "pauli",
    "cy": "pauli",
    "cz": "pauli",
    "toffoli": "pauli",
    "ccnot": "pauli",
    "s_gate": "phase",
    "S": "phase",
    "t_gate": "phase",
    "T": "phase",
    "rx": "rotation",
    "ry": "rotation",
    "rz": "rotation",
    "crx": "rotation",
    "cry": "rotation",
    "crz": "rotation",
    "rzz": "rotation",
    "rxx": "rotation",
    "u2": "unitary",
    "u3": "unitary",
    "unitary": "unitary",
    "measure": "measure",
    "measurement": "measure",
}

# Geometry, in data units (the axes use an equal aspect ratio).
_BOX = 0.62          # gate-box side length
_DOT = 0.10          # control-dot radius
_OPLUS = 0.22        # CNOT target circle radius
_ROUND = 0.12        # corner rounding of gate boxes
_INNER_SUBLABEL_OFFSET = 0.236  # golden-ratio lower text position inside a gate box


def _style_for(gate_type: str) -> tuple[str, str]:
    return _PALETTE[_FAMILY.get(gate_type, "default")]


def _measure_targets(gate: dict) -> list[int]:
    """Qubits a measure gate reads out (supports ``qubits`` list or ``target_qubit``)."""
    qubits = gate.get("qubits")
    if not qubits and "target_qubit" in gate:
        qubits = [gate["target_qubit"]]
    return [int(q) for q in (qubits or [])]


def _pretty_angle(value: Any) -> str:
    """Format an angle, preferring ``π`` fractions over decimals."""
    text = _format_angle_value(value)
    return text.replace("pi", "π")


def _gate_label(gate: dict) -> str:
    gate_type = gate["type"]
    symbol = _single_gate_symbol(gate_type) or _controlled_target_symbol(gate_type)
    if symbol is None:
        symbol = str(gate_type).upper()[:3]
    return symbol


def _angle_sublabel(gate: dict) -> str | None:
    gate_type = gate.get("type")
    if gate_type in {"rx", "ry", "rz", "crx", "cry", "crz", "rzz", "rxx"}:
        param = gate.get("parameter")
        if param is not None:
            return _pretty_angle(param)
    if gate_type in {"u2", "u3"}:
        params = gate.get("parameter")
        if params is not None:
            angles = [_pretty_angle(value) for value in params]
            if gate_type == "u2":
                return ", ".join(angles)
            if gate_type == "u3" and len(angles) >= 3:
                return f"{angles[0]}, {angles[1]}\n{angles[2]}"
            return "\n".join(angles)
    return None


def _sublabel_scale(gate: dict) -> float:
    return 0.40 if gate.get("type") in {"u2", "u3"} else 0.50


def _gate_qubits(gate: dict, n_qubits: int) -> list[int]:
    """All wires a gate touches (used for layer packing)."""
    gate_type = gate["type"]
    if gate_type in {"swap", "rzz", "rxx"}:
        return [int(gate["qubit_1"]), int(gate["qubit_2"])]
    if gate_type in {"identity", "I", "unitary"}:
        return list(range(n_qubits))
    if gate_type in {"measure", "measurement"}:
        return _measure_targets(gate) or [0]
    qubits: list[int] = []
    controls = gate.get("control_qubits")
    if controls:
        qubits.extend(int(q) for q in controls)
    if "target_qubit" in gate:
        qubits.append(int(gate["target_qubit"]))
    return qubits or [0]


def _pack_layers(circuit: Any) -> list[int]:
    """Assign gates to compact columns while preserving wire-span order."""
    n_qubits = int(circuit.n_qubits)
    next_available = [0] * n_qubits
    columns: list[int] = []
    for gate in circuit.gates:
        touched = _gate_qubits(gate, n_qubits)
        span = set(range(min(touched), max(touched) + 1))
        col = max(next_available[q] for q in span)
        for q in span:
            next_available[q] = col + 1
        columns.append(col)
    return columns


# --- Primitive shapes -----------------------------------------------------


def _yy(qubit: int, n_qubits: int) -> float:
    """Map qubit index to a y-coordinate (qubit 0 on top)."""
    return float(n_qubits - 1 - qubit)


def _draw_box(ax, x, y, label, facecolor, edgecolor, *, fontsize, sublabel=None,
              sublabel_inside=False,
              sublabel_scale=0.62,
              width=_BOX, height=_BOX):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={_ROUND}",
        linewidth=1.6,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=3,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    label_y = y + height * 0.14 if sublabel and sublabel_inside else y
    ax.text(x, label_y, label, ha="center", va="center", fontsize=fontsize,
            color=edgecolor, fontweight="bold", zorder=4)
    if sublabel:
        if sublabel_inside:
            ax.text(x, y - height * _INNER_SUBLABEL_OFFSET, sublabel, ha="center", va="center",
                    fontsize=fontsize * sublabel_scale, color=edgecolor,
                    zorder=5)
        else:
            ax.text(x, y - height / 2 - 0.04, sublabel, ha="center", va="top",
                    fontsize=fontsize * sublabel_scale, color=edgecolor, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.05", facecolor="white",
                              edgecolor="none"))


def _draw_control(ax, x, y, color):
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), _DOT, facecolor=color, edgecolor=color, zorder=4))


def _draw_oplus(ax, x, y, color):
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), _OPLUS, facecolor="white", edgecolor=color,
                         linewidth=1.6, zorder=4))
    ax.plot([x - _OPLUS, x + _OPLUS], [y, y], color=color, linewidth=1.6, zorder=5)
    ax.plot([x, x], [y - _OPLUS, y + _OPLUS], color=color, linewidth=1.6, zorder=5)


def _draw_swap_mark(ax, x, y, color):
    d = _OPLUS * 0.8
    ax.plot([x - d, x + d], [y - d, y + d], color=color, linewidth=2.0, zorder=4)
    ax.plot([x - d, x + d], [y + d, y - d], color=color, linewidth=2.0, zorder=4)


def _draw_measure(ax, x, y, facecolor, edgecolor):
    from matplotlib.patches import Arc, FancyBboxPatch

    _draw_box(ax, x, y, "", facecolor, edgecolor, fontsize=1)
    ax.add_patch(Arc((x, y - 0.08), _BOX * 0.62, _BOX * 0.62, theta1=0, theta2=180,
                     edgecolor=edgecolor, linewidth=1.6, zorder=4))
    ax.plot([x, x + 0.16], [y - 0.08, y + 0.14], color=edgecolor,
            linewidth=1.6, zorder=4)


def _draw_connector(ax, x, q_lo, q_hi, n_qubits, color):
    ax.plot([x, x], [_yy(q_hi, n_qubits), _yy(q_lo, n_qubits)], color=color,
            linewidth=1.8, zorder=2)


# --- Per-gate dispatch ----------------------------------------------------


def _render_gate(ax, gate, x, n_qubits, fontsize):
    gate_type = gate["type"]
    facecolor, edgecolor = _style_for(gate_type)

    if gate_type in {"measure", "measurement"}:
        for target in (_measure_targets(gate) or [0]):
            _draw_measure(ax, x, _yy(target, n_qubits), facecolor, edgecolor)
        return

    if gate_type in {"identity", "I"}:
        for q in range(n_qubits):
            _draw_box(ax, x, _yy(q, n_qubits), "I", facecolor, edgecolor,
                      fontsize=fontsize)
        return

    if gate_type == "unitary":
        n = int(gate.get("n_qubits", n_qubits))
        top, bottom = _yy(0, n_qubits), _yy(n - 1, n_qubits)
        height = (top - bottom) + _BOX
        _draw_box(ax, x, (top + bottom) / 2, "U", facecolor, edgecolor,
                  fontsize=fontsize, height=height)
        return

    if gate_type == "swap":
        q1, q2 = int(gate["qubit_1"]), int(gate["qubit_2"])
        _draw_connector(ax, x, min(q1, q2), max(q1, q2), n_qubits, edgecolor)
        _draw_swap_mark(ax, x, _yy(q1, n_qubits), edgecolor)
        _draw_swap_mark(ax, x, _yy(q2, n_qubits), edgecolor)
        return

    if gate_type in {"rzz", "rxx"}:
        q1, q2 = int(gate["qubit_1"]), int(gate["qubit_2"])
        _draw_connector(ax, x, min(q1, q2), max(q1, q2), n_qubits, edgecolor)
        label = "Rzz" if gate_type == "rzz" else "Rxx"
        # Two squares share one angle, shown once below the lower square (the
        # larger qubit index maps to the smaller y, i.e. the visually lower box).
        upper_qubit, lower_qubit = min(q1, q2), max(q1, q2)
        _draw_box(ax, x, _yy(upper_qubit, n_qubits), label, facecolor, edgecolor,
                  fontsize=fontsize)
        _draw_box(ax, x, _yy(lower_qubit, n_qubits), label, facecolor, edgecolor,
                  fontsize=fontsize, sublabel=_angle_sublabel(gate),
                  sublabel_inside=False, sublabel_scale=_sublabel_scale(gate))
        return

    controls = [int(q) for q in gate.get("control_qubits", [])]
    target = int(gate["target_qubit"])

    if controls:
        involved = controls + [target]
        _draw_connector(ax, x, min(involved), max(involved), n_qubits, edgecolor)
        for c in controls:
            _draw_control(ax, x, _yy(c, n_qubits), edgecolor)
        if gate_type in {"cx", "cnot", "toffoli", "ccnot"}:
            _draw_oplus(ax, x, _yy(target, n_qubits), edgecolor)
        else:
            _draw_box(ax, x, _yy(target, n_qubits), _gate_label(gate),
                      facecolor, edgecolor, fontsize=fontsize,
                      sublabel=_angle_sublabel(gate),
                      sublabel_inside=False,
                      sublabel_scale=_sublabel_scale(gate))
        return

    _draw_box(ax, x, _yy(target, n_qubits), _gate_label(gate), facecolor,
              edgecolor, fontsize=fontsize, sublabel=_angle_sublabel(gate),
              sublabel_inside=False,
              sublabel_scale=_sublabel_scale(gate))


# --- Public API -----------------------------------------------------------


def _render_figure(
    circuit: Any,
    *,
    ax,
    layered: bool,
    fontsize: int,
    title: str | None,
    qubit_labels: list[str] | None,
    wire_color: str,
):
    """Draw an already-resolved ``Circuit`` and return ``(fig, ax)``."""
    plt = require_matplotlib()
    n_qubits = int(circuit.n_qubits)
    gates = list(circuit.gates)

    if layered and gates:
        columns = _pack_layers(circuit)
    else:
        columns = list(range(len(gates)))
    n_cols = (max(columns) + 1) if columns else 1

    if ax is None:
        width = max(3.5, n_cols * 0.95 + 1.4)
        height = max(1.6, n_qubits * 0.9 + (0.9 if title else 0.5))
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    x_left, x_right = -0.85, n_cols - 1 + 0.85

    # Find the leftmost measurement column for each qubit (wire ends there).
    measure_col: dict[int, float] = {}
    for gate, col in zip(gates, columns):
        if gate["type"] in {"measure", "measurement"}:
            for q in _measure_targets(gate):
                if q not in measure_col or col < measure_col[q]:
                    measure_col[q] = float(col)

    for q in range(n_qubits):
        y = _yy(q, n_qubits)
        wire_end = measure_col[q] if q in measure_col else x_right
        ax.plot([x_left, wire_end], [y, y], color=wire_color, linewidth=1.4,
                zorder=1)
        label = qubit_labels[q] if qubit_labels else f"q{q}"
        ax.text(x_left - 0.15, y, label, ha="right", va="center",
                fontsize=fontsize * 0.8, color="#444444")

    for gate, col in zip(gates, columns):
        _render_gate(ax, gate, float(col), n_qubits, fontsize)

    ax.set_xlim(x_left - 0.9, x_right + 0.3)
    ax.set_ylim(-0.9, n_qubits - 1 + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize * 1.05, pad=10)
    fig.tight_layout()
    return fig, ax


def _load_from_path(path: Path) -> Circuit:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_circuit_json(path)
    if suffix in {".qasm", ".qasm2", ".qasm3"}:
        return load_circuit_qasm(path)
    text = path.read_text(encoding="utf-8")
    if "OPENQASM" in text:
        return circuit_from_qasm(text)
    return circuit_from_json(text)


def _coerce_circuit(source: Any) -> tuple[Any, str | None]:
    """Normalise any aicir circuit form to ``(circuit, default_name)``.

    Accepted forms: a ``Circuit``; a named wrapper exposing ``.circuit``
    (e.g. ``ArchitectureSpec``); a single gate dict or a sequence of gate
    dicts; a circuit-JSON dict; a JSON or OPENQASM string; or a path to a
    ``.json`` / ``.qasm`` file.
    """
    # Named wrapper around a circuit (e.g. QAS ArchitectureSpec).
    inner = getattr(source, "circuit", None)
    if inner is not None and hasattr(inner, "gates") and hasattr(inner, "n_qubits"):
        return inner, getattr(source, "name", None)

    # A Circuit (or any object exposing gates/n_qubits).
    if hasattr(source, "gates") and hasattr(source, "n_qubits"):
        return source, getattr(source, "name", None)

    # Mapping forms: circuit-JSON dict or a single gate dict.
    if isinstance(source, dict):
        if "gates" in source and "n_qubits" in source:
            return circuit_from_json_dict(source), source.get("name")
        if "type" in source:
            return Circuit(source), None
        raise TypeError("dict is neither a circuit-JSON dict nor a gate dict")

    # A sequence of gate dicts.
    if isinstance(source, (list, tuple)):
        if source and all(isinstance(g, dict) and "type" in g for g in source):
            return Circuit(*source), None
        raise TypeError("sequence must be non-empty gate dicts with a 'type' key")

    # Path / string forms (file path, JSON text, or OPENQASM text).
    if isinstance(source, Path):
        return _load_from_path(source), source.stem
    if isinstance(source, str):
        text = source.strip()
        # Only short, single-line strings are considered file paths; JSON and
        # OPENQASM payloads are multi-line content, not paths.
        if "\n" not in source and len(source) < 512:
            candidate = Path(source)
            try:
                is_file = candidate.suffix.lower() in {".json", ".qasm", ".qasm2", ".qasm3"} or candidate.exists()
            except OSError:
                is_file = False
            if is_file:
                return _load_from_path(candidate), candidate.stem
        if "OPENQASM" in text:
            return circuit_from_qasm(source), None
        if text.startswith("{"):
            return circuit_from_json(source), None
        return circuit_from_qasm(source), None

    raise TypeError(f"Unsupported circuit form: {type(source).__name__}")


# --- Caller-aware output naming -------------------------------------------

# Per executing-file counter for circuits that are not bound to a variable.
_DEFAULT_COUNTERS: dict[str, int] = {}
# Cache of parsed caller sources keyed by filename -> (mtime, ast.Module).
_TREE_CACHE: dict[str, tuple[float, ast.Module]] = {}


def _parse_caller_source(filename: str) -> ast.Module | None:
    try:
        mtime = os.path.getmtime(filename)
    except OSError:
        return None
    cached = _TREE_CACHE.get(filename)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        source = "".join(linecache.getlines(filename))
        if not source:
            source = Path(filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return None
    _TREE_CACHE[filename] = (mtime, tree)
    return tree


def _is_plot_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "plot"
    if isinstance(func, ast.Attribute):
        return func.attr == "plot"
    return False


def _first_circuit_arg(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "circuit":
            return keyword.value
    return None


def _circuit_expr_for_plot_call(call: ast.Call) -> ast.expr | None:
    arg = _first_circuit_arg(call)
    if arg is not None:
        return arg
    if isinstance(call.func, ast.Attribute) and call.func.attr == "plot":
        return call.func.value
    return None


def _arg_variable_name(filename: str, lineno: int) -> str | None:
    """Return the variable name passed as the circuit to ``plot`` at ``lineno``.

    Returns ``None`` when the argument is not a plain variable (e.g. a literal,
    a call, an attribute, or an indexed expression). For method-call syntax,
    the receiver name in ``cir.plot()`` is treated as the circuit variable.
    """
    tree = _parse_caller_source(filename)
    if tree is None:
        return None

    candidates: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or node.lineno
        if start <= lineno <= end and _circuit_expr_for_plot_call(node) is not None:
            candidates.append(node)
    if not candidates:
        return None

    # Prefer calls that clearly target ``plot`` and that start on this line.
    candidates.sort(
        key=lambda n: (not _is_plot_call(n), n.lineno != lineno, n.col_offset)
    )
    arg = _circuit_expr_for_plot_call(candidates[0])
    if isinstance(arg, ast.Name):
        return arg.id
    return None


def _default_target(name: str | None, caller) -> tuple[Path, str]:
    """Resolve the output directory and base filename from the caller frame."""
    filename = caller.f_code.co_filename if caller is not None else "<unknown>"
    exe_path = Path(filename)
    has_source = caller is not None and not filename.startswith("<") and exe_path.exists()

    if has_source:
        exe_dir = exe_path.parent
        exe_stem = exe_path.stem
        var = _arg_variable_name(filename, caller.f_lineno)
    else:  # interactive sessions, exec(), etc.
        exe_dir = Path.cwd()
        exe_stem = "circuit"
        var = None

    if name:
        return exe_dir, name
    if var:
        return exe_dir, f"{exe_stem}_{var}"

    index = _DEFAULT_COUNTERS.get(filename, 0)
    _DEFAULT_COUNTERS[filename] = index + 1
    return exe_dir, f"{exe_stem}_{index}"


def _caller_dir(caller) -> Path:
    filename = caller.f_code.co_filename if caller is not None else "<unknown>"
    exe_path = Path(filename)
    if caller is not None and not filename.startswith("<") and exe_path.exists():
        return exe_path.parent
    return Path.cwd()


def _resolve_output_path(path: Any, name: str | None, caller) -> Path:
    # Relative explicit paths are resolved from the caller script directory, not
    # from the shell's current working directory.
    if path is not None:
        out = Path(path)
        if not out.is_absolute():
            out = _caller_dir(caller) / out
        if not out.is_dir():
            return out if out.suffix.lower() == ".png" else out.with_suffix(".png")

    # No path, or a directory to drop the default-named file into.
    exe_dir, base = _default_target(name, caller)
    if path is not None:
        return out / f"{base}.png"
    return exe_dir / f"{base}.png"


def plot(
    circuit: Any,
    path: Any = None,
    *,
    name: str | None = None,
    save: bool = True,
    dpi: int = 200,
    ax=None,
    layered: bool = True,
    fontsize: int = 15,
    title: str | None = None,
    qubit_labels: list[str] | None = None,
    wire_color: str = "#9AA0A6",
    _caller=None,
):
    """Plot a quantum circuit as a coloured, rounded-box PNG figure.

    ``circuit`` may be given in any form aicir defines:

    * a :class:`~aicir.core.circuit.Circuit`;
    * a named wrapper exposing ``.circuit`` (e.g. a QAS ``ArchitectureSpec``);
    * a single gate dict or a sequence of gate dicts;
    * a circuit-JSON dict, or a JSON / OPENQASM string;
    * a path to a ``.json`` or ``.qasm`` file.

    Output location
    ---------------
    Pass an explicit destination as the optional second argument to control
    exactly where the figure is written::

        plot(cir, "figures/bell.png")     # saves to figures/bell.png
        plot(cir, "/abs/path/qft.png")    # any directory is created as needed

    When ``path`` is omitted, the figure is written **next to the script that
    calls** ``plot``, named ``<executing-file>_<circuit-variable>.png``. For
    example, in ``demo.py``::

        cir = Circuit(...)
        plot(cir)            # writes demo_cir.png beside demo.py

    When the circuit is not a plain variable (a literal, a call result, etc.),
    successive anonymous calls fall back to ``demo_0.png``, ``demo_1.png``,
    and so on (also beside the script).

    Parameters
    ----------
    path:
        Optional output destination. A file name such as
        ``"<dir>/<name>.png"`` is used verbatim (a ``.png`` suffix is enforced
        and missing parent directories are created); an existing directory
        receives the default-named file; ``None`` (default) uses the
        caller-aware name described above.
    name:
        Override the output base name (written beside the caller, or into
        ``path`` if ``path`` is a directory). Ignored when ``path`` is an
        explicit file name.
    save:
        When ``False``, only build the figure and skip writing a file.
    dpi:
        Resolution of the saved PNG.
    ax, layered, fontsize, title, qubit_labels, wire_color:
        Rendering options (see :func:`_render_figure`); ``layered`` packs
        independent gates into shared columns.

    Returns
    -------
    (fig, ax) tuple.
    """
    frame = inspect.currentframe()
    caller = _caller if _caller is not None else (frame.f_back if frame is not None else None)
    circuit_obj, _ = _coerce_circuit(circuit)
    fig, axes = _render_figure(
        circuit_obj,
        ax=ax,
        layered=layered,
        fontsize=fontsize,
        title=title,
        qubit_labels=qubit_labels,
        wire_color=wire_color,
    )
    if save:
        out = _resolve_output_path(path, name, caller)
        if str(out.parent):
            out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, axes

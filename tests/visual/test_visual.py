import sys
from pathlib import Path

import numpy as np
import pytest

from aicir import Circuit, cnot, crz, cz, hadamard, measure, reset, rxx, rzz, rx, rz, s_gate, swap, t_gate, toffoli, u2, u3
from aicir.visual import (
    circuit_to_text,
    draw_circuit,
    gate_histogram,
    plot,
    plot_density_matrix,
    plot_state_amplitudes,
    plot_state_probs,
)


def test_circuit_to_text_and_draw_circuit_text():
    circuit = Circuit(hadamard(0), cnot(1, [0]), rzz(np.pi / 2, 0, 1), n_qubits=2)

    diagram = circuit_to_text(circuit)

    assert diagram == draw_circuit(circuit)
    assert "H" in diagram
    assert "X" in diagram
    assert "Rzz" in diagram
    assert "pi/2" in diagram


def test_circuit_to_text_shows_reset_marker():
    circuit = Circuit(measure(0), reset(0), n_qubits=1)

    diagram = circuit_to_text(circuit)

    assert "|0>" in diagram


def test_gate_histogram_counts_gate_types():
    circuit = Circuit(rx(0.1, 0), rx(0.2, 1), cnot(1, [0]), n_qubits=2)

    assert gate_histogram(circuit) == {"cx": 1, "rx": 2}


def test_circuit_helpers_reject_invalid_inputs():
    with pytest.raises(TypeError):
        circuit_to_text(object())
    with pytest.raises(TypeError):
        gate_histogram(object())


@pytest.fixture()
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_plot_state_probs_accepts_probability_vector(plt):
    fig, ax = plot_state_probs(np.array([0.5, 0.0, 0.0, 0.5]))

    assert fig is ax.figure
    assert len(ax.patches) == 4
    assert ax.get_ylabel() == "Probability"


def test_plot_state_probs_accepts_state_vector(plt):
    fig, ax = plot_state_probs(np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)]))

    heights = [patch.get_height() for patch in ax.patches]
    assert fig is ax.figure
    assert heights[0] == pytest.approx(0.5)
    assert heights[-1] == pytest.approx(0.5)


def test_plot_state_amplitudes(plt):
    fig, ax = plot_state_amplitudes(np.array([1.0, 1.0j]) / np.sqrt(2))

    assert fig is ax.figure
    assert len(ax.patches) == 6
    assert ax.get_ylabel() == "Amplitude"


def test_plot_density_matrix(plt):
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

    fig, ax = plot_density_matrix(rho, part="real")

    assert fig is ax.figure
    assert ax.get_title() == "Density matrix (real)"


def test_plot_density_matrix_rejects_bad_part(plt):
    with pytest.raises(ValueError):
        plot_density_matrix(np.eye(2), part="bad")


# Stem of this test file; default output names are prefixed with it.
STEM = Path(__file__).stem


def _reset_default_counters():
    sys.modules["aicir.visual.plot"]._DEFAULT_COUNTERS.clear()


def test_plot_returns_fig_and_draws_patches(plt, tmp_path):
    circuit = Circuit(
        hadamard(0),
        cnot(1, [0]),
        cz(2, [1]),
        crz(np.pi / 2, 0, [1]),
        rzz(np.pi / 2, 0, 2),
        swap(1, 2),
        toffoli(2, (0, 1)),
        n_qubits=3,
    )

    fig, ax = plot(circuit, path=tmp_path, name="bell", title="circuit")

    assert fig is ax.figure
    assert ax.get_title() == "circuit"
    assert len(ax.patches) > 0
    assert (tmp_path / "bell.png").exists()


def test_plot_colors_clifford_and_nonclifford_gate_families():
    from matplotlib.colors import to_rgb

    style_for = sys.modules["aicir.visual.plot"]._style_for
    clifford_style = style_for("hadamard")
    nonclifford_style = style_for("t_gate")
    previous_clifford_style = ("#CBE8C6", "#4C9A4A")

    assert clifford_style != ("#CBDDF0", "#3B6FB5")
    assert sum(to_rgb(clifford_style[0])) < sum(to_rgb(previous_clifford_style[0]))
    assert sum(to_rgb(clifford_style[1])) < sum(to_rgb(previous_clifford_style[1]))
    assert style_for("pauli_x") == clifford_style
    assert style_for("cx") == clifford_style
    assert style_for("cz") == clifford_style
    assert style_for("swap") == clifford_style
    assert style_for("t_gate") == ("#CBDDF0", "#3B6FB5")
    assert style_for("t_gate") == nonclifford_style
    assert style_for("toffoli") == nonclifford_style

    assert style_for("rx") != nonclifford_style
    assert style_for("rx") == style_for("u3")
    assert style_for("rx") == style_for("rzz")


def test_plot_parameterized_gate_colors_are_lighter_than_base_family(plt):
    from matplotlib.colors import to_rgb

    circuit = Circuit(t_gate(0), rx(0.1, 1), n_qubits=2)

    fig, ax = plot(circuit, layered=False, save=False)

    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]
    t_patch, rx_patch = box_patches

    assert fig is ax.figure
    assert sum(to_rgb(rx_patch.get_facecolor())) > sum(to_rgb(t_patch.get_facecolor()))
    assert sum(to_rgb(rx_patch.get_edgecolor())) > sum(to_rgb(t_patch.get_edgecolor()))


def test_plot_parameterized_nonclifford_light_color_is_only_slightly_lighter():
    plot_module = sys.modules["aicir.visual.plot"]
    nonclifford_style = plot_module._PALETTE["hadamard"]
    light_nonclifford = plot_module._light_style(nonclifford_style, amount=0.20)
    old_light_nonclifford = plot_module._light_style(nonclifford_style, amount=0.30)

    assert light_nonclifford != old_light_nonclifford
    assert light_nonclifford[0] < old_light_nonclifford[0]
    assert light_nonclifford[1] < old_light_nonclifford[1]


def test_plot_layered_packs_columns(plt):
    circuit = Circuit(hadamard(0), hadamard(1), hadamard(2), n_qubits=3)

    fig, ax = plot(circuit, layered=True, save=False)

    # All three independent Hadamards share one column, so every box is at x=0.
    xs = {round(patch.get_x() + patch.get_width() / 2, 6) for patch in ax.patches}
    assert xs == {0.0}


def test_plot_layered_preserves_order_across_wire_spans(plt):
    circuit = Circuit(
        cnot(0, [1, 2]),
        cnot(1, [0, 2]),
        cnot(2, [0]),
        s_gate(2),
        n_qubits=3,
    )

    fig, ax = plot(circuit, layered=True, save=False)

    s_patch = ax.patches[-1]
    s_x = round(s_patch.get_x() + s_patch.get_width() / 2, 6)
    cnot_xs = [
        round(line.get_xdata()[0], 6)
        for line in ax.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == line.get_xdata()[1]
        and line.get_ydata()[0] != line.get_ydata()[1]
    ]
    assert s_x > max(cnot_xs)


def test_plot_u2_u3_show_smaller_parameter_sublabels(plt):
    circuit = Circuit(
        rz(np.pi / 2, 0),
        u2(np.pi / 3, np.pi / 5, 1),
        u3(np.pi, 0.0, np.pi / 7, 2),
        n_qubits=3,
    )

    fig, ax = plot(circuit, layered=False, save=False)

    text_by_value = {text.get_text(): text for text in ax.texts}
    rz_label = text_by_value["π/2"]
    u2_label = text_by_value["π/3, 0.628"]
    u3_label = text_by_value["π, 0.000\n0.449"]

    # Angle sublabels render below the box (box half-height + small gap).
    below_box = 0.62 / 2 + 0.04
    assert u2_label.get_fontsize() < rz_label.get_fontsize()
    assert u3_label.get_fontsize() < rz_label.get_fontsize()
    assert rz_label.get_position()[1] == pytest.approx(2.0 - below_box)
    assert u2_label.get_position()[1] < 1.0
    assert u3_label.get_position()[1] < 0.0


def test_plot_fits_gate_text_to_box_size(plt):
    circuit = Circuit(
        {"type": "identity", "n_qubits": 1},
        rz(123.456789, 0),
        u2(123.456789, 234.567891, 1),
        u3(123.456789, 234.567891, 345.678912, 2),
        rzz(123.456789, 0, 2),
        n_qubits=3,
    )

    fig, ax = plot(circuit, layered=False, save=False, fontsize=60)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    marked = [
        text for text in ax.texts
        if getattr(text, "_aicir_max_width_data", None) is not None
        and getattr(text, "_aicir_max_height_data", None) is not None
    ]
    assert marked
    for text in marked:
        max_width_data = text._aicir_max_width_data
        max_height_data = text._aicir_max_height_data
        x, y = text.get_position()
        left = ax.transData.transform((x - max_width_data / 2, y))[0]
        right = ax.transData.transform((x + max_width_data / 2, y))[0]
        bottom = ax.transData.transform((x, y - max_height_data / 2))[1]
        top = ax.transData.transform((x, y + max_height_data / 2))[1]
        extent = text.get_window_extent(renderer=renderer)
        assert extent.width <= abs(right - left) + 1.0
        assert extent.height <= abs(top - bottom) + 1.0


def test_plot_marks_gate_box_labels_with_three_quarter_height_limit(plt):
    circuit = Circuit(hadamard(0), rz(np.pi / 2, 1), n_qubits=2)

    fig, ax = plot(circuit, layered=False, save=False)

    label_texts = [text for text in ax.texts if text.get_text() in {"H", "Rz"}]
    assert {text.get_text() for text in label_texts} == {"H", "Rz"}
    for text in label_texts:
        assert text._aicir_max_height_data == pytest.approx(0.62 * 0.75)


def test_plot_rzz_gate_label_is_rzz(plt):
    circuit = Circuit(rzz(np.pi / 2, 0, 1), n_qubits=2)

    fig, ax = plot(circuit, save=False)

    labels = [text.get_text() for text in ax.texts]
    assert labels.count("Rzz") == 2
    # The shared angle is shown once, below the lower (qubit 1) square.
    assert labels.count("π/2") == 1
    assert "ZZ" not in labels

    value_ys = [text.get_position()[1] for text in ax.texts if text.get_text() == "π/2"]
    assert value_ys == pytest.approx([-(0.62 / 2 + 0.04)])


def test_plot_rxx_gate_label_is_rxx(plt):
    circuit = Circuit(rxx(np.pi / 2, 0, 1), n_qubits=2)

    fig, ax = plot(circuit, save=False)

    labels = [text.get_text() for text in ax.texts]
    assert labels.count("Rxx") == 2
    # The shared angle is shown once, below the lower (qubit 1) square.
    assert labels.count("π/2") == 1

    value_ys = [text.get_position()[1] for text in ax.texts if text.get_text() == "π/2"]
    assert value_ys == pytest.approx([-(0.62 / 2 + 0.04)])


def test_plot_joint_measure_draws_boxes_connected_by_solid_line(plt):
    circuit = Circuit(measure(0, 2), n_qubits=3)

    fig, ax = plot(circuit, layered=False, save=False)

    vertical_solid = [
        line for line in ax.lines
        if line.get_linestyle() in {"-", "solid"}
        and np.allclose(line.get_xdata(), [0.0, 0.0], atol=1e-6)
        and np.allclose(line.get_ydata(), [0.0, 2.0], atol=1e-6)
    ]
    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]

    assert fig is ax.figure
    assert len(box_patches) == 2
    assert vertical_solid


def test_plot_measure_with_following_gate_does_not_draw_separator(plt):
    circuit = Circuit(hadamard(0), measure(0, 2), rz(0.2, 1), n_qubits=3)

    fig, ax = plot(circuit, layered=False, save=False)

    box = sys.modules["aicir.visual.plot"]._BOX
    measure_edge = sys.modules["aicir.visual.plot"]._style_for("measure")[1]
    separators = [
        line for line in ax.lines
        if line.get_linestyle() == "--"
        and line.get_color() == measure_edge
        and np.allclose(line.get_ydata(), [0.0, 2.0], atol=1e-6)
    ]

    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]
    h_patch = next(p for p in box_patches if p.get_y() + p.get_height() / 2 == pytest.approx(2.0))
    rz_patch = next(p for p in box_patches if p.get_y() + p.get_height() / 2 == pytest.approx(1.0))

    assert fig is ax.figure
    assert separators == []
    assert h_patch.get_x() + h_patch.get_width() < 1.0 - box / 2
    assert rz_patch.get_x() > 1.0 + box / 2


def test_plot_measured_qubit_wire_continues_when_later_gate_uses_it(plt):
    circuit = Circuit(hadamard(0), measure(0), rz(0.2, 0), n_qubits=1)

    fig, ax = plot(circuit, layered=False, save=False)

    solid_spans = []
    for line in ax.lines:
        if line.get_linestyle() not in {"-", "solid"}:
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.shape != (2,) or ydata.shape != (2,):
            continue
        if np.allclose(ydata, [0.0, 0.0], atol=1e-6):
            solid_spans.append((min(xdata), max(xdata)))

    assert fig is ax.figure
    assert any(start <= 1.0 and end >= 2.0 for start, end in solid_spans)


def test_plot_measure_reset_are_global_layer_barriers(plt):
    circuit = Circuit(hadamard(0), measure(1), rx(0.2, 0), reset(1), rz(0.3, 0), n_qubits=2)

    fig, ax = plot(circuit, layered=True, save=False)

    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]
    centers = [round(patch.get_x() + patch.get_width() / 2, 6) for patch in box_patches]

    assert fig is ax.figure
    assert centers == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_plot_rejects_invalid_input(plt):
    with pytest.raises(TypeError):
        plot(object(), save=False)


def test_plot_default_name_uses_variable_name(plt, tmp_path):
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    plot(cir, path=tmp_path)

    # <executing-file>_<variable>.png
    assert (tmp_path / f"{STEM}_cir.png").exists()


def test_plot_anonymous_falls_back_to_default_counter(plt, tmp_path):
    _reset_default_counters()

    plot(Circuit(hadamard(0), n_qubits=1), path=tmp_path)
    plot(Circuit(hadamard(0), n_qubits=1), path=tmp_path)

    assert (tmp_path / f"{STEM}_0.png").exists()
    assert (tmp_path / f"{STEM}_1.png").exists()


def test_plot_explicit_path_and_filename(plt, tmp_path):
    _reset_default_counters()
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    # "<dir>/<name>.png" used verbatim, creating missing parent directories.
    target = tmp_path / "nested" / "subdir" / "my_figure.png"
    plot(cir, str(target))
    assert target.exists()

    # A path without a suffix gets ".png" appended.
    plot(cir, str(tmp_path / "no_suffix"))
    assert (tmp_path / "no_suffix.png").exists()

    # An explicit file path must not advance the anonymous default counter.
    plot(Circuit(hadamard(0), n_qubits=1), str(tmp_path / "anon.png"))
    assert (tmp_path / "anon.png").exists()
    assert sys.modules["aicir.visual.plot"]._DEFAULT_COUNTERS == {}


def test_circuit_plot_method_uses_visual_plot(plt, tmp_path):
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    fig, ax = cir.plot(tmp_path / "method_output")

    assert fig is ax.figure
    assert (tmp_path / "method_output.png").exists()


def test_circuit_plot_method_default_saves_next_to_calling_script(plt, tmp_path):
    script = tmp_path / "call_circuit_plot.py"
    source = "\n".join(
        [
            "from aicir import Circuit, hadamard",
            "cir = Circuit(hadamard(0), n_qubits=1)",
            "cir.plot()",
        ]
    )
    script.write_text(source, encoding="utf-8")

    exec(compile(source, str(script), "exec"), {})

    assert (tmp_path / "call_circuit_plot_cir.png").exists()


def test_circuit_plot_method_relative_path_uses_calling_script_dir(plt, tmp_path):
    script = tmp_path / "call_circuit_plot_relative.py"
    source = "\n".join(
        [
            "from aicir import Circuit, hadamard",
            "cir = Circuit(hadamard(0), n_qubits=1)",
            "cir.plot('figures/named')",
        ]
    )
    script.write_text(source, encoding="utf-8")

    exec(compile(source, str(script), "exec"), {})

    assert (tmp_path / "figures" / "named.png").exists()


def test_plot_accepts_gate_list_and_qasm_and_json(plt, tmp_path):
    from aicir.core.io.json_io import circuit_to_json
    from aicir.core.io.qasm import circuit_to_qasm

    circuit = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    # list of gate dicts
    plot([hadamard(0), cnot(1, [0])], path=tmp_path / "gates.png")
    # circuit-JSON string
    plot(circuit_to_json(circuit), path=tmp_path / "fromjson.png")
    # OPENQASM string
    plot(circuit_to_qasm(circuit), path=tmp_path / "fromqasm.png")
    # single gate dict
    fig, ax = plot(hadamard(0), save=False)

    assert (tmp_path / "gates.png").exists()
    assert (tmp_path / "fromjson.png").exists()
    assert (tmp_path / "fromqasm.png").exists()
    assert fig is ax.figure


def test_plot_accepts_architecture_spec(plt, tmp_path):
    from aicir.qas._types import ArchitectureSpec

    spec = ArchitectureSpec.from_gates("my_ansatz", [hadamard(0), cnot(1, [0])], n_qubits=2)

    plot(spec, path=tmp_path)

    # Variable name wins for the filename; the spec is still a valid input form.
    assert (tmp_path / f"{STEM}_spec.png").exists()


def test_plot_handles_measurement_and_unitary(plt):
    circuit = Circuit(
        {"type": "unitary", "n_qubits": 2},
        {"type": "measure", "target_qubit": 0},
        n_qubits=2,
    )

    fig, ax = plot(circuit, save=False)

    assert fig is ax.figure
    assert len(ax.patches) > 0


def test_plot_handles_reset_marker(plt):
    circuit = Circuit(measure(0), reset(0), n_qubits=1)

    fig, ax = plot(circuit, save=False)

    labels = [text.get_text() for text in ax.texts]
    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]
    from matplotlib.colors import to_rgba
    measure_edge = sys.modules["aicir.visual.plot"]._style_for("measure")[1]
    reset_arcs = [
        patch for patch in ax.patches
        if patch.__class__.__name__ == "Arc"
        and patch.center[0] == pytest.approx(1.0)
    ]
    dashed = [
        line for line in ax.lines
        if line.get_linestyle() == "--"
        and not np.allclose(line.get_xdata(), [line.get_xdata()[0], line.get_xdata()[0]], atol=1e-6)
    ]

    assert fig is ax.figure
    assert "Reset" not in labels
    assert "|0>" not in labels
    assert len(box_patches) == 2
    assert len(reset_arcs) == 2
    assert box_patches[-1].get_edgecolor() == pytest.approx(to_rgba(measure_edge))
    for arc in reset_arcs:
        assert arc.get_edgecolor() == pytest.approx(to_rgba(measure_edge))
    assert dashed == []


def test_plot_reset_arcs_are_split_before_arrowheads(plt):
    circuit = Circuit(reset(0), n_qubits=1)

    fig, ax = plot(circuit, layered=False, save=False)

    box = sys.modules["aicir.visual.plot"]._BOX
    radius = box * 0.23
    min_gap_degrees = 16.0
    measure_edge = sys.modules["aicir.visual.plot"]._style_for("measure")[1]
    arrow_lines = [
        line for line in ax.lines
        if line.get_zorder() == 5
        and line.get_color() == measure_edge
        and np.mean(np.asarray(line.get_xdata(), dtype=float)) == pytest.approx(0.0, abs=0.25)
    ]

    assert fig is ax.figure
    assert len(arrow_lines) == 4
    for line in arrow_lines:
        x0 = float(line.get_xdata()[0])
        y0 = float(line.get_ydata()[0])
        distance = np.hypot(x0, y0)
        assert distance == pytest.approx(radius)

    reset_arcs = [
        patch for patch in ax.patches
        if patch.__class__.__name__ == "Arc"
        and patch.center[0] == pytest.approx(0.0)
    ]
    assert len(reset_arcs) == 2
    arc_ends = sorted(round(arc.theta2, 6) for arc in reset_arcs)
    assert arc_ends == [210.0, 390.0]
    for arc in reset_arcs:
        assert (arc.theta2 - arc.theta1) == pytest.approx(180.0 - min_gap_degrees)


def test_plot_reset_arrowheads_are_slightly_larger(plt):
    circuit = Circuit(reset(0), n_qubits=1)

    fig, ax = plot(circuit, layered=False, save=False)

    box = sys.modules["aicir.visual.plot"]._BOX
    measure_edge = sys.modules["aicir.visual.plot"]._style_for("measure")[1]
    arrow_lines = [
        line for line in ax.lines
        if line.get_zorder() == 5
        and line.get_color() == measure_edge
        and np.mean(np.asarray(line.get_xdata(), dtype=float)) == pytest.approx(0.0, abs=0.25)
    ]
    lengths = [
        np.hypot(
            float(line.get_xdata()[1]) - float(line.get_xdata()[0]),
            float(line.get_ydata()[1]) - float(line.get_ydata()[0]),
        )
        for line in arrow_lines
    ]

    assert fig is ax.figure
    assert len(lengths) == 4
    assert min(lengths) > box * 0.075


def test_plot_draws_reset_link_as_plain_solid_wire_before_next_gate(plt):
    circuit = Circuit(measure(0), reset(0), rz(0.2, 0), n_qubits=1)

    fig, ax = plot(circuit, layered=False, save=False)

    labels = [text.get_text() for text in ax.texts]
    box_patches = [patch for patch in ax.patches if hasattr(patch, "get_y")]
    from matplotlib.colors import to_rgba
    measure_edge = sys.modules["aicir.visual.plot"]._style_for("measure")[1]
    reset_arcs = [
        patch for patch in ax.patches
        if patch.__class__.__name__ == "Arc"
        and patch.center[0] == pytest.approx(1.0)
    ]
    dashed = [
        line for line in ax.lines
        if line.get_linestyle() == "--"
        and not np.allclose(line.get_xdata(), [line.get_xdata()[0], line.get_xdata()[0]], atol=1e-6)
    ]

    assert fig is ax.figure
    assert "Reset" not in labels
    assert "|0>" not in labels
    assert len(reset_arcs) == 2
    assert box_patches[1].get_edgecolor() == pytest.approx(to_rgba(measure_edge))
    for arc in reset_arcs:
        assert arc.get_edgecolor() == pytest.approx(to_rgba(measure_edge))
    assert dashed == []

    solid_wire_spans = []
    for line in ax.lines:
        if line.get_linestyle() not in {"-", "solid"}:
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.shape != (2,) or ydata.shape != (2,):
            continue
        if not np.allclose(ydata, [0.0, 0.0], atol=1e-6):
            continue
        solid_wire_spans.append((min(xdata), max(xdata)))
    assert any(np.isclose(start, 0.0) and np.isclose(end, 2.0) for start, end in solid_wire_spans)


@pytest.mark.parametrize("next_gate", [cnot(1, [0]), swap(0, 1)])
def test_plot_reset_dash_stops_at_virtual_box_for_boxless_next_gate(plt, next_gate):
    circuit = Circuit(measure(0), reset(0), next_gate, n_qubits=2)

    fig, ax = plot(circuit, layered=False, save=False)

    box = sys.modules["aicir.visual.plot"]._BOX
    solid_segments = []
    for line in ax.lines:
        if line.get_linestyle() not in {"-", "solid"}:
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.shape != (2,) or ydata.shape != (2,):
            continue
        if np.allclose(ydata, [1.0, 1.0], atol=1e-6):
            solid_segments.append((min(xdata), max(xdata)))

    expected_stop = 2.0 - box / 2
    assert fig is ax.figure
    assert any(np.isclose(start, 0.0) and np.isclose(end, expected_stop) for start, end in solid_segments)
    assert any(np.isclose(start, expected_stop) and end > 2.0 for start, end in solid_segments)

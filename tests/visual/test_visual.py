import sys
from pathlib import Path

import numpy as np
import pytest

from aicir import Circuit, cnot, crz, cz, hadamard, rzz, rx, s_gate, swap, toffoli
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
    assert "ZZ" in diagram
    assert "pi/2" in diagram


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

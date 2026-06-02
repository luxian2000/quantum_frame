import numpy as np
import pytest

from aicir import Circuit, cnot, hadamard, rzz, rx
from aicir.visual import (
    circuit_to_text,
    draw_circuit,
    gate_histogram,
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

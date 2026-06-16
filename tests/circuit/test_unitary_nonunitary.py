import pytest
import numpy as np
from aicir import Circuit, hadamard, cnot, measure, reset


def test_unitary_raises_on_measure_by_default():
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    with pytest.raises(ValueError):
        cir.unitary()


def test_unitary_raises_on_reset_by_default():
    cir = Circuit(hadamard(0), reset(0), n_qubits=1)
    with pytest.raises(ValueError):
        cir.unitary()


def test_unitary_ignore_nonunitary_drops_markers():
    cir = Circuit(hadamard(0), measure(0), reset(0), n_qubits=1)
    u = cir.unitary(ignore_nonunitary=True)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    assert np.allclose(np.asarray(u), h, atol=1e-6)


def test_pure_circuit_unitary_unchanged():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    assert np.asarray(cir.unitary()).shape == (4, 4)

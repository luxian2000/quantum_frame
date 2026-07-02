import numpy as np
import pytest
from aicir import Circuit, Hamiltonian, Measure, NumpyBackend, cnot, ry


def _circuit():
    return Circuit(ry(0.4, 0), cnot(1, [0]), ry(0.9, 1), n_qubits=2)


def test_measure_tensor_matches_statevector_probs():
    bk = NumpyBackend()
    c = _circuit()
    ref = Measure(bk).run(c, shots=None, method="statevector")
    tn = Measure(bk).run(c, shots=None, method="tensor")
    assert np.allclose(ref.state.probabilities(), tn.state.probabilities(), atol=1e-5)


def test_measure_tensor_expectation_matches():
    bk = NumpyBackend()
    c = _circuit()
    H = {"H": Hamiltonian([("ZI", 1.0)]).to_matrix(bk)}
    ref = Measure(bk).run(c, shots=None, observables=H, method="statevector")
    tn = Measure(bk).run(c, shots=None, observables=H, method="tensor")
    assert np.isclose(ref.expectation_values["H"], tn.expectation_values["H"], atol=1e-5)


def test_measure_tensor_rejects_bad_method():
    bk = NumpyBackend()
    with pytest.raises(ValueError):
        Measure(bk).run(_circuit(), shots=None, method="bogus")

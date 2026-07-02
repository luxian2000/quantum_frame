import numpy as np
import pytest
from aicir import (Circuit, NumpyBackend, State, cnot, ry, rzz,
                   single_amplitude, partial_amplitude, tn_statevector)


def _ref_state(circuit, bk):
    return State.zero_state(circuit.n_qubits, bk).evolve(circuit.unitary(backend=bk)).to_numpy()


def _demo_circuit():
    return Circuit(ry(0.4, 0), cnot(1, [0]), ry(0.9, 1), rzz(0.3, 0, 1), n_qubits=2)


def test_tn_statevector_matches_reference():
    bk = NumpyBackend()
    c = _demo_circuit()
    assert np.allclose(tn_statevector(c, backend=bk).to_numpy(), _ref_state(c, bk), atol=1e-5)


def test_single_amplitude_matches_reference():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)
    for i, bits in enumerate(["00", "01", "10", "11"]):
        assert np.isclose(single_amplitude(c, bits, backend=bk), ref[i], atol=1e-5)


def test_partial_amplitude_open_qubits():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)  # qubit0 MSB: index = b0*2 + b1
    # 固定 qubit0=0，开放 qubit1 -> [<00|psi>, <01|psi>]
    vec = partial_amplitude(c, open_qubits=[1], backend=bk)
    assert np.allclose(vec, [ref[0], ref[1]], atol=1e-5)


def test_partial_amplitude_bitstrings():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)
    vec = partial_amplitude(c, bitstrings=["11", "00"], backend=bk)
    assert np.allclose(vec, [ref[3], ref[0]], atol=1e-5)


def test_partial_amplitude_requires_exactly_one_mode():
    bk = NumpyBackend()
    c = _demo_circuit()
    with pytest.raises(ValueError):
        partial_amplitude(c, backend=bk)
    with pytest.raises(ValueError):
        partial_amplitude(c, open_qubits=[0], bitstrings=["00"], backend=bk)


def test_tn_statevector_matches_reference_gpu():
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    c = _demo_circuit()
    assert np.allclose(tn_statevector(c, backend=bk).to_numpy(), _ref_state(c, bk), atol=1e-4)

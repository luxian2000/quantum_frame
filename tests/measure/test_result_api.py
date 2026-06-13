import numpy as np
import pytest
from aicir.measure.result import Result, MeasureSpec


def _empty_probs(n):
    p = np.zeros(1 << n); p[0] = 1.0
    return p


def test_output_by_index_id_and_terminal():
    specs = [MeasureSpec(op_index=2, id="m0", qubits=[0, 1], basis="Z")]
    r = Result(n_qubits=2, backend_name="numpy", probabilities=_empty_probs(2),
               shots=4, measurement_specs=specs,
               incircuit_outputs={2: np.array([[1], [1], [-1], [1]])},
               terminal_output=np.array([[1, 1], [1, -1], [-1, -1], [1, 1]]),
               terminal_qubits=[0, 1])
    assert r.output(2).shape == (4, 1)
    assert np.array_equal(r.output("m0"), r.output(2))
    assert r.output(-1).shape == (4, 2)


def test_output_invalid_index_raises():
    r = Result(n_qubits=1, backend_name="numpy", probabilities=_empty_probs(1),
               shots=2, measurement_specs=[], incircuit_outputs={}, terminal_output=None,
               terminal_qubits=None)
    with pytest.raises(ValueError):
        r.output(0)
    with pytest.raises(ValueError):
        r.output(-1)


def test_counts_prob_sampling_only():
    specs = [MeasureSpec(op_index=1, id=None, qubits=[0], basis="Z")]
    r = Result(n_qubits=1, backend_name="numpy", probabilities=_empty_probs(1),
               shots=None, measurement_specs=specs, incircuit_outputs={1: 1},
               terminal_output=None, terminal_qubits=None)
    with pytest.raises(RuntimeError):
        r.counts(1)


def test_reduce_partial_trace():
    rho = np.zeros((4, 4), dtype=complex); rho[0, 0] = rho[3, 3] = 0.5
    r = Result(n_qubits=2, backend_name="numpy", probabilities=_empty_probs(2),
               shots=8, measurement_specs=[], incircuit_outputs={}, terminal_output=None,
               terminal_qubits=None, final_state=rho, final_state_kind="density_matrix")
    red = r.reduce([0], pos="final")
    assert np.allclose(red, np.array([[0.5, 0], [0, 0.5]]), atol=1e-6)

import numpy as np
from aicir.core.state import State
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.measure.trajectory import TrajectoryResult
from aicir.measure.aggregate import aggregate_avg


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_aggregate_two_shots_outputs_stacked_and_state_is_density():
    t0 = TrajectoryResult(pre=_sv([1, 0]), post=_sv([1, 0]), incircuit={1: 1}, terminal=[1])
    t1 = TrajectoryResult(pre=_sv([0, 1]), post=_sv([0, 1]), incircuit={1: -1}, terminal=[-1])
    agg = aggregate_avg([t0, t1], n_qubits=1, measurement_specs=[], terminal_qubits=[0])
    assert agg["state"].shape == (2, 2)
    assert np.allclose(agg["state"], np.array([[0.5, 0], [0, 0.5]]), atol=1e-6)
    assert agg["incircuit_outputs"][1].shape == (2, 1)
    assert agg["terminal_output"].shape == (2, 1)
    assert agg["incircuit_counts"][1] == {1: 1, -1: 1}

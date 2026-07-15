import numpy as np
from aicir import Circuit, hadamard, cnot, measure, reset, pauli_x
from aicir.core.state import State
from aicir.backends.numpy_backend import NumpyBackend
from aicir.measure.trajectory import run_trajectory


def _init(n):
    b = NumpyBackend()
    return State.zero_state(n, b), b


def test_trajectory_records_incircuit_outcome_and_terminal():
    cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1]), n_qubits=2)  # ops: 0,1,2
    st, b = _init(2)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=True, measure_qubits=[0, 1],
                        snap_ops=set(), rng=rng, noise_model=None)
    assert tr.incircuit[2] == 1            # Bell 的 Z⊗Z 恒 +1
    assert len(tr.terminal) == 2
    assert set(tr.terminal) <= {1, -1}


def test_trajectory_reset_zeroes_qubit():
    cir = Circuit(pauli_x(0), reset(0), n_qubits=1)  # ops 0,1
    st, b = _init(1)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=False, measure_qubits=None,
                        snap_ops={1}, rng=rng, noise_model=None)
    snap = tr.snaps[1].to_numpy().reshape(-1)
    assert np.allclose(snap, [1, 0], atol=1e-6)
    assert tr.terminal is None


def test_trajectory_snap_after_op_index():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    st, b = _init(2)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=False, measure_qubits=None,
                        snap_ops={0}, rng=rng, noise_model=None)
    v = tr.snaps[0].to_numpy().reshape(-1)
    assert np.allclose(np.abs(v), [1/np.sqrt(2), 0, 1/np.sqrt(2), 0], atol=1e-6)


def test_density_path_gate_matrices_built_once_across_shots(monkeypatch):
    # 密度/噪声路径全尺寸门矩阵只应按门构建一次，不随每条轨迹重建
    import numpy as np
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.core.circuit import Circuit, cnot, hadamard
    from aicir.measure import trajectory as traj_mod
    from aicir.measure.measure import Measure
    from aicir.noise import BitFlipChannel, NoiseModel

    calls = []
    original = traj_mod.gate_to_matrix

    def spying(gate, *a, **k):
        calls.append(True)
        return original(gate, *a, **k)

    monkeypatch.setattr(traj_mod, "gate_to_matrix", spying)

    backend = NumpyBackend()
    circ = Circuit(hadamard(0), cnot(1, [0]), hadamard(1), n_qubits=2, backend=backend)
    circ.noise_model = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=0.2))

    result = Measure(backend).run(circ, shots=8, seed=2, return_state=False)
    assert np.isclose(float(np.sum(result.probabilities)), 1.0, atol=1e-6)
    assert len(calls) == 3  # 3 个门各一次，而非 8 shots × 3

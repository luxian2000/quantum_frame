import numpy as np

from aicir import (
    BitFlipChannel,
    Circuit,
    Measure,
    NoiseModel,
    NumpyBackend,
    cnot,
    hadamard,
    rx,
    ry,
    rzz,
)
from aicir.core import State
from aicir.core.gates import apply_gate_to_state


class RecordingNumpyBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.local_calls = []

    def apply_statevector_local(self, state, local_matrix, axes, n_qubits):
        self.local_calls.append((tuple(int(axis) for axis in axes), int(n_qubits)))
        return super().apply_statevector_local(state, local_matrix, axes, n_qubits)


def test_apply_gate_to_state_uses_backend_local_hook_for_one_and_two_qubit_gates():
    backend = RecordingNumpyBackend()
    state = State.zero_state(4, backend).data

    state = apply_gate_to_state(hadamard(0), state, 4, backend)
    state = apply_gate_to_state(cnot(3, [1]), state, 4, backend)
    state = apply_gate_to_state(rzz(0.25, 0, 2), state, 4, backend)

    assert backend.local_calls == [((0,), 4), ((1, 3), 4), ((0, 2), 4)]


def test_measure_run_statevector_uses_chunked_hook_and_matches_unitary_reference():
    backend = RecordingNumpyBackend()
    circuit = Circuit(
        hadamard(0),
        ry(0.3, 1),
        cnot(2, [0]),
        rzz(0.2, 1, 3),
        rx(-0.4, 2),
        n_qubits=4,
        backend=backend,
    )

    result = Measure(backend).run(circuit, shots=None, return_state=True)
    actual = result.final_state.to_numpy()

    reference_backend = NumpyBackend()
    reference = State.zero_state(4, reference_backend).evolve(
        circuit.unitary(backend=reference_backend)
    ).to_numpy()

    np.testing.assert_allclose(actual, reference, atol=1e-6)
    assert len(backend.local_calls) == 5


def test_noisy_measure_run_does_not_use_statevector_local_hook():
    backend = RecordingNumpyBackend()
    circuit = Circuit(hadamard(0), n_qubits=1, backend=backend)
    circuit.noise_model = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=0.0))

    result = Measure(backend).run(circuit, shots=None, return_state=True)

    assert backend.local_calls == []
    assert result.final_state.is_density
    np.testing.assert_allclose(np.trace(result.final_state.to_numpy()), 1.0, atol=1e-6)

import unittest

import numpy as np

from aicir import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    Circuit,
    Measure,
    NoiseModel,
    PhaseFlipChannel,
    NumpyBackend,
)


class TestNoiseModel(unittest.TestCase):
    def setUp(self):
        self.backend = NumpyBackend()
        self.measure = Measure(self.backend)

    def test_bit_flip_full_probability(self):
        # 从 |0><0| 开始，经过 identity 后施加 bit flip(p=1) -> |1><1|
        circ = Circuit({"type": "identity", "n_qubits": 1}, n_qubits=1)
        noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=1.0))
        circ.noise_model = noise

        result = self.measure.run(circ, shots=None)
        self.assertAlmostEqual(result.probabilities[0], 0.0, places=6)
        self.assertAlmostEqual(result.probabilities[1], 1.0, places=6)
        self.assertEqual(result.metadata.get("noise_model"), "NoiseModel")

    def test_amplitude_damping_from_excited_state(self):
        # 从 |1><1| 开始，gamma=1 完全衰减到 |0><0|
        circ = Circuit({"type": "identity", "n_qubits": 1}, n_qubits=1)
        rho1 = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex64)
        noise = NoiseModel().add_channel(AmplitudeDampingChannel(target_qubit=0, gamma=1.0))
        circ.noise_model = noise

        result = self.measure.run(
            circ,
            initial_density_matrix=rho1,
            shots=None,
        )
        self.assertAlmostEqual(result.probabilities[0], 1.0, places=6)
        self.assertAlmostEqual(result.probabilities[1], 0.0, places=6)

    def test_phase_flip_changes_coherence(self):
        # 手动准备 |+><+|，相位翻转 p=1 后应变成 |-><-|
        circ = Circuit({"type": "identity", "n_qubits": 1}, n_qubits=1)
        plus = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]], dtype=np.complex64)
        noise = NoiseModel().add_channel(PhaseFlipChannel(target_qubit=0, p=1.0))
        circ.noise_model = noise

        result = self.measure.run(
            circ,
            initial_density_matrix=plus,
            shots=None,
            return_state=True,
        )
        rho = result.final_state.matrix.reshape(2, 2)
        self.assertAlmostEqual(np.real(rho[0, 1]), -0.5, places=6)
        self.assertAlmostEqual(np.real(rho[1, 0]), -0.5, places=6)

    def test_rule_gate_filter(self):
        # 规则仅在 hadamard 后触发；此电路是 X，不应触发 bit flip
        circ = Circuit({"type": "pauli_x", "target_qubit": 0}, n_qubits=1)
        noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=1.0), after_gates=["hadamard"])
        circ.noise_model = noise

        result = self.measure.run(circ, shots=None)
        # 只受 X 作用：|0> -> |1>
        self.assertAlmostEqual(result.probabilities[0], 0.0, places=6)
        self.assertAlmostEqual(result.probabilities[1], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from aicir.noise import (
    CorrelatedTwoQubitPauliChannel,
    ErasureChannel,
    GeneralizedAmplitudeDampingChannel,
    KrausChannel,
    PauliChannel,
    PhaseDampingChannel,
    ReadoutErrorChannel,
    ResetChannel,
    ThermalRelaxationChannel,
    TwoQubitDepolarizingChannel,
)
from aicir.backends.numpy_backend import NumpyBackend


def apply_channel(channel, rho, n_qubits):
    backend = NumpyBackend()
    out = backend.zeros(rho.shape)
    for k in channel.kraus_operators(n_qubits, backend):
        out = out + backend.matmul(backend.matmul(k, rho), backend.dagger(k))
    return np.asarray(out)


class TestNoiseChannels(unittest.TestCase):
    def test_pauli_channel_applies_independent_pauli_probabilities(self):
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

        out = apply_channel(PauliChannel(target_qubit=0, px=0.25, py=0.25, pz=0.0), rho0, 1)

        self.assertAlmostEqual(float(np.real(out[0, 0])), 0.5, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 0.5, places=6)

    def test_phase_damping_removes_coherence_without_changing_population(self):
        plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex64)

        out = apply_channel(PhaseDampingChannel(target_qubit=0, gamma=1.0), plus, 1)

        self.assertAlmostEqual(float(np.real(out[0, 0])), 0.5, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 0.5, places=6)
        self.assertAlmostEqual(float(np.real(out[0, 1])), 0.0, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 0])), 0.0, places=6)

    def test_generalized_amplitude_damping_can_heat_ground_state(self):
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

        out = apply_channel(
            GeneralizedAmplitudeDampingChannel(target_qubit=0, gamma=1.0, p_excited=1.0),
            rho0,
            1,
        )

        self.assertAlmostEqual(float(np.real(out[0, 0])), 0.0, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 1.0, places=6)

    def test_two_qubit_depolarizing_maps_pure_state_to_maximally_mixed_at_full_probability(self):
        rho00 = np.zeros((4, 4), dtype=np.complex64)
        rho00[0, 0] = 1.0

        out = apply_channel(TwoQubitDepolarizingChannel(qubit_1=0, qubit_2=1, p=1.0), rho00, 2)

        np.testing.assert_allclose(np.real(np.diag(out)), np.full(4, 0.25), atol=1e-6)
        self.assertAlmostEqual(float(np.trace(out).real), 1.0, places=6)

    def test_kraus_channel_embeds_targeted_single_qubit_operators(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
        rho00 = np.zeros((4, 4), dtype=np.complex64)
        rho00[0, 0] = 1.0

        out = apply_channel(KrausChannel([x], target_qubits=(1,), channel_name="x_on_q1"), rho00, 2)

        self.assertAlmostEqual(float(np.real(out[1, 1])), 1.0, places=6)

    def test_reset_channel_resets_excited_state_to_zero(self):
        rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)

        out = apply_channel(ResetChannel(target_qubit=0, p=1.0), rho1, 1)

        self.assertAlmostEqual(float(np.real(out[0, 0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 0.0, places=6)

    def test_erasure_channel_replaces_target_with_configured_mixed_state(self):
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

        out = apply_channel(ErasureChannel(target_qubit=0, p=1.0, erase_to=0.5), rho0, 1)

        self.assertAlmostEqual(float(np.real(out[0, 0])), 0.5, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 0.5, places=6)
        self.assertAlmostEqual(float(np.trace(out).real), 1.0, places=6)

    def test_readout_error_channel_applies_asymmetric_classical_confusion_proxy(self):
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

        out = apply_channel(ReadoutErrorChannel(target_qubit=0, p01=0.25, p10=0.0), rho0, 1)

        self.assertAlmostEqual(float(np.real(out[0, 0])), 0.75, places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), 0.25, places=6)

    def test_correlated_two_qubit_pauli_channel_applies_joint_error(self):
        rho00 = np.zeros((4, 4), dtype=np.complex64)
        rho00[0, 0] = 1.0

        out = apply_channel(
            CorrelatedTwoQubitPauliChannel(qubit_1=0, qubit_2=1, probabilities={("x", "x"): 1.0}),
            rho00,
            2,
        )

        self.assertAlmostEqual(float(np.real(out[3, 3])), 1.0, places=6)

    def test_thermal_relaxation_channel_relaxes_excited_state(self):
        rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)

        out = apply_channel(
            ThermalRelaxationChannel(target_qubit=0, t1=1.0, t2=2.0, gate_time=1.0, excited_population=0.0),
            rho1,
            1,
        )

        expected_ground = 1.0 - np.exp(-1.0)
        self.assertAlmostEqual(float(np.real(out[0, 0])), float(expected_ground), places=6)
        self.assertAlmostEqual(float(np.real(out[1, 1])), float(np.exp(-1.0)), places=6)


if __name__ == "__main__":
    unittest.main()

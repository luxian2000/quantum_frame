import math
import unittest

import numpy as np

from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.channel.noise.model import NoiseModel
from nexq.algorithms.qas import load_default_ion_trap_noise_config, load_ion_trap_noise_config


class _CountingIdentityChannel:
    name = "counting_identity"

    def __init__(self, target_qubit):
        self.target_qubit = target_qubit
        self.calls = 0

    def kraus_operators(self, n_qubits, backend):
        self.calls += 1
        return [backend.eye(1 << n_qubits)]


class TestIonTrapNoiseConfig(unittest.TestCase):
    def test_default_parameters_match_si1000_values(self):
        config = load_default_ion_trap_noise_config()
        resolved = config.resolved_parameters()

        self.assertEqual(resolved["rounds"], 25)
        self.assertEqual(resolved["basis"], "z")
        self.assertEqual(resolved["twoq_gate"], "zz_opt")
        self.assertEqual(resolved["data_qubits"], [0, 1, 6, 3, 4, 2, 7])
        self.assertEqual(resolved["ancillas"], [5])
        self.assertEqual(resolved["logical_label_mode"], "parity")

        self.assertAlmostEqual(resolved["oneq_depol"], 1.5e-3)
        self.assertAlmostEqual(resolved["twoq_depol"], 9.375e-3)
        self.assertAlmostEqual(resolved["cross_talk"], 2.8e-6)
        self.assertAlmostEqual(resolved["meas_bitflip"], 2.5e-4)
        self.assertAlmostEqual(resolved["reset_bitflip"], 2.5e-4)
        self.assertAlmostEqual(resolved["T2"], 0.2)
        self.assertAlmostEqual(resolved["oneq_gate_time"], 1.0e-4)
        self.assertAlmostEqual(resolved["twoq_gate_time"], 6.0e-4)

    def test_idle_dephasing_uses_document_formula_per_gate_family(self):
        config = load_default_ion_trap_noise_config()

        expected_oneq = 0.5 * (1.0 - math.exp(-1.0e-4 / 0.2))
        expected_twoq = 0.5 * (1.0 - math.exp(-6.0e-4 / 0.2))

        self.assertAlmostEqual(config.idle_dephasing_probability(gate_family="oneq"), expected_oneq)
        self.assertAlmostEqual(config.idle_dephasing_probability(gate_family="twoq"), expected_twoq)

    def test_noise_switches_control_default_model_rules(self):
        config = load_ion_trap_noise_config(
            overrides={
                "parameters": {
                    "enable_idle_dephasing_noise": False,
                    "enable_crosstalk_noise": False,
                }
            }
        )
        model = config.build_noise_model(qubits=[0, 1])
        channel_names = [rule.channel.name for rule in model.rules]

        self.assertNotIn("phase_flip", channel_names)
        self.assertIn("depolarizing", channel_names)

    def test_idle_rules_exclude_active_gate_qubits(self):
        config = load_default_ion_trap_noise_config()
        model = config.build_noise_model(qubits=[0, 1, 2])
        idle_rules = [rule for rule in model.rules if rule.channel.name == "phase_flip"]

        self.assertTrue(idle_rules)
        self.assertTrue(all(rule.exclude_gate_qubits for rule in idle_rules))

    def test_noise_model_skips_excluded_active_qubit(self):
        backend = NumpyBackend()
        model = NoiseModel()
        active_channel = _CountingIdentityChannel(target_qubit=0)
        idle_channel = _CountingIdentityChannel(target_qubit=1)
        model.add_channel(active_channel, after_gates=["hadamard"], exclude_gate_qubits=True)
        model.add_channel(idle_channel, after_gates=["hadamard"], exclude_gate_qubits=True)

        rho = np.eye(4, dtype=np.complex64) / 4.0
        _ = model.apply(
            rho,
            n_qubits=2,
            backend=backend,
            gate={"type": "hadamard", "target_qubit": 0},
        )

        self.assertEqual(active_channel.calls, 0)
        self.assertEqual(idle_channel.calls, 1)


if __name__ == "__main__":
    unittest.main()
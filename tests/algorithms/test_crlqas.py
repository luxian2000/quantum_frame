import unittest

import numpy as np

from nexq.algorithms.qas import AdamSPSAConfig, CRLQASConfig, crlqas, train_crlqas


class TestCRLQAS(unittest.TestCase):
    def test_train_crlqas_runs_and_returns_result(self):
        # Zero Hamiltonian keeps the test deterministic and very fast.
        hamiltonian = np.zeros((2, 2), dtype=np.complex64)
        config = CRLQASConfig(
            max_episodes=4,
            n_act=2,
            batch_size=2,
            replay_capacity=32,
            train_interval=1,
            target_update_interval=5,
            log_interval=0,
            adam_spsa=AdamSPSAConfig(iterations=2),
            seed=7,
        )

        result = train_crlqas(hamiltonian=hamiltonian, config=config)

        self.assertEqual(result.circuit.n_qubits, 1)
        self.assertTrue(np.isfinite(result.minimum_energy))
        self.assertAlmostEqual(result.minimum_energy, 0.0, places=5)
        self.assertEqual(len(result.episode_best_energies), config.max_episodes)

    def test_crlqas_invalid_hamiltonian_dimension_raises(self):
        bad = np.zeros((3, 3), dtype=np.complex64)
        config = CRLQASConfig(max_episodes=1, n_act=1, seed=1)

        with self.assertRaises(ValueError):
            _ = crlqas(hamiltonian=bad, config=config)


if __name__ == "__main__":
    unittest.main()

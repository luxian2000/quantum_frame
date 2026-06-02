import unittest

import numpy as np

from aicir.encoder.basis import BasisEncoder


class TestBasisEncoder(unittest.TestCase):
    def test_redundant_false_deduplicates_before_encoding(self):
        encoder = BasisEncoder(redundant=False)
        _, state = encoder.encode([1, 2, 2, 2])

        probs = np.abs(state.to_numpy()) ** 2

        self.assertEqual(len(probs), 2)
        self.assertTrue(np.allclose(np.sort(probs), np.array([0.5, 0.5]), atol=1e-6))

    def test_redundant_true_preserves_frequency_weights(self):
        encoder = BasisEncoder(redundant=True)
        _, state = encoder.encode([1, 2, 2, 2])

        probs = np.abs(state.to_numpy()) ** 2

        self.assertEqual(len(probs), 2)
        self.assertTrue(np.allclose(np.sort(probs), np.array([0.25, 0.75]), atol=1e-6))
        self.assertAlmostEqual(float(np.max(probs) / np.min(probs)), 3.0, places=6)

    def test_scaling_keeps_all_unique_values(self):
        encoder = BasisEncoder(redundant=False)
        _, state = encoder.encode([1, 1.1, 5.1, 4, 3])

        probs = np.abs(state.to_numpy()) ** 2
        non_zero = probs[probs > 1e-12]

        self.assertGreaterEqual(int(state.n_qubits), 3)
        self.assertEqual(len(non_zero), 5)
        self.assertTrue(np.allclose(non_zero, np.full(5, 0.2), atol=1e-6))

    def test_larger_n_qubits_keeps_extra_high_bits_zero(self):
        encoder = BasisEncoder(n_qubits=6, redundant=False)
        _, state = encoder.encode([1, 1.1, 5.1, 4, 3])

        probs = np.abs(state.to_numpy()) ** 2
        occupied = np.where(probs > 1e-12)[0]

        self.assertEqual(int(state.n_qubits), 6)
        self.assertTrue(np.all(occupied < 32))
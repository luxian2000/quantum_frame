import unittest

import numpy as np

from Circuit import Circuit, cnot, hadamard
from quantum_sim import ExecutionEngine, TorchBackend
from quantum_sim.core.operators import Hamiltonian


class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.backend = TorchBackend(device="cpu")
        self.engine = ExecutionEngine(self.backend)
        self.bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    def test_run_state_vector_probabilities_and_counts(self):
        result = self.engine.run(self.bell, shots=2000)

        self.assertEqual(result.n_qubits, 2)
        self.assertIsNotNone(result.counts)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)
        self.assertAlmostEqual(float(np.sum(result.probabilities)), 1.0, places=6)

    def test_run_state_vector_expectation_and_variance(self):
        # Bell 态对 ZZ 的期望值是 +1，方差是 0
        h = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        result = self.engine.run(self.bell, shots=None, observables={"ZZ": op})

        self.assertIn("ZZ", result.expectation_values)
        self.assertIn("ZZ", result.expectation_variances)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)
        self.assertAlmostEqual(result.expectation_variances["ZZ"], 0.0, places=5)
        self.assertAlmostEqual(result.stddev("ZZ"), 0.0, places=5)

    def test_run_density_matrix_path(self):
        h = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        result = self.engine.run_density_matrix(self.bell, shots=1500, observables={"ZZ": op})

        self.assertEqual(result.metadata.get("state_mode"), "density_matrix")
        self.assertIsNotNone(result.counts)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)
        self.assertAlmostEqual(result.expectation_variances["ZZ"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

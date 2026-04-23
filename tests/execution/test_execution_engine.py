import unittest

import numpy as np
import torch

from quantum_sim import Circuit, Measure, TorchBackend, cnot, hadamard, ry
from quantum_sim.core.operators import Hamiltonian


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.backend = TorchBackend(device="cpu")
        self.engine = Measure(self.backend)
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

    def test_run_batch_multi_circuits(self):
        # |00> --H0--> (|00>+|10>)/sqrt2, 对 ZZ 的期望值应为 0
        c1 = Circuit(hadamard(0), n_qubits=2)
        # Bell 态，对 ZZ 的期望值应为 +1
        c2 = self.bell

        h = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        results = self.engine.run_batch(
            [c1, c2],
            shots=1200,
            observables={"ZZ": op},
            per_circuit_options=[
                {"label": "single_h"},
                {"label": "bell"},
            ],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metadata.get("batch_index"), 0)
        self.assertEqual(results[1].metadata.get("batch_index"), 1)
        self.assertEqual(results[0].metadata.get("label"), "single_h")
        self.assertEqual(results[1].metadata.get("label"), "bell")
        self.assertAlmostEqual(results[0].expectation_values["ZZ"], 0.0, places=5)
        self.assertAlmostEqual(results[1].expectation_values["ZZ"], 1.0, places=5)

    def test_scan_parameters(self):
        # 扫描 theta: 电路 = RY(theta) on q0
        # 对 Z0 的期望值理论上为 cos(theta)
        h = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0]})
        op = h.to_matrix(self.backend)

        params = [0.0, np.pi / 2, np.pi]

        def build(theta):
            return Circuit(ry(torch.tensor(theta, dtype=torch.float64), 0), n_qubits=2)

        results = self.engine.scan_parameters(
            build,
            params,
            shots=None,
            observables={"Z0": op},
            return_state=False,
        )

        self.assertEqual(len(results), len(params))
        for i, theta in enumerate(params):
            self.assertEqual(results[i].metadata.get("scan_index"), i)
            self.assertAlmostEqual(float(results[i].metadata.get("scan_param")), theta, places=12)
            self.assertAlmostEqual(results[i].expectation_values["Z0"], float(np.cos(theta)), places=5)


if __name__ == "__main__":
    unittest.main()

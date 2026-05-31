import unittest

import numpy as np
import torch

from nexq import Circuit, Measure, TorchBackend, cnot, hadamard, ry
from nexq.channel.backends import NumpyBackend
from nexq.channel.operators import Hamiltonian
from nexq.core.circuit import crx, swap, toffoli
from nexq.core.gates import apply_gate_to_state, gate_to_matrix


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.backend = TorchBackend(device="cpu")
        self.measure = Measure(self.backend)
        self.bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    def test_run_state_vector_probabilities_and_counts(self):
        result = self.measure.run(self.bell, shots=2000)

        self.assertEqual(result.n_qubits, 2)
        self.assertIsNotNone(result.counts)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)
        self.assertAlmostEqual(float(np.sum(result.probabilities)), 1.0, places=6)

    def test_run_state_vector_expectation_and_variance(self):
        h = Hamiltonian(n_qubits=2).term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        result = self.measure.run(self.bell, shots=None, observables={"ZZ": op})

        self.assertIn("ZZ", result.expectation_values)
        self.assertIn("ZZ", result.expectation_variances)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)
        self.assertAlmostEqual(result.expectation_variances["ZZ"], 0.0, places=5)
        self.assertAlmostEqual(result.stddev("ZZ"), 0.0, places=5)

    def test_run_density_matrix_path(self):
        h = Hamiltonian(n_qubits=2).term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        result = self.measure.run_density_matrix(self.bell, shots=1500, observables={"ZZ": op})

        self.assertEqual(result.metadata.get("state_mode"), "density_matrix")
        self.assertIsNotNone(result.counts)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)
        self.assertAlmostEqual(result.expectation_variances["ZZ"], 0.0, places=5)

    def test_run_batch_multi_circuits(self):
        c1 = Circuit(hadamard(0), n_qubits=2)
        c2 = self.bell

        h = Hamiltonian(n_qubits=2).term(1.0, {"Z": [0, 1]})
        op = h.to_matrix(self.backend)

        results = self.measure.run_batch(
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
        h = Hamiltonian(n_qubits=2).term(1.0, {"Z": [0]})
        op = h.to_matrix(self.backend)

        params = [0.0, np.pi / 2, np.pi]

        def build(theta):
            return Circuit(ry(torch.tensor(theta, dtype=torch.float64), 0), n_qubits=2)

        results = self.measure.scan_parameters(
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

    def test_run_prefers_circuit_bound_backend(self):
        np_backend = NumpyBackend()
        circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=np_backend)

        result = self.measure.run(circ, shots=200)

        self.assertTrue(result.backend_name.startswith("NumpyBackend("))
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)

    def test_run_gatewise_path_does_not_require_unitary(self):
        class GateOnlyCircuit:
            def __init__(self):
                self.n_qubits = 2
                self.gates = [hadamard(0), cnot(1, [0])]

            def unitary(self, backend=None):
                raise AssertionError("gatewise path should not call unitary()")

        result = self.measure.run(GateOnlyCircuit(), shots=None)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)

    def test_local_gate_application_matches_full_matrix_path(self):
        rng = np.random.default_rng(7)
        gates = [
            hadamard(0),
            ry(0.3, 2),
            cnot(0, [2], [0]),
            crx(0.2, 2, [1]),
            swap(0, 2),
            toffoli(2, [0, 1]),
        ]

        for backend in (NumpyBackend(), TorchBackend(device="cpu")):
            for gate in gates:
                n_qubits = 3
                state_np = rng.normal(size=(1 << n_qubits, 1)) + 1j * rng.normal(size=(1 << n_qubits, 1))
                state_np = (state_np / np.linalg.norm(state_np)).astype(np.complex64)
                state = backend.cast(state_np)

                direct = apply_gate_to_state(gate, state, n_qubits, backend)
                full = backend.apply_unitary(state, gate_to_matrix(gate, n_qubits, backend=backend))

                self.assertTrue(
                    np.allclose(backend.to_numpy(direct), backend.to_numpy(full), atol=1e-5),
                    msg=f"local gate mismatch for {backend.name}: {gate}",
                )


if __name__ == "__main__":
    unittest.main()

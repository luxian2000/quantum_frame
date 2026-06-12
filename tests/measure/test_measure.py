import unittest

import numpy as np
import torch

from aicir import Circuit, Measure, TorchBackend, cnot, hadamard, measure, pauli_x, reset, ry
from aicir.channel.backends import NumpyBackend
from aicir.channel.operators import Hamiltonian
from aicir.core.circuit import crx, rxx, swap, toffoli
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.core.state import State
from aicir.measure.result import Result


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
        h = Hamiltonian(n_qubits=2, terms=[("ZZ", 1.0)])
        op = h.to_matrix(self.backend)

        result = self.measure.run(self.bell, shots=None, observables={"ZZ": op})

        self.assertIn("ZZ", result.expectation_values)
        self.assertIn("ZZ", result.expectation_variances)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)
        self.assertAlmostEqual(result.expectation_variances["ZZ"], 0.0, places=5)
        self.assertAlmostEqual(result.stddev("ZZ"), 0.0, places=5)

    def test_run_density_matrix_path(self):
        h = Hamiltonian(n_qubits=2, terms=[("ZZ", 1.0)])
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

        h = Hamiltonian(n_qubits=2, terms=[("ZZ", 1.0)])
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

    def test_run_batch_uses_distributed_backend_partition_and_gather(self):
        class PartitionBackend(NumpyBackend):
            def __init__(self):
                super().__init__()
                self.local_indices = []

            def should_run_batch_index(self, index):
                return index % 2 == 0

            def gather_indexed_results(self, indexed_results):
                self.local_indices = [idx for idx, _ in indexed_results]
                remote_result = Result(
                    n_qubits=1,
                    backend_name="remote",
                    probabilities=np.array([1.0, 0.0]),
                    metadata={"batch_index": 1, "label": "remote"},
                )
                return sorted(indexed_results + [(1, remote_result)], key=lambda item: item[0])

        backend = PartitionBackend()
        measure = Measure(backend)
        circuits = [
            Circuit(hadamard(0), n_qubits=1),
            Circuit(ry(0.1, 0), n_qubits=1),
            Circuit(ry(0.2, 0), n_qubits=1),
        ]

        results = measure.run_batch(circuits, shots=None, per_circuit_options=[
            {"label": "local_0"},
            {"label": "skipped_remote"},
            {"label": "local_2"},
        ])

        self.assertEqual(backend.local_indices, [0, 2])
        self.assertEqual([r.metadata["batch_index"] for r in results], [0, 1, 2])
        self.assertEqual(results[1].metadata["label"], "remote")

    def test_scan_parameters(self):
        h = Hamiltonian(n_qubits=2, terms=[("ZI", 1.0)])
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
            rxx(0.4, 0, 2),
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

    def test_local_gate_application_reuses_cached_backend_matrix(self):
        backend = NumpyBackend()
        state = backend.zeros_state(1)
        gate = hadamard(0)

        with unittest.mock.patch.object(backend, "cast", wraps=backend.cast) as cast:
            state = apply_gate_to_state(gate, state, 1, backend)
            state = apply_gate_to_state(gate, state, 1, backend)

        self.assertEqual(cast.call_count, 1)
        np.testing.assert_allclose(
            backend.to_numpy(state),
            np.array([[1.0 + 0j], [0.0 + 0j]], dtype=np.complex64),
            atol=1e-6,
        )


    def test_explicit_measure_qubits_reads_subset(self):
        # Approach 1: tell run() which qubits to read out, no in-circuit gates.
        result = self.measure.run(self.bell, shots=2000, measure_qubits=[0])

        self.assertEqual(result.metadata["measured_qubits"], [0])
        self.assertTrue(all(len(b.strip("|>")) == 1 for b in result.counts))

    def test_explicit_measure_qubits_default_reports_none(self):
        result = self.measure.run(self.bell, shots=None, measure_qubits=[0, 1])

        # Selecting every qubit is equivalent to the plain default readout.
        self.assertEqual(result.metadata["measured_qubits"], [0, 1])

    def test_measure_qubits_validates_indices(self):
        with self.assertRaises(ValueError):
            self.measure.run(self.bell, shots=10, measure_qubits=[5])

    def test_approaches_are_mutually_exclusive(self):
        # Approach 2: circuit carries a measure() gate.
        circ = Circuit(hadamard(0), cnot(1, [0]), measure(0), n_qubits=2)

        # Combining it with the Approach 1 measure_qubits override is rejected.
        with self.assertRaises(ValueError):
            self.measure.run(circ, shots=100, measure_qubits=[1])
        with self.assertRaises(ValueError):
            self.measure.run_density_matrix(circ, shots=100, measure_qubits=[1])

    def test_in_circuit_measure_still_works_without_override(self):
        circ = Circuit(hadamard(0), cnot(1, [0]), measure(0), n_qubits=2)
        result = self.measure.run(circ, shots=1000)

        self.assertEqual(result.metadata["measured_qubits"], [0])

    def test_reset_sets_measured_qubit_to_zero(self):
        circ = Circuit(pauli_x(0), measure(0), reset(0), n_qubits=1)

        result = self.measure.run(circ, shots=None)

        np.testing.assert_allclose(result.state.reshape(-1), [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.probabilities, [1.0, 0.0], atol=1e-6)

    def test_reset_density_matrix_sets_measured_qubit_to_zero(self):
        circ = Circuit(pauli_x(0), measure(0), reset(0), n_qubits=1)

        result = self.measure.run_density_matrix(circ, shots=None)

        np.testing.assert_allclose(
            result.state.reshape(2, 2),
            np.array([[1.0 + 0.0j, 0.0], [0.0, 0.0]], dtype=np.complex64),
            atol=1e-6,
        )
        np.testing.assert_allclose(result.probabilities, [1.0, 0.0], atol=1e-6)

    def test_reset_requires_previous_measure_on_same_qubit(self):
        with self.assertRaises(ValueError):
            self.measure.run(Circuit(reset(0), n_qubits=1), shots=None)

    def test_reset_rejects_gate_between_measure_and_reset(self):
        circ = Circuit(measure(0), hadamard(0), reset(0), n_qubits=1)

        with self.assertRaises(ValueError):
            self.measure.run(circ, shots=None)

    def test_build_initial_state_rejects_density_form(self):
        """密度矩阵形态 State 传入态矢路径应抛出 TypeError。"""
        backend = NumpyBackend()
        m = Measure(backend)
        density_state = State.zero_state(1, backend).to_density_matrix()
        with self.assertRaises(TypeError) as ctx:
            m._build_initial_state(1, backend, initial_state=density_state)
        self.assertIn("密度矩阵形态", str(ctx.exception))

    def test_build_initial_density_matrix_rejects_vector_form(self):
        """向量形态 State 传入密度矩阵路径应抛出 TypeError。"""
        backend = NumpyBackend()
        m = Measure(backend)
        vector_state = State.zero_state(1, backend)
        with self.assertRaises(TypeError) as ctx:
            m._build_initial_density_matrix(1, backend, initial_density_matrix=vector_state)
        self.assertIn("向量形态", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

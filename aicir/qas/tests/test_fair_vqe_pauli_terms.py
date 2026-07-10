"""Regression tests for fair VQE Pauli-term energy evaluation."""

from __future__ import annotations

from io import StringIO
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aicir.qas.core._types import ArchitectureSpec
from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene, architecture_from_operator_sequence_gene
from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.fair_vqe import (
    CPU_PAULI_EXPECTATION_MIN_TERMS,
    _evaluate_state_pauli_energy,
    _numpy_pauli_expectation,
    evaluate_vqe_energy,
    optimize_vqe_energy,
)


class FairVqePauliTermsTest(unittest.TestCase):
    def test_operator_sequence_xy_basis_changes_run_in_shared_fair_vqe(self):
        architecture = architecture_from_operator_sequence_gene(
            OperatorSequenceAnsatzGene(n_qubits=2, operators=("XY",))
        )
        problem = VQEProblem(
            name="xy_operator_2q",
            n_qubits=2,
            hamiltonian=((1.0, "ZI"),),
            reference_energy=-1.0,
        )

        energy = evaluate_vqe_energy(architecture, problem, parameters=[0.0])

        self.assertAlmostEqual(energy, 1.0, places=6)

    def test_evaluate_pauli_problem_does_not_build_dense_hamiltonian(self):
        n_qubits = 9
        architecture = ArchitectureSpec.from_gates("empty_9q", [], n_qubits=n_qubits)
        problem = VQEProblem(
            name="z0_9q",
            n_qubits=n_qubits,
            hamiltonian=((1.0, "Z" + "I" * (n_qubits - 1)),),
            reference_energy=-1.0,
        )

        with patch(
            "aicir.qas.vqe_loop.fair_vqe.hamiltonian_matrix",
            side_effect=AssertionError("dense Hamiltonian construction should not be used"),
        ):
            energy = evaluate_vqe_energy(architecture, problem)
            result = optimize_vqe_energy(
                architecture,
                problem,
                seed=7,
                n_starts=1,
                max_evaluations=1,
                budget_override=1,
            )

        self.assertAlmostEqual(energy, 1.0, places=6)
        self.assertAlmostEqual(result.energy, 1.0, places=6)
        self.assertEqual(result.evaluations, 1)

    def test_optimizer_always_keeps_zero_parameter_candidate(self):
        architecture = ArchitectureSpec.from_gates(
            "rx_1q",
            [{"type": "rx", "target_qubit": 0, "parameter": 0.123}],
            n_qubits=1,
        )
        problem = VQEProblem(name="z0_1q", n_qubits=1, hamiltonian=((1.0, "Z"),), reference_energy=-1.0)

        class BadCoblya:
            def __init__(self, *args, **kwargs):
                pass

            def minimize(self, _objective, start):
                return SimpleNamespace(fun=-24.0, x=start, best_fun=-24.0, best_x=start, nfev=1)

        def fake_energy(_architecture, _problem, parameters, *, backend):
            return -53.0 if np.allclose(parameters, [0.0]) else -24.0

        with patch("aicir.qas.vqe_loop.fair_vqe.COBYLA", BadCoblya), patch(
            "aicir.qas.vqe_loop.fair_vqe._evaluate_pauli_state_energy",
            side_effect=fake_energy,
        ):
            result = optimize_vqe_energy(
                architecture,
                problem,
                seed=7,
                n_starts=1,
                max_evaluations=1,
                budget_override=1,
                init_mode="random_uniform_pi",
            )

        self.assertAlmostEqual(result.energy, -53.0)
        self.assertEqual(result.best_parameters, [0.0])
        self.assertTrue(result.metadata["zero_parameter_candidate"]["included"])

    def test_numpy_pauli_expectation_rejects_backend_tensor_like_input(self):
        class BackendTensorLike:
            device = "npu:0"

            def detach(self):
                return self

            def cpu(self):
                return self

            def __array__(self, dtype=None):
                raise AssertionError("implicit NumPy conversion should not be attempted")

        with self.assertRaisesRegex(TypeError, "backend tensor"):
            _numpy_pauli_expectation(BackendTensorLike(), [(0, 0, 0, 1.0, 0.0)])


    def test_large_npu_pauli_expectation_explicitly_copies_state_to_cpu(self):
        class BackendTensorLike:
            device = "npu:0"

            def __init__(self, array):
                self.array = np.asarray(array, dtype=np.complex64)

            def detach(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *_shape):
                return self

            def __array__(self, dtype=None):
                raise AssertionError("implicit NumPy conversion should not be attempted")

        class FakeNpuBackend:
            _device = "npu:0"

            def __init__(self):
                self.to_numpy_calls = 0

            def to_numpy(self, value):
                self.to_numpy_calls += 1
                return value.array

        backend = FakeNpuBackend()
        state = BackendTensorLike([1.0 + 0.0j, 0.0 + 0.0j])
        pauli_cache = [(0, 0, 0, 1.0, 0.0)] * CPU_PAULI_EXPECTATION_MIN_TERMS

        energy = _evaluate_state_pauli_energy(state, pauli_cache, backend=backend)

        self.assertEqual(backend.to_numpy_calls, 1)
        self.assertAlmostEqual(energy, float(CPU_PAULI_EXPECTATION_MIN_TERMS))


    def test_numpy_pauli_expectation_traces_progress(self):
        state = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        pauli_cache = [(0, 0, 0, 1.0, 0.0)] * 3
        stream = StringIO()

        with patch("sys.stdout", stream), patch("aicir.qas.vqe_loop.fair_vqe._fair_trace_enabled", return_value=True):
            energy = _numpy_pauli_expectation(state, pauli_cache)

        self.assertAlmostEqual(energy, 3.0)
        output = stream.getvalue()
        self.assertIn("stage=numpy_pauli_expectation_begin", output)
        self.assertIn("stage=numpy_pauli_expectation_progress", output)
        self.assertIn("term_index=3", output)
        self.assertIn("stage=numpy_pauli_expectation_end", output)

    def test_numpy_pauli_expectation_error_trace_identifies_term(self):
        state = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        pauli_cache = [(999, 0, 0, 1.0, 0.0)]
        stream = StringIO()

        with patch("sys.stdout", stream), patch("aicir.qas.vqe_loop.fair_vqe._fair_trace_enabled", return_value=True):
            with self.assertRaises(IndexError):
                _numpy_pauli_expectation(state, pauli_cache)

        output = stream.getvalue()
        self.assertIn("stage=numpy_pauli_expectation_term_error", output)
        self.assertIn("term_index=1", output)
        self.assertIn("flip_mask=999", output)

    def test_fair_vqe_trace_prints_gate_and_expectation_stages(self):
        architecture = ArchitectureSpec.from_gates("x_1q", [{"type": "pauli_x", "qubits": [0]}], n_qubits=1)
        problem = VQEProblem(
            name="z0_1q",
            n_qubits=1,
            hamiltonian=((1.0, "Z"),),
            reference_energy=-1.0,
        )
        stream = StringIO()

        with patch("sys.stdout", stream), patch("aicir.qas.vqe_loop.fair_vqe._fair_trace_enabled", return_value=True):
            evaluate_vqe_energy(architecture, problem)

        output = stream.getvalue()
        self.assertIn("stage=simulate_gate_begin", output)
        self.assertIn("gate_index=0", output)
        self.assertIn("gate_type=pauli_x", output)
        self.assertIn("stage=simulate_gate_end", output)
        self.assertIn("stage=pauli_expectation_begin", output)

if __name__ == "__main__":
    unittest.main()

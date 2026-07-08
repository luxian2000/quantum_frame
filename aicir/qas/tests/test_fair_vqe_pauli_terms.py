"""Regression tests for fair VQE Pauli-term energy evaluation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from aicir.qas.core._types import ArchitectureSpec
from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.fair_vqe import (
    CPU_PAULI_EXPECTATION_MIN_TERMS,
    _evaluate_state_pauli_energy,
    _numpy_pauli_expectation,
    evaluate_vqe_energy,
    optimize_vqe_energy,
)


class FairVqePauliTermsTest(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()

"""Regression tests for fair VQE Pauli-term energy evaluation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from aicir.qas.core._types import ArchitectureSpec
from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy, optimize_vqe_energy


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


if __name__ == "__main__":
    unittest.main()

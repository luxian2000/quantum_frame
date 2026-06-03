import math
import unittest

from aicir.qas import (
    HEAMask,
    architecture_from_hea_mask,
    evaluate_h2_energy,
    exact_ground_energy,
    ising4_demo_problem,
    mutate_hea_mask,
    run_ising4_trainability_prior_demo,
    run_vqe_hea_demo,
    run_vqe_ising4_demo,
)
from aicir.qas.task_evaluation import parameter_count


class TestVQEHEADemo(unittest.TestCase):
    def test_mutate_hea_mask_changes_exactly_one_dimension(self):
        import numpy as np

        mask = HEAMask(
            n_qubits=2,
            layers=1,
            rotation_block="ry",
            entangler="cx",
            final_rotation="ry",
            entangle_pattern="linear",
        )
        mutated = mutate_hea_mask(mask, np.random.default_rng(7))

        self.assertEqual(mutated.n_qubits, mask.n_qubits)
        changed = sum(left != right for left, right in zip(mask.key()[1:], mutated.key()[1:]))
        self.assertEqual(changed, 1)

    def test_h2_energy_evaluates_hea_architecture(self):
        architecture = architecture_from_hea_mask(HEAMask(n_qubits=2, layers=1))
        n_params = parameter_count(architecture.circuit)
        energy = evaluate_h2_energy(architecture, [0.0] * n_params)

        self.assertTrue(math.isfinite(energy))

    def test_vqe_hea_demo_runs_small_pipeline(self):
        report = run_vqe_hea_demo(candidate_limit=8, stage1_keep_top=4, sa_steps=2, seed=7)

        self.assertTrue(report.stage1_rows)
        self.assertTrue(any(row.kept for row in report.stage1_rows))
        self.assertTrue(any(not row.kept for row in report.stage1_rows))
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.final_results)
        self.assertIn("metric | min | p25 | max", "\n".join(report.summary_lines()))
        self.assertIn("Final VQE validation", "\n".join(report.summary_lines()))

    def test_ising4_problem_has_exact_reference(self):
        problem = ising4_demo_problem()

        self.assertEqual(problem.n_qubits, 4)
        self.assertAlmostEqual(problem.reference_energy, exact_ground_energy(problem.hamiltonian), places=10)

    def test_vqe_ising4_demo_runs_small_pipeline(self):
        report = run_vqe_ising4_demo(candidate_limit=8, stage1_keep_top=4, sa_steps=2, seed=11)

        self.assertIn("tfim_chain_4q", "\n".join(report.summary_lines()))
        self.assertTrue(report.stage1_rows)
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.final_results)

    def test_ising4_trainability_prior_demo_runs(self):
        report = run_ising4_trainability_prior_demo(
            seed=29,
            candidate_limit=8,
            stage1_keep_top=6,
            trainability_top_k=3,
            sa_steps=2,
            search_max_evaluations=6,
            final_n_starts=1,
            fair_evals_per_param=2,
            fair_min_evaluations=4,
        )
        summary = "\n".join(report.summary_lines())

        self.assertTrue(report.trainability_top_results)
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.baseline_results)
        self.assertIn("Trainability top fair final", summary)
        self.assertIn("Diagnostic: SA final vs baselines", summary)


if __name__ == "__main__":
    unittest.main()

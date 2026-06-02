import math
import unittest

from nexq.qas import (
    HEAMask,
    architecture_from_hea_mask,
    evaluate_h2_energy,
    exact_ground_energy,
    ising4_demo_problem,
    mutate_hea_mask,
    run_ising4_budget_sweep,
    run_ising4_fitness_correlation,
    run_ising4_multistart_sa,
    run_vqe_hea_demo,
    run_vqe_ising4_demo,
)
from nexq.qas.task_evaluation import parameter_count


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

    def test_ising4_budget_sweep_reports_capped_and_fair_validation(self):
        report = run_ising4_budget_sweep(
            seed=13,
            steps=(2, 3),
            candidate_limit=8,
            stage1_keep_top=4,
            search_max_evaluations=6,
            final_n_starts=1,
            capped_max_evaluations=8,
            fair_evals_per_param=2,
            fair_min_evaluations=4,
        )
        summary = "\n".join(report.summary_lines())

        self.assertEqual(len(report.sweep_rows), 2)
        self.assertTrue(report.capped_results)
        self.assertTrue(report.fair_results)
        self.assertIn("SA budget sweep", summary)
        self.assertIn("Final validation: per-param fair budget", summary)

    def test_ising4_multistart_sa_uses_diverse_seeds(self):
        report = run_ising4_multistart_sa(
            seed=17,
            n_starts=3,
            steps_per_start=2,
            candidate_limit=12,
            stage1_keep_top=8,
            search_max_evaluations=6,
            final_n_starts=1,
            capped_max_evaluations=8,
            fair_evals_per_param=2,
            fair_min_evaluations=4,
        )
        summary = "\n".join(report.summary_lines())
        start_keys = {row.start_mask.key() for row in report.restart_rows}

        self.assertGreaterEqual(len(start_keys), 2)
        self.assertTrue(report.capped_results)
        self.assertTrue(report.fair_results)
        self.assertIn("Multi-start SA", summary)

    def test_ising4_fitness_correlation_reports_spearman(self):
        report = run_ising4_fitness_correlation(
            seed=19,
            top_k=4,
            candidate_limit=8,
            stage1_keep_top=6,
            short_max_evaluations=6,
            fair_n_starts=1,
            fair_evals_per_param=2,
            fair_min_evaluations=4,
        )
        summary = "\n".join(report.summary_lines())
        correlations = report.correlations()

        self.assertEqual(len(report.rows), 4)
        self.assertIn("spearman_short_vs_fair", correlations)
        self.assertIn("top4_overlap", correlations)
        self.assertIn("VQE-QAS short/fair fitness validation", summary)


if __name__ == "__main__":
    unittest.main()

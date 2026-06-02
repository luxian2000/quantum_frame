import math
import unittest

from nexq.qas import (
    HEAMask,
    architecture_from_hea_mask,
    evaluate_h2_energy,
    mutate_hea_mask,
    run_vqe_hea_demo,
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
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.final_results)
        self.assertIn("Final VQE validation", "\n".join(report.summary_lines()))


if __name__ == "__main__":
    unittest.main()

import unittest

from nexq.algorithms.qas import ArchitectureSpec, evaluate_architectures
from nexq.channel.backends.numpy_backend import NumpyBackend


class TestArchitectureEvaluation(unittest.TestCase):
    def setUp(self):
        self.backend = NumpyBackend()

    def test_architecture_spec_summarizes_candidate(self):
        architecture = ArchitectureSpec.from_gates(
            name="single_rotation",
            gates=[{"type": "rx", "target_qubit": 0, "parameter": 0.25}],
            n_qubits=1,
            backend=self.backend,
        )

        self.assertEqual(architecture.name, "single_rotation")
        self.assertEqual(architecture.n_qubits, 1)
        self.assertEqual(architecture.n_gates, 1)
        self.assertEqual(architecture.parameter_count, 1)
        self.assertEqual(architecture.two_qubit_gate_count, 0)

    def test_evaluate_architectures_lists_active_and_todo_metrics(self):
        architecture = ArchitectureSpec.from_gates(
            name="bell_like",
            gates=[
                {"type": "hadamard", "target_qubit": 0},
                {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
            ],
            n_qubits=2,
            backend=self.backend,
        )

        [score] = evaluate_architectures([architecture], backend=self.backend, n_samples=4)

        self.assertEqual(score.rank, 1)
        self.assertGreaterEqual(score.weighted_score, 0.0)
        self.assertLessEqual(score.weighted_score, 1.0)
        self.assertEqual(score.expressibility.active_metric, "kl_haar")

        expressibility_metrics = {metric.name: metric for metric in score.expressibility.metrics}
        self.assertTrue(expressibility_metrics["kl_haar"].active)
        self.assertEqual(expressibility_metrics["kl_haar"].status, "implemented")
        self.assertFalse(expressibility_metrics["frame_potential"].active)
        self.assertEqual(expressibility_metrics["frame_potential"].status, "todo")

        self.assertEqual(score.trainability.active_metric, "structure_proxy")
        self.assertEqual(score.noise_robustness.active_metric, "ion_trap_error_budget_proxy")
        self.assertIn("total_error_budget", score.noise_robustness.raw_values)
        self.assertEqual(score.hardware_efficiency.active_metric, "native_depth_twoq_efficiency")

        row = score.to_row()
        self.assertEqual(row["name"], "bell_like")
        self.assertIn("weighted_score", row)


if __name__ == "__main__":
    unittest.main()
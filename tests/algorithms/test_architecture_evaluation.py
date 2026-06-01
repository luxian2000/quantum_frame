import unittest

from nexq.qas import ArchitectureSpec, HardwareProfile, evaluate_architectures
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.metrics.trainability import local_probe_gradient_statistics


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

    def test_zero_cost_gradient_trainability_metric(self):
        trainable = ArchitectureSpec.from_gates(
            name="ry_probe",
            gates=[{"type": "ry", "target_qubit": 0, "parameter": 0.0}],
            n_qubits=1,
            backend=self.backend,
        )
        flat = ArchitectureSpec.from_gates(
            name="h_only",
            gates=[{"type": "hadamard", "target_qubit": 0}],
            n_qubits=1,
            backend=self.backend,
        )

        stats = local_probe_gradient_statistics(trainable.circuit, samples=4, seed=7)
        self.assertGreater(stats["mean_gradient_norm"], 0.0)

        trainable_score, flat_score = evaluate_architectures(
            [trainable, flat],
            backend=self.backend,
            n_samples=4,
            active_metrics={"trainability": "gradient_norm"},
        )

        self.assertEqual(trainable_score.trainability.active_metric, "gradient_norm")
        self.assertGreater(trainable_score.trainability.score, flat_score.trainability.score)
        self.assertIn("mean_gradient_norm", trainable_score.trainability.raw_values)

    def test_topology_mapping_hardware_metric(self):
        local = ArchitectureSpec.from_gates(
            name="local_cx",
            gates=[{"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]}],
            n_qubits=3,
            backend=self.backend,
        )
        nonlocal_arch = ArchitectureSpec.from_gates(
            name="nonlocal_cx",
            gates=[{"type": "cx", "target_qubit": 2, "control_qubits": [0], "control_states": [1]}],
            n_qubits=3,
            backend=self.backend,
        )
        profile = HardwareProfile(
            coupling_map=[(0, 1), (1, 2)],
            edge_fidelity={(0, 1): 0.99, (1, 2): 0.98},
            max_depth=8,
        )

        local_score, nonlocal_score = evaluate_architectures(
            [local, nonlocal_arch],
            backend=self.backend,
            hardware_profile=profile,
            active_metrics={"hardware_efficiency": "topology_mapping_efficiency"},
        )

        self.assertEqual(local_score.hardware_efficiency.active_metric, "topology_mapping_efficiency")
        self.assertGreater(local_score.hardware_efficiency.score, nonlocal_score.hardware_efficiency.score)
        self.assertEqual(nonlocal_score.hardware_efficiency.raw_values["connectivity_violation_count"], 1)
        self.assertEqual(local_score.hardware_efficiency.raw_values["mapping_fidelity_score"], 0.99)


if __name__ == "__main__":
    unittest.main()

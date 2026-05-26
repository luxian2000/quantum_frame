import unittest

from nexq.algorithms.qas import build_common_architectures, common_architecture_names, evaluate_architectures
from nexq.channel.backends.numpy_backend import NumpyBackend


class TestArchitectureCandidates(unittest.TestCase):
    def setUp(self):
        self.backend = NumpyBackend()

    def test_common_architecture_library_has_expected_presets(self):
        names = common_architecture_names()

        self.assertIn("hea_linear", names)
        self.assertIn("qaoa_chain", names)
        self.assertIn("strongly_entangling_crx", names)
        self.assertEqual(len(names), len(set(names)))

    def test_build_common_architectures_returns_specs(self):
        architectures = build_common_architectures(
            n_qubits=3,
            layers=1,
            backend=self.backend,
            names=["hea_linear", "real_amplitudes_linear", "qaoa_chain"],
        )

        self.assertEqual(len(architectures), 3)
        self.assertEqual([architecture.n_qubits for architecture in architectures], [3, 3, 3])
        self.assertTrue(all(architecture.n_gates > 0 for architecture in architectures))
        self.assertTrue(all(architecture.tags for architecture in architectures))

    def test_candidate_library_runs_through_architecture_evaluator(self):
        architectures = build_common_architectures(
            n_qubits=3,
            layers=1,
            backend=self.backend,
            names=["hea_linear", "qaoa_chain", "brickwork_cx"],
        )

        scores = evaluate_architectures(architectures, backend=self.backend, n_samples=4)

        self.assertEqual([score.rank for score in scores], [1, 2, 3])
        self.assertEqual(len({score.architecture.name for score in scores}), 3)
        self.assertTrue(all(0.0 <= score.weighted_score <= 1.0 for score in scores))
        self.assertTrue(all("total_error_budget" in score.noise_robustness.raw_values for score in scores))


if __name__ == "__main__":
    unittest.main()
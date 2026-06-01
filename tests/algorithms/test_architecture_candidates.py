import unittest

from nexq.qas import ArchitectureSearch, SearchConfig, build_common_architectures, common_architecture_names, evaluate_architectures
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

    def test_supercircuit_strategy_generates_subcircuits(self):
        search = ArchitectureSearch(backend=self.backend)
        result = search.run(
            SearchConfig(
                n_qubits=3,
                candidate_layers=2,
                n_samples=4,
                include_common_candidates=False,
                search_strategy="supercircuit",
                population_size=6,
                top_k=3,
            )
        )

        self.assertEqual(result.metadata["search_strategy"], "supercircuit")
        self.assertEqual(len(result.scores), 3)
        self.assertTrue(result.candidates)
        self.assertTrue(all("supercircuit_mask" in candidate.metadata for candidate in result.candidates))
        self.assertTrue(all(0.0 <= score.weighted_score <= 1.0 for score in result.scores))


if __name__ == "__main__":
    unittest.main()

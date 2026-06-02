import unittest

from nexq.qas import (
    ArchitectureSearch,
    OptimizerConfig,
    SearchConfig,
    maxcut_line,
    run_hybrid_qas_validation_experiment,
    run_multi_seed_validation_experiment,
    run_random_proxy_validation_experiment,
    run_search_strategy_comparison,
    run_task_feedback_validation_experiment,
    run_validation_experiment,
    small_resource_allocation,
)


class TestQASTaskValidation(unittest.TestCase):
    def test_maxcut_problem_has_bruteforce_reference(self):
        problem = maxcut_line(n_qubits=4)

        self.assertEqual(problem.n_qubits, 4)
        self.assertEqual(problem.classical_optimum, 3.0)
        self.assertEqual(problem.evaluate_bitstring("0101"), 3.0)

    def test_resource_allocation_problem_constructs(self):
        problem = small_resource_allocation()

        self.assertEqual(problem.n_qubits, 4)
        self.assertGreater(problem.classical_optimum, 0.0)

    def test_architecture_search_honors_resource_filters(self):
        search = ArchitectureSearch()
        result = search.run(
            SearchConfig(
                n_qubits=3,
                candidate_layers=1,
                n_samples=4,
                max_two_qubit_gates=2,
                top_k=2,
            )
        )

        self.assertLessEqual(len(result.scores), 2)
        self.assertTrue(all(score.architecture.two_qubit_gate_count <= 2 for score in result.scores))

    def test_validation_runner_compares_baselines_and_qas(self):
        report = run_validation_experiment(
            maxcut_line(n_qubits=3),
            search_config=SearchConfig(n_qubits=3, candidate_layers=1, n_samples=4, candidate_budget=4),
            optimizer_config=OptimizerConfig(max_evaluations=4, seed=7),
            qas_top_k=2,
        )

        self.assertEqual(len(report.baseline_results), 3)
        self.assertEqual(len(report.qas_results), 2)
        self.assertIsNotNone(report.best_result)
        self.assertIn("name | prior | optimized", "\n".join(report.summary_lines()))

    def test_multi_seed_validation_report_aggregates_runs(self):
        report = run_multi_seed_validation_experiment(
            maxcut_line(n_qubits=3),
            seeds=[7, 8],
            search_config=SearchConfig(n_qubits=3, candidate_layers=1, n_samples=4, candidate_budget=4),
            optimizer_config=OptimizerConfig(max_evaluations=4),
            qas_top_k=1,
        )

        summary = report.architecture_summary()
        self.assertEqual(report.n_seeds, 2)
        self.assertTrue(summary)
        self.assertIn("architecture_summary", report.to_dict())
        self.assertIn("win_rate", summary[0])
        self.assertIn("group | name | runs | mean", "\n".join(report.summary_lines()))
        self.assertTrue(all("result_group" in row for row in summary))

    def test_random_proxy_validation_reports_correlation(self):
        report = run_random_proxy_validation_experiment(
            maxcut_line(n_qubits=3),
            n_random_samples=5,
            optimizer_config=OptimizerConfig(max_evaluations=3, seed=19),
            random_seed=19,
        )

        self.assertTrue(report.rows)
        self.assertIn("pearson_best", report.correlations())
        self.assertIn("random_best", report.rows[0].to_dict())
        self.assertIn("short_optimized", report.rows[0].to_dict())
        self.assertIn("correlation | pearson_random_best_vs_short", "\n".join(report.summary_lines()))

    def test_task_feedback_validation_mutates_supercircuit_masks(self):
        report = run_task_feedback_validation_experiment(
            maxcut_line(n_qubits=3),
            search_config=SearchConfig(
                n_qubits=3,
                candidate_layers=1,
                n_samples=4,
                include_common_candidates=False,
                search_strategy="supercircuit_evolution",
                population_size=4,
                search_generations=1,
                beam_width=2,
                mutation_rate=0.5,
                top_k=3,
            ),
            optimizer_config=OptimizerConfig(max_evaluations=3, seed=11),
            qas_top_k=2,
            feedback_generations=2,
            feedback_population_size=4,
            feedback_elite_count=2,
        )

        self.assertEqual(len(report.baseline_results), 3)
        self.assertEqual(len(report.qas_results), 2)
        self.assertEqual(report.metadata["feedback_generations"], 2)
        self.assertGreaterEqual(report.metadata["task_feedback_evaluated"], 4)
        self.assertTrue(all(result.metadata["result_group"] == "qas_task_feedback" for result in report.qas_results))
        self.assertTrue(all("supercircuit_mask" in result.metadata for result in report.qas_results))

    def test_search_strategy_comparison_summarizes_strategies(self):
        report = run_search_strategy_comparison(
            maxcut_line(n_qubits=3),
            search_config=SearchConfig(
                n_qubits=3,
                candidate_layers=1,
                n_samples=4,
                include_common_candidates=True,
                population_size=4,
                search_generations=1,
                beam_width=2,
                mutation_rate=0.5,
                top_k=3,
            ),
            optimizer_config=OptimizerConfig(max_evaluations=3, seed=13),
            qas_top_k=1,
            strategies=(
                "supercircuit_progressive",
                "supercircuit_evolution",
                "supercircuit_reflective",
                "task_feedback",
                "hybrid",
            ),
            feedback_generations=1,
            feedback_population_size=3,
            feedback_elite_count=1,
        )

        rows = report.strategy_summary()
        self.assertEqual(
            set(report.reports),
            {"supercircuit_progressive", "supercircuit_evolution", "supercircuit_reflective", "task_feedback", "hybrid"},
        )
        self.assertEqual(len(rows), 5)
        self.assertIn("strategy | baseline_best | qas_best", "\n".join(report.summary_lines()))
        self.assertTrue(all("strategy" in row for row in rows))

    def test_hybrid_validation_chains_all_three_search_stages(self):
        report = run_hybrid_qas_validation_experiment(
            maxcut_line(n_qubits=3),
            search_config=SearchConfig(
                n_qubits=3,
                candidate_layers=1,
                n_samples=4,
                include_common_candidates=False,
                population_size=4,
                search_generations=1,
                beam_width=2,
                mutation_rate=0.5,
                top_k=3,
            ),
            optimizer_config=OptimizerConfig(max_evaluations=3, seed=17),
            qas_top_k=2,
            feedback_generations=1,
            feedback_population_size=3,
            feedback_elite_count=1,
        )

        self.assertEqual(report.metadata["hybrid_pipeline"], "progressive->reflective_evolution->task_feedback")
        self.assertEqual(len(report.baseline_results), 3)
        self.assertEqual(len(report.qas_results), 2)
        self.assertTrue(all(result.metadata["result_group"] == "qas_hybrid" for result in report.qas_results))


if __name__ == "__main__":
    unittest.main()

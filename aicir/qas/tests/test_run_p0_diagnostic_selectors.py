import csv
import builtins
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aicir.qas.demos import run_p0_diagnostic as runner


class SelectorOutputTests(unittest.TestCase):
    def test_selector_summary_reports_fair_quality_of_proxy_top_k(self):
        rows = [
            {"architecture_id": "fair_best", "E2": "1.0", "E5": "9.0", "fair_high": "-5.0"},
            {"architecture_id": "fair_second", "E2": "2.0", "E5": "1.0", "fair_high": "-4.0"},
            {"architecture_id": "proxy_best_bad", "E2": "0.5", "E5": "0.2", "fair_high": "-1.0"},
            {"architecture_id": "fair_third", "E2": "3.0", "E5": "2.0", "fair_high": "-3.0"},
        ]

        summary = runner.selector_level_summary(rows, "E2", top_k=2)

        self.assertEqual(summary["selected_architecture_ids"], ["proxy_best_bad", "fair_best"])
        self.assertEqual(summary["fair_best_in_topK"], -5.0)
        self.assertEqual(summary["fair_mean_in_topK"], -3.0)
        self.assertEqual(summary["fair_median_in_topK"], -3.0)
        self.assertEqual(summary["fair_topK_hit"], 1)
        self.assertEqual(summary["fair_topK_hit_rate"], 0.5)
        self.assertEqual(summary["fair_rank_of_proxy_best"], 4)

    def test_write_selector_outputs_respects_selector_choice(self):
        rows = [
            {"architecture_id": "a", "E2": "0.1", "E5": "9.0", "fair_high": "-4.0"},
            {"architecture_id": "b", "E2": "0.2", "E5": "0.1", "fair_high": "-5.0"},
            {"architecture_id": "c", "E2": "0.3", "E5": "0.2", "fair_high": "-3.0"},
        ]

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            comparison = runner.write_selector_outputs(rows, output_dir, selector="both", top_k=2)

            self.assertEqual(set(comparison["selectors"]), {"E2", "E5"})
            self.assertTrue((output_dir / "queue_e2_topK.csv").exists())
            self.assertTrue((output_dir / "queue_e5_topK.csv").exists())
            self.assertTrue((output_dir / "selector_comparison.json").exists())

            with (output_dir / "queue_e5_topK.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual([row["architecture_id"] for row in selected], ["b", "c"])

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            comparison = runner.write_selector_outputs(rows, output_dir, selector="e2", top_k=1)

            self.assertEqual(set(comparison["selectors"]), {"E2"})
            self.assertTrue((output_dir / "queue_e2_topK.csv").exists())
            self.assertFalse((output_dir / "queue_e5_topK.csv").exists())

    def test_default_pairs_fall_back_when_supernet_dependency_is_unavailable(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "aicir.qas.algorithms.supernet":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = blocked_import
            self.assertEqual(runner.default_two_qubit_pairs(4), ((0, 1), (1, 2), (2, 3)))
        finally:
            builtins.__import__ = original_import

    def test_problem_aware_sampler_mode_biases_entangler_pairs(self):
        sampler = runner.make_architecture_sampler(
            n_architectures=1,
            n_qubits=3,
            depth=3,
            pairs=((0, 1), (1, 2)),
            terms=((-1.0, "ZZI"),),
            reference_energy=-1.0,
            preset="toy",
            seed=7,
            sampling_mode="problem_aware_supernet_native",
            problem_aware_entangler_floor=0.0,
        )

        row = next(iter(sampler(None)))
        gene = runner.SupernetAnsatzGene.from_jsonable(row["ansatz_gene"])

        self.assertEqual(row["sampling_mode"], "problem_aware_supernet_native")
        self.assertTrue(all(layer[0] != "none" for layer in gene.two_qubit_layers))
        self.assertTrue(all(layer[1] == "none" for layer in gene.two_qubit_layers))

    def test_e5_selector_does_not_build_light_vqe_registry(self):
        originals = {
            "load_hamiltonian_terms": runner.load_hamiltonian_terms,
            "default_two_qubit_pairs": runner.default_two_qubit_pairs,
            "make_architecture_sampler": runner.make_architecture_sampler,
            "build_native_supernet_e5_evaluator": runner.build_native_supernet_e5_evaluator,
            "build_light_vqe_evaluator_registry": runner.build_light_vqe_evaluator_registry,
            "make_adaptive_fair_runner": runner.make_adaptive_fair_runner,
            "run_experiment": runner.run_experiment,
        }

        def fail_light_registry(*_args, **_kwargs):
            raise AssertionError("E5-only selector must not build E1/E2 light VQE evaluators")

        def fake_run_experiment(_config, output_csv, *, evaluator_registry, architecture_sampler, fair_vqe_runner):
            self.assertEqual(set(evaluator_registry), {"E5"})
            with Path(output_csv).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(runner.EXPERIMENT_FIELDS))
                writer.writeheader()
                writer.writerow(
                    {
                        "architecture_id": "arch_e5",
                        "E5": "-1.0",
                        "fair_high": "-1.2",
                        "evaluation_order_index": "0",
                    }
                )

        try:
            runner.load_hamiltonian_terms = lambda *_args, **_kwargs: (((-1.0, "ZI"),), 2, -1.0)
            runner.default_two_qubit_pairs = lambda _n_qubits: ((0, 1),)
            runner.make_architecture_sampler = lambda **_kwargs: (lambda _config: ())
            runner.build_native_supernet_e5_evaluator = lambda **_kwargs: (lambda _row: {"E5": -1.0})
            runner.build_light_vqe_evaluator_registry = fail_light_registry
            runner.make_adaptive_fair_runner = lambda **_kwargs: (lambda _row: {"fair_high": -1.2})
            runner.run_experiment = fake_run_experiment
            with tempfile.TemporaryDirectory() as temp:
                runner.main(["--selector", "e5", "--selector-top-k", "1", "--output-dir", temp])
        finally:
            for name, value in originals.items():
                setattr(runner, name, value)


if __name__ == "__main__":
    unittest.main()

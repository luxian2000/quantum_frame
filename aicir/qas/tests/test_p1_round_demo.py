import csv
import json
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.library.ansatz import SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS


def make_gene():
    return SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=(("ry", "rz"), ("rx", "ry")),
        two_qubit_layers=(("cx",), ("rzz",)),
        two_qubit_pairs=((0, 1),),
    )


def write_bootstrap_labels(path: Path) -> None:
    gene = make_gene()
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": "bootstrap_parent",
            "canonical_arch_hash": json.dumps(gene.to_jsonable(), sort_keys=True),
            "protocol_version": "fair_vqe_protocol_v2",
            "batch_id": "bootstrap",
            "source": "initial_train",
            "label_status": "completed",
            "n_qubits": "2",
            "hamiltonian_id": "toy_h2",
            "hamiltonian_class": "molecular",
            "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]),
            "ansatz_gene": json.dumps(gene.to_jsonable()),
            "fair_best_energy": "-3.0",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS))
        writer.writeheader()
        writer.writerow(row)


class P1RoundDemoTests(unittest.TestCase):
    def test_load_hamiltonian_for_demo_supports_pauli_terms_spec_file(self):
        from aicir.qas.demos.run_p1_round_demo import load_hamiltonian_for_demo

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ham.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "pauli_terms",
                        "terms": [[1.0, "ZI"], [-0.5, "IZ"]],
                        "metadata": {"electronic_reference_energy_old_thread": -3.25},
                    }
                ),
                encoding="utf-8",
            )
            terms, n_qubits, reference_energy = load_hamiltonian_for_demo(
                "custom",
                hamiltonian_file=str(path),
            )

        self.assertEqual(terms, ((1.0, "ZI"), (-0.5, "IZ")))
        self.assertEqual(n_qubits, 2)
        self.assertEqual(reference_energy, -3.25)

    def test_demo_defaults_to_e5_fallback_selector(self):
        from aicir.qas.demos.run_p1_round_demo import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
            ]
        )

        self.assertEqual(args.selector, "e5")
        self.assertTrue(args.enable_training_free_pruning)
        self.assertEqual(args.fallback_audit_multiplier, 4)
        self.assertEqual(args.selection_policy, "no_regret")
        self.assertIsNone(args.oracle_max_neighbor_std)
        self.assertEqual(args.min_previous_oracle_hit_rate, 0.5)

    def test_demo_accepts_no_regret_lite_policy_and_oracle_extra_budget(self):
        from aicir.qas.demos.run_p1_round_demo import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--selection-policy",
                "no_regret_lite",
                "--oracle-extra-top-k",
                "2",
            ]
        )

        self.assertEqual(args.selection_policy, "no_regret_lite")
        self.assertEqual(args.oracle_extra_top_k, 2)
    def test_demo_accepts_task_and_graph_proxy_selectors(self):
        from aicir.qas.demos.run_p1_round_demo import _needed_registry_fields, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--selector",
                "task_proxy",
                "--baseline-selectors",
                "E2,E5,VQE_TASK_PROXY,GNN_PROXY,ENSEMBLE",
            ]
        )

        self.assertEqual(args.selector, "task_proxy")
        self.assertEqual(
            set(_needed_registry_fields(args)),
            {"E2", "E5", "VQE_TASK_PROXY", "GNN_PROXY", "ENSEMBLE"},
        )

    def test_auto_selector_registers_all_candidate_evaluators(self):
        from aicir.qas.demos.run_p1_round_demo import _needed_registry_fields, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--selector",
                "auto",
                "--baseline-selectors",
                "E2",
            ]
        )

        self.assertEqual(args.selector, "auto")
        self.assertEqual(
            set(_needed_registry_fields(args)),
            {"E2", "E5", "VQE_TASK_PROXY", "GNN_PROXY", "ENSEMBLE"},
        )
    def test_ensemble_selector_builds_all_component_evaluators(self):
        from aicir.qas.demos.run_p1_round_demo import _needed_registry_fields, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--selector",
                "ensemble",
                "--baseline-selectors",
                "ENSEMBLE",
            ]
        )

        self.assertEqual(
            set(_needed_registry_fields(args)),
            {"E2", "VQE_TASK_PROXY", "GNN_PROXY", "ENSEMBLE"},
        )

    def test_real_e2_registry_scores_chemistry_rows_without_supernet_e5_path(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        args = demo.build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--growth-route",
                "line_b_chemistry_excitation",
                "--selector",
                "e2",
                "--baseline-selectors",
                "E2",
                "--e2-max-evals",
                "1",
            ]
        )
        row = {
            "architecture_id": "chem_parent",
            "family": "chemistry_excitation",
            "n_qubits": "2",
            "hamiltonian_id": "toy_chem",
            "hamiltonian_terms": json.dumps([[1.0, "ZI"]]),
            "reference_energy": "-1.0",
            "ansatz_gene": json.dumps(
                {
                    "kind": "chemistry_excitation",
                    "n_qubits": 2,
                    "hf_occupied_qubits": [1],
                    "excitations": [{"type": "single_excitation", "qubits": [0, 1]}],
                    "active_electrons": 1,
                    "active_spatial_orbitals": 1,
                }
            ),
        }

        class FakeResult:
            energy = -0.75
            evaluations = 1
            n_starts = 1
            best_parameters = [0.0]
            metadata = {"budget_per_start": 1}

        with patch("aicir.qas.demos.run_p1_round_demo.optimize_vqe_energy", return_value=FakeResult(), create=True), patch(
            "aicir.qas.demos.run_p0_diagnostic._run_torch_pauli_proxy",
            side_effect=AssertionError("chemistry E2 must not use the supernet torch_pauli proxy"),
        ):
            registry = demo.build_real_evaluator_registry(args, [(1.0, "ZI")], 2, -1.0)
            scored = registry["E2"](row)

        self.assertEqual(scored["E2"], -0.75)
        self.assertEqual(scored["E2_nfev"], 1)
    def test_real_labeling_runner_uses_architecture_stable_seed_mode(self):
        from aicir.qas.demos.run_p1_round_demo import run_labeling_queue

        with patch("aicir.qas.demos.run_p1_round_demo.subprocess.run") as run:
            run_labeling_queue(
                queue_path="queue.csv",
                output_path="labels.csv",
                protocol="protocol.json",
                seed=2026,
                n_seeds=1,
                max_evals=64,
                backend="numpy",
                dtype="complex128",
                dry_run=False,
            )

        command = run.call_args.args[0]
        self.assertIn("--seed-by-architecture-id", command)

    def test_compare_label_outputs_reports_selector_level_fair_metrics(self):
        from aicir.qas.demos.run_p1_round_demo import compare_label_outputs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p1 = root / "p1_labels.csv"
            random = root / "random_labels.csv"
            for path, energy in ((p1, -2.5), (random, -1.0)):
                row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
                row.update(
                    {
                        "architecture_id": path.stem,
                        "label_status": "completed",
                        "fair_best_energy": str(energy),
                        "batch_id": f"{path.stem}_r2",
                    }
                )
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS))
                    writer.writeheader()
                    writer.writerow(row)

            comparison = compare_label_outputs({"p1": p1, "random": random})

        self.assertEqual(comparison["strategies"]["p1"]["fair_call_count"], 1)
        self.assertEqual(comparison["strategies"]["p1"]["fair_best_in_queue"], -2.5)
        self.assertEqual(comparison["strategies"]["p1"]["fair_best_round"], 2)
        self.assertEqual(comparison["strategies"]["random"]["fair_best_in_queue"], -1.0)
        self.assertEqual(comparison["best_strategy"], "p1")

    def test_aggregate_csvs_deduplicates_architectures_by_best_label(self):
        from aicir.qas.demos.run_p1_round_demo import _aggregate_csvs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "merged.csv"
            for path, energy in ((first, -1.0), (second, -2.0)):
                row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
                row.update(
                    {
                        "architecture_id": "duplicate_arch",
                        "canonical_arch_hash": "same",
                        "label_status": "completed",
                        "fair_best_energy": str(energy),
                    }
                )
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS))
                    writer.writeheader()
                    writer.writerow(row)

            _aggregate_csvs([first, second], output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fair_best_energy"], "-2.0")

    def test_demo_writes_four_queues_and_runs_labeling_with_same_budget(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        label_calls = []

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {
                "E2": lambda row: {"E2": -2.0 if row.get("mutation_type") == "connectivity_mutation" else -1.0},
                "E5": lambda row: {"E5": -1.5 if row.get("mutation_type") == "connectivity_mutation" else -0.5},
            }

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            label_calls.append(
                {
                    "queue": Path(queue_path).name,
                    "seed": seed,
                    "max_evals": max_evals,
                    "n_seeds": n_seeds,
                    "backend": backend,
                    "dtype": dtype,
                    "dry_run": dry_run,
                }
            )
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows):
                row["label_status"] = "completed"
                row["fair_best_energy"] = str(-2.0 - 0.1 * index)
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            output = root / "demo"
            write_bootstrap_labels(bootstrap)

            result = demo.main(
                [
                    "--preset",
                    "h2_sto3g_jw_r0735_4q",
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(output),
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--selector",
                    "e2",
                    "--baseline-selectors",
                    "E2,E5",
                    "--run-labeling",
                    "--fair-max-evals",
                    "7",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

            comparison_path = output / "comparison.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

        self.assertTrue(result["queues"]["p1"].endswith("p1_queue.csv"))
        self.assertEqual(set(result["queues"]), {"p1", "random", "E2", "E5"})
        self.assertEqual(len(label_calls), 4)
        self.assertEqual({call["seed"] for call in label_calls}, {2026})
        self.assertEqual(set(result["label_seeds"].values()), {2026})
        self.assertEqual({call["max_evals"] for call in label_calls}, {7})
        self.assertEqual({item["fair_call_count"] for item in comparison["strategies"].values()}, {1})

    def test_demo_runs_three_rounds_with_p1_label_feedback_and_equal_budget(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        label_calls = []

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {
                "E2": lambda row: {"E2": -2.0 if row.get("mutation_type") == "connectivity_mutation" else -1.0},
                "E5": lambda row: {"E5": -1.5 if row.get("mutation_type") == "connectivity_mutation" else -0.5},
            }

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            label_calls.append({"queue": Path(queue_path).name, "round": Path(queue_path).parent.name})
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows):
                row["label_status"] = "completed"
                row["fair_best_energy"] = str(-2.0 - 0.01 * len(label_calls) - 0.001 * index)
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            output = root / "demo"
            write_bootstrap_labels(bootstrap)

            result = demo.main(
                [
                    "--preset",
                    "h2_sto3g_jw_r0735_4q",
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(output),
                    "--rounds",
                    "3",
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--selector",
                    "e2",
                    "--baseline-selectors",
                    "E2,E5",
                    "--run-labeling",
                    "--fair-max-evals",
                    "7",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

            with (output / "labels_p1.csv").open(newline="", encoding="utf-8") as handle:
                p1_labels = list(csv.DictReader(handle))

        self.assertEqual(len(label_calls), 12)
        self.assertEqual([round_info["labeled_count_before_round"] for round_info in result["rounds"]], [1, 2, 3])
        self.assertEqual(result["summary"]["expected_total_fair_calls_per_strategy"], 3)
        self.assertEqual(result["summary"]["actual_fair_calls_per_strategy"], {"p1": 3, "random": 3, "E2": 3, "E5": 3})
        self.assertEqual({item["fair_call_count"] for item in result["comparison"]["strategies"].values()}, {3})
        self.assertEqual(len(p1_labels), 3)

    def test_demo_stops_after_two_plateau_rounds(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {"E2": lambda row: {"E2": -1.0}, "E5": lambda row: {"E5": -1.0}}

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["label_status"] = "completed"
                row["fair_best_energy"] = "-2.0"
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            write_bootstrap_labels(bootstrap)
            result = demo.main(
                [
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(root / "demo"),
                    "--rounds",
                    "5",
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--run-labeling",
                    "--early-stop-epsilon",
                    "0.01",
                    "--early-stop-patience",
                    "2",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual(result["summary"]["stop_reason"], "plateau")

    def test_no_regret_lite_multi_round_summary_uses_actual_plan_fair_calls(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {"E2": lambda row: {"E2": -1.0}, "E5": lambda row: {"E5": -1.0}}

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows):
                row["label_status"] = "completed"
                row["fair_best_energy"] = str(-2.0 - 0.01 * index)
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            write_bootstrap_labels(bootstrap)
            result = demo.main(
                [
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(root / "demo"),
                    "--rounds",
                    "2",
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--selector",
                    "e2",
                    "--baseline-selectors",
                    "E2",
                    "--selection-policy",
                    "no_regret_lite",
                    "--oracle-extra-top-k",
                    "1",
                    "--k-min",
                    "1",
                    "--d-max",
                    "1.0",
                    "--run-labeling",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

        round_fair_calls = [item["summary"]["budget"]["total_fair_calls"] for item in result["rounds"]]
        self.assertEqual(round_fair_calls, [2, 2])
        self.assertEqual(result["summary"]["expected_total_fair_calls_per_strategy"], 4)
        self.assertEqual(result["summary"]["actual_fair_calls_per_strategy"], {"p1": 4, "random": 4, "E2": 4})
        self.assertEqual(result["summary"]["p1_plan_fair_calls_by_round"], [2, 2])
        self.assertEqual(result["summary"]["p1_fallback_fair_calls_by_round"], [1, 1])
        self.assertEqual(result["summary"]["p1_oracle_extra_fair_calls_by_round"], [1, 1])
    def test_demo_stops_before_exceeding_max_total_fair_calls(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {"E2": lambda row: {"E2": -1.0}, "E5": lambda row: {"E5": -1.0}}

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["label_status"] = "completed"
                row["fair_best_energy"] = "-2.0"
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            write_bootstrap_labels(bootstrap)
            result = demo.main(
                [
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(root / "demo"),
                    "--rounds",
                    "5",
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--run-labeling",
                    "--max-total-fair-calls",
                    "2",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["summary"]["stop_reason"], "max_total_fair_calls")
        self.assertEqual(result["summary"]["actual_fair_calls_per_strategy"]["p1"], 2)

    def test_demo_continues_when_baseline_labeling_fails(self):
        from aicir.qas.demos import run_p1_round_demo as demo

        def evaluator_builder(_args, _terms, _n_qubits, _reference_energy):
            return {"E2": lambda row: {"E2": -1.0}, "E5": lambda row: {"E5": -1.0}}

        def labeling_runner(*, queue_path, output_path, protocol, seed, n_seeds, max_evals, backend, dtype, dry_run):
            if Path(queue_path).name == "queue_e5_baseline.csv":
                raise subprocess.CalledProcessError(returncode=2, cmd="label-e5")
            with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["label_status"] = "completed"
                row["fair_best_energy"] = "-2.0"
            with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bootstrap = root / "bootstrap_labels.csv"
            write_bootstrap_labels(bootstrap)
            result = demo.main(
                [
                    "--bootstrap-labels-csv",
                    str(bootstrap),
                    "--output-dir",
                    str(root / "demo"),
                    "--rounds",
                    "2",
                    "--parent-count",
                    "1",
                    "--children-per-parent",
                    "2",
                    "--fair-top-k",
                    "1",
                    "--run-labeling",
                    "--label-n-seeds",
                    "1",
                ],
                evaluator_registry_builder=evaluator_builder,
                labeling_runner=labeling_runner,
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -3.0),
            )

        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(len(result["label_errors"]), 2)
        self.assertNotIn("E5", result["comparison"]["strategies"])



    def test_demo_accepts_growth_route_weight_controls(self):
        from aicir.qas.demos.run_p1_round_demo import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--bootstrap-labels-csv",
                "bootstrap.csv",
                "--growth-route",
                "line_b_chemistry_excitation",
                "--operator-genetic-weight",
                "0.25",
                "--operator-adapt-growth-weight",
                "0.75",
                "--chemistry-genetic-weight",
                "0.4",
                "--chemistry-adapt-growth-weight",
                "0.6",
                "--chemistry-growth-mode",
                "mixed",
            ]
        )

        self.assertEqual(args.growth_route, "line_b_chemistry_excitation")
        self.assertEqual(args.operator_genetic_weight, 0.25)
        self.assertEqual(args.operator_adapt_growth_weight, 0.75)
        self.assertEqual(args.chemistry_genetic_weight, 0.4)
        self.assertEqual(args.chemistry_adapt_growth_weight, 0.6)
        self.assertEqual(args.chemistry_growth_mode, "mixed")
if __name__ == "__main__":
    unittest.main()


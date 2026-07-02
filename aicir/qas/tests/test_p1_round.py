import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene, SupernetAnsatzGene


def make_gene(layers=2):
    return SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=tuple(("ry", "rz") for _ in range(layers)),
        two_qubit_layers=tuple(("cx",) for _ in range(layers)),
        two_qubit_pairs=((0, 1),),
    )


def labeled_row(architecture_id, fair):
    gene = make_gene()
    return {
        "architecture_id": architecture_id,
        "canonical_arch_hash": json.dumps(gene.to_jsonable(), sort_keys=True),
        "protocol_version": "fair_vqe_protocol_v2",
        "label_status": "completed",
        "n_qubits": "2",
        "hamiltonian_id": "toy_h2",
        "hamiltonian_class": "molecular",
        "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]),
        "family": "supernet_native",
        "depth_group": f"L{gene.layers}",
        "entangler_type": "mixed_supernet",
        "topology": "supernet_pairs",
        "ansatz_gene": json.dumps(gene.to_jsonable()),
        "fair_best_energy": str(fair),
    }


def operator_labeled_row(architecture_id, fair, operators=("XI",)):
    gene = OperatorSequenceAnsatzGene(n_qubits=2, operators=tuple(operators))
    return {
        "architecture_id": architecture_id,
        "canonical_arch_hash": json.dumps(gene.to_jsonable(), sort_keys=True),
        "protocol_version": "fair_vqe_protocol_v2",
        "label_status": "completed",
        "n_qubits": "2",
        "hamiltonian_id": "toy_operator",
        "hamiltonian_class": "pauli_terms",
        "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "XX"]]),
        "family": "operator_sequence",
        "depth_group": f"L{gene.layers}",
        "topology": "pauli_operator_sequence",
        "ansatz_gene": json.dumps(gene.to_jsonable()),
        "fair_best_energy": str(fair),
    }


def control_row(architecture_id, score="-10.0"):
    gene = SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=(("h", "h"), ("rx", "rx")),
        two_qubit_layers=(("rzz",), ("none",)),
        two_qubit_pairs=((0, 1),),
    )
    return {
        "architecture_id": architecture_id,
        "canonical_arch_hash": json.dumps(gene.to_jsonable(), sort_keys=True),
        "protocol_version": "fair_vqe_protocol_v2",
        "label_status": "",
        "n_qubits": "2",
        "hamiltonian_id": "toy_h2",
        "hamiltonian_class": "molecular",
        "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]),
        "family": "supernet_native",
        "depth_group": f"L{gene.layers}",
        "entangler_type": "mixed_supernet",
        "topology": "supernet_pairs",
        "ansatz_gene": json.dumps(gene.to_jsonable()),
        "E2": score,
    }


class P1RoundTests(unittest.TestCase):
    def test_protocol_schema_preserves_p1_metadata_columns(self):
        from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS

        for field in (
            "parent_architecture_id",
            "crossover_parent_architecture_id",
            "mutation_type",
            "p1_selection_source",
            "predicted_fair_energy",
            "oracle_reason",
            "oracle_neighbor_target_std",
            "fallback_selector",
            "fallback_score",
            "VQE_TASK_PROXY",
            "GNN_PROXY",
            "ENSEMBLE",
            "predictor_confidence",
        ):
            self.assertIn(field, BENCHMARK_TABLE_FIELDS)

    def test_plan_p1_round_generates_operator_sequence_children(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        def e2(row):
            return {"E2": -2.0 if "YY" in str(row.get("ansatz_gene", "")) else -1.0}

        plan = plan_p1_round(
            labeled_rows=[operator_labeled_row("parent", -1.0)],
            parent_count=1,
            children_per_parent=1,
            fair_top_k=1,
            evaluator_registry={"E2": e2},
            selector="e2",
            cheap_eval_selector="e2",
            mutation_types=("operator_insert",),
            operator_pool=("YY",),
            k_min=99,
            selection_policy="no_regret",
            baseline_selector_fields=("E2",),
        )

        self.assertEqual(len(plan.queue_rows), 1)
        child_gene = OperatorSequenceAnsatzGene.from_jsonable(json.loads(plan.child_rows[0]["ansatz_gene"]))
        self.assertEqual(plan.child_rows[0]["mutation_type"], "operator_insert")
        self.assertIn("YY", child_gene.operators)
        self.assertEqual(plan.queue_rows[0]["source"], "p1_fallback")

    def test_operator_sequence_selector_e5_downgrades_to_e2_when_e5_is_not_applicable(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        calls = []

        def e5(_row):
            raise ValueError("supernet_native evaluator requires a supernet_native ansatz gene")

        def e2(row):
            calls.append(str(row["architecture_id"]))
            return {"E2": -2.0 if "YY" in str(row.get("ansatz_gene", "")) else -1.0}

        plan = plan_p1_round(
            labeled_rows=[operator_labeled_row("parent", -1.0)],
            parent_count=1,
            children_per_parent=1,
            fair_top_k=1,
            evaluator_registry={"E2": e2, "E5": e5},
            selector="e5",
            cheap_eval_selector="e5",
            mutation_types=("operator_insert",),
            operator_pool=("YY",),
            k_min=99,
            selection_policy="no_regret",
            baseline_selector_fields=("E2",),
        )

        self.assertEqual(len(plan.queue_rows), 1)
        self.assertEqual(calls, [plan.child_rows[0]["architecture_id"]])
        self.assertEqual(plan.fallback_rows[0]["fallback_selector"], "E2")
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_field"], "E5")
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_downgrade_calls"], 1)

    def test_plan_p1_round_accepts_task_proxy_as_independent_selector(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        def task_proxy(row):
            return {"VQE_TASK_PROXY": -2.0 if "YY" in str(row.get("ansatz_gene", "")) else -1.0}

        plan = plan_p1_round(
            labeled_rows=[operator_labeled_row("parent", -1.0)],
            parent_count=1,
            children_per_parent=1,
            fair_top_k=1,
            evaluator_registry={"VQE_TASK_PROXY": task_proxy},
            selector="task_proxy",
            cheap_eval_selector="e2",
            mutation_types=("operator_insert",),
            operator_pool=("YY",),
            k_min=99,
            selection_policy="no_regret",
            baseline_selector_fields=("VQE_TASK_PROXY",),
        )

        self.assertEqual(len(plan.queue_rows), 1)
        self.assertEqual(plan.fallback_rows[0]["fallback_selector"], "VQE_TASK_PROXY")
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_field"], "VQE_TASK_PROXY")

    def test_ensemble_selector_uses_rank_aggregation_not_raw_mean(self):
        from aicir.qas.vqe_loop.p1_round import _score_rows

        rows = [
            {"architecture_id": "a", "canonical_arch_hash": "a", "E2": "-100.0", "VQE_TASK_PROXY": "10.0"},
            {"architecture_id": "b", "canonical_arch_hash": "b", "E2": "-99.0", "VQE_TASK_PROXY": "-10.0"},
            {"architecture_id": "c", "canonical_arch_hash": "c", "E2": "-98.0", "VQE_TASK_PROXY": "-9.0"},
        ]

        scored = _score_rows(rows, "ENSEMBLE", {}, {})
        by_id = {row["architecture_id"]: row for row in scored}

        self.assertEqual(by_id["a"]["ENSEMBLE"], "1.000000000000")
        self.assertEqual(by_id["b"]["ENSEMBLE"], "0.500000000000")
        self.assertEqual(by_id["c"]["ENSEMBLE"], "1.500000000000")
        self.assertEqual(by_id["b"]["fallback_selector"], "ENSEMBLE")

    def test_plan_p1_round_merges_oracle_and_fallback_with_equal_budget_baselines(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        def evaluator(field):
            def run(row):
                base = -2.0 if row["mutation_type"] == "connectivity_mutation" else -1.0
                return {field: base if field == "E2" else base + 0.25}

            return run

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=2,
            mutation_types=("gate_mutation", "connectivity_mutation"),
            fair_top_k=2,
            selector="e2",
            evaluator_registry={"E2": evaluator("E2"), "E5": evaluator("E5")},
            k_min=1,
            d_max=0.09,
            batch_id="p1_demo",
            seed=7,
            selection_policy="quota",
        )

        self.assertEqual(len(plan.queue_rows), 2)
        self.assertEqual(plan.summary["budget"]["total_fair_calls"], 2)
        self.assertEqual(plan.summary["budget"]["expected_total_fair_calls"], 2)
        self.assertEqual(plan.summary["n_oracle_trusted"], 1)
        self.assertEqual(plan.summary["n_oracle_abstain"], 1)
        self.assertEqual(
            {row["p1_selection_source"] for row in plan.queue_rows},
            {"oracle_trusted", "fallback_selector"},
        )
        self.assertEqual(set(plan.baseline_queues), {"random", "E2", "E5"})
        self.assertTrue(all(len(rows) == 2 for rows in plan.baseline_queues.values()))
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_calls"], 1)
        self.assertEqual(plan.summary["cheap_eval"]["cheap_eval_calls_saved"], 1)

    def test_plan_p1_round_audits_trusted_oracle_rows_when_abstain_pool_is_empty(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        calls = []

        def evaluator(row):
            calls.append(row["architecture_id"])
            base = -2.0 if row["mutation_type"] == "connectivity_mutation" else -1.0
            return {"E2": base}

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=4,
            mutation_types=("gate_mutation", "connectivity_mutation", "layer_mutation", "depth_mutation"),
            fair_top_k=4,
            selector="e2",
            evaluator_registry={"E2": evaluator},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=7,
            baseline_selector_fields=(),
            selection_policy="quota",
        )

        self.assertEqual(plan.summary["n_oracle_abstain"], 0)
        self.assertEqual(plan.summary["quota"], {"q0": 2, "q1": 2, "qc": 0})
        self.assertEqual(plan.summary["source_counts"], {"oracle_trusted": 2, "fallback_selector": 2})
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_calls"], 4)
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_audit_calls"], 4)
        self.assertEqual(len(calls), 4)

    def test_no_regret_p1_keeps_fallback_topk_when_oracle_rows_are_trusted(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        scores = {
            "gate_mutation": -4.0,
            "connectivity_mutation": -3.0,
            "layer_mutation": -2.0,
            "depth_mutation": -1.0,
        }

        def evaluator(row):
            return {"E2": scores[str(row["mutation_type"])]}

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=4,
            mutation_types=("gate_mutation", "connectivity_mutation", "layer_mutation", "depth_mutation"),
            fair_top_k=2,
            selector="e2",
            evaluator_registry={"E2": evaluator},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=7,
            baseline_selector_fields=(),
            selection_policy="no_regret",
        )

        self.assertGreater(plan.summary["n_oracle_trusted"], 0)
        self.assertEqual(plan.summary["selection_policy"], "no_regret")
        self.assertEqual(plan.summary["source_counts"], {"fallback_selector": 2})
        self.assertEqual(
            [row["mutation_type"] for row in plan.queue_rows],
            ["gate_mutation", "connectivity_mutation"],
        )

    def test_no_regret_lite_keeps_fallback_topk_and_adds_oracle_extra_without_cheap_eval(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        calls = []

        def evaluator(row):
            calls.append(row["architecture_id"])
            return {"E2": -1.0}

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=2,
            mutation_types=("gate_mutation", "connectivity_mutation"),
            fair_top_k=1,
            selector="e2",
            evaluator_registry={"E2": evaluator},
            k_min=1,
            d_max=0.09,
            batch_id="p1_demo",
            seed=7,
            baseline_selector_fields=(),
            selection_policy="no_regret_lite",
            oracle_extra_top_k=1,
        )

        self.assertEqual(plan.summary["selection_policy"], "no_regret_lite")
        self.assertIn("supernet_native", plan.summary["ansatz_families"])
        self.assertTrue(plan.summary["ansatz_families"]["supernet_native"]["capabilities"]["evaluators"]["E5"])
        self.assertEqual(plan.summary["source_counts"], {"fallback_selector": 1, "oracle_trusted_extra": 1})
        self.assertEqual(plan.summary["budget"]["fallback_top_k_per_round"], 1)
        self.assertEqual(plan.summary["budget"]["oracle_extra_fair_calls"], 1)
        self.assertEqual(plan.summary["budget"]["total_fair_calls"], 2)
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_calls"], 1)
        self.assertEqual(plan.summary["cheap_eval"]["cheap_eval_calls_saved"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["p1_selection_source"] for row in plan.queue_rows], ["fallback_selector", "oracle_trusted_extra"])
    def test_plan_p1_round_uses_wide_default_audit_window_for_all_trusted_oracle_pool(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        calls = []

        def evaluator(row):
            calls.append(row["architecture_id"])
            return {"E2": -1.0}

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=12,
            mutation_types=("gate_mutation", "connectivity_mutation", "layer_mutation", "depth_mutation"),
            fair_top_k=4,
            selector="e2",
            evaluator_registry={"E2": evaluator},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=7,
            baseline_selector_fields=(),
        )

        self.assertEqual(plan.summary["n_oracle_abstain"], 0)
        self.assertEqual(plan.summary["n_oracle_trusted"], 11)
        self.assertEqual(
            plan.summary["cheap_eval"]["p1_fallback_audit_calls"],
            plan.summary["n_oracle_trusted"],
        )
        self.assertEqual(
            plan.summary["cheap_eval"]["p1_fallback_audit_target"],
            plan.summary["n_oracle_trusted"],
        )
        self.assertEqual(len(calls), plan.summary["n_oracle_trusted"])

    def test_training_free_hard_reject_runs_before_oracle_and_fallback(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        calls = []

        def evaluator(row):
            calls.append(row["architecture_id"])
            return {"E2": -1.0}

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=1,
            mutation_types=("gate_mutation",),
            fair_top_k=1,
            selector="e2",
            evaluator_registry={"E2": evaluator},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=19,
            baseline_selector_fields=("E2",),
            enable_training_free_pruning=True,
            expressibility_hard_floor=2.0,
        )

        self.assertEqual(calls, [])
        self.assertEqual(plan.queue_rows, [])
        self.assertEqual(plan.summary["training_free"]["hard_reject_count"], 1)
        self.assertEqual(plan.summary["n_oracle_trusted"], 0)
        self.assertEqual(plan.summary["n_oracle_abstain"], 0)
        self.assertEqual(plan.summary["cheap_eval"]["p1_fallback_calls"], 0)

    def test_baseline_queues_use_same_candidate_pool_including_controls(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            control_rows=[control_row("control_best", score="-99.0")],
            parent_count=1,
            children_per_parent=1,
            mutation_types=("gate_mutation",),
            fair_top_k=1,
            selector="e2",
            evaluator_registry={"E2": lambda row: {"E2": -1.0 if row["architecture_id"] != "control_best" else -99.0}},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=17,
            baseline_selector_fields=("E2",),
        )

        self.assertEqual(plan.summary["baseline_candidate_pool_size"], 2)
        self.assertEqual(plan.baseline_queues["E2"][0]["architecture_id"], "control_best")
        self.assertEqual(plan.baseline_queues["E2"][0]["source"], "baseline_e2")

    def test_p1_selection_module_ranks_rows_by_score(self):
        from aicir.qas.vqe_loop.p1_selection import rank_by_score

        rows = [
            {"architecture_id": "bad", "E2": "2.0"},
            {"architecture_id": "good", "E2": "1.0"},
            {"architecture_id": "missing", "E2": ""},
        ]
        ranked = rank_by_score(rows, "E2")
        self.assertEqual([row["architecture_id"] for row in ranked], ["good", "bad"])

    def test_write_p1_round_outputs_writes_p1_and_baseline_queues(self):
        from aicir.qas.vqe_loop.p1_round import plan_p1_round, write_p1_round_outputs

        plan = plan_p1_round(
            labeled_rows=[labeled_row("parent", -3.0)],
            parent_count=1,
            children_per_parent=2,
            mutation_types=("gate_mutation", "connectivity_mutation"),
            fair_top_k=1,
            selector="e2",
            evaluator_registry={"E2": lambda _row: {"E2": -1.0}},
            k_min=1,
            d_max=1.0,
            batch_id="p1_demo",
            seed=11,
            baseline_selector_fields=("E2",),
        )

        with tempfile.TemporaryDirectory() as temp:
            paths = write_p1_round_outputs(plan, temp)
            self.assertTrue(paths["queue"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["baseline_E2"].exists())
            with paths["queue"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["batch_id"], "p1_demo")
        self.assertEqual(rows[0]["label_status"], "pending")
        self.assertEqual(rows[0]["parent_architecture_id"], "parent")


if __name__ == "__main__":
    unittest.main()





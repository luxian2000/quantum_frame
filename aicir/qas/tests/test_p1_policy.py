import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class P1PolicyTests(unittest.TestCase):
    def test_budget_tracker_makes_fair_calls_explicit(self):
        from aicir.qas.vqe_loop.benchmark_table import FairBudgetTracker

        tracker = FairBudgetTracker(rounds=3, fair_top_k_per_round=4)
        tracker.record_round("round1", 4)
        tracker.record_round("round2", 3)

        self.assertEqual(tracker.expected_total_fair_calls, 12)
        self.assertEqual(tracker.total_fair_calls, 7)
        self.assertEqual(tracker.remaining_fair_calls, 5)
        self.assertEqual(
            tracker.to_jsonable(),
            {
                "rounds": 3,
                "fair_top_k_per_round": 4,
                "expected_total_fair_calls": 12,
                "total_fair_calls": 7,
                "remaining_fair_calls": 5,
                "round_fair_calls": {"round1": 4, "round2": 3},
            },
        )

    def test_deduplicate_children_reuses_labeled_and_skips_known_unlabeled(self):
        from aicir.qas.vqe_loop.benchmark_table import deduplicate_children

        labeled = [{"architecture_id": "labeled", "canonical_arch_hash": "same", "fair_best_energy": "-2.0"}]
        pending = [{"architecture_id": "pending", "canonical_arch_hash": "pending_key"}]
        children = [
            {"architecture_id": "child_reuses_label", "canonical_arch_hash": "same"},
            {"architecture_id": "child_pending_dup", "canonical_arch_hash": "pending_key"},
            {"architecture_id": "child_new", "canonical_arch_hash": "new_key"},
            {"architecture_id": "child_new_dup", "canonical_arch_hash": "new_key"},
        ]

        result = deduplicate_children(children, labeled_rows=labeled, known_unlabeled_rows=pending)

        self.assertEqual([row["architecture_id"] for row in result.reused_labeled], ["labeled"])
        self.assertEqual([row["architecture_id"] for row in result.new_children], ["child_new"])
        self.assertEqual(result.skipped_duplicate_architecture_ids, ["child_pending_dup", "child_new_dup"])

    def test_p1_both_selector_delegates_to_configured_cheap_selector(self):
        from aicir.qas.vqe_loop.benchmark_table import resolve_p1_selector_fields

        self.assertEqual(resolve_p1_selector_fields("e2", cheap_eval_selector="e5"), ("E2",))
        self.assertEqual(resolve_p1_selector_fields("e5", cheap_eval_selector="e2"), ("E5",))
        self.assertEqual(resolve_p1_selector_fields("task_proxy", cheap_eval_selector="e2"), ("VQE_TASK_PROXY",))
        self.assertEqual(resolve_p1_selector_fields("gnn_proxy", cheap_eval_selector="e2"), ("GNN_PROXY",))
        self.assertEqual(resolve_p1_selector_fields("ensemble", cheap_eval_selector="e2"), ("ENSEMBLE",))
        self.assertEqual(resolve_p1_selector_fields("both", cheap_eval_selector="e5"), ("E5",))
        self.assertEqual(resolve_p1_selector_fields("both", cheap_eval_selector="e2"), ("E2",))

    def test_auto_selector_uses_p0_fair_labels_as_target(self):
        from aicir.qas.vqe_loop.p1_selection import choose_p1_auto_selector

        rows = [
            {"architecture_id": "best", "fair_best_energy": "-5.0", "E2": "-4.0", "GNN_PROXY": "-1.0"},
            {"architecture_id": "mid", "fair_best_energy": "-3.0", "E2": "-2.0", "GNN_PROXY": "-5.0"},
            {"architecture_id": "bad", "fair_best_energy": "-1.0", "E2": "-1.0", "GNN_PROXY": "-3.0"},
        ]

        decision = choose_p1_auto_selector(rows, candidates=("E2", "GNN_PROXY"), top_k=1, min_completed=2)

        self.assertEqual(decision.selector, "e2")
        self.assertEqual(decision.field, "E2")
        self.assertEqual(decision.reason, "p0_fair_alignment")
        self.assertGreater(decision.scores["E2"]["top_k_hit_rate"], decision.scores["GNN_PROXY"]["top_k_hit_rate"])

    def test_auto_selector_falls_back_when_p0_labels_are_sparse(self):
        from aicir.qas.vqe_loop.p1_selection import choose_p1_auto_selector

        decision = choose_p1_auto_selector(
            [{"architecture_id": "only_one", "fair_best_energy": "-1.0", "E2": "-1.0"}],
            candidates=("E2", "GNN_PROXY"),
            min_completed=2,
            fallback_selector="task_proxy",
        )

        self.assertEqual(decision.selector, "task_proxy")
        self.assertEqual(decision.reason, "insufficient_p0_labels")

    def test_quota_policy_handles_cold_start_and_bad_oracle_feedback(self):
        from aicir.qas.vqe_loop.benchmark_table import choose_quota

        self.assertEqual(choose_quota(10, oracle_trusted_count=0).to_jsonable(), {"q0": 0, "q1": 10, "qc": 0})
        self.assertEqual(choose_quota(10, oracle_trusted_count=20).to_jsonable(), {"q0": 6, "q1": 3, "qc": 1})
        self.assertEqual(
            choose_quota(
                10,
                oracle_trusted_count=20,
                previous_oracle_trusted_fair_mean=-1.0,
                previous_fallback_fair_mean=-2.0,
            ).to_jsonable(),
            {"q0": 3, "q1": 6, "qc": 1},
        )

    def test_quota_merge_does_not_mix_oracle_and_proxy_score_scales(self):
        from aicir.qas.vqe_loop.benchmark_table import P1Quota, merge_quota_candidates

        oracle_rows = [
            {"architecture_id": "oracle_good", "predicted_fair_energy": "-3.0"},
            {"architecture_id": "oracle_bad", "predicted_fair_energy": "-1.0"},
        ]
        fallback_rows = [
            {"architecture_id": "fallback_good", "E2": "-100.0"},
            {"architecture_id": "fallback_bad", "E2": "-2.0"},
        ]
        control_rows = [{"architecture_id": "control"}]

        merged = merge_quota_candidates(
            oracle_rows,
            fallback_rows,
            control_rows,
            quota=P1Quota(q0=1, q1=1, qc=1),
            fallback_score_field="E2",
        )

        self.assertEqual([row["architecture_id"] for row in merged], ["oracle_good", "fallback_good", "control"])
        self.assertEqual([row["p1_selection_source"] for row in merged], ["oracle_trusted", "fallback_selector", "control"])

    def test_quota_merge_can_sort_control_rows_by_optional_score(self):
        from aicir.qas.vqe_loop.benchmark_table import P1Quota, merge_quota_candidates

        control_rows = [
            {"architecture_id": "fifo_first", "diversity_distance": "0.1"},
            {"architecture_id": "diverse", "diversity_distance": "0.9"},
        ]

        fifo = merge_quota_candidates(
            [],
            [],
            control_rows,
            quota=P1Quota(q0=0, q1=0, qc=1),
            fallback_score_field="E2",
        )
        scored = merge_quota_candidates(
            [],
            [],
            control_rows,
            quota=P1Quota(q0=0, q1=0, qc=1),
            fallback_score_field="E2",
            control_score_field="diversity_distance",
            control_score_descending=True,
        )

        self.assertEqual([row["architecture_id"] for row in fifo], ["fifo_first"])
        self.assertEqual([row["architecture_id"] for row in scored], ["diverse"])

    def test_soft_prefilter_prefers_pass_rows_from_top_window(self):
        from aicir.qas.vqe_loop.benchmark_table import rank_with_zero_cost_soft_prefilter

        rows = [
            {"architecture_id": "soft_best", "fallback_score": "-5.0", "zero_cost_status": "soft_flag"},
            {"architecture_id": "pass_second", "fallback_score": "-4.0", "zero_cost_status": "pass"},
            {"architecture_id": "soft_third", "fallback_score": "-3.0", "zero_cost_status": "soft_flag"},
            {"architecture_id": "pass_outside", "fallback_score": "-2.0", "zero_cost_status": "pass"},
        ]

        ranked = rank_with_zero_cost_soft_prefilter(
            rows,
            "fallback_score",
            target_count=2,
            window_multiplier=1,
        )

        self.assertEqual([row["architecture_id"] for row in ranked[:2]], ["pass_second", "soft_best"])


if __name__ == "__main__":
    unittest.main()

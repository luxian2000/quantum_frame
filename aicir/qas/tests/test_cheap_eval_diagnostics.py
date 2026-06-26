import csv
import json
import math

from aicir.qas.vqe_loop.cheap_eval_diagnostics import (
    cost_curve,
    decide_proxy_status,
    kendall_pairwise_accuracy,
    proxy_quality_cost_frontier,
    spearman_correlation,
    stratified_proxy_summary,
    summarize_proxy_diagnostics,
    temporal_analysis,
    top_k_enrichment,
    warm_start_gain_summary,
)


def _rows():
    return [
        {
            "architecture_id": "a",
            "fair_high": "-4.0",
            "E1": "-1.0",
            "E2": "-3.8",
            "E5": "-3.9",
            "fair_warm": "-3.7",
            "fair_random": "-3.2",
            "hit_rate": "0.8",
        },
        {
            "architecture_id": "b",
            "fair_high": "-3.0",
            "E1": "-2.0",
            "E2": "-2.8",
            "E5": "-3.1",
            "fair_warm": "-2.9",
            "fair_random": "-2.7",
            "hit_rate": "0.7",
        },
        {
            "architecture_id": "c",
            "fair_high": "-2.0",
            "E1": "-3.0",
            "E2": "-2.2",
            "E5": "-2.1",
            "fair_warm": "-2.0",
            "fair_random": "-1.8",
            "hit_rate": "0.2",
        },
        {
            "architecture_id": "d",
            "fair_high": "-1.0",
            "E1": "-4.0",
            "E2": "-1.1",
            "E5": "-0.9",
            "fair_warm": "-1.0",
            "fair_random": "-0.9",
            "hit_rate": "0.1",
        },
    ]


def test_rank_metrics_detect_good_and_reversed_proxies():
    rows = _rows()

    assert round(spearman_correlation(rows, "E2", "fair_high"), 6) == 1.0
    assert round(spearman_correlation(rows, "E1", "fair_high"), 6) == -1.0
    assert round(kendall_pairwise_accuracy(rows, "E2", "fair_high"), 6) == 1.0
    assert round(kendall_pairwise_accuracy(rows, "E1", "fair_high"), 6) == 0.0


def test_top_k_enrichment_measures_screening_usefulness():
    rows = _rows()

    e2 = top_k_enrichment(rows, "E2", "fair_high", k=2)
    e1 = top_k_enrichment(rows, "E1", "fair_high", k=2)

    assert e2["top_k_recall"] == 1.0
    assert e2["fair_top_recall"] == 1.0
    assert e2["target_mean_proxy_top_k"] == -3.5
    assert e2["enrichment"] > 0.0
    assert e1["top_k_recall"] == 0.0
    assert e1["fair_top_recall"] == 0.0
    assert e1["enrichment"] < 0.0


def test_top_k_enrichment_uses_architecture_identity_not_target_value():
    rows = [
        {"architecture_id": "fair_best", "fair_high": "-5.0", "proxy": "-1.0"},
        {"architecture_id": "fair_second", "fair_high": "-4.0", "proxy": "-2.0"},
        {"architecture_id": "duplicate_energy_not_fair_top", "fair_high": "-4.0", "proxy": "-9.0"},
        {"architecture_id": "bad", "fair_high": "-1.0", "proxy": "-8.0"},
    ]

    result = top_k_enrichment(rows, "proxy", "fair_high", k=2)

    assert result["intersection_count"] == 0.0
    assert result["top_k_recall"] == 0.0
    assert result["proxy_top_recall"] == 0.0
    assert result["fair_top_recall"] == 0.0


def test_top_k_enrichment_distinguishes_missing_proxy_recall_directions():
    rows = [
        {"architecture_id": "fair_best_missing_proxy", "fair_high": "-5.0", "proxy": ""},
        {"architecture_id": "fair_second_with_proxy", "fair_high": "-4.0", "proxy": "-4.0"},
        {"architecture_id": "other_missing_proxy", "fair_high": "-3.0", "proxy": ""},
    ]

    result = top_k_enrichment(rows, "proxy", "fair_high", k=2)

    assert result["intersection_count"] == 1.0
    assert result["top_k_recall"] == 0.5
    assert result["proxy_top_recall"] == 0.5
    assert result["fair_top_recall"] == 1.0


def test_cost_curve_is_n_dependent():
    curve = cost_curve(upfront_cost=100.0, per_arch_cost=5.0, n_values=(10, 20))

    assert curve["10"]["total_cost"] == 150.0
    assert curve["10"]["amortized_cost"] == 15.0
    assert curve["20"]["total_cost"] == 200.0
    assert curve["20"]["amortized_cost"] == 10.0


def test_quality_cost_frontier_keeps_cost_and_quality_together():
    summary = proxy_quality_cost_frontier(
        _rows(),
        proxy_fields=("E1", "E2", "E5"),
        target_field="fair_high",
        cost_models={
            "E1": {"upfront_cost": 0.0, "per_arch_cost": 1.0},
            "E5": {"upfront_cost": 100.0, "per_arch_cost": 2.0},
        },
        k=2,
        n_values=(10,),
    )

    assert summary["E1"]["spearman"] < 0.0
    assert round(summary["E2"]["spearman"], 6) == 1.0
    assert summary["E5"]["cost"]["10"]["total_cost"] == 120.0
    assert summary["E5"]["top_k"]["top_k_recall"] == 1.0


def test_warm_start_gain_summary_uses_random_minus_warm():
    summary = warm_start_gain_summary(_rows())

    assert summary["count"] == 4.0
    assert round(summary["mean_gain"], 6) == 0.25
    assert summary["std_gain"] > 0.0


def test_warm_start_gain_cv_is_infinite_for_zero_mean_nonzero_variance():
    rows = [
        {"fair_warm": "-2.0", "fair_random": "-1.0"},
        {"fair_warm": "-1.0", "fair_random": "-2.0"},
    ]

    summary = warm_start_gain_summary(rows)

    assert summary["mean_gain"] == 0.0
    assert summary["std_gain"] == 1.0
    assert summary["gain_variance"] == 1.0
    assert math.isinf(summary["gain_cv_abs"])


def test_stratified_proxy_summary_reports_high_and_low_hit_groups():
    summary = stratified_proxy_summary(
        _rows(),
        proxy_field="E5",
        target_field="fair_high",
        strata_field="hit_rate",
        threshold=0.5,
        k=1,
    )

    assert summary["low"]["count"] == 2.0
    assert summary["high"]["count"] == 2.0
    assert summary["high"]["top_k"]["top_k_recall"] == 1.0


def test_decision_gate_passes_repairs_and_falls_back():
    passing = {
        "spearman": 0.75,
        "top_k": {"enrichment": 0.2},
    }
    failing = {
        "spearman": 0.25,
        "top_k": {"enrichment": -0.1},
    }

    assert decide_proxy_status(passing) == "pass"
    assert decide_proxy_status(failing, repair_round=1, max_repair_rounds=3) == "repair"
    assert decide_proxy_status(failing, repair_round=3, max_repair_rounds=3) == "fallback"


def test_summarize_proxy_diagnostics_marks_each_proxy():
    summary = summarize_proxy_diagnostics(
        _rows(),
        proxy_fields=("E1", "E2"),
        target_field="fair_high",
        cost_models={"E2": {"upfront_cost": 0.0, "per_arch_cost": 5.0}},
        k=2,
        n_values=(10,),
        warm_fields=("fair_warm", "fair_random"),
        strata_field="hit_rate",
        strata_threshold=0.5,
    )

    assert summary["row_count"] == 4
    assert summary["proxies"]["E1"]["status"] == "repair"
    assert summary["proxies"]["E2"]["status"] == "pass"
    assert summary["proxies"]["E2"]["cost"]["10"]["total_cost"] == 50.0
    assert summary["warm_start_gain"]["count"] == 4.0
    assert "gain_cv_abs" in summary["warm_start_gain"]
    assert summary["strata"]["field"] == "hit_rate"
    assert summary["strata"]["proxies"]["E2"]["high"]["count"] == 2.0


def test_temporal_analysis_reports_order_window_correlations():
    rows = [
        {"evaluation_order_index": "0", "fair_high": "-1.0", "E5": "-4.0"},
        {"evaluation_order_index": "1", "fair_high": "-2.0", "E5": "-3.0"},
        {"evaluation_order_index": "2", "fair_high": "-3.0", "E5": "-3.0"},
        {"evaluation_order_index": "3", "fair_high": "-4.0", "E5": "-4.0"},
    ]

    summary = temporal_analysis(rows, "E5", "fair_high", order_field="evaluation_order_index", window_size=2)

    assert summary["count"] == 4.0
    assert summary["window_size"] == 2.0
    assert len(summary["windows"]) == 2
    assert summary["windows"][0]["start_order"] == 0.0
    assert summary["windows"][1]["end_order"] == 3.0
    assert round(summary["windows"][0]["spearman"], 6) == -1.0
    assert round(summary["windows"][1]["spearman"], 6) == 1.0


def test_summarize_proxy_diagnostics_includes_temporal_analysis():
    rows = []
    for index, row in enumerate(_rows()):
        copied = dict(row)
        copied["evaluation_order_index"] = str(index)
        rows.append(copied)

    summary = summarize_proxy_diagnostics(
        rows,
        proxy_fields=("E5",),
        target_field="fair_high",
        temporal_order_field="evaluation_order_index",
        temporal_window_size=2,
    )

    assert summary["temporal"]["order_field"] == "evaluation_order_index"
    assert summary["temporal"]["proxies"]["E5"]["window_size"] == 2.0


def test_cli_writes_json_summary(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_diagnostics import main

    csv_path = tmp_path / "rows.csv"
    output_path = tmp_path / "summary.json"
    fieldnames = [
        "architecture_id",
        "fair_high",
        "E1",
        "E2",
        "fair_warm",
        "fair_random",
        "hit_rate",
        "evaluation_order_index",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(_rows()):
            copied = dict(row)
            copied["evaluation_order_index"] = str(index)
            writer.writerow({field: copied[field] for field in fieldnames})

    main(
        [
            "--input",
            str(csv_path),
            "--target",
            "fair_high",
            "--proxies",
            "E1,E2",
            "--output",
            str(output_path),
            "--cost-models",
            json.dumps({"E2": {"upfront_cost": 0, "per_arch_cost": 5}}),
            "--warm-fields",
            "fair_warm,fair_random",
            "--strata-field",
            "hit_rate",
            "--strata-threshold",
            "0.5",
            "--temporal-order-field",
            "evaluation_order_index",
            "--temporal-window-size",
            "2",
            "--top-k",
            "2",
            "--n-values",
            "10",
        ]
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["target_field"] == "fair_high"
    assert loaded["proxies"]["E2"]["status"] == "pass"
    assert loaded["warm_start_gain"]["count"] == 4.0
    assert loaded["strata"]["proxies"]["E2"]["high"]["count"] == 2.0
    assert loaded["temporal"]["proxies"]["E2"]["window_size"] == 2.0


def test_cli_accepts_cost_models_file(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_diagnostics import main

    csv_path = tmp_path / "rows.csv"
    output_path = tmp_path / "summary.json"
    cost_path = tmp_path / "cost.json"
    fieldnames = ["architecture_id", "fair_high", "E2"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in _rows():
            writer.writerow({field: row[field] for field in fieldnames})
    cost_path.write_text(json.dumps({"E2": {"upfront_cost": 7, "per_arch_cost": 5}}), encoding="utf-8")

    main(
        [
            "--input",
            str(csv_path),
            "--target",
            "fair_high",
            "--proxies",
            "E2",
            "--output",
            str(output_path),
            "--cost-models-file",
            str(cost_path),
            "--top-k",
            "2",
            "--n-values",
            "10",
        ]
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["proxies"]["E2"]["cost"]["10"]["total_cost"] == 57.0

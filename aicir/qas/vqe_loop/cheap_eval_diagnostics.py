"""Diagnostics for cheap VQE architecture evaluators.

The functions in this module analyze rows that were already produced by a P0
experiment.  They do not run VQE or train a supernet; instead they compare
candidate cheap-proxy columns such as ``E1``, ``E2``, and ``E5`` against a
high-budget fair-VQE target column.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Row = Mapping[str, Any]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _paired_values(rows: Iterable[Row], proxy_field: str, target_field: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        proxy = _as_float(row.get(proxy_field))
        target = _as_float(row.get(target_field))
        if proxy is None or target is None:
            continue
        pairs.append((proxy, target))
    return pairs


def _rank(values: Sequence[float]) -> list[float]:
    """Return average ranks for ``values`` with low values ranked best."""

    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _value in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = sqrt(sum(value * value for value in left_centered))
    right_norm = sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def spearman_correlation(rows: Iterable[Row], proxy_field: str, target_field: str) -> float | None:
    """Compute Spearman rank correlation between a cheap proxy and fair energy."""

    pairs = _paired_values(rows, proxy_field, target_field)
    if len(pairs) < 2:
        return None
    proxy_ranks = _rank([proxy for proxy, _target in pairs])
    target_ranks = _rank([target for _proxy, target in pairs])
    return _pearson(proxy_ranks, target_ranks)


def kendall_pairwise_accuracy(rows: Iterable[Row], proxy_field: str, target_field: str) -> float | None:
    """Return the fraction of candidate pairs ordered the same by proxy and target."""

    pairs = _paired_values(rows, proxy_field, target_field)
    total = 0
    correct = 0
    for left_index in range(len(pairs)):
        for right_index in range(left_index + 1, len(pairs)):
            proxy_delta = pairs[left_index][0] - pairs[right_index][0]
            target_delta = pairs[left_index][1] - pairs[right_index][1]
            if proxy_delta == 0.0 or target_delta == 0.0:
                continue
            total += 1
            if proxy_delta * target_delta > 0.0:
                correct += 1
    if total == 0:
        return None
    return correct / float(total)


def _row_identity(row: Row, index: int) -> str:
    architecture_id = row.get("architecture_id")
    if architecture_id is not None and str(architecture_id).strip():
        return str(architecture_id)
    return f"row:{index}"


def top_k_enrichment(
    rows: Iterable[Row],
    proxy_field: str,
    target_field: str,
    k: int,
) -> dict[str, float]:
    """Measure whether cheap top-K covers fair top-K and beats random average."""

    target_rows: list[dict[str, float | int | str]] = []
    proxy_rows: list[dict[str, float | int | str]] = []
    for index, row in enumerate(rows):
        target = _as_float(row.get(target_field))
        if target is None:
            continue
        record: dict[str, float | int | str] = {
            "id": _row_identity(row, index),
            "index": index,
            "target": target,
        }
        target_rows.append(record)
        proxy = _as_float(row.get(proxy_field))
        if proxy is not None:
            proxy_record = dict(record)
            proxy_record["proxy"] = proxy
            proxy_rows.append(proxy_record)

    if not target_rows or int(k) <= 0:
        return {
            "count": 0.0,
            "k": 0.0,
            "fair_k": 0.0,
            "proxy_k": 0.0,
            "target_count": 0.0,
            "intersection_count": 0.0,
            "top_k_recall": 0.0,
            "proxy_top_recall": 0.0,
            "fair_top_recall": 0.0,
            "target_mean_proxy_top_k": 0.0,
            "target_mean_fair_top_k": 0.0,
            "target_mean_all": 0.0,
            "enrichment": 0.0,
        }
    fair_limit = min(int(k), len(target_rows))
    proxy_limit = min(int(k), len(proxy_rows))
    fair_top = sorted(target_rows, key=lambda row: (float(row["target"]), int(row["index"])))[:fair_limit]
    proxy_top = sorted(proxy_rows, key=lambda row: (float(row["proxy"]), int(row["index"])))[:proxy_limit]
    all_mean = sum(float(row["target"]) for row in target_rows) / len(target_rows)
    proxy_mean = (
        sum(float(row["target"]) for row in proxy_top) / len(proxy_top)
        if proxy_top
        else 0.0
    )
    fair_mean = sum(float(row["target"]) for row in fair_top) / len(fair_top)
    fair_top_ids = {str(row["id"]) for row in fair_top}
    proxy_top_ids = {str(row["id"]) for row in proxy_top}
    intersection_count = len(fair_top_ids & proxy_top_ids)
    proxy_top_recall = intersection_count / float(len(fair_top)) if fair_top else 0.0
    fair_top_recall = intersection_count / float(len(proxy_top)) if proxy_top else 0.0
    return {
        "count": float(len(proxy_rows)),
        "k": float(fair_limit),
        "fair_k": float(fair_limit),
        "proxy_k": float(proxy_limit),
        "target_count": float(len(target_rows)),
        "intersection_count": float(intersection_count),
        "top_k_recall": float(proxy_top_recall),
        "proxy_top_recall": float(proxy_top_recall),
        "fair_top_recall": float(fair_top_recall),
        "target_mean_proxy_top_k": float(proxy_mean),
        "target_mean_fair_top_k": float(fair_mean),
        "target_mean_all": float(all_mean),
        "enrichment": float(all_mean - proxy_mean) if proxy_top else 0.0,
    }


def cost_curve(
    upfront_cost: float,
    per_arch_cost: float,
    n_values: Sequence[int] = (10, 20, 40, 80),
) -> dict[str, dict[str, float]]:
    """Return total and amortized costs for different screening batch sizes."""

    curve: dict[str, dict[str, float]] = {}
    for n_value in n_values:
        n = max(1, int(n_value))
        total = float(upfront_cost) + float(per_arch_cost) * n
        curve[str(n)] = {
            "n": float(n),
            "total_cost": float(total),
            "amortized_cost": float(total / n),
        }
    return curve


def proxy_quality_cost_frontier(
    rows: Sequence[Row],
    proxy_fields: Sequence[str],
    target_field: str,
    cost_models: Mapping[str, Mapping[str, float]],
    *,
    k: int = 3,
    n_values: Sequence[int] = (10, 20, 40, 80),
) -> dict[str, dict[str, Any]]:
    """Summarize quality/cost metrics for each cheap proxy."""

    summaries: dict[str, dict[str, Any]] = {}
    for proxy_field in proxy_fields:
        cost_model = cost_models.get(proxy_field, {})
        summaries[proxy_field] = {
            "spearman": spearman_correlation(rows, proxy_field, target_field),
            "kendall_pairwise_accuracy": kendall_pairwise_accuracy(rows, proxy_field, target_field),
            "top_k": top_k_enrichment(rows, proxy_field, target_field, k),
            "cost": cost_curve(
                upfront_cost=float(cost_model.get("upfront_cost", 0.0)),
                per_arch_cost=float(cost_model.get("per_arch_cost", 1.0)),
                n_values=n_values,
            ),
        }
    return summaries


def warm_start_gain_summary(
    rows: Iterable[Row],
    warm_field: str = "fair_warm",
    random_field: str = "fair_random",
) -> dict[str, float]:
    gains: list[float] = []
    for row in rows:
        warm = _as_float(row.get(warm_field))
        random = _as_float(row.get(random_field))
        if warm is None or random is None:
            continue
        gains.append(random - warm)
    if not gains:
        return {
            "count": 0.0,
            "mean_gain": 0.0,
            "gain_variance": 0.0,
            "std_gain": 0.0,
            "gain_cv_abs": 0.0,
        }
    mean = sum(gains) / len(gains)
    variance = sum((gain - mean) ** 2 for gain in gains) / len(gains)
    std = sqrt(variance)
    if mean == 0.0:
        gain_cv_abs = float("inf") if std != 0.0 else 0.0
    else:
        gain_cv_abs = std / abs(mean)
    return {
        "count": float(len(gains)),
        "mean_gain": float(mean),
        "gain_variance": float(variance),
        "std_gain": float(std),
        "gain_cv_abs": float(gain_cv_abs),
    }


def stratified_proxy_summary(
    rows: Sequence[Row],
    proxy_field: str,
    target_field: str,
    strata_field: str,
    *,
    threshold: float,
    k: int = 3,
) -> dict[str, dict[str, Any]]:
    low = [row for row in rows if (_as_float(row.get(strata_field)) or 0.0) < float(threshold)]
    high = [row for row in rows if (_as_float(row.get(strata_field)) or 0.0) >= float(threshold)]
    return {
        "low": {
            "count": float(len(low)),
            "spearman": spearman_correlation(low, proxy_field, target_field),
            "top_k": top_k_enrichment(low, proxy_field, target_field, k),
        },
        "high": {
            "count": float(len(high)),
            "spearman": spearman_correlation(high, proxy_field, target_field),
            "top_k": top_k_enrichment(high, proxy_field, target_field, k),
        },
    }


def temporal_analysis(
    rows: Sequence[Row],
    proxy_field: str,
    target_field: str,
    *,
    order_field: str = "evaluation_order_index",
    window_size: int = 20,
) -> dict[str, Any]:
    """Compute proxy quality in order windows to expose supernet warmup effects."""

    ordered: list[dict[str, Any]] = []
    for row in rows:
        order = _as_float(row.get(order_field))
        proxy = _as_float(row.get(proxy_field))
        target = _as_float(row.get(target_field))
        if order is None or proxy is None or target is None:
            continue
        ordered.append({**dict(row), order_field: order, proxy_field: proxy, target_field: target})
    ordered.sort(key=lambda row: float(row[order_field]))
    size = max(1, int(window_size))
    windows: list[dict[str, float | None]] = []
    for start in range(0, len(ordered), size):
        window = ordered[start : start + size]
        if len(window) < 2:
            continue
        windows.append(
            {
                "start_order": float(window[0][order_field]),
                "end_order": float(window[-1][order_field]),
                "count": float(len(window)),
                "spearman": spearman_correlation(window, proxy_field, target_field),
                "kendall_pairwise_accuracy": kendall_pairwise_accuracy(window, proxy_field, target_field),
            }
        )
    return {
        "count": float(len(ordered)),
        "order_field": str(order_field),
        "window_size": float(size),
        "windows": windows,
    }


def decide_proxy_status(
    summary: Mapping[str, Any],
    *,
    spearman_threshold: float = 0.6,
    min_enrichment: float = 0.0,
    repair_round: int = 0,
    max_repair_rounds: int = 3,
) -> str:
    """Return ``pass``, ``repair``, or ``fallback`` for a proxy summary."""

    spearman = _as_float(summary.get("spearman"))
    top_k = summary.get("top_k", {})
    enrichment = _as_float(top_k.get("enrichment") if isinstance(top_k, Mapping) else None)
    passes = (
        spearman is not None
        and spearman >= float(spearman_threshold)
        and enrichment is not None
        and enrichment >= float(min_enrichment)
    )
    if passes:
        return "pass"
    if int(repair_round) >= int(max_repair_rounds):
        return "fallback"
    return "repair"


def summarize_proxy_diagnostics(
    rows: Sequence[Row],
    proxy_fields: Sequence[str],
    target_field: str,
    cost_models: Mapping[str, Mapping[str, float]] | None = None,
    *,
    k: int = 3,
    n_values: Sequence[int] = (10, 20, 40, 80),
    repair_round: int = 0,
    warm_fields: tuple[str, str] | None = None,
    strata_field: str | None = None,
    strata_threshold: float | None = None,
    temporal_order_field: str | None = None,
    temporal_window_size: int = 20,
) -> dict[str, Any]:
    cost_models = cost_models or {}
    proxies = proxy_quality_cost_frontier(
        rows,
        proxy_fields,
        target_field,
        cost_models,
        k=k,
        n_values=n_values,
    )
    for proxy_summary in proxies.values():
        proxy_summary["status"] = decide_proxy_status(proxy_summary, repair_round=repair_round)
    result: dict[str, Any] = {
        "row_count": len(rows),
        "target_field": target_field,
        "proxy_fields": list(proxy_fields),
        "proxies": proxies,
    }
    if warm_fields is not None:
        result["warm_start_gain"] = warm_start_gain_summary(
            rows,
            warm_field=warm_fields[0],
            random_field=warm_fields[1],
        )
    if strata_field is not None and strata_threshold is not None:
        result["strata"] = {
            "field": str(strata_field),
            "threshold": float(strata_threshold),
            "proxies": {
                proxy_field: stratified_proxy_summary(
                    rows,
                    proxy_field,
                    target_field,
                    str(strata_field),
                    threshold=float(strata_threshold),
                    k=k,
                )
                for proxy_field in proxy_fields
            },
        }
    if temporal_order_field is not None:
        result["temporal"] = {
            "order_field": str(temporal_order_field),
            "window_size": float(max(1, int(temporal_window_size))),
            "proxies": {
                proxy_field: temporal_analysis(
                    rows,
                    proxy_field,
                    target_field,
                    order_field=str(temporal_order_field),
                    window_size=int(temporal_window_size),
                )
                for proxy_field in proxy_fields
            },
        }
    return result


def _parse_cost_models(raw: str | None, path: str | None = None) -> dict[str, dict[str, float]]:
    if path:
        raw = Path(path).read_text(encoding="utf-8")
    if not raw:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("--cost-models must be a JSON object")
    parsed: dict[str, dict[str, float]] = {}
    for proxy, model in loaded.items():
        if not isinstance(model, Mapping):
            raise ValueError("each cost model must be an object")
        parsed[str(proxy)] = {
            "upfront_cost": float(model.get("upfront_cost", 0.0)),
            "per_arch_cost": float(model.get("per_arch_cost", 1.0)),
        }
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize cheap-evaluation diagnostics from CSV rows.")
    parser.add_argument("--input", required=True, help="CSV file with proxy and fair target columns.")
    parser.add_argument("--target", required=True, help="High-budget fair-VQE target column.")
    parser.add_argument("--proxies", required=True, help="Comma-separated proxy columns, e.g. E1,E2,E5.")
    parser.add_argument("--output", required=True, help="JSON summary output path.")
    parser.add_argument("--cost-models", default=None, help="JSON object keyed by proxy with upfront/per_arch costs.")
    parser.add_argument("--cost-models-file", default=None, help="Path to JSON cost models keyed by proxy.")
    parser.add_argument("--warm-fields", default=None, help="Comma-separated warm/random fair fields, e.g. fair_warm,fair_random.")
    parser.add_argument("--strata-field", default=None, help="Numeric field for hit-rate/exposure stratification.")
    parser.add_argument("--strata-threshold", type=float, default=None, help="Threshold for low/high strata split.")
    parser.add_argument("--temporal-order-field", default=None, help="Order field for temporal warmup analysis.")
    parser.add_argument("--temporal-window-size", type=int, default=20, help="Window size for temporal analysis.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-values", default="10,20,40,80")
    args = parser.parse_args(argv)

    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    proxy_fields = [part.strip() for part in str(args.proxies).split(",") if part.strip()]
    n_values = [int(part.strip()) for part in str(args.n_values).split(",") if part.strip()]
    warm_fields = None
    if args.warm_fields:
        parts = [part.strip() for part in str(args.warm_fields).split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("--warm-fields must contain exactly two comma-separated fields")
        warm_fields = (parts[0], parts[1])
    summary = summarize_proxy_diagnostics(
        rows,
        proxy_fields,
        str(args.target),
        _parse_cost_models(args.cost_models, args.cost_models_file),
        k=int(args.top_k),
        n_values=n_values,
        warm_fields=warm_fields,
        strata_field=args.strata_field,
        strata_threshold=args.strata_threshold,
        temporal_order_field=args.temporal_order_field,
        temporal_window_size=int(args.temporal_window_size),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

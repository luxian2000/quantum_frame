#!/usr/bin/env python
"""Plan and optionally label one or more P1 mutation/oracle/fallback rounds.

The script is intentionally thin: ``plan_p1_round`` owns queue construction,
and ``fair_labeling.py`` owns fair COBYLA labels.  This entry point wires real E2/E5
evaluators to the planner and keeps P1/random/E2-only/E5-only fair budgets equal.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aicir.qas.library.ansatz import SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import architecture_key, resolve_p1_selector_fields, scoped_architecture_key
from aicir.qas.vqe_loop.benchmark_table import read_csv_rows, write_csv_rows
from aicir.qas.vqe_loop.graph_predictor import build_graph_predictor_evaluator
from aicir.qas.vqe_loop.growth_routes import get_growth_route_config
from aicir.qas.vqe_loop.p1_round import plan_p1_round, write_p1_round_outputs
from aicir.qas.vqe_loop.p1_selection import choose_p1_auto_selector
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS
from aicir.qas.vqe_loop.benchmark_table import architecture_from_candidate_row, problem_from_row_terms, problem_from_terms
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float
from aicir.qas.vqe_loop.fair_vqe import optimize_vqe_energy


DEFAULT_PRESET = "h2_sto3g_jw_r0735_4q"
DEFAULT_OUTPUT_DIR = Path("outputs/p1_round_demo")
DEFAULT_PROTOCOL = "default"


def _parse_csv_list(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = [str(part).strip() for part in raw]
    return tuple(value for value in values if value)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("integer list cannot be empty")
    return values



def _mean(values: Sequence[float]) -> float | None:
    items = [float(value) for value in values]
    return sum(items) / float(len(items)) if items else None


def _source_energy_feedback(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    oracle_energies: list[float] = []
    fallback_energies: list[float] = []
    for row in rows:
        energy = _as_float(row.get("fair_best_energy"))
        if energy is None:
            continue
        source = str(row.get("p1_selection_source") or row.get("source") or "").strip().lower()
        if source in {"oracle_trusted", "p1_oracle"}:
            oracle_energies.append(float(energy))
        elif source in {"fallback_selector", "p1_fallback"}:
            fallback_energies.append(float(energy))

    fallback_mean = _mean(fallback_energies)
    oracle_hit_rate = None
    if oracle_energies and fallback_mean is not None:
        oracle_hit_rate = sum(1 for energy in oracle_energies if energy <= fallback_mean) / float(len(oracle_energies))
    return {
        "oracle_mean": _mean(oracle_energies),
        "fallback_mean": fallback_mean,
        "oracle_hit_rate": oracle_hit_rate,
    }


def _canonical_hash(gene: SupernetAnsatzGene) -> str:
    return json.dumps(gene.to_jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _gene_row(
    *,
    architecture_id: str,
    gene: SupernetAnsatzGene,
    preset: str,
    terms: Sequence[tuple[float, str]],
    reference_energy: float,
) -> dict[str, Any]:
    return {
        "architecture_id": architecture_id,
        "canonical_arch_hash": _canonical_hash(gene),
        "protocol_version": "fair_vqe_protocol_v2",
        "source": "p1_demo_control",
        "label_status": "",
        "n_qubits": str(gene.n_qubits),
        "hamiltonian_id": preset,
        "hamiltonian_class": "molecular_preset",
        "family": "supernet_native",
        "depth_group": f"L{gene.layers}",
        "entangler_type": "mixed_supernet",
        "topology": "supernet_pairs",
        "hamiltonian_terms": json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms]),
        "ansatz_gene": json.dumps(gene.to_jsonable(), ensure_ascii=False),
        "reference_energy": "" if reference_energy != reference_energy else f"{float(reference_energy):.12f}",
    }


def _make_control_rows(
    *,
    count: int,
    preset: str,
    terms: Sequence[tuple[float, str]],
    n_qubits: int,
    depth: int,
    reference_energy: float,
    seed: int,
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        return []
    import random

    from aicir.qas.demos.run_p0_diagnostic import (
        default_two_qubit_pairs,
        sample_uniform_supernet_gene,
    )

    rng = random.Random(int(seed))
    pairs = default_two_qubit_pairs(int(n_qubits))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < int(count) and attempts < int(count) * 10:
        attempts += 1
        gene = sample_uniform_supernet_gene(int(n_qubits), int(depth), pairs, rng)
        key = _canonical_hash(gene)
        if key in seen:
            continue
        rows.append(
            _gene_row(
                architecture_id=f"{preset}_control_{len(rows):03d}",
                gene=gene,
                preset=preset,
                terms=terms,
                reference_energy=reference_energy,
            )
        )
        seen.add(key)
    return rows


def load_hamiltonian_for_demo(
    preset: str,
    *,
    hamiltonian_file: str | None = None,
    reference_energy_override: float | None = None,
) -> tuple[tuple[tuple[float, str], ...], int, float]:
    from aicir.qas.demos.run_p0_diagnostic import load_hamiltonian_terms

    return load_hamiltonian_terms(
        preset,
        hamiltonian_file=hamiltonian_file,
        reference_energy_override=reference_energy_override,
    )


def _needed_registry_fields(args: argparse.Namespace) -> tuple[str, ...]:
    fields = set(_parse_csv_list(args.baseline_selectors))
    if str(args.selector).strip().lower() == "auto":
        fields.update(_parse_csv_list(args.auto_selector_candidates))
    else:
        fields.update(resolve_p1_selector_fields(args.selector, cheap_eval_selector=args.cheap_eval_selector))
    normalized: set[str] = set()
    for field in fields:
        upper = str(field).strip().upper()
        if upper in {"E2", "E5", "VQE_TASK_PROXY", "GNN_PROXY", "ENSEMBLE"}:
            normalized.add(upper)
    if "ENSEMBLE" in normalized:
        normalized.update({"E2", "VQE_TASK_PROXY", "GNN_PROXY"})
    return tuple(sorted(normalized))


def _option_is_explicit(argv: Sequence[str], option: str) -> bool:
    return any(str(token) == option or str(token).startswith(f"{option}=") for token in argv)


def _apply_growth_route_defaults(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if not _option_is_explicit(argv, "--growth-route"):
        return
    config = get_growth_route_config(args.growth_route)
    defaults = (
        ("--rounds", "rounds", config.rounds),
        ("--early-stop-epsilon", "early_stop_epsilon", config.early_stop_epsilon),
        ("--early-stop-patience", "early_stop_patience", config.early_stop_patience),
        ("--max-total-fair-calls", "max_total_fair_calls", config.max_total_fair_calls),
        ("--parent-count", "parent_count", config.parent_count),
        ("--diversity-count", "diversity_count", config.diversity_count),
        ("--children-per-parent", "children_per_parent", config.children_per_parent),
        ("--fair-top-k", "fair_top_k", config.fair_top_k),
        ("--selector", "selector", config.selector),
        ("--baseline-selectors", "baseline_selectors", ",".join(config.baseline_selectors)),
        ("--mutation-types", "mutation_types", ",".join(config.mutation_types)),
        ("--min-layers", "min_layers", config.min_layers),
        ("--max-layers", "max_layers", config.max_layers),
        ("--operator-pool-limit", "operator_pool_limit", config.operator_pool_limit),
        ("--chemistry-adapt-append-k", "chemistry_adapt_append_k", config.chemistry_adapt_append_k),
        ("--chemistry-adapt-pool-limit", "chemistry_adapt_pool_limit", config.chemistry_adapt_pool_limit),
    )
    for option, destination, value in defaults:
        if not _option_is_explicit(argv, option):
            setattr(args, destination, value)
    if config.name == "line_a_operator_sequence":
        if not _option_is_explicit(argv, "--operator-genetic-weight"):
            args.operator_genetic_weight = config.genetic_weight
        if not _option_is_explicit(argv, "--operator-adapt-growth-weight"):
            args.operator_adapt_growth_weight = config.adapt_growth_weight
    else:
        if not _option_is_explicit(argv, "--chemistry-genetic-weight"):
            args.chemistry_genetic_weight = config.genetic_weight
        if not _option_is_explicit(argv, "--chemistry-adapt-growth-weight"):
            args.chemistry_adapt_growth_weight = config.adapt_growth_weight


def _route_mutation_weights(args: argparse.Namespace, mutation_types: Sequence[str]) -> dict[str, float] | None:
    mutation_set = tuple(str(item).strip().lower() for item in mutation_types if str(item).strip())
    if not mutation_set:
        return None
    route = str(args.growth_route).strip().lower()
    config = get_growth_route_config(route)
    documented_genetic_weights = config.genetic_weights()
    if route == "line_a_operator_sequence":
        genetic = tuple(item for item in mutation_set if item.startswith("operator_") and item != "operator_adapt_growth")
        weights: dict[str, float] = {}
        if genetic and float(args.operator_genetic_weight) > 0.0:
            total = sum(documented_genetic_weights.get(item, 0.0) for item in genetic)
            if total > 0.0:
                weights.update(
                    {
                        item: float(args.operator_genetic_weight) * documented_genetic_weights[item] / total
                        for item in genetic
                        if item in documented_genetic_weights
                    }
                )
        if "operator_adapt_growth" in mutation_set and float(args.operator_adapt_growth_weight) > 0.0:
            weights["operator_adapt_growth"] = float(args.operator_adapt_growth_weight)
        return weights or None
    if route == "line_b_chemistry_excitation":
        mode = str(args.chemistry_growth_mode).strip().lower()
        genetic_weight = 0.0 if mode == "adapt" else float(args.chemistry_genetic_weight)
        adapt_weight = 0.0 if mode == "genetic" else float(args.chemistry_adapt_growth_weight)
        genetic = tuple(item for item in mutation_set if item.startswith("chemistry_") and item != "chemistry_adapt_growth")
        weights: dict[str, float] = {}
        if genetic and genetic_weight > 0.0:
            total = sum(documented_genetic_weights.get(item, 0.0) for item in genetic)
            if total > 0.0:
                weights.update(
                    {
                        item: genetic_weight * documented_genetic_weights[item] / total
                        for item in genetic
                        if item in documented_genetic_weights
                    }
                )
        if "chemistry_adapt_growth" in mutation_set and adapt_weight > 0.0:
            weights["chemistry_adapt_growth"] = adapt_weight
        return weights or None
    return None

def _build_shared_vqe_e2_evaluator(
    args: argparse.Namespace,
    terms: Sequence[tuple[float, str]],
    n_qubits: int,
    reference_energy: float,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Low-budget shared VQE selector for non-supernet ansatz families."""

    from aicir.qas.core.backend_utils import resolve_qas_backend

    default_problem = problem_from_terms(
        terms,
        n_qubits=int(n_qubits),
        name=str(getattr(args, "preset", "p1_e2")),
        reference_energy=float(reference_energy),
    )
    seed_offsets = _parse_int_tuple(args.proxy_seed_offsets)
    starts = max(1, len(seed_offsets))
    max_evals = max(1, int(args.e2_max_evals))

    def evaluator(row: Mapping[str, Any]) -> Mapping[str, Any]:
        mutable_row = dict(row)
        architecture = architecture_from_candidate_row(mutable_row)
        problem = problem_from_row_terms(
            mutable_row,
            n_qubits=int(mutable_row.get("n_qubits") or architecture.n_qubits),
            default_problem=default_problem,
            default_name_prefix="p1_e2",
        )
        backend = resolve_qas_backend(
            kind=str(getattr(args, "backend", None) or "numpy"),
            fallback_to_cpu=False,
            dtype=str(getattr(args, "dtype", None) or "complex128"),
        )
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=int(args.seed),
            n_starts=starts,
            evals_per_param=10,
            max_evaluations=max_evals,
            budget_override=max_evals,
            backend=backend,
            init_mode="zero_then_random",
        )
        return {
            "E2": float(result.energy),
            "E2_nfev": int(result.evaluations),
            "E2_n_starts": int(result.n_starts),
            "E2_budget_per_start": result.metadata.get("budget_per_start", max_evals),
            "E2_selector_engine": "shared_vqe_low_budget",
        }

    return evaluator

def build_real_evaluator_registry(
    args: argparse.Namespace,
    terms: Sequence[tuple[float, str]],
    n_qubits: int,
    reference_energy: float,
) -> dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    """Build real E2/E5 evaluators for H2/LiH/larger molecular presets."""

    from aicir.qas.demos.run_p0_diagnostic import (
        SINGLE_QUBIT_CHOICES,
        build_light_vqe_evaluator_registry,
        build_torch_pauli_evaluator_registry,
    )
    from aicir.qas.vqe_loop.ansatz_family import ansatz_family_from_row
    from aicir.qas.vqe_loop.p0_supernet_native import build_native_supernet_e5_evaluator
    from aicir.qas.vqe_loop.task_proxy import build_vqe_task_proxy_evaluator

    needed = set(_needed_registry_fields(args))
    registry: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
    proxy_seed_offsets = _parse_int_tuple(args.proxy_seed_offsets)
    fair_seed_offsets = _parse_int_tuple(args.fair_seed_offsets)

    if "E2" in needed:
        if args.light_evaluator == "torch_pauli":
            light_registry, _fair = build_torch_pauli_evaluator_registry(
                e1_max_evals=args.e1_max_evals,
                e2_max_evals=args.e2_max_evals,
                seed=args.seed,
                proxy_seed_offsets=proxy_seed_offsets,
                fair_seed_offsets=fair_seed_offsets,
                fair_max_evals=args.fair_max_evals,
                device=args.device,
            )
        else:
            light_registry, _fair = build_light_vqe_evaluator_registry(
                problem=None,
                e1_max_evals=args.e1_max_evals,
                e2_max_evals=args.e2_max_evals,
                fair_max_evals=args.fair_max_evals or 1000,
                seed=args.seed,
                n_starts=1,
                proxy_seed_offsets=proxy_seed_offsets,
            )
        supernet_e2 = light_registry["E2"]
        shared_vqe_e2 = _build_shared_vqe_e2_evaluator(args, terms, n_qubits, reference_energy)

        def e2(row: Mapping[str, Any]) -> Mapping[str, Any]:
            family = ansatz_family_from_row(row)
            if family in {"chemistry_excitation", "operator_sequence", "explicit_gate_sequence"}:
                return shared_vqe_e2(row)
            return supernet_e2(row)

        registry["E2"] = e2

    if "E5" in needed:
        registry["E5"] = build_native_supernet_e5_evaluator(
            problem=None,
            supernet_num=args.e5_supernet_num,
            supernet_steps=args.e5_supernet_steps,
            finetune_steps=args.e5_finetune_steps,
            seed=args.seed,
            single_qubit_gates=SINGLE_QUBIT_CHOICES,
            two_qubit_gates=("cx", "rzz"),
        )

    if "VQE_TASK_PROXY" in needed:
        registry["VQE_TASK_PROXY"] = build_vqe_task_proxy_evaluator(operator_pool=_parse_csv_list(args.operator_pool))

    if "GNN_PROXY" in needed:
        bootstrap_rows = read_csv_rows(args.bootstrap_labels_csv) if args.bootstrap_labels_csv else []
        registry["GNN_PROXY"] = build_graph_predictor_evaluator(bootstrap_rows)

    return registry


def run_labeling_queue(
    *,
    queue_path: str | Path,
    output_path: str | Path,
    protocol: str,
    seed: int,
    n_seeds: int,
    max_evals: int | None,
    backend: str,
    dtype: str,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "aicir.qas.vqe_loop.fair_labeling",
        "--queue",
        str(queue_path),
        "--output",
        str(output_path),
        "--protocol",
        str(protocol),
        "--seed",
        str(int(seed)),
        "--n-seeds",
        str(int(n_seeds)),
        "--backend",
        str(backend),
        "--dtype",
        str(dtype),
        "--seed-by-architecture-id",
    ]
    if max_evals is not None:
        command.extend(["--max-evals", str(int(max_evals))])
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=str(REPO), check=True)


def _completed_rows(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [row for row in rows if _as_float(row.get("fair_best_energy")) is not None]


def _round_from_row(row: Mapping[str, Any]) -> int | None:
    batch_id = str(row.get("batch_id", ""))
    match = re.search(r"(?:^|_)r(\d+)(?:$|_)", batch_id)
    if match:
        return int(match.group(1))
    match = re.search(r"round(\d+)", batch_id)
    return int(match.group(1)) if match else None


def compare_label_outputs(label_paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Compare fair labels from equal-budget P1/random/E2/E5 queues."""

    rows_by_strategy = {name: _completed_rows(path) for name, path in label_paths.items()}
    best_energy_by_arch: dict[str, float] = {}
    for rows in rows_by_strategy.values():
        for index, row in enumerate(rows):
            architecture_id = str(row.get("architecture_id") or f"row:{index}")
            energy = _as_float(row.get("fair_best_energy"))
            if energy is None:
                continue
            best_energy_by_arch[architecture_id] = min(energy, best_energy_by_arch.get(architecture_id, energy))

    global_rank = {
        architecture_id: rank + 1
        for rank, (architecture_id, _energy) in enumerate(
            sorted(best_energy_by_arch.items(), key=lambda item: (item[1], item[0]))
        )
    }

    strategies: dict[str, Any] = {}
    for name, rows in rows_by_strategy.items():
        completed = [
            (float(row["fair_best_energy"]), row)
            for row in rows
            if _as_float(row.get("fair_best_energy")) is not None
        ]
        energies = [energy for energy, _row in completed]
        ids = [str(row.get("architecture_id") or f"row:{index}") for index, row in enumerate(rows)]
        top_k_ids = {architecture_id for architecture_id, rank in global_rank.items() if rank <= max(1, len(rows))}
        proxy_best_id = ids[0] if ids else ""
        best_row = min(completed, key=lambda item: item[0])[1] if completed else {}
        best = min(energies) if energies else None
        mean = sum(energies) / len(energies) if energies else None
        med = median(energies) if energies else None
        strategies[name] = {
            "fair_call_count": len(energies),
            "selected_architecture_ids": ids,
            "fair_best_architecture_id": best_row.get("architecture_id", ""),
            "fair_best_batch_id": best_row.get("batch_id", ""),
            "fair_best_round": _round_from_row(best_row) if best_row else None,
            "fair_best_in_queue": best,
            "fair_mean_in_queue": mean,
            "fair_median_in_queue": med,
            "fair_best_in_topK": best,
            "fair_mean_in_topK": mean,
            "fair_median_in_topK": med,
            "fair_topK_hit": len(set(ids) & top_k_ids),
            "fair_rank_of_proxy_best": global_rank.get(proxy_best_id),
        }

    comparable = [
        (metrics["fair_best_in_queue"], name)
        for name, metrics in strategies.items()
        if metrics["fair_best_in_queue"] is not None
    ]
    best_strategy = min(comparable, key=lambda item: (item[0], item[1]))[1] if comparable else None
    return {"strategies": strategies, "best_strategy": best_strategy}


def _load_bootstrap_labels(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.bootstrap_labels_csv:
        raise ValueError("--bootstrap-labels-csv is required for this P1 round demo")
    rows = read_csv_rows(args.bootstrap_labels_csv)
    completed = [row for row in rows if _as_float(row.get("fair_best_energy")) is not None]
    if not completed:
        raise ValueError("--bootstrap-labels-csv must contain at least one completed fair_best_energy label")
    return completed


def _queue_paths(paths: Mapping[str, Path], baseline_selectors: Sequence[str]) -> dict[str, str]:
    queues = {"p1": str(paths["queue"]), "random": str(paths["baseline_random"])}
    discovered = [
        key.removeprefix("baseline_")
        for key in paths
        if key.startswith("baseline_") and key != "baseline_random"
    ]
    selectors = dict.fromkeys([str(selector) for selector in baseline_selectors] + discovered)
    for selector in selectors:
        key = f"baseline_{selector}"
        if key in paths:
            queues[str(selector)] = str(paths[key])
    return queues


def _append_unique_completed_labels(
    labeled_rows: list[dict[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    existing = {scoped_architecture_key(row) for row in labeled_rows if architecture_key(row)}
    appended: list[dict[str, Any]] = []
    for row in new_rows:
        if _as_float(row.get("fair_best_energy")) is None:
            continue
        key = scoped_architecture_key(row)
        if not key or key in existing:
            continue
        copied = dict(row)
        labeled_rows.append(copied)
        appended.append(copied)
        existing.add(key)
    return appended


def _append_unique_known_unlabeled(
    known_rows: list[dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    existing = {scoped_architecture_key(row) for row in known_rows if architecture_key(row)}
    appended: list[dict[str, Any]] = []
    for row in rows:
        if _as_float(row.get("fair_best_energy")) is not None:
            continue
        key = scoped_architecture_key(row)
        if not architecture_key(row) or key in existing:
            continue
        copied = dict(row)
        known_rows.append(copied)
        appended.append(copied)
        existing.add(key)
    return appended


def _merge_key(row: Mapping[str, Any]) -> str:
    return architecture_key(row) or str(row.get("architecture_id", ""))


def _prefer_label_row(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_energy = _as_float(left.get("fair_best_energy"))
    right_energy = _as_float(right.get("fair_best_energy"))
    if right_energy is not None and (left_energy is None or right_energy < left_energy):
        return dict(right)
    return dict(left)


def _aggregate_csvs(paths: Sequence[str | Path], output_path: str | Path, *, deduplicate: bool = True) -> str:
    rows_by_key: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv_rows(path):
            if not deduplicate:
                rows.append(dict(row))
                continue
            key = _merge_key(row)
            if not key:
                rows.append(dict(row))
            elif key in rows_by_key:
                rows_by_key[key] = _prefer_label_row(rows_by_key[key], row)
            else:
                rows_by_key[key] = dict(row)
    if deduplicate:
        rows.extend(rows_by_key.values())
    write_csv_rows(output_path, rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    return str(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan P1 mutation/oracle/fallback rounds and optionally run equal-budget fair labels.",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="Chemistry preset, e.g. H2, LiH, or a larger molecule preset.")
    parser.add_argument("--hamiltonian-file", default=None, help="Optional JSON Pauli-term list for larger/custom molecules.")
    parser.add_argument("--reference-energy", type=float, default=None)
    parser.add_argument("--bootstrap-labels-csv", default=None, help="Completed fair labels used as P1 parents/oracle training data.")
    parser.add_argument("--known-unlabeled-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rounds", type=int, default=1, help="Number of P1 mutation/oracle/fallback rounds to run.")
    parser.add_argument("--early-stop-epsilon", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-total-fair-calls", type=int, default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--parent-count", type=int, default=4)
    parser.add_argument("--diversity-count", type=int, default=0)
    parser.add_argument("--children-per-parent", type=int, default=8)
    parser.add_argument("--control-count", type=int, default=0)
    parser.add_argument("--fair-top-k", type=int, default=4)
    parser.add_argument("--selector", choices=("e2", "e5", "both", "task_proxy", "gnn_proxy", "ensemble", "auto"), default="e5")
    parser.add_argument("--cheap-eval-selector", choices=("e2", "e5"), default="e2")
    parser.add_argument("--auto-selector-candidates", default="E2,E5,VQE_TASK_PROXY,GNN_PROXY,ENSEMBLE")
    parser.add_argument("--auto-selector-top-k", type=int, default=3)
    parser.add_argument("--auto-selector-min-completed", type=int, default=3)
    parser.add_argument("--auto-selector-fallback", choices=("e2", "e5", "task_proxy", "gnn_proxy", "ensemble"), default="e2")
    parser.add_argument("--baseline-selectors", default="E2,E5")
    parser.add_argument(
        "--mutation-types",
        default="gate_mutation,connectivity_mutation,layer_mutation,depth_mutation",
        help=(
            "Comma-separated P1 mutations. operator_adapt_growth is opt-in and only applies "
            "to OperatorSequenceAnsatzGene rows; pass it explicitly together with --operator-pool."
        ),
    )
    parser.add_argument(
        "--operator-pool",
        default="",
        help="Comma-separated Pauli strings used by operator_insert/operator_big_mutation, e.g. XI,YY,ZZ.",
    )
    parser.add_argument("--operator-pool-limit", type=int, default=None)
    parser.add_argument("--growth-route", choices=("line_a_operator_sequence", "line_b_chemistry_excitation"), default="line_a_operator_sequence")
    parser.add_argument("--operator-genetic-weight", type=float, default=0.5)
    parser.add_argument("--operator-adapt-growth-weight", type=float, default=0.5)
    parser.add_argument("--chemistry-genetic-weight", type=float, default=0.5)
    parser.add_argument("--chemistry-adapt-growth-weight", type=float, default=0.5)
    parser.add_argument("--chemistry-growth-mode", choices=("genetic", "adapt", "mixed"), default="mixed")
    parser.add_argument("--chemistry-adapt-append-k", type=int, default=1)
    parser.add_argument("--chemistry-adapt-pool-limit", type=int, default=None)
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=0.1)
    parser.add_argument(
        "--fallback-audit-multiplier",
        type=int,
        default=4,
        help="When oracle trusted rows exceed abstain rows, score up to multiplier*K trusted rows with the fallback selector.",
    )
    parser.add_argument("--selection-policy", choices=("no_regret", "no_regret_lite", "quota"), default="no_regret")
    parser.add_argument("--oracle-extra-top-k", type=int, default=0)
    parser.add_argument("--oracle-max-neighbor-std", type=float, default=None)
    parser.add_argument("--min-previous-oracle-hit-rate", type=float, default=0.5)
    parser.set_defaults(enable_training_free_pruning=True)
    parser.add_argument("--enable-training-free-pruning", dest="enable_training_free_pruning", action="store_true")
    parser.add_argument("--disable-training-free-pruning", dest="enable_training_free_pruning", action="store_false")
    parser.add_argument("--training-free-soft-prefilter-multiplier", type=int, default=2)
    parser.add_argument("--trainability-soft-quantile", type=float, default=0.10)
    parser.add_argument("--expressibility-soft-quantile", type=float, default=0.05)
    parser.add_argument("--trainability-hard-floor", type=float, default=0.01)
    parser.add_argument("--expressibility-hard-floor", type=float, default=0.01)
    parser.add_argument("--entanglement-soft-floor", type=float, default=0.0)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--max-two-q", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-id", default="p1_round_demo")
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--light-evaluator", choices=("basic_vqe", "torch_pauli"), default="torch_pauli")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--e1-max-evals", type=int, default=20)
    parser.add_argument("--e2-max-evals", type=int, default=250)
    parser.add_argument("--e5-supernet-num", type=int, default=5)
    parser.add_argument("--e5-supernet-steps", type=int, default=250)
    parser.add_argument("--e5-finetune-steps", type=int, default=250)
    parser.add_argument("--proxy-seed-offsets", default="0,17,43")
    parser.add_argument("--fair-seed-offsets", default="1000")
    parser.add_argument("--fair-max-evals", type=int, default=None, help="Fair COBYLA max-evals used for every queue label run.")
    parser.add_argument("--run-labeling", action="store_true")
    parser.add_argument("--dry-run-labels", action="store_true")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--label-seed", type=int, default=2026)
    parser.add_argument("--label-n-seeds", type=int, default=3)
    parser.add_argument(
        "--label-seed-step",
        type=int,
        default=0,
        help="Optional per-queue seed increment; default 0 keeps shared architectures directly comparable.",
    )
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dtype", default="complex128")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_registry_builder: Callable[[argparse.Namespace, Sequence[tuple[float, str]], int, float], Mapping[str, Any]] = build_real_evaluator_registry,
    labeling_runner: Callable[..., None] = run_labeling_queue,
    hamiltonian_loader: Callable[..., tuple[Sequence[tuple[float, str]], int, float]] = load_hamiltonian_for_demo,
) -> dict[str, Any]:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    _apply_growth_route_defaults(args, raw_argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terms, n_qubits, reference_energy = hamiltonian_loader(
        args.preset,
        hamiltonian_file=args.hamiltonian_file,
        reference_energy_override=args.reference_energy,
    )
    terms = tuple((float(coeff), str(pauli)) for coeff, pauli in terms)
    baseline_selectors = _parse_csv_list(args.baseline_selectors)
    mutation_types = _parse_csv_list(args.mutation_types)
    mutation_weights = _route_mutation_weights(args, mutation_types)
    operator_pool = _parse_csv_list(args.operator_pool)
    labeled_rows = _load_bootstrap_labels(args)
    requested_selector = str(args.selector)
    auto_selector_decision = None
    if requested_selector.lower() == "auto":
        auto_selector_decision = choose_p1_auto_selector(
            labeled_rows,
            candidates=_parse_csv_list(args.auto_selector_candidates),
            top_k=int(args.auto_selector_top_k),
            min_completed=int(args.auto_selector_min_completed),
            fallback_selector=str(args.auto_selector_fallback),
        )
        args.selector = auto_selector_decision.selector
    known_unlabeled_rows = read_csv_rows(args.known_unlabeled_csv) if args.known_unlabeled_csv else []
    control_rows = _make_control_rows(
        count=args.control_count,
        preset=args.preset,
        terms=terms,
        n_qubits=int(n_qubits),
        depth=int(args.depth),
        reference_energy=float(reference_energy),
        seed=int(args.seed) + 7919,
    )
    registry = dict(evaluator_registry_builder(args, terms, int(n_qubits), float(reference_energy)))
    from aicir.qas.core.backend_utils import resolve_qas_backend

    growth_backend = resolve_qas_backend(
        kind=str(args.backend),
        fallback_to_cpu=False,
        dtype=str(args.dtype),
    )
    rounds = max(1, int(args.rounds))
    if rounds > 1 and not args.run_labeling:
        raise ValueError("--rounds > 1 requires --run-labeling so P1 fair labels can feed the next round")

    current_labeled_rows = [dict(row) for row in labeled_rows]
    round_results: list[dict[str, Any]] = []
    aggregate_queue_paths: dict[str, list[str]] = {}
    aggregate_label_paths: dict[str, list[str]] = {}
    label_seeds: dict[str, int] = {}
    label_errors: list[dict[str, Any]] = []
    p1_fair_calls_so_far = 0
    bootstrap_energies = [
        float(row["fair_best_energy"])
        for row in current_labeled_rows
        if _as_float(row.get("fair_best_energy")) is not None
    ]
    best_p1_energy: float | None = min(bootstrap_energies) if bootstrap_energies else None
    plateau_count = 0
    stop_reason: str | None = None
    fair_best_by_round: list[float | None] = []
    fair_best_improvement_by_round: list[float | None] = []
    previous_oracle_trusted_fair_mean: float | None = None
    previous_fallback_fair_mean: float | None = None
    previous_oracle_hit_rate: float | None = None
    oracle_hit_rate_by_round: list[float | None] = []
    oracle_trusted_fair_mean_by_round: list[float | None] = []
    fallback_fair_mean_by_round: list[float | None] = []

    for round_index in range(rounds):
        round_number = round_index + 1

        round_fair_top_k = max(0, int(args.fair_top_k))
        round_oracle_extra_top_k = max(0, int(args.oracle_extra_top_k))
        if args.max_total_fair_calls is not None:
            remaining_fair_calls = max(0, int(args.max_total_fair_calls) - p1_fair_calls_so_far)
            if remaining_fair_calls <= 0:
                stop_reason = "max_total_fair_calls"
                break
            round_fair_top_k = min(round_fair_top_k, remaining_fair_calls)
            round_oracle_extra_top_k = min(
                round_oracle_extra_top_k,
                max(0, remaining_fair_calls - round_fair_top_k),
            )

        round_output_dir = output_dir if rounds == 1 else output_dir / f"round{round_number}"
        round_batch_id = str(args.batch_id) if rounds == 1 else f"{args.batch_id}_r{round_number}"
        labeled_count_before_round = len(current_labeled_rows)
        plan = plan_p1_round(
            labeled_rows=current_labeled_rows,
            known_unlabeled_rows=known_unlabeled_rows,
            control_rows=control_rows,
            parent_count=args.parent_count,
            diversity_count=args.diversity_count,
            children_per_parent=args.children_per_parent,
            fair_top_k=round_fair_top_k,
            selector=args.selector,
            cheap_eval_selector=args.cheap_eval_selector,
            evaluator_registry=registry,
            k_min=args.k_min,
            d_max=args.d_max,
            batch_id=round_batch_id,
            protocol_version=args.protocol_version,
            mutation_types=mutation_types,
            mutation_weights=mutation_weights,
            operator_pool=operator_pool or None,
            operator_pool_limit=args.operator_pool_limit,
            chemistry_adapt_append_k=int(args.chemistry_adapt_append_k),
            chemistry_adapt_pool_limit=args.chemistry_adapt_pool_limit,
            growth_backend=growth_backend,
            min_layers=args.min_layers,
            max_layers=args.max_layers,
            seed=int(args.seed) + round_index,
            baseline_selector_fields=baseline_selectors,
            previous_oracle_trusted_fair_mean=previous_oracle_trusted_fair_mean,
            previous_fallback_fair_mean=previous_fallback_fair_mean,
            previous_oracle_hit_rate=previous_oracle_hit_rate,
            enable_training_free_pruning=bool(args.enable_training_free_pruning),
            training_free_soft_prefilter_multiplier=args.training_free_soft_prefilter_multiplier,
            fallback_audit_multiplier=args.fallback_audit_multiplier,
            selection_policy=args.selection_policy,
            oracle_extra_top_k=round_oracle_extra_top_k,
            min_previous_oracle_hit_rate=args.min_previous_oracle_hit_rate,
            oracle_max_neighbor_std=args.oracle_max_neighbor_std,
            trainability_soft_quantile=args.trainability_soft_quantile,
            expressibility_soft_quantile=args.expressibility_soft_quantile,
            trainability_hard_floor=args.trainability_hard_floor,
            expressibility_hard_floor=args.expressibility_hard_floor,
            entanglement_soft_floor=args.entanglement_soft_floor,
            max_params=args.max_params,
            max_two_q=args.max_two_q,
        )
        planned_p1_fair_calls = int(plan.summary.get("budget", {}).get("total_fair_calls", len(plan.queue_rows)))
        if not plan.child_rows:
            stop_reason = "no_new_child"
            break
        if planned_p1_fair_calls <= 0:
            stop_reason = "no_fair_candidates"
            break
        paths = write_p1_round_outputs(plan, round_output_dir)
        round_queues = _queue_paths(paths, baseline_selectors)
        for name, queue_path in round_queues.items():
            aggregate_queue_paths.setdefault(str(name), []).append(str(queue_path))

        round_label_paths: dict[str, str] = {}
        if args.run_labeling:
            for index, (name, queue_path) in enumerate(round_queues.items()):
                output_path = round_output_dir / f"labels_{str(name).lower()}.csv"
                label_seed = int(args.label_seed)
                if int(args.label_seed_step) != 0:
                    label_seed += index * int(args.label_seed_step)
                label_seeds[str(name)] = label_seed
                try:
                    labeling_runner(
                        queue_path=queue_path,
                        output_path=output_path,
                        protocol=args.protocol,
                        seed=label_seed,
                        n_seeds=args.label_n_seeds,
                        max_evals=args.fair_max_evals,
                        backend=args.backend,
                        dtype=args.dtype,
                        dry_run=args.dry_run_labels,
                    )
                except Exception as exc:
                    label_errors.append(
                        {
                            "round": round_number,
                            "strategy": str(name),
                            "queue": str(queue_path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if str(name) == "p1" and output_path.exists():
                        _append_unique_known_unlabeled(known_unlabeled_rows, read_csv_rows(output_path))
                    continue
                round_label_paths[str(name)] = str(output_path)
                aggregate_label_paths.setdefault(str(name), []).append(str(output_path))
            p1_fair_calls_so_far += planned_p1_fair_calls

        p1_labels_added: list[dict[str, Any]] = []
        round_p1_best: float | None = None
        round_improvement: float | None = None
        if args.run_labeling and "p1" in round_label_paths:
            round_p1_output_rows = read_csv_rows(round_label_paths["p1"])
            _append_unique_known_unlabeled(known_unlabeled_rows, round_p1_output_rows)
            round_p1_rows = [
                row for row in round_p1_output_rows if _as_float(row.get("fair_best_energy")) is not None
            ]
            feedback = _source_energy_feedback(round_p1_rows)
            previous_oracle_trusted_fair_mean = feedback["oracle_mean"]
            previous_fallback_fair_mean = feedback["fallback_mean"]
            previous_oracle_hit_rate = feedback["oracle_hit_rate"]
            oracle_hit_rate_by_round.append(previous_oracle_hit_rate)
            oracle_trusted_fair_mean_by_round.append(previous_oracle_trusted_fair_mean)
            fallback_fair_mean_by_round.append(previous_fallback_fair_mean)
            round_energies = [
                float(row["fair_best_energy"])
                for row in round_p1_rows
                if _as_float(row.get("fair_best_energy")) is not None
            ]
            round_p1_best = min(round_energies) if round_energies else None
            p1_labels_added = _append_unique_completed_labels(
                current_labeled_rows,
                round_p1_rows,
            )
            if round_p1_best is not None:
                if best_p1_energy is None:
                    best_p1_energy = round_p1_best
                else:
                    candidate_best = min(best_p1_energy, round_p1_best)
                    round_improvement = best_p1_energy - candidate_best
                    best_p1_energy = candidate_best
                    if round_improvement < float(args.early_stop_epsilon):
                        plateau_count += 1
                    else:
                        plateau_count = 0
        elif args.run_labeling:
            oracle_hit_rate_by_round.append(None)
            oracle_trusted_fair_mean_by_round.append(None)
            fallback_fair_mean_by_round.append(None)
            stop_reason = "p1_labeling_failed"
        else:
            oracle_hit_rate_by_round.append(None)
            oracle_trusted_fair_mean_by_round.append(None)
            fallback_fair_mean_by_round.append(None)

        round_results.append(
            {
                "round": round_number,
                "batch_id": round_batch_id,
                "output_dir": str(round_output_dir),
                "queues": round_queues,
                "labels": round_label_paths,
                "summary": plan.summary,
                "labeled_count_before_round": labeled_count_before_round,
                "p1_labels_added": len(p1_labels_added),
                "p1_fair_best": round_p1_best,
                "p1_fair_best_improvement": round_improvement,
                "plateau_count": plateau_count,
            }
        )
        fair_best_by_round.append(round_p1_best)
        fair_best_improvement_by_round.append(round_improvement)
        if stop_reason == "p1_labeling_failed":
            break
        if int(args.early_stop_patience) > 0 and plateau_count >= int(args.early_stop_patience):
            stop_reason = "plateau"
            break

    if stop_reason is None:
        stop_reason = "max_rounds"

    queues: dict[str, str]
    label_paths: dict[str, str] = {}
    comparison: dict[str, Any] | None = None
    if rounds == 1 and round_results:
        queues = dict(round_results[0]["queues"])
        label_paths = dict(round_results[0]["labels"])
    elif rounds == 1:
        queues = {}
    else:
        queues = {
            name: _aggregate_csvs(
                paths,
                output_dir / ("p1_queue.csv" if name == "p1" else f"queue_{str(name).lower()}_baseline.csv"),
            )
            for name, paths in aggregate_queue_paths.items()
        }
        if args.run_labeling:
            label_paths = {
                name: _aggregate_csvs(paths, output_dir / f"labels_{str(name).lower()}.csv")
                for name, paths in aggregate_label_paths.items()
            }

    if args.run_labeling:
        comparison = compare_label_outputs(label_paths)
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_csv_rows(output_dir / "p1_benchmark_table.csv", current_labeled_rows, fieldnames=BENCHMARK_TABLE_FIELDS)

    actual_fair_calls = {
        name: len(read_csv_rows(path))
        for name, path in label_paths.items()
    } if args.run_labeling else {}
    p1_plan_fair_calls_by_round = [
        int(item["summary"].get("budget", {}).get("total_fair_calls", 0))
        for item in round_results
    ]
    p1_fallback_fair_calls_by_round = [
        int(item["summary"].get("budget", {}).get("fallback_fair_calls", 0))
        for item in round_results
    ]
    p1_oracle_extra_fair_calls_by_round = [
        int(item["summary"].get("budget", {}).get("oracle_extra_fair_calls", 0))
        for item in round_results
    ]
    multi_round_summary = {
        "rounds": rounds,
        "fair_top_k_per_round": int(args.fair_top_k),
        "expected_total_fair_calls_per_strategy": sum(p1_plan_fair_calls_by_round),
        "actual_fair_calls_per_strategy": actual_fair_calls,
        "p1_plan_fair_calls_by_round": p1_plan_fair_calls_by_round,
        "p1_fallback_fair_calls_by_round": p1_fallback_fair_calls_by_round,
        "p1_oracle_extra_fair_calls_by_round": p1_oracle_extra_fair_calls_by_round,
        "rounds_completed": len(round_results),
        "stop_reason": stop_reason,
        "early_stop_epsilon": float(args.early_stop_epsilon),
        "early_stop_patience": int(args.early_stop_patience),
        "max_total_fair_calls": args.max_total_fair_calls,
        "fair_best_by_round": fair_best_by_round,
        "fair_best_improvement_by_round": fair_best_improvement_by_round,
        "oracle_trusted_by_round": [item["summary"].get("n_oracle_trusted", 0) for item in round_results],
        "oracle_abstain_by_round": [item["summary"].get("n_oracle_abstain", 0) for item in round_results],
        "oracle_hit_rate_by_round": oracle_hit_rate_by_round,
        "oracle_trusted_fair_mean_by_round": oracle_trusted_fair_mean_by_round,
        "fallback_fair_mean_by_round": fallback_fair_mean_by_round,
        "p1_labels_added_by_round": [item["p1_labels_added"] for item in round_results],
        "labeled_count_before_round": [item["labeled_count_before_round"] for item in round_results],
        "round_summaries": [item["summary"] for item in round_results],
    }

    result = {
        "output_dir": str(output_dir),
        "preset": args.preset,
        "n_qubits": int(n_qubits),
        "reference_energy": float(reference_energy),
        "queues": queues,
        "labels": label_paths,
        "label_seeds": label_seeds,
        "label_errors": label_errors,
        "comparison": comparison,
        "summary": round_results[0]["summary"] if rounds == 1 and round_results else multi_round_summary,
        "rounds": round_results,
        "requested_selector": requested_selector,
        "resolved_selector": str(args.selector),
        "auto_selector": auto_selector_decision.to_jsonable() if auto_selector_decision is not None else None,
    }
    (output_dir / "demo_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    main()



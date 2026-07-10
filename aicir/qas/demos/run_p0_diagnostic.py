#!/usr/bin/env python
"""P0 cheap-eval diagnostic: 4q H2, L3, E1/E2/E5/fair_high.

Reusable entry point: change PRESET / DEPTH / N_ARCHITECTURES to run
LiH, Ising, or other benchmarks with the same pipeline.

Usage (inside Docker qml_env):

    /opt/conda/envs/qml_env/bin/python \
        -m aicir.qas.demos.run_p0_diagnostic

Or with custom config:

    /opt/conda/envs/qml_env/bin/python \
        -m aicir.qas.demos.run_p0_diagnostic \
        --n-architectures 40 --depth 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from aicir.qas.library.ansatz import SupernetAnsatzGene
from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.cheap_eval_experiment import (
    CheapEvalExperimentConfig,
    EXPERIMENT_FIELDS,
    _architecture_from_experiment_row,
    _hamiltonian_terms_from_experiment_row,
    _mean_min_std,
    _resolve_initial_parameters,
    _row_order_seed_offset,
    _supernet_architecture_from_gene,
    _supernet_gene_from_experiment_row,
    _supernet_gene_static_counts,
    _run_vqe_proxy,
    build_light_vqe_evaluator_registry,
    run_experiment,
)
from aicir.qas.vqe_loop.fair_vqe import fair_vqe_final_maxfev, optimize_vqe_energy
from aicir.qas.vqe_loop.benchmark_table import hamiltonian_from_terms, problem_from_row_terms
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float
from aicir.qas.vqe_loop.p0_problem_aware import sample_problem_aware_supernet_gene
from aicir.qas.vqe_loop.p0_supernet_native import build_native_supernet_e5_evaluator


# 閳光偓閳光偓 Defaults 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

PRESET = "h2_sto3g_jw_r0735_4q"
REFERENCE_ENERGY = -1.414403075643
N_QUBITS = 4
DEPTH = 3
N_ARCHITECTURES = 30
TWO_QUBIT_PAIRS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
SINGLE_QUBIT_CHOICES = ("i", "h", "rx", "ry", "rz")
TWO_QUBIT_CHOICES = ("none", "cx", "rzz")
SEED = 42

E1_MAX_EVALS = 20
E2_MAX_EVALS = 250
E3_MAX_EVALS = 20
E4_MAX_EVALS = 250
E5_SUPERNET_NUM = 5
E5_SUPERNET_STEPS = 250
E5_FINETUNE_STEPS = 250
PROXY_SEED_OFFSETS = (0, 17, 43)
FAIR_SEED_OFFSETS = (1000,)

OUTPUT_DIR = Path("outputs/h2_4q_p0_e1e2e5_fair_l3")
SELECTOR_PROXY_FIELDS = {
    "e2": ("E2",),
    "e5": ("E5",),
    "both": ("E2", "E5"),
}


# 閳光偓閳光偓 Hamiltonian loading 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def load_hamiltonian_terms(
    preset: str,
    *,
    hamiltonian_file: str | None = None,
    reference_energy_override: float | None = None,
) -> tuple[tuple[tuple[float, str], ...], int, float]:
    """Load Pauli terms from aicir.chemistry preset.

    Returns (terms, n_qubits, reference_energy).
    """
    if hamiltonian_file:
        loaded = json.loads(Path(hamiltonian_file).read_text(encoding="utf-8-sig"))
        if isinstance(loaded, list):
            terms = tuple((float(coeff), str(pauli)) for coeff, pauli in loaded)
            n_qubits = len(terms[0][1])
            reference_energy = float(reference_energy_override) if reference_energy_override is not None else float("nan")
            return terms, n_qubits, reference_energy

        from aicir.chemistry.spec import load_hamiltonian_input

        generated = load_hamiltonian_input(hamiltonian_file)
        terms = tuple((float(coeff), str(pauli)) for coeff, pauli in generated.terms)
        n_qubits = int(generated.n_qubits)
        metadata = dict(getattr(generated, "metadata", {}) or {})
        reference_energy = reference_energy_override
        if reference_energy is None:
            for key in (
                "reference_energy",
                "electronic_reference_energy",
                "electronic_reference_energy_old_thread",
                "fci_energy",
                "exact_energy",
            ):
                value = metadata.get(key)
                if value is not None and str(value).strip():
                    reference_energy = float(value)
                    break
        reference = float(reference_energy) if reference_energy is not None else float("nan")
        return terms, n_qubits, reference

    from aicir.chemistry.spec import generate_hamiltonian
    generated = generate_hamiltonian({"preset": preset})
    terms = tuple((float(coeff), str(pauli)) for coeff, pauli in generated.terms)
    n_qubits = int(generated.n_qubits)
    reference_energy = float(reference_energy_override) if reference_energy_override is not None else REFERENCE_ENERGY
    return terms, n_qubits, reference_energy


# 閳光偓閳光偓 Architecture sampler 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def sample_uniform_supernet_gene(
    n_qubits: int,
    depth: int,
    pairs: Sequence[tuple[int, int]],
    rng: random.Random,
) -> SupernetAnsatzGene:
    """Generate one uniform random SupernetAnsatzGene."""
    single_layers = []
    two_layers = []
    for _ in range(depth):
        single_layers.append(
            tuple(rng.choice(SINGLE_QUBIT_CHOICES) for _ in range(n_qubits))
        )
        two_layers.append(
            tuple(rng.choice(TWO_QUBIT_CHOICES) for _ in range(len(pairs)))
        )
    return SupernetAnsatzGene(
        n_qubits=n_qubits,
        single_qubit_layers=tuple(single_layers),
        two_qubit_layers=tuple(two_layers),
        two_qubit_pairs=tuple(tuple(p) for p in pairs),
    )


def make_architecture_sampler(
    n_architectures: int,
    n_qubits: int,
    depth: int,
    pairs: Sequence[tuple[int, int]],
    terms: Sequence[tuple[float, str]],
    reference_energy: float,
    preset: str,
    seed: int,
    sampling_mode: str = "uniform_supernet_native",
    problem_aware_fraction: float = 0.5,
    problem_aware_entangler_floor: float = 0.10,
):
    """Return a sampler callable for run_experiment."""
    rng = random.Random(seed)
    normalized_sampling_mode = str(sampling_mode).strip().lower()

    def sampler(config):
        for i in range(n_architectures):
            use_problem_aware = normalized_sampling_mode == "problem_aware_supernet_native"
            if normalized_sampling_mode == "mixed_supernet_native":
                use_problem_aware = rng.random() < max(0.0, min(1.0, float(problem_aware_fraction)))
            if use_problem_aware:
                gene = sample_problem_aware_supernet_gene(
                    n_qubits=n_qubits,
                    depth=depth,
                    pairs=pairs,
                    hamiltonian_terms=terms,
                    rng=rng,
                    single_qubit_gates=SINGLE_QUBIT_CHOICES,
                    two_qubit_gates=tuple(gate for gate in TWO_QUBIT_CHOICES if gate != "none"),
                    entangler_probability_floor=float(problem_aware_entangler_floor),
                )
                row_sampling_mode = "problem_aware_supernet_native"
            else:
                gene = sample_uniform_supernet_gene(n_qubits, depth, pairs, rng)
                row_sampling_mode = "uniform_supernet_native"
            yield {
                "problem_id": preset,
                "hamiltonian_class": "molecular_preset",
                "hamiltonian_id": preset,
                "reference_energy": reference_energy,
                "hamiltonian_terms": json.dumps(
                    [[coeff, pauli] for coeff, pauli in terms]
                ),
                "sampling_mode": row_sampling_mode if normalized_sampling_mode == "mixed_supernet_native" else normalized_sampling_mode,
                "depth": depth,
                "architecture_id": f"{preset}_L{depth}_{i:03d}",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": n_qubits,
            }

    return sampler


def default_two_qubit_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    try:
        from aicir.qas.algorithms.supernet import _default_two_qubit_pairs
    except ModuleNotFoundError:
        return tuple((index, index + 1) for index in range(max(0, int(n_qubits) - 1)))
    return tuple((int(left), int(right)) for left, right in _default_two_qubit_pairs(int(n_qubits)))


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("seed offset list cannot be empty")
    return values


def selector_proxy_fields(selector: str) -> tuple[str, ...]:
    try:
        return SELECTOR_PROXY_FIELDS[str(selector).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown selector: {selector}") from exc



def _row_id(row: Mapping[str, Any], fallback: int) -> str:
    value = str(row.get("architecture_id") or "").strip()
    return value if value else f"row:{fallback}"


def _rank_rows_by_field(rows: Sequence[Mapping[str, Any]], field: str) -> list[Mapping[str, Any]]:
    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        value = _as_float(row.get(field))
        if value is not None:
            scored.append((float(value), index, row))
    return [row for _value, _index, row in sorted(scored, key=lambda item: (item[0], _row_id(item[2], item[1])))]


def selector_level_summary(
    rows: Sequence[Mapping[str, Any]],
    proxy_field: str,
    *,
    target_field: str = "fair_high",
    top_k: int = 5,
) -> dict[str, Any]:
    """Report fair-label quality of one proxy selector's top-K architectures."""

    limit = max(1, int(top_k))
    proxy_ranked = _rank_rows_by_field(rows, proxy_field)
    selected = proxy_ranked[:limit]
    fair_ranked = _rank_rows_by_field(rows, target_field)
    fair_rank_by_id = {
        _row_id(row, index): index + 1
        for index, row in enumerate(fair_ranked)
    }
    fair_top_ids = {
        _row_id(row, index)
        for index, row in enumerate(fair_ranked[:limit])
    }
    selected_ids = [_row_id(row, index) for index, row in enumerate(selected)]
    selected_fair = [
        float(value)
        for row in selected
        for value in [_as_float(row.get(target_field))]
        if value is not None
    ]
    hit_count = len(set(selected_ids) & fair_top_ids)
    proxy_best_id = selected_ids[0] if selected_ids else ""
    return {
        "proxy_field": str(proxy_field),
        "target_field": str(target_field),
        "top_k": int(limit),
        "selected_architecture_ids": selected_ids,
        "fair_best_in_topK": min(selected_fair) if selected_fair else None,
        "fair_mean_in_topK": (sum(selected_fair) / len(selected_fair)) if selected_fair else None,
        "fair_median_in_topK": median(selected_fair) if selected_fair else None,
        "fair_topK_hit": int(hit_count),
        "fair_topK_hit_rate": hit_count / float(limit) if limit else 0.0,
        "fair_rank_of_proxy_best": fair_rank_by_id.get(proxy_best_id),
    }


def _selector_queue_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    extras = sorted(
        key
        for row in rows
        for key in row
        if not str(key).startswith("_") and key not in EXPERIMENT_FIELDS
    )
    return list(EXPERIMENT_FIELDS) + [key for key in extras if key not in EXPERIMENT_FIELDS]


def _write_selector_queue(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _selector_queue_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_selector_outputs(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    selector: str,
    top_k: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, Any] = {
        "selector": str(selector).lower(),
        "top_k": int(max(1, int(top_k))),
        "selectors": {},
        "queue_files": {},
    }
    for proxy_field in selector_proxy_fields(selector):
        selected = _rank_rows_by_field(rows, proxy_field)[: comparison["top_k"]]
        queue_path = output_dir / f"queue_{proxy_field.lower()}_topK.csv"
        _write_selector_queue(queue_path, selected)
        comparison["selectors"][proxy_field] = selector_level_summary(
            rows,
            proxy_field,
            top_k=comparison["top_k"],
        )
        comparison["queue_files"][proxy_field] = str(queue_path)
    (output_dir / "selector_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return comparison


# 閳光偓閳光偓 Adaptive-budget fair runner 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def make_adaptive_fair_runner(
    *,
    seed: int,
    seed_offsets: Sequence[int] = FAIR_SEED_OFFSETS,
    n_starts: int = 1,
    max_evals_override: int | None = None,
    optimizer=optimize_vqe_energy,
    backend: Any = None,
):
    """Fair VQE runner with per-architecture adaptive budget.

    budget = max(1000, 200 * n_params)  via fair_vqe_final_maxfev.
    """

    def fair(row: Mapping[str, Any]) -> Mapping[str, Any]:
        architecture = _architecture_from_experiment_row(row)
        budget = int(max_evals_override) if max_evals_override is not None else fair_vqe_final_maxfev(architecture.parameter_count)
        return _run_vqe_proxy(
            row,
            output_field="fair_high",
            problem=None,
            budget=budget,
            seed=seed,
            seed_offsets=seed_offsets,
            n_starts=n_starts,
            optimizer=optimizer,
            backend=backend,
            init_mode="random_uniform_pi",
        )

    return fair


def _prepare_torch_pauli_energy(row: Mapping[str, Any], *, device: str = "cpu"):
    import torch
    from aicir.qas.algorithms.supernet import (
        Supernet,
        SupernetConfig,
        _SINGLE_QUBIT_GATES,
        _TWO_QUBIT_GATES,
    )

    gene = _supernet_gene_from_experiment_row(row)
    terms = _hamiltonian_terms_from_experiment_row(row, None)
    hamiltonian = hamiltonian_from_terms(terms, n_qubits=gene.n_qubits)
    config = SupernetConfig(
        n_qubits=gene.n_qubits,
        layers=gene.layers,
        two_qubit_pairs=gene.two_qubit_pairs,
        supernet_num=1,
        supernet_steps=0,
        ranking_num=1,
        finetune_steps=0,
        device=device,
        task="vqe",
    )
    helper = Supernet(config)

    def energy(theta: Sequence[float]) -> float:
        values = [float(value) for value in theta]
        cursor = 0
        gates: list[dict[str, Any]] = []
        for single_layer, two_layer in zip(gene.single_qubit_layers, gene.two_qubit_layers):
            for qubit, gate_type in enumerate(single_layer):
                spec = _SINGLE_QUBIT_GATES[gate_type]
                params = values[cursor : cursor + spec.n_params]
                cursor += spec.n_params
                gate = spec.builder(params, (qubit,))
                if gate is not None:
                    gates.append(gate)
            for pair_index, gate_type in enumerate(two_layer):
                if gate_type == "none":
                    continue
                spec = _TWO_QUBIT_GATES[gate_type]
                params = values[cursor : cursor + spec.n_params]
                cursor += spec.n_params
                gate = spec.builder(params, gene.two_qubit_pairs[pair_index])
                if gate is not None:
                    gates.append(gate)
        if cursor != len(values):
            raise ValueError(f"Expected {cursor} theta values, got {len(values)}")
        with torch.no_grad():
            state = helper._simulate_gates(gates)
            return float(helper._hamiltonian_expectation(state, hamiltonian).detach().cpu())

    return energy


def _run_torch_pauli_proxy(
    row: Mapping[str, Any],
    *,
    output_field: str,
    budget: int,
    seed: int,
    seed_offsets: Sequence[int],
    device: str = "cpu",
    initial_parameters=None,
) -> Mapping[str, Any]:
    import numpy as np
    from scipy.optimize import minimize

    gene = _supernet_gene_from_experiment_row(row)
    n_params, two_q_count = _supernet_gene_static_counts(gene)
    energy_fn = _prepare_torch_pauli_energy(row, device=device)
    order_offset = _row_order_seed_offset(row)
    offsets = tuple(int(offset) for offset in seed_offsets) or (0,)
    results: list[tuple[float, np.ndarray, int]] = []
    resolved_initial = _resolve_initial_parameters(row, initial_parameters)
    for seed_offset in offsets:
        rng = np.random.default_rng(int(seed) + int(seed_offset) + order_offset)
        if resolved_initial is None:
            start = rng.uniform(-float(np.pi), float(np.pi), size=n_params)
        else:
            start = np.asarray(resolved_initial, dtype=float).reshape(-1)
            if start.size != n_params:
                raise ValueError(f"initial_parameters must have length {n_params}, got {start.size}")
        if n_params == 0:
            energy = energy_fn(start)
            results.append((energy, start, 1))
            continue
        result = minimize(
            energy_fn,
            start,
            method="COBYLA",
            options={"maxiter": int(budget), "rhobeg": 1.0, "tol": 1e-6},
        )
        nfev = int(getattr(result, "nfev", int(budget)))
        results.append((float(result.fun), np.asarray(result.x, dtype=float), nfev))
    best_energy, best_theta, total_nfev = min(results, key=lambda item: item[0])
    return {
        output_field: float(best_energy),
        "n_qubits": int(gene.n_qubits),
        "n_params": int(n_params),
        "two_q_count": int(two_q_count),
        f"{output_field}_nfev": int(total_nfev),
    }


def build_torch_pauli_evaluator_registry(
    *,
    e1_max_evals: int,
    e2_max_evals: int,
    seed: int,
    proxy_seed_offsets: Sequence[int],
    fair_seed_offsets: Sequence[int],
    fair_max_evals: int | None,
    device: str = "cpu",
    e3_max_evals: int | None = None,
    e4_max_evals: int | None = None,
    warm_start_parameters=None,
) -> tuple[dict[str, Any], Any]:
    def e1(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_torch_pauli_proxy(
            row,
            output_field="E1",
            budget=int(e1_max_evals),
            seed=int(seed),
            seed_offsets=proxy_seed_offsets,
            device=device,
        )

    def e2(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_torch_pauli_proxy(
            row,
            output_field="E2",
            budget=int(e2_max_evals),
            seed=int(seed),
            seed_offsets=proxy_seed_offsets,
            device=device,
        )

    def fair(row: Mapping[str, Any]) -> Mapping[str, Any]:
        gene = _supernet_gene_from_experiment_row(row)
        n_params, _two_q_count = _supernet_gene_static_counts(gene)
        budget = int(fair_max_evals) if fair_max_evals is not None else fair_vqe_final_maxfev(n_params)
        return _run_torch_pauli_proxy(
            row,
            output_field="fair_high",
            budget=budget,
            seed=int(seed),
            seed_offsets=fair_seed_offsets,
            device=device,
        )

    registry = {"E1": e1, "E2": e2}

    if e3_max_evals is not None:
        def e3(row: Mapping[str, Any]) -> Mapping[str, Any]:
            return _run_torch_pauli_proxy(
                row,
                output_field="E3",
                budget=int(e3_max_evals),
                seed=int(seed),
                seed_offsets=(0,),
                device=device,
                initial_parameters=warm_start_parameters,
            )

        registry["E3"] = e3

    if e4_max_evals is not None:
        def e4(row: Mapping[str, Any]) -> Mapping[str, Any]:
            return _run_torch_pauli_proxy(
                row,
                output_field="E4",
                budget=int(e4_max_evals),
                seed=int(seed),
                seed_offsets=(0,),
                device=device,
                initial_parameters=warm_start_parameters,
            )

        registry["E4"] = e4

    return registry, fair


# 閳光偓閳光偓 Main 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P0 cheap-eval diagnostic runner.")
    parser.add_argument("--preset", default=PRESET)
    parser.add_argument("--hamiltonian-file", default=None)
    parser.add_argument("--reference-energy", type=float, default=None)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--n-architectures", type=int, default=N_ARCHITECTURES)
    parser.add_argument("--e1-max-evals", type=int, default=E1_MAX_EVALS)
    parser.add_argument("--e2-max-evals", type=int, default=E2_MAX_EVALS)
    parser.add_argument("--e3-max-evals", type=int, default=E3_MAX_EVALS)
    parser.add_argument("--e4-max-evals", type=int, default=E4_MAX_EVALS)
    parser.add_argument("--e5-supernet-num", type=int, default=E5_SUPERNET_NUM)
    parser.add_argument("--e5-supernet-steps", type=int, default=E5_SUPERNET_STEPS)
    parser.add_argument("--e5-finetune-steps", type=int, default=E5_FINETUNE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--proxy-seed-offsets", default=",".join(str(item) for item in PROXY_SEED_OFFSETS))
    parser.add_argument("--fair-seed-offsets", default=",".join(str(item) for item in FAIR_SEED_OFFSETS))
    parser.add_argument("--fair-max-evals", type=int, default=None)
    parser.add_argument("--light-evaluator", choices=("basic_vqe", "torch_pauli"), default="basic_vqe")
    parser.add_argument("--selector", choices=("e2", "e5", "both"), default="both")
    parser.add_argument("--selector-top-k", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform_supernet_native", "problem_aware_supernet_native", "mixed_supernet_native"),
        default="uniform_supernet_native",
    )
    parser.add_argument("--problem-aware-fraction", type=float, default=0.5)
    parser.add_argument("--problem-aware-entangler-floor", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_seed_offsets = parse_int_tuple(args.proxy_seed_offsets)
    fair_seed_offsets = parse_int_tuple(args.fair_seed_offsets)
    active_proxy_fields = selector_proxy_fields(args.selector)

    # 1. Load Hamiltonian
    terms, n_qubits, reference_energy = load_hamiltonian_terms(
        args.preset,
        hamiltonian_file=args.hamiltonian_file,
        reference_energy_override=args.reference_energy,
    )
    print(f"Loaded {args.preset}: {len(terms)} terms, {n_qubits}q, E_ref={reference_energy:.6f}")
    two_qubit_pairs = default_two_qubit_pairs(n_qubits)

    # 2. Config
    config = CheapEvalExperimentConfig(
        benchmark_set=(args.preset,),
        depths=(args.depth,),
        n_architectures=args.n_architectures,
        sampling_mode=args.sampling_mode,
        proxy_fields=active_proxy_fields,
        target_field="fair_high",
    )

    # 3. Architecture sampler
    architecture_sampler = make_architecture_sampler(
        n_architectures=args.n_architectures,
        n_qubits=n_qubits,
        depth=args.depth,
        pairs=two_qubit_pairs,
        terms=terms,
        reference_energy=reference_energy,
        preset=args.preset,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        problem_aware_fraction=args.problem_aware_fraction,
        problem_aware_entangler_floor=args.problem_aware_entangler_floor,
    )

    # 4. Optional native supernet screening selector.
    e5_evaluator = None
    if "E5" in active_proxy_fields:
        e5_evaluator = build_native_supernet_e5_evaluator(
            problem=None,
            supernet_num=args.e5_supernet_num,
            supernet_steps=args.e5_supernet_steps,
            finetune_steps=args.e5_finetune_steps,
            seed=args.seed,
            single_qubit_gates=SINGLE_QUBIT_CHOICES,
            two_qubit_gates=("cx", "rzz"),
        )

    # 5. Build only the selector evaluators that this run can actually use.
    registry: dict[str, Any] = {}
    needs_light_selector = any(field in active_proxy_fields for field in ("E1", "E2"))
    if args.light_evaluator == "torch_pauli":
        if needs_light_selector:
            registry, fair_runner = build_torch_pauli_evaluator_registry(
                e1_max_evals=args.e1_max_evals,
                e2_max_evals=args.e2_max_evals,
                seed=args.seed,
                proxy_seed_offsets=proxy_seed_offsets,
                fair_seed_offsets=fair_seed_offsets,
                fair_max_evals=args.fair_max_evals,
            )
        else:
            def fair_runner(row: Mapping[str, Any]) -> Mapping[str, Any]:
                gene = _supernet_gene_from_experiment_row(row)
                n_params, _two_q_count = _supernet_gene_static_counts(gene)
                budget = int(args.fair_max_evals) if args.fair_max_evals is not None else fair_vqe_final_maxfev(n_params)
                return _run_torch_pauli_proxy(
                    row,
                    output_field="fair_high",
                    budget=budget,
                    seed=int(args.seed),
                    seed_offsets=fair_seed_offsets,
                )
    else:
        if needs_light_selector:
            registry, _ = build_light_vqe_evaluator_registry(
                problem=None,
                e1_max_evals=args.e1_max_evals,
                e2_max_evals=args.e2_max_evals,
                fair_max_evals=1000,  # not used; we override with adaptive fair runner
                seed=args.seed,
                n_starts=1,
                proxy_seed_offsets=proxy_seed_offsets,
            )
        fair_runner = make_adaptive_fair_runner(
            seed=args.seed,
            seed_offsets=fair_seed_offsets,
            max_evals_override=args.fair_max_evals,
        )
    if e5_evaluator is not None:
        registry["E5"] = e5_evaluator

    # 6. Run experiment
    rows_csv = output_dir / "rows.csv"
    print(f"Running experiment: selector={args.selector}, {args.n_architectures} archs, L{args.depth}, "
          f"E2={args.e2_max_evals}x{len(proxy_seed_offsets)}, "
          f"E5=sn{args.e5_supernet_num}/s{args.e5_supernet_steps}/f{args.e5_finetune_steps}, "
          f"fair=adaptive")
    start = time.time()

    run_experiment(
        config,
        rows_csv,
        evaluator_registry=registry,
        architecture_sampler=architecture_sampler,
        fair_vqe_runner=fair_runner,
    )

    elapsed = time.time() - start
    print(f"Experiment completed in {elapsed:.1f}s")

    # 8. Analyze
    with rows_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Read {len(rows)} rows for analysis")
    hamiltonian_terms_json = json.dumps([[coeff, pauli] for coeff, pauli in terms])
    for row in rows:
        row.setdefault("hamiltonian_terms", hamiltonian_terms_json)
        if not row.get("hamiltonian_terms"):
            row["hamiltonian_terms"] = hamiltonian_terms_json
        row.setdefault("reference_energy", reference_energy)
        if not row.get("reference_energy"):
            row["reference_energy"] = reference_energy
        row.setdefault("n_qubits", n_qubits)
        if not row.get("n_qubits"):
            row["n_qubits"] = n_qubits

    # Cost models: budget-based (not walltime, for cross-run comparability)
    cost_models = {
        "E1": {
            "upfront_cost": 0,
            "per_arch_cost": args.e1_max_evals * len(proxy_seed_offsets),
        },
        "E2": {
            "upfront_cost": 0,
            "per_arch_cost": args.e2_max_evals * len(proxy_seed_offsets),
        },
        "E5": {
            "upfront_cost": args.e5_supernet_num * args.e5_supernet_steps,
            "per_arch_cost": args.e5_finetune_steps,
        },
        "E3": {
            "upfront_cost": args.e5_supernet_num * args.e5_supernet_steps,
            "per_arch_cost": args.e3_max_evals,
        },
        "E4": {
            "upfront_cost": args.e5_supernet_num * args.e5_supernet_steps,
            "per_arch_cost": args.e4_max_evals,
        },
    }

    selector_comparison = write_selector_outputs(
        rows,
        output_dir,
        selector=args.selector,
        top_k=min(max(1, int(args.selector_top_k)), max(1, len(rows))),
    )
    summary = {
        "diagnostics_removed": True,
        "proxy_fields": list(active_proxy_fields),
        "selector_comparison": selector_comparison,
    }

    summary["run_notes"] = {
        "problem": args.preset,
        "reference_energy": reference_energy,
        "n_architectures": args.n_architectures,
        "depth": args.depth,
        "budgets": {
            "E1": args.e1_max_evals,
            "E2": args.e2_max_evals,
            "E3": args.e3_max_evals,
            "E4": args.e4_max_evals,
            "E5_supernet_num": args.e5_supernet_num,
            "E5_supernet_steps": args.e5_supernet_steps,
            "E5_finetune_steps": args.e5_finetune_steps,
            "fair": "adaptive max(1000, 200*n_params)",
            "fair_max_evals_override": args.fair_max_evals,
        },
        "selector": args.selector,
        "selector_top_k": args.selector_top_k,
        "sampling_mode": args.sampling_mode,
        "problem_aware_fraction": args.problem_aware_fraction,
        "problem_aware_entangler_floor": args.problem_aware_entangler_floor,
        "proxy_seed_offsets": list(proxy_seed_offsets),
        "fair_seed_offsets": list(fair_seed_offsets),
        "walltime_seconds": elapsed,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Summary written to {summary_path}")

    # 9. Print quick selector summary
    def _fmt(value: Any, digits: int = 4) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    print("\n=== Selector Comparison ===")
    for proxy, selector_summary in selector_comparison["selectors"].items():
        print(
            f"  {proxy}: fair_best_in_topK={_fmt(selector_summary['fair_best_in_topK'])}  "
            f"fair_mean_in_topK={_fmt(selector_summary['fair_mean_in_topK'])}  "
            f"fair_topK_hit={selector_summary['fair_topK_hit']}/{selector_summary['top_k']}  "
            f"fair_rank_of_proxy_best={selector_summary['fair_rank_of_proxy_best']}"
        )


if __name__ == "__main__":
    main()





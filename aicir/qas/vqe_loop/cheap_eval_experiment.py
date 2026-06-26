"""Scaffold for P0 cheap-evaluation experiments.

This module defines the canonical row schema, manifest, and injectable runner
for E1-E5 diagnostic experiments.  It does not implement VQE or supernet
training itself; heavy evaluators are supplied as callables and write rows
conforming to ``EXPERIMENT_FIELDS``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from aicir.qas.core._types import ArchitectureSpec
from aicir.qas.primitives.ansatz import (
    LayerwiseAnsatzGene,
    SupernetAnsatzGene,
    architecture_from_layerwise_gene,
    architecture_from_supernet_gene,
)
from aicir.qas.problems.hamiltonians import VQEProblem, exact_ground_energy
from aicir.qas.vqe_loop.fair_vqe import VQEOptimizationResult, optimize_vqe_energy
from aicir.qas.vqe_loop.geometry import parse_pauli_hamiltonian_terms


EXPERIMENT_FIELDS: tuple[str, ...] = (
    "problem_id",
    "hamiltonian_class",
    "sampling_mode",
    "depth",
    "architecture_id",
    "ansatz_gene",
    "n_qubits",
    "n_params",
    "two_q_count",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E5_mean",
    "E5_min",
    "E5_std",
    "fair_high",
    "fair_warm",
    "fair_random",
    "hit_rate",
    "exposure_count",
    "evaluation_order_index",
    "error_log",
    "walltime_E1",
    "walltime_E2",
    "walltime_E3",
    "walltime_E4",
    "walltime_E5",
    "walltime_fair_high",
)


PROXY_NOTES: dict[str, str] = {
    "E1": "random init low-budget light VQE",
    "E2": "random init high-budget light VQE",
    "E3": "supernet warm-start low-budget light VQE",
    "E4": "supernet warm-start high-budget finetune",
    "E5": "current native supernet screening",
}


@dataclass(frozen=True)
class CheapEvalExperimentConfig:
    benchmark_set: tuple[str, ...]
    depths: tuple[int, ...]
    n_architectures: int
    sampling_mode: str = "uniform"
    proxy_fields: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
    target_field: str = "fair_high"

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["benchmark_set"] = list(self.benchmark_set)
        payload["depths"] = [int(depth) for depth in self.depths]
        payload["proxy_fields"] = list(self.proxy_fields)
        payload["notes"] = {field: PROXY_NOTES[field] for field in self.proxy_fields if field in PROXY_NOTES}
        payload["experiment_fields"] = list(EXPERIMENT_FIELDS)
        return payload


def write_empty_experiment_csv(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(EXPERIMENT_FIELDS)).writeheader()
    return output


def write_experiment_manifest(config: CheapEvalExperimentConfig, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_jsonable(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


ArchitectureSampler = Callable[[CheapEvalExperimentConfig], Iterable[Mapping[str, Any]]]
EvaluatorRegistry = Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any] | float]]
FairVqeRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]
VQEOptimizer = Callable[..., VQEOptimizationResult]
InitialParametersProvider = Sequence[float] | Callable[[Mapping[str, Any]], Sequence[float] | None] | None


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _set_if_empty(row: dict[str, Any], key: str, value: Any) -> None:
    if key not in row or _is_empty(row.get(key)):
        row[key] = value


def _append_error(row: dict[str, Any], label: str, exc: Exception, *, limit: int = 200) -> None:
    message = str(exc).replace("\n", " ").replace("\r", " ")
    entry = f"{label}:{type(exc).__name__}:{message[:limit]}"
    current = str(row.get("error_log") or "").strip()
    row["error_log"] = f"{current}; {entry}" if current else entry


def _merge_evaluator_result(row: dict[str, Any], field: str, result: Mapping[str, Any] | float) -> None:
    if isinstance(result, Mapping):
        for key, value in result.items():
            _set_if_empty(row, str(key), value)
        if _is_empty(row.get(field)) and "energy" in result:
            row[field] = result["energy"]
        if _is_empty(row.get(field)) and "value" in result:
            row[field] = result["value"]
        return
    _set_if_empty(row, field, result)


def _architecture_from_experiment_row(row: Mapping[str, Any]) -> ArchitectureSpec:
    cached = row.get("_cached_architecture")
    if isinstance(cached, ArchitectureSpec):
        return cached
    raw_gene = row.get("ansatz_gene", "")
    if raw_gene is None or str(raw_gene).strip() in {"", '""', "null"}:
        raise ValueError("experiment row requires ansatz_gene for real VQE evaluation")
    parsed = json.loads(raw_gene) if isinstance(raw_gene, str) else raw_gene
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError("ansatz_gene must decode to a JSON object")
    if str(parsed.get("kind", "")).lower() == "supernet_native":
        architecture = architecture_from_supernet_gene(SupernetAnsatzGene.from_jsonable(parsed))
    else:
        architecture = architecture_from_layerwise_gene(LayerwiseAnsatzGene.from_jsonable(parsed))
    if isinstance(row, dict):
        row["_cached_architecture"] = architecture
    return architecture


def _problem_from_experiment_row(row: Mapping[str, Any], default_problem: VQEProblem | None) -> VQEProblem:
    cached = row.get("_cached_problem")
    if isinstance(cached, VQEProblem):
        return cached
    raw_terms = row.get("hamiltonian_terms", "")
    if raw_terms is None or str(raw_terms).strip() == "":
        if default_problem is None:
            raise ValueError("experiment row requires hamiltonian_terms when no default problem is provided")
        if isinstance(row, dict):
            row["_cached_problem"] = default_problem
        return default_problem
    loaded = json.loads(str(raw_terms))
    terms = parse_pauli_hamiltonian_terms(loaded)
    if not terms:
        raise ValueError("hamiltonian_terms must contain at least one Pauli term")
    widths = {len(pauli) for _coeff, pauli in terms}
    n_qubits = int(row.get("n_qubits") or max(widths))
    if widths != {n_qubits}:
        raise ValueError(f"hamiltonian_terms width must match n_qubits={n_qubits}; found widths={sorted(widths)}")
    problem = VQEProblem(
        name=str(row.get("hamiltonian_id") or f"row_pauli_{n_qubits}q"),
        n_qubits=n_qubits,
        hamiltonian=terms,
        reference_energy=exact_ground_energy(terms),
    )
    if isinstance(row, dict):
        row["_cached_problem"] = problem
    return problem


def _row_order_seed_offset(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("evaluation_order_index") or 0) * 1000
    except (TypeError, ValueError):
        return 0


def _resolve_initial_parameters(
    row: Mapping[str, Any],
    provider: InitialParametersProvider,
) -> Sequence[float] | None:
    if provider is None:
        return None
    if callable(provider):
        return provider(row)
    return provider


def _run_vqe_proxy(
    row: Mapping[str, Any],
    *,
    output_field: str,
    problem: VQEProblem | None,
    budget: int,
    seed: int,
    seed_offsets: Sequence[int],
    n_starts: int,
    optimizer: VQEOptimizer,
    backend: Any = None,
    init_mode: str = "random_uniform_pi",
    initial_parameters: InitialParametersProvider = None,
) -> dict[str, Any]:
    architecture = _architecture_from_experiment_row(row)
    resolved_problem = _problem_from_experiment_row(row, problem)
    order_offset = _row_order_seed_offset(row)
    offsets = tuple(int(offset) for offset in seed_offsets) or (0,)
    results = []
    for seed_offset in offsets:
        results.append(
            optimizer(
                architecture,
                resolved_problem,
                seed=int(seed) + int(seed_offset) + order_offset,
                n_starts=int(n_starts),
                evals_per_param=int(budget),
                max_evaluations=int(budget),
                budget_override=int(budget),
                backend=backend,
                init_mode=init_mode,
                initial_parameters=_resolve_initial_parameters(row, initial_parameters),
            )
        )
    result = min(results, key=lambda item: float(item.energy))
    return {
        output_field: float(result.energy),
        "n_qubits": architecture.n_qubits,
        "n_params": architecture.parameter_count,
        "two_q_count": architecture.two_qubit_gate_count,
    }


def build_light_vqe_evaluator_registry(
    *,
    problem: VQEProblem | None,
    e1_max_evals: int = 20,
    e2_max_evals: int = 250,
    fair_max_evals: int = 1000,
    seed: int = 1234,
    n_starts: int = 1,
    fair_n_starts: int | None = None,
    proxy_seed_offsets: Sequence[int] = (0,),
    fair_seed_offsets: Sequence[int] = (1000,),
    optimizer: VQEOptimizer = optimize_vqe_energy,
    backend: Any = None,
    init_mode: str = "random_uniform_pi",
    initial_parameters: InitialParametersProvider = None,
) -> tuple[dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]], Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    """Build real E1/E2 light-VQE evaluators plus a high-budget fair runner.

    E3/E4/E5 are intentionally left out: those need supernet warm-start and
    native-screening state that should be wired separately.
    """

    def e1(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_vqe_proxy(
            row,
            output_field="E1",
            problem=problem,
            budget=int(e1_max_evals),
            seed=int(seed),
            seed_offsets=proxy_seed_offsets,
            n_starts=int(n_starts),
            optimizer=optimizer,
            backend=backend,
            init_mode=init_mode,
            initial_parameters=initial_parameters,
        )

    def e2(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_vqe_proxy(
            row,
            output_field="E2",
            problem=problem,
            budget=int(e2_max_evals),
            seed=int(seed),
            seed_offsets=proxy_seed_offsets,
            n_starts=int(n_starts),
            optimizer=optimizer,
            backend=backend,
            init_mode=init_mode,
            initial_parameters=initial_parameters,
        )

    def fair(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return _run_vqe_proxy(
            row,
            output_field="fair_high",
            problem=problem,
            budget=int(fair_max_evals),
            seed=int(seed),
            seed_offsets=fair_seed_offsets,
            n_starts=int(fair_n_starts if fair_n_starts is not None else n_starts),
            optimizer=optimizer,
            backend=backend,
            init_mode=init_mode,
            initial_parameters=initial_parameters,
        )

    return {"E1": e1, "E2": e2}, fair


def run_experiment(
    config: CheapEvalExperimentConfig,
    output_csv: str | Path,
    *,
    evaluator_registry: EvaluatorRegistry,
    architecture_sampler: ArchitectureSampler,
    fair_vqe_runner: FairVqeRunner,
    flush_callback: Callable[[], None] | None = None,
) -> None:
    """Run the P0 cheap-eval experiment and fill an ``EXPERIMENT_FIELDS`` CSV.

    The callable contract is pinned here so the real evaluator wiring can be
    added without reshaping the analyzer schema:
    architecture sampler -> E1-E5 evaluator registry -> high-budget fair VQE.
    """

    missing = [field for field in config.proxy_fields if field not in evaluator_registry]
    if missing:
        raise KeyError(f"missing evaluator(s) for proxy field(s): {', '.join(missing)}")

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit = max(0, int(config.n_architectures))
    rows_written = 0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPERIMENT_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for order_index, architecture in enumerate(architecture_sampler(config)):
            if rows_written >= limit:
                break
            row: dict[str, Any] = {field: "" for field in EXPERIMENT_FIELDS}
            row.update(dict(architecture))
            row.setdefault("sampling_mode", config.sampling_mode)
            if not row.get("sampling_mode"):
                row["sampling_mode"] = config.sampling_mode
            row["evaluation_order_index"] = row.get("evaluation_order_index") or order_index

            for proxy_field in config.proxy_fields:
                started = time.perf_counter()
                try:
                    result = evaluator_registry[proxy_field](row)
                except Exception as exc:
                    _set_if_empty(row, proxy_field, "")
                    _append_error(row, proxy_field, exc)
                else:
                    _merge_evaluator_result(row, proxy_field, result)
                finally:
                    elapsed = time.perf_counter() - started
                walltime_field = f"walltime_{proxy_field}"
                if walltime_field in EXPERIMENT_FIELDS and _is_empty(row.get(walltime_field)):
                    row[walltime_field] = elapsed

            started = time.perf_counter()
            try:
                fair_result = fair_vqe_runner(row)
            except Exception as exc:
                _set_if_empty(row, config.target_field, "")
                _append_error(row, config.target_field, exc)
            else:
                _merge_evaluator_result(row, config.target_field, dict(fair_result))
            finally:
                fair_elapsed = time.perf_counter() - started
            if _is_empty(row.get("walltime_fair_high")):
                row["walltime_fair_high"] = fair_elapsed

            writer.writerow(row)
            handle.flush()
            if flush_callback is not None:
                flush_callback()
            rows_written += 1


def _parse_csv_text(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_depths(raw: str) -> tuple[int, ...]:
    depths = tuple(int(part) for part in _parse_csv_text(raw))
    if not depths:
        raise ValueError("--depths must contain at least one integer")
    for depth in depths:
        if depth <= 0:
            raise ValueError("--depths must be positive")
    return depths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a P0 cheap-eval experiment manifest and row CSV.")
    parser.add_argument("--manifest", required=True, help="Output manifest JSON path.")
    parser.add_argument("--output-csv", required=True, help="Output empty experiment CSV path.")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark ids.")
    parser.add_argument("--depths", required=True, help="Comma-separated fixed depths.")
    parser.add_argument("--n-architectures", type=int, required=True)
    parser.add_argument("--sampling-mode", default="uniform")
    args = parser.parse_args(argv)

    config = CheapEvalExperimentConfig(
        benchmark_set=_parse_csv_text(args.benchmarks),
        depths=_parse_depths(args.depths),
        n_architectures=int(args.n_architectures),
        sampling_mode=str(args.sampling_mode),
    )
    write_experiment_manifest(config, args.manifest)
    write_empty_experiment_csv(args.output_csv)


if __name__ == "__main__":
    main()

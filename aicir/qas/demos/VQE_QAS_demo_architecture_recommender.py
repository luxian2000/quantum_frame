"""Demo: recommend VQE/QAS architectures for a chemistry ground-state task.

Input: a JSON scenario with a problem description, molecular/PySCF-style
Hamiltonian input, and an optional current architecture.

Output: a TXT report with one or more architecture/search recommendations.

Run from the repository root:
    python -m aicir.qas.demos.VQE_QAS_demo_architecture_recommender \
        --input aicir/qas/demos/vqe_qas_architecture_recommender_input_lih.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aicir.chemistry.spec import generate_hamiltonian, load_hamiltonian_input
from aicir.qas.primitives.ansatz import SupernetAnsatzGene, architecture_from_supernet_gene
from aicir.qas.problems.hamiltonians import VQEProblem, exact_ground_energy
from aicir.qas.primitives.backend_utils import resolve_qas_backend
from aicir.qas.vqe_loop import ClosedLoopConfig, run_vqe_qas_closed_loop
from aicir.qas.vqe_loop.fair_vqe import optimize_vqe_energy


DEFAULT_INPUT_PATH = Path(__file__).with_name("vqe_qas_architecture_recommender_input_lih.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("vqe_qas_architecture_recommender_results.txt")


def _load_json(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("architecture recommender input must be a JSON object")
    return loaded


def _resolve_existing_path(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [
        raw,
        REPO_ROOT / raw,
        REPO_ROOT / "aicir" / "qas" / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def _molecule_name(spec: Mapping[str, Any]) -> str:
    molecule = str(spec.get("molecule") or spec.get("name") or "").strip()
    if molecule:
        return molecule
    geometry = spec.get("geometry") or spec.get("atom")
    if isinstance(geometry, list) and geometry:
        symbols = []
        for atom in geometry:
            if isinstance(atom, (list, tuple)) and atom:
                symbols.append(str(atom[0]))
        return "".join(symbols) if symbols else "molecule"
    return "molecule"


def _estimated_from_molecular_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    molecule = _molecule_name(spec)
    n_qubits = spec.get("estimated_n_qubits")
    term_count = spec.get("estimated_terms", spec.get("estimated_term_count"))
    if n_qubits is None:
        if molecule.lower() == "lih":
            n_qubits = 12
        elif molecule.lower() == "h2":
            n_qubits = 4
    if term_count is None:
        if molecule.lower() == "lih":
            term_count = 631
        elif molecule.lower() == "h2":
            term_count = 19
    return {
        "molecule": molecule,
        "n_qubits": int(n_qubits) if n_qubits is not None else None,
        "term_count": int(term_count) if term_count is not None else None,
    }


def summarize_hamiltonian(request: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve or estimate the Hamiltonian summary for the request."""

    hamiltonian_file = request.get("hamiltonian_file")
    if hamiltonian_file:
        generated = load_hamiltonian_input(_resolve_existing_path(str(hamiltonian_file)))
        return {
            "resolved": True,
            "resolution_error": "",
            "input_kind": "hamiltonian_file",
            "hamiltonian_id": generated.hamiltonian_id,
            "hamiltonian_class": generated.hamiltonian_class,
            "molecule": str(generated.metadata.get("molecule", generated.metadata.get("preset", "")) or ""),
            "n_qubits": int(generated.n_qubits),
            "term_count": len(generated.terms),
            "estimated": False,
            "metadata": dict(generated.metadata),
            "terms": tuple(generated.terms),
        }

    hamiltonian_spec = request.get("hamiltonian")
    molecular_spec = request.get("molecular_spec")
    raw_spec = hamiltonian_spec if hamiltonian_spec is not None else molecular_spec
    if not isinstance(raw_spec, Mapping):
        raise ValueError("input must provide either 'molecular_spec' or 'hamiltonian'")

    try:
        generated = generate_hamiltonian(raw_spec)
    except ImportError as exc:
        if not isinstance(molecular_spec, Mapping):
            raise
        estimated = _estimated_from_molecular_spec(molecular_spec)
        return {
            "resolved": False,
            "resolution_error": str(exc),
            "input_kind": "molecular",
            "hamiltonian_id": "",
            "hamiltonian_class": "molecular_unresolved",
            "molecule": estimated["molecule"],
            "n_qubits": estimated["n_qubits"],
            "term_count": estimated["term_count"],
            "estimated": True,
            "metadata": dict(molecular_spec),
        }

    metadata = dict(generated.metadata)
    return {
        "resolved": True,
        "resolution_error": "",
        "input_kind": str(metadata.get("input_kind", raw_spec.get("kind", "unknown"))),
        "hamiltonian_id": generated.hamiltonian_id,
        "hamiltonian_class": generated.hamiltonian_class,
        "molecule": str(metadata.get("molecule", raw_spec.get("molecule", "")) or ""),
        "n_qubits": int(generated.n_qubits),
        "term_count": len(generated.terms),
        "estimated": False,
        "metadata": metadata,
        "terms": tuple(generated.terms),
    }


def _current_architecture_diagnosis(current: Mapping[str, Any] | None, hamiltonian: Mapping[str, Any]) -> list[str]:
    if not current:
        return [
            "No current architecture was provided.",
            "Use the first recommendation as a bootstrap candidate and keep the others as fair-VQE controls.",
        ]
    notes: list[str] = []
    name = str(current.get("name", current.get("architecture_id", "current_architecture")))
    notes.append(f"current architecture: {name}")
    n_params = current.get("n_params")
    two_q = current.get("two_q_count")
    if n_params is not None:
        notes.append(f"reported parameters: {n_params}")
    if two_q is not None:
        notes.append(f"reported two-qubit gates: {two_q}")
    if hamiltonian.get("term_count") and int(hamiltonian["term_count"]) > 100:
        notes.append("large Pauli-term count: prefer warm-started search and verify only top candidates.")
    if two_q is not None and int(float(two_q)) > max(8, int(hamiltonian.get("n_qubits") or 0)):
        notes.append("two-qubit count is relatively high: include a lower-entangler control architecture.")
    if current.get("fair_best_energy") is not None:
        notes.append(f"known fair best energy: {current['fair_best_energy']}")
    return notes


def _qubit_count(hamiltonian: Mapping[str, Any]) -> int:
    value = hamiltonian.get("n_qubits")
    return int(value) if value is not None else 12


def recommend_architectures(
    request: Mapping[str, Any],
    hamiltonian: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return vqe-loop supernet search jobs, not hand-authored architectures."""

    n_qubits = _qubit_count(hamiltonian)
    requested_top_k = max(1, int(request.get("top_k", 3)))
    bootstrap_count = max(3, requested_top_k)
    hamiltonian_name = str(hamiltonian.get("hamiltonian_id") or "hamiltonian_input")
    output_slug = str(request.get("output_slug") or hamiltonian_name or "qas_supernet_search")
    output_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in output_slug).strip("_")

    search_jobs = [
        {
            "rank": 1,
            "name": "vqe_loop_supernet_bootstrap",
            "kind": "vqe_loop_supernet_native_search",
            "architecture_source": "generated by aicir.qas.vqe_loop supernet_native",
            "why": (
                "Run vqe-loop with initial_labels=0 so native supernet performs random sampling, cheap ranking, "
                f"and writes the top {bootstrap_count} architectures into the oracle/queue before fair VQE verifies them."
            ),
            "output_dir": f"outputs/{output_slug}_supernet_bootstrap",
            "vqe_loop_args": {
                "--initial-labels": 0,
                "--rounds": 1,
                "--k-min": 3,
                "--supernet-native-count": bootstrap_count,
                "--supernet-native-layers": "auto",
                "--supernet-native-supernet-num": 1,
                "--supernet-native-steps": 120 if n_qubits <= 8 else 150,
                "--supernet-native-ranking-num": 40 if n_qubits <= 8 else 80,
                "--supernet-native-finetune-steps": 100 if n_qubits <= 8 else 150,
                "--backend": "numpy",
                "--dtype": "complex128",
            },
            "architecture_files": [
                "supernet_bootstrap_queue.csv",
                f"benchmark_table_{n_qubits}q_v2.csv",
                "round1/round1_queue.csv",
                "round1/round1_benchmark_table.csv",
            ],
            "how_to_read_architecture": (
                "Read ansatz_gene/canonical_arch_hash from completed rows sorted by fair_best_energy. "
                "screening_energy is only a cheap ranking signal."
            ),
            "validation": "screening_energy_is_final_label=false; accept only fair_best_energy and delta_ref as final labels.",
        },
        {
            "rank": 2,
            "name": "vqe_loop_supernet_wider_ranking",
            "kind": "vqe_loop_supernet_native_search",
            "architecture_source": "generated by aicir.qas.vqe_loop supernet_native",
            "why": (
                "Use the same vqe-loop path with a larger ranking pool when the first search finds no fair-label "
                "improvement. This expands candidate coverage without introducing hand-authored architectures."
            ),
            "output_dir": f"outputs/{output_slug}_supernet_wide",
            "vqe_loop_args": {
                "--initial-labels": 0,
                "--rounds": 1,
                "--k-min": 3,
                "--supernet-native-count": max(5, bootstrap_count),
                "--supernet-native-layers": "auto",
                "--supernet-native-supernet-num": 2,
                "--supernet-native-steps": 200 if n_qubits > 8 else 150,
                "--supernet-native-ranking-num": 160 if n_qubits > 8 else 80,
                "--supernet-native-finetune-steps": 200 if n_qubits > 8 else 120,
                "--backend": "numpy",
                "--dtype": "complex128",
            },
            "architecture_files": [
                "supernet_bootstrap_queue.csv",
                f"benchmark_table_{n_qubits}q_v2.csv",
                "round1/round1_benchmark_table.csv",
            ],
            "how_to_read_architecture": "Select completed trackB_supernet rows by fair_best_energy, then inspect ansatz_gene.",
            "validation": "Compare against the first search using fair_best_energy, not rank score.",
        },
        {
            "rank": 3,
            "name": "vqe_loop_supernet_second_seed",
            "kind": "vqe_loop_supernet_native_search",
            "architecture_source": "generated by aicir.qas.vqe_loop supernet_native",
            "why": (
                "Run the same vqe-loop supernet search with a different supernet seed when the first run needs "
                "more diversity. The final architecture still comes from completed fair labels."
            ),
            "output_dir": f"outputs/{output_slug}_supernet_seed2",
            "vqe_loop_args": {
                "--initial-labels": 0,
                "--rounds": 1,
                "--supernet-native-count": bootstrap_count,
                "--supernet-native-layers": "auto",
                "--supernet-native-supernet-num": 1,
                "--supernet-native-steps": 120,
                "--supernet-native-ranking-num": 40,
                "--supernet-native-finetune-steps": 100,
                "--supernet-native-seed": 29,
                "--backend": "numpy",
                "--dtype": "complex128",
            },
            "architecture_files": [
                "supernet_bootstrap_queue.csv",
                f"benchmark_table_{n_qubits}q_v2.csv",
                "round1/round1_benchmark_table.csv",
            ],
            "how_to_read_architecture": "Use completed rows with the lowest fair_best_energy across both seed runs.",
            "validation": "Compare seed runs only by fair labels, never by screening energy.",
        },
    ]
    return search_jobs[:requested_top_k]


def _format_args(args: Mapping[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                parts.append(str(key))
            continue
        parts.append(f"{key} {value}")
    return " ".join(parts)


def _result_dirs(request: Mapping[str, Any], recommendations: list[Mapping[str, Any]]) -> list[Path]:
    raw_dirs = request.get("result_dirs", request.get("search_result_dirs", []))
    dirs: list[Path] = []
    if isinstance(raw_dirs, (str, os.PathLike)):
        raw_dirs = [raw_dirs]
    if isinstance(raw_dirs, list):
        dirs.extend(Path(item) for item in raw_dirs if item)
    if dirs:
        return dirs
    dirs.extend(Path(str(item["output_dir"])) for item in recommendations if item.get("output_dir"))
    return dirs


def _resolve_result_dir(result_dir: Path) -> Path:
    if result_dir.is_absolute():
        return result_dir
    candidates = [
        REPO_ROOT / result_dir,
        REPO_ROOT / "aicir" / "qas" / result_dir,
        Path.cwd() / result_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _architecture_csv_files(result_dir: Path) -> list[Path]:
    base = _resolve_result_dir(result_dir)
    if not base.exists():
        return []
    patterns = [
        "benchmark_table_*q_v2.csv",
        "round*/round*_benchmark_table.csv",
        "supernet_bootstrap_queue.csv",
        "round*/round*_queue.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(base.glob(pattern)))
    return files


def _queue_csv_files(result_dir: Path) -> list[Path]:
    base = _resolve_result_dir(result_dir)
    if not base.exists():
        return []
    return sorted(base.glob("supernet_bootstrap_queue.csv")) + sorted(base.glob("round*/round*_queue.csv"))


def _random_baseline_csv_files(result_dir: Path) -> list[Path]:
    base = _resolve_result_dir(result_dir)
    if not base.exists():
        return []
    return sorted(base.glob("supernet_random_baseline.csv"))


def _benchmark_csv_files(result_dir: Path) -> list[Path]:
    base = _resolve_result_dir(result_dir)
    if not base.exists():
        return []
    return sorted(base.glob("benchmark_table_*q_v2.csv")) + sorted(base.glob("round*/round*_benchmark_table.csv"))


def _architecture_sort_key(row: Mapping[str, str]) -> tuple[int, float]:
    fair = _as_float(row.get("fair_best_energy"))
    delta = _as_float(row.get("delta_ref"))
    screening = _as_float(row.get("screening_energy"))
    rank = _as_float(row.get("supernet_rank_score"))
    if fair is not None:
        return (0, fair)
    if delta is not None:
        return (1, delta)
    if screening is not None:
        return (2, screening)
    if rank is not None:
        return (3, rank)
    return (4, float("inf"))


def collect_searched_architectures(
    request: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Read architectures already produced by vqe-loop supernet outputs."""

    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    for result_dir in _result_dirs(request, recommendations):
        for csv_file in _architecture_csv_files(result_dir):
            with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    gene = str(row.get("ansatz_gene") or row.get("canonical_arch_hash") or "").strip()
                    if not gene:
                        continue
                    architecture_id = str(row.get("architecture_id") or row.get("canonical_arch_hash") or "").strip()
                    key = architecture_id or gene
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "architecture_id": architecture_id,
                            "label_status": str(row.get("label_status") or ""),
                            "source": str(row.get("source") or ""),
                            "n_params": str(row.get("n_params") or ""),
                            "two_q_count": str(row.get("two_q_count") or ""),
                            "fair_best_energy": str(row.get("fair_best_energy") or ""),
                            "delta_ref": str(row.get("delta_ref") or ""),
                            "reference_energy": str(row.get("reference_energy") or ""),
                            "screening_energy": str(row.get("screening_energy") or ""),
                            "screening_energy_is_final_label": str(
                                row.get("screening_energy_is_final_label") or ""
                            ),
                            "ansatz_gene": gene,
                            "supernet_rank_score": str(row.get("supernet_rank_score") or ""),
                            "file": str(csv_file),
                        }
                    )
    candidates.sort(key=_architecture_sort_key)
    return candidates


def _row_to_architecture(row: Mapping[str, str], csv_file: Path) -> dict[str, str] | None:
    raw_gene = str(row.get("ansatz_gene") or "").strip()
    gene = "" if raw_gene in {"", '""'} else raw_gene
    if not gene:
        gene = str(row.get("canonical_arch_hash") or "").strip()
    if gene in {"", '""'}:
        gene = ""
    if not gene:
        return None
    return {
        "architecture_id": str(row.get("architecture_id") or row.get("canonical_arch_hash") or "").strip(),
        "label_status": str(row.get("label_status") or ""),
        "source": str(row.get("source") or ""),
        "n_params": str(row.get("n_params") or ""),
        "two_q_count": str(row.get("two_q_count") or ""),
        "fair_best_energy": str(row.get("fair_best_energy") or ""),
        "delta_ref": str(row.get("delta_ref") or ""),
        "reference_energy": str(row.get("reference_energy") or ""),
        "screening_energy": str(row.get("screening_energy") or ""),
        "screening_energy_is_final_label": str(row.get("screening_energy_is_final_label") or ""),
        "ansatz_gene": gene,
        "supernet_rank_score": str(row.get("supernet_rank_score") or ""),
        "file": str(csv_file),
    }


def collect_random_architectures(
    request: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for result_dir in _result_dirs(request, recommendations):
        baseline_files = _random_baseline_csv_files(result_dir)
        candidate_files = baseline_files if baseline_files else _queue_csv_files(result_dir)
        for csv_file in candidate_files:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    architecture = _row_to_architecture(row, csv_file)
                    if architecture is not None and architecture["source"].startswith("trackB_supernet"):
                        key = architecture["architecture_id"] or architecture["ansatz_gene"]
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(architecture)
    if not any("supernet_random_baseline.csv" in row["file"] for row in rows):
        rows.sort(key=_architecture_sort_key)
    return rows


def collect_optimized_architectures(
    request: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for result_dir in _result_dirs(request, recommendations):
        for csv_file in _benchmark_csv_files(result_dir):
            with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    architecture = _row_to_architecture(row, csv_file)
                    if (
                        architecture is not None
                        and architecture["source"] == "trackB_supernet"
                        and architecture["fair_best_energy"]
                    ):
                        key = architecture["architecture_id"] or architecture["ansatz_gene"]
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(architecture)
    rows.sort(key=_architecture_sort_key)
    return rows


def _load_parameter_vector(row: Mapping[str, str]) -> list[float] | None:
    params_ref = str(row.get("supernet_init_params_ref") or "").strip()
    source_file = str(row.get("file") or "").strip()
    if not params_ref or not source_file:
        return None
    params_path = Path(source_file).parent / params_ref
    if not params_path.exists():
        return None
    loaded = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        return None
    return [float(value) for value in loaded]


def _fair_vqe_energy_for_random_architecture(
    row: Mapping[str, str],
    hamiltonian: Mapping[str, Any],
    request: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> str:
    fallback = str(row.get("screening_energy") or "n/a")
    try:
        raw_gene = str(row.get("ansatz_gene") or "").strip()
        if not raw_gene:
            return fallback
        parsed_gene = json.loads(raw_gene)
        architecture = architecture_from_supernet_gene(SupernetAnsatzGene.from_jsonable(parsed_gene))
        terms = hamiltonian.get("terms")
        if not terms:
            return fallback
        problem_terms = [(float(coeff), str(pauli)) for coeff, pauli in terms]
        problem = VQEProblem(
            name=str(hamiltonian.get("hamiltonian_id") or "literal_hamiltonian"),
            n_qubits=int(hamiltonian["n_qubits"]),
            hamiltonian=problem_terms,
            reference_energy=float(hamiltonian.get("reference_energy") or exact_ground_energy(problem_terms)),
        )
        search = request.get("search", {})
        search = search if isinstance(search, Mapping) else {}
        planned_args = recommendations[0].get("vqe_loop_args", {}) if recommendations else {}
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=int(search.get("label_seed", 5200)),
            n_starts=int(search.get("n_seeds", 1)),
            evals_per_param=200,
            max_evaluations=int(search.get("max_evals", 100)),
            budget_override=int(search.get("max_evals", 100)),
            backend=resolve_qas_backend(
                kind=str(search.get("backend", planned_args.get("--backend", "numpy"))),
                fallback_to_cpu=False,
                dtype=str(search.get("dtype", planned_args.get("--dtype", "complex128"))),
            ),
            init_mode="random_uniform_pi",
            init_scale=3.141592653589793,
            initial_parameters=_load_parameter_vector(row),
        )
        return f"{float(result.energy):.12f}"
    except Exception:
        return fallback


def _auto_layer_list(n_qubits: int) -> list[int]:
    if int(n_qubits) <= 8:
        return [1, 2, 3]
    return [1, 2]


def _search_layer_list(search: Mapping[str, Any], planned_args: Mapping[str, Any], n_qubits: int) -> list[int]:
    if "supernet_native_layers_list" in search:
        raw = search["supernet_native_layers_list"]
        if isinstance(raw, (list, tuple)):
            layers = [int(item) for item in raw]
        else:
            layers = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    else:
        raw_layers = search.get("supernet_native_layers", planned_args.get("--supernet-native-layers", "auto"))
        if str(raw_layers).strip().lower() == "auto":
            layers = _auto_layer_list(int(n_qubits))
        else:
            layers = [int(raw_layers)]
    unique: list[int] = []
    for layer in layers:
        if int(layer) <= 0:
            raise ValueError("supernet_native_layers must be positive")
        if int(layer) not in unique:
            unique.append(int(layer))
    return unique


def _supernet_gene_structure(raw_gene: str) -> str:
    try:
        gene = SupernetAnsatzGene.from_jsonable(json.loads(str(raw_gene)))
    except Exception:
        return "n/a"
    single_layers = ["[" + ",".join(layer) + "]" for layer in gene.single_qubit_layers]
    two_layers = ["[" + ",".join(layer) + "]" for layer in gene.two_qubit_layers]
    pairs = "[" + ",".join(f"{left}-{right}" for left, right in gene.two_qubit_pairs) + "]"
    topology = "linear" if gene.two_qubit_pairs == tuple((index, index + 1) for index in range(gene.n_qubits - 1)) else "custom"
    return (
        f"layers={gene.layers}; topology={topology}; pairs={pairs}; "
        f"single_layers={';'.join(single_layers)}; two_qubit_layers={';'.join(two_layers)}"
    )


def _current_architecture_structure(current: Mapping[str, Any] | None) -> str:
    if not current:
        return "n/a"
    if current.get("ansatz_gene"):
        return _supernet_gene_structure(str(current["ansatz_gene"]))
    name = str(current.get("name") or current.get("architecture_id") or "current_architecture")
    layers = current.get("layers", current.get("depth_group", "n/a"))
    topology = str(current.get("entangle_pattern") or current.get("topology") or "n/a")
    rotation = str(current.get("rotation_block") or current.get("single_blocks") or "n/a")
    entangler = str(current.get("entangler") or current.get("entangler_type") or "n/a")
    final_rotation = str(current.get("final_rotation") or "n/a")
    return (
        f"name={name}; layers={layers}; topology={topology}; "
        f"per_layer_single={rotation}; per_layer_entangler={entangler}; final_rotation={final_rotation}"
    )


def execute_search_if_requested(
    request: dict[str, Any],
    hamiltonian: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> Path | None:
    if not bool(request.get("run_search", False)):
        return None
    terms = hamiltonian.get("terms")
    if not terms:
        raise ValueError(
            "run_search=true requires resolved Pauli terms. Provide hamiltonian_file or hamiltonian.kind=pauli_terms/preset."
        )

    search = request.get("search", {})
    search = search if isinstance(search, Mapping) else {}
    planned_args = recommendations[0].get("vqe_loop_args", {}) if recommendations else {}
    base_output_dir = Path(str(request.get("output_dir") or recommendations[0]["output_dir"]))
    layer_list = _search_layer_list(search, planned_args, int(hamiltonian["n_qubits"]))
    k_min = int(search.get("k_min", planned_args.get("--k-min", 3)))
    count = int(search.get("supernet_native_count", planned_args.get("--supernet-native-count", max(1, k_min))))
    result_dirs = list(request.get("result_dirs", []))
    first_output_dir: Path | None = None
    for layer in layer_list:
        output_dir = base_output_dir if len(layer_list) == 1 else base_output_dir.with_name(f"{base_output_dir.name}_L{layer}")
        if first_output_dir is None:
            first_output_dir = output_dir
        config = ClosedLoopConfig(
            output_dir=output_dir,
            n_qubits=int(hamiltonian["n_qubits"]),
            hamiltonian_terms=list(terms),
            hamiltonian_id=str(hamiltonian.get("hamiltonian_id") or "literal_hamiltonian"),
            hamiltonian_class=str(hamiltonian.get("hamiltonian_class") or "literal"),
            rounds=int(search.get("rounds", planned_args.get("--rounds", 1))),
            initial_labels=0,
            k_min=k_min,
            supernet_native_count=count,
            supernet_native_layers=int(layer),
            supernet_native_supernet_num=int(
                search.get("supernet_native_supernet_num", planned_args.get("--supernet-native-supernet-num", 1))
            ),
            supernet_native_steps=int(search.get("supernet_native_steps", planned_args.get("--supernet-native-steps", 20))),
            supernet_native_ranking_num=int(
                search.get("supernet_native_ranking_num", planned_args.get("--supernet-native-ranking-num", max(4, count)))
            ),
            supernet_native_finetune_steps=int(
                search.get("supernet_native_finetune_steps", planned_args.get("--supernet-native-finetune-steps", 0))
            ),
            supernet_native_seed=int(search.get("supernet_native_seed", 11)) + (int(layer) * 1000 if len(layer_list) > 1 else 0),
            supernet_native_single_qubit_gates=(
                tuple(search["single_qubit_gates"]) if "single_qubit_gates" in search else None
            ),
            label_seed=int(search.get("label_seed", 5200)) + (int(layer) * 1000 if len(layer_list) > 1 else 0),
            n_seeds=int(search.get("n_seeds", 1)),
            max_evals=int(search.get("max_evals", 100)),
            backend=str(search.get("backend", planned_args.get("--backend", "numpy"))),
            dtype=str(search.get("dtype", planned_args.get("--dtype", "complex128"))),
            include_layerwise=False,
        )
        run_vqe_qas_closed_loop(config)
        if str(output_dir) not in result_dirs:
            result_dirs.insert(0, str(output_dir))
    request["result_dirs"] = result_dirs
    return first_output_dir


def render_report(
    request: Mapping[str, Any],
    hamiltonian: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> str:
    random_architectures = collect_random_architectures(request, recommendations)
    optimized_architectures = collect_optimized_architectures(request, recommendations)
    random_item = random_architectures[0] if random_architectures else None
    optimized_item = optimized_architectures[0] if optimized_architectures else None
    random_energy = (
        _fair_vqe_energy_for_random_architecture(random_item, hamiltonian, request, recommendations)
        if random_item
        else "n/a"
    )
    exact_energy = (
        optimized_item.get("reference_energy")
        if optimized_item and optimized_item.get("reference_energy")
        else str(request.get("reference_energy") or hamiltonian.get("reference_energy") or "n/a")
    )
    problem_summary = " - ".join(
        part
        for part in [
            str(request.get("scenario", "")).strip(),
            str(request.get("problem", "")).strip(),
        ]
        if part
    )
    if not problem_summary:
        problem_summary = f"{hamiltonian.get('hamiltonian_class', 'unknown')} {hamiltonian.get('n_qubits', 'n/a')}q"

    lines = [
        "=== VQE/QAS Architecture Result ===",
        f"Problem Summary: {problem_summary}",
        f"Exact Energy: {exact_energy}",
        f"Initial Architecture: {str((request.get('current_architecture') or {}).get('name') or 'n/a')}",
        f"Initial Architecture Structure: {_current_architecture_structure(request.get('current_architecture'))}",
        f"Random Architecture: {random_item['architecture_id'] if random_item else 'n/a'}",
        f"Random Architecture Energy: {random_energy}",
        f"Random Architecture Structure: {_supernet_gene_structure(random_item['ansatz_gene']) if random_item else 'n/a'}",
        f"Random Architecture Gene: {random_item['ansatz_gene'] if random_item else 'n/a'}",
        f"Optimized Architecture: {optimized_item['architecture_id'] if optimized_item else 'n/a'}",
        f"Optimized Architecture Energy: {optimized_item['fair_best_energy'] if optimized_item else 'n/a'}",
        f"Optimized Architecture Structure: {_supernet_gene_structure(optimized_item['ansatz_gene']) if optimized_item else 'n/a'}",
        f"Optimized Architecture Gene: {optimized_item['ansatz_gene'] if optimized_item else 'n/a'}",
    ]
    return "\n".join(line for line in lines if line != "")


def run_recommender(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    request = _load_json(input_path)
    hamiltonian = summarize_hamiltonian(request)
    recommendations = recommend_architectures(request, hamiltonian)
    search_output_dir = execute_search_if_requested(request, hamiltonian, recommendations)
    searched_architectures = collect_searched_architectures(request, recommendations)
    report = render_report(request, hamiltonian, recommendations)
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output),
        "hamiltonian": hamiltonian,
        "recommendations": recommendations,
        "searched_architectures": searched_architectures,
        "search_executed": search_output_dir is not None,
        "search_output_dir": str(search_output_dir) if search_output_dir is not None else "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend VQE/QAS architectures for a chemistry scenario")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Scenario JSON input")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="TXT report output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_recommender(args.input, args.output)
    print(Path(result["output"]).read_text(encoding="utf-8"))
    print(f"results_file: {result['output']}")


if __name__ == "__main__":
    main()

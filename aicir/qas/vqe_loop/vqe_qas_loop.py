"""One-call orchestration for the VQE-QAS closed loop.

The lower modules stay focused: `protocol` owns benchmark table semantics,
`geometry` owns trust-region distances, and `selection_ops` owns Stage-2
selection operators.
This module wires those pieces into an end-to-end workflow that can start from
literal Hamiltonian terms instead of a demo-specific hard-coded problem.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PauliTerm = tuple[float, str]


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Configuration for a small VQE-QAS closed-loop run."""

    output_dir: Path
    n_qubits: int
    hamiltonian_terms: Sequence[PauliTerm] | None = None
    hamiltonian_id: str = "literal_hamiltonian"
    hamiltonian_class: str = "literal"
    rounds: int = 1
    initial_labels: int = 24
    holdout_fraction: float = 0.25
    k_min: int = 3
    d_max: float = 0.28125
    local: int = 4
    boundary: int = 2
    sparse: int = 2
    control: int = 0
    ea_population: int = 32
    ea_generations: int = 8
    ea_seed_count: int = 12
    ea_seed: int = 101
    label_seed: int = 5200
    n_seeds: int = 1
    max_evals: int = 100
    backend: str = "numpy"
    dtype: str = "complex128"
    protocol: Path = Path("aicir/qas/vqe_loop/fair_label_protocol.json")


@dataclass(frozen=True)
class ClosedLoopResult:
    """Paths produced by a VQE-QAS closed-loop run."""

    output_dir: Path
    candidates: Path
    initial_queue: Path
    initial_benchmark_table: Path
    final_benchmark_table: Path
    round_summaries: tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_module(module: str, args: Sequence[str], *, cwd: Path) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=str(cwd), check=True)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def stamp_literal_hamiltonian_terms(
    csv_path: str | Path,
    terms: Sequence[PauliTerm],
    *,
    hamiltonian_id: str,
    hamiltonian_class: str = "literal",
) -> None:
    """Write literal Pauli terms into every row of a queue-like CSV file."""

    path = Path(csv_path)
    fieldnames, rows = _read_csv(path)
    required = ["hamiltonian_terms", "hamiltonian_id", "hamiltonian_class"]
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)

    encoded_terms = json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms])
    for row in rows:
        row["hamiltonian_terms"] = encoded_terms
        row["hamiltonian_id"] = hamiltonian_id
        row["hamiltonian_class"] = hamiltonian_class

    _write_csv(path, fieldnames, rows)


def run_vqe_qas_closed_loop(config: ClosedLoopConfig) -> ClosedLoopResult:
    """Run Stage 0/1.5 plus one or more Stage-2 VQE-QAS rounds."""

    repo_root = _repo_root()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_module(
        "aicir.qas.vqe_loop.preparation",
        [
            "--scales",
            str(config.n_qubits),
            "--initial-labels",
            str(config.initial_labels),
            "--holdout-fraction",
            str(config.holdout_fraction),
            "--k-min",
            str(config.k_min),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
    )

    candidates = output_dir / "stage0_candidates.csv"
    initial_queue = output_dir / "stage1_5_initial_label_queue.csv"
    if config.hamiltonian_terms:
        stamp_literal_hamiltonian_terms(
            initial_queue,
            config.hamiltonian_terms,
            hamiltonian_id=config.hamiltonian_id,
            hamiltonian_class=config.hamiltonian_class,
        )

    initial_benchmark = output_dir / f"benchmark_table_{config.n_qubits}q_v2.csv"
    _run_module(
        "aicir.qas.vqe_loop.labeling",
        [
            "--queue",
            str(initial_queue),
            "--output",
            str(initial_benchmark),
            "--protocol",
            str(config.protocol),
            "--seed",
            str(config.label_seed),
            "--n-seeds",
            str(config.n_seeds),
            "--max-evals",
            str(config.max_evals),
            "--backend",
            config.backend,
            "--dtype",
            config.dtype,
        ],
        cwd=repo_root,
    )

    benchmark = initial_benchmark
    summaries: list[Path] = []
    for round_index in range(1, int(config.rounds) + 1):
        batch_id = f"round{round_index}"
        round_dir = output_dir / batch_id
        _run_module(
            "aicir.qas.vqe_loop.stage2",
            [
                "--candidates",
                str(candidates),
                "--benchmark-table",
                str(benchmark),
                "--output-dir",
                str(round_dir),
                "--batch-id",
                batch_id,
                "--k-min",
                str(config.k_min),
                "--d-max",
                str(config.d_max),
                "--local",
                str(config.local),
                "--boundary",
                str(config.boundary),
                "--sparse",
                str(config.sparse),
                "--control",
                str(config.control),
                "--ea-population",
                str(config.ea_population),
                "--ea-generations",
                str(config.ea_generations),
                "--ea-seed-count",
                str(config.ea_seed_count),
                "--ea-seed",
                str(config.ea_seed + round_index - 1),
                "--label-seed",
                str(config.label_seed + 1000 * round_index),
                "--n-seeds",
                str(config.n_seeds),
                "--max-evals",
                str(config.max_evals),
                "--backend",
                config.backend,
                "--dtype",
                config.dtype,
            ],
            cwd=repo_root,
        )
        benchmark = round_dir / f"{batch_id}_benchmark_table.csv"
        summaries.append(round_dir / f"{batch_id}_loop_summary.json")

    return ClosedLoopResult(
        output_dir=output_dir,
        candidates=candidates,
        initial_queue=initial_queue,
        initial_benchmark_table=initial_benchmark,
        final_benchmark_table=benchmark,
        round_summaries=tuple(summaries),
    )

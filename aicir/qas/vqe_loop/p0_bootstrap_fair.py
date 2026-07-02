"""One-call orchestration for the P0 bootstrap plus fair-label VQE-QAS path.

The lower modules stay focused: `protocol` owns benchmark table semantics,
`rows` owns flat CSV/benchmark-row parsing, and the current P1 planner lives in
`p1_round.py` plus `demos/run_p1_round_demo.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from aicir.qas.vqe_loop.benchmark_table import read_csv_with_fieldnames, write_csv_rows
from aicir.qas.vqe_loop.benchmark_table import _resolve_batch_quotas
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, LabelSource, LabelStatus


PauliTerm = tuple[float, str]


@dataclass(frozen=True)
class ClosedLoopResolvedDefaults:
    """Concrete loop knobs after resolving qubit-scaled defaults."""

    initial_labels: int
    rounds: int
    local: int
    boundary: int
    sparse: int
    control: int


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Configuration for the P0 bootstrap plus fair-label run.

    ``ClosedLoopConfig`` is kept as a compatibility name; new code should use
    ``P0BootstrapConfig`` for this one-shot P0/fair path.
    """

    output_dir: Path
    n_qubits: int
    hamiltonian_terms: Sequence[PauliTerm] | None = None
    hamiltonian_id: str = "literal_hamiltonian"
    hamiltonian_class: str = "literal"
    rounds: int | None = None
    initial_labels: int | None = None
    batch_size: int | None = None
    holdout_fraction: float = 0.25
    k_min: int = 3
    d_max: float = 0.28125
    local: int | None = None
    boundary: int | None = None
    sparse: int | None = None
    control: int | None = None
    patience: int = 2
    min_improvement: float = 1.0e-8
    ea_population: int = 32
    ea_generations: int = 8
    ea_seed_count: int = 12
    ea_seed: int = 101
    supernet_native_count: int = 0
    supernet_native_layers: int = 3
    supernet_native_supernet_num: int = 2
    supernet_native_steps: int = 20
    supernet_native_ranking_num: int = 24
    supernet_native_finetune_steps: int = 0
    supernet_native_seed: int = 11
    supernet_native_device: str = "cpu"
    supernet_native_single_qubit_gates: tuple[str, ...] | None = None
    use_chemistry_excitation_pool: bool = False
    active_electrons: int | None = None
    active_spatial_orbitals: int | None = None
    chemistry_excitation_count: int = 0
    chemistry_excitation_max_excitations: int = 4
    chemistry_excitation_seed: int = 17
    enable_p0_zero_cost: bool = True
    label_seed: int = 5200
    n_seeds: int = 1
    max_evals: int = 100
    backend: str = "numpy"
    dtype: str = "complex128"
    protocol: Path = Path("default")
    include_layerwise: bool = True
    layerwise_count: int | None = None
    layerwise_layers: int = 3


@dataclass(frozen=True)
class ClosedLoopResult:
    """Paths produced by the P0 bootstrap plus fair-label run.

    ``ClosedLoopResult`` is kept as a compatibility name; new code should use
    ``P0BootstrapResult``.
    """

    output_dir: Path
    candidates: Path
    initial_queue: Path
    initial_benchmark_table: Path
    final_benchmark_table: Path
    round_summaries: tuple[Path, ...]


def default_initial_labels_for_qubits(n_qubits: int) -> int:
    """Return the default initial fair-label budget for this qubit scale."""

    n = int(n_qubits)
    if n <= 4:
        return 12
    if n <= 8:
        return 24
    if n <= 12:
        return 36
    return 48


def default_max_rounds_for_qubits(n_qubits: int) -> int:
    """Return the default maximum Stage-2 round count for this qubit scale."""

    n = int(n_qubits)
    if n <= 4:
        return 4
    if n <= 8:
        return 4
    if n <= 12:
        return 2
    return 2


def resolve_closed_loop_defaults(
    *,
    n_qubits: int,
    initial_labels: int | None = None,
    rounds: int | None = None,
    batch_size: int | None = None,
    local: int | None = None,
    boundary: int | None = None,
    sparse: int | None = None,
    control: int | None = None,
) -> ClosedLoopResolvedDefaults:
    """Resolve auto budgets while preserving explicit user overrides."""

    resolved_local, resolved_boundary, resolved_sparse, resolved_control = _resolve_batch_quotas(
        n_qubits=n_qubits,
        batch_size=batch_size,
        local=local,
        boundary=boundary,
        sparse=sparse,
        control=control,
    )
    return ClosedLoopResolvedDefaults(
        initial_labels=int(initial_labels) if initial_labels is not None else default_initial_labels_for_qubits(n_qubits),
        rounds=int(rounds) if rounds is not None else default_max_rounds_for_qubits(n_qubits),
        local=resolved_local,
        boundary=resolved_boundary,
        sparse=resolved_sparse,
        control=resolved_control,
    )


def effective_supernet_bootstrap_count(
    config: ClosedLoopConfig,
    resolved: ClosedLoopResolvedDefaults,
) -> int:
    """Return the bootstrap top-K needed to make later oracle rounds usable."""

    requested = int(config.supernet_native_count)
    if (
        int(resolved.initial_labels) <= 0
        and requested > 0
        and int(resolved.rounds) > 0
    ):
        return max(requested, int(config.k_min))
    return requested



def chemistry_bootstrap_enabled(config: ClosedLoopConfig) -> bool:
    """Return whether the chemistry excitation P0 bootstrap path is requested."""

    return bool(config.use_chemistry_excitation_pool) or int(config.chemistry_excitation_count) > 0


def p0_bootstrap_enabled(config: ClosedLoopConfig) -> bool:
    """Return whether closed-loop can create a P0 bootstrap fair-label queue."""

    return chemistry_bootstrap_enabled(config) or int(config.supernet_native_count) > 0
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_module(module: str, args: Sequence[str], *, cwd: Path) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=str(cwd), check=True)



def _best_completed_label(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, ""
    _fieldnames, rows = read_csv_with_fieldnames(path)
    completed = [
        row for row in rows
        if row.get("label_status") == "completed" and row.get("fair_best_energy") not in {"", None}
    ]
    if not completed:
        return None, ""
    best = min(completed, key=lambda row: float(row["fair_best_energy"]))
    return float(best["fair_best_energy"]), best.get("architecture_id", "")


def stamp_literal_hamiltonian_terms(
    csv_path: str | Path,
    terms: Sequence[PauliTerm],
    *,
    hamiltonian_id: str,
    hamiltonian_class: str = "literal",
) -> None:
    """Write literal Pauli terms into every row of a queue-like CSV file."""

    path = Path(csv_path)
    fieldnames, rows = read_csv_with_fieldnames(path)
    required = ["hamiltonian_terms", "hamiltonian_id", "hamiltonian_class"]
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)

    encoded_terms = json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms])
    for row in rows:
        row["hamiltonian_terms"] = encoded_terms
        row["hamiltonian_id"] = hamiltonian_id
        row["hamiltonian_class"] = hamiltonian_class

    write_csv_rows(path, rows, fieldnames=fieldnames)


def _encoded_hamiltonian_terms(terms: Sequence[PauliTerm]) -> str:
    return json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms])


def _queue_row_from_candidate_row(
    candidate_row: dict[str, object],
    *,
    protocol_version: str,
    batch_id: str,
    hamiltonian_terms: Sequence[PauliTerm],
    hamiltonian_id: str,
    hamiltonian_class: str,
    source: str = LabelSource.TRACKB_SUPERNET.value,
) -> dict[str, object]:
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update({field: candidate_row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})
    row.update(
        {
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": source,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "hamiltonian_terms": _encoded_hamiltonian_terms(hamiltonian_terms),
            "hamiltonian_id": hamiltonian_id,
            "hamiltonian_class": hamiltonian_class,
            "hamiltonian_coverage_features": row.get("hamiltonian_coverage_features")
            or row.get("hamiltonian_coverage", ""),
        }
    )
    return row


def write_supernet_bootstrap_queue(
    config: ClosedLoopConfig,
    *,
    output_dir: Path,
    protocol_version: str = "fair_vqe_protocol_v2",
) -> tuple[Path, Path, Path]:
    """Write supernet cheap-ranking top candidates as bootstrap oracle records.

    This path intentionally does not need an initial fair-label benchmark.  It
    samples/ranks in the native supernet, writes the top rows into the vqe_loop
    queue schema, and leaves fair VQE as a later verification step for only
    those selected top candidates.
    """

    if not config.hamiltonian_terms:
        raise ValueError("supernet bootstrap requires literal hamiltonian_terms")
    if int(config.supernet_native_count) <= 0:
        raise ValueError("supernet bootstrap requires supernet_native_count > 0")

    from aicir.qas.vqe_loop.p0_supernet_native import build_supernet_native_rows
    from aicir.qas.vqe_loop.training_free import annotate_p0_bootstrap_rows

    output_dir = Path(output_dir)
    queue_path = output_dir / "supernet_bootstrap_queue.csv"
    oracle_records_path = output_dir / "supernet_bootstrap_oracle_records.csv"
    random_baseline_path = output_dir / "supernet_random_baseline.csv"
    summary_path = output_dir / "supernet_bootstrap_plan_summary.json"
    rows, supernet_summary = build_supernet_native_rows(
        hamiltonian_terms=list(config.hamiltonian_terms),
        hamiltonian_id=str(config.hamiltonian_id),
        hamiltonian_class=str(config.hamiltonian_class),
        count=int(config.supernet_native_count),
        layers=int(config.supernet_native_layers),
        supernet_num=int(config.supernet_native_supernet_num),
        supernet_steps=int(config.supernet_native_steps),
        ranking_num=int(config.supernet_native_ranking_num),
        finetune_steps=int(config.supernet_native_finetune_steps),
        seed=int(config.supernet_native_seed),
        device=str(config.supernet_native_device),
        single_qubit_gates=config.supernet_native_single_qubit_gates,
        excluded_ids=set(),
        params_dir=output_dir,
    )
    rows, training_free_summary = annotate_p0_bootstrap_rows(rows, enabled=bool(config.enable_p0_zero_cost))
    queue_rows = [
        _queue_row_from_candidate_row(
            row,
            protocol_version=protocol_version,
            batch_id="supernet_bootstrap",
            hamiltonian_terms=list(config.hamiltonian_terms),
            hamiltonian_id=str(config.hamiltonian_id),
            hamiltonian_class=str(config.hamiltonian_class),
        )
        for row in rows
    ]
    write_csv_rows(queue_path, queue_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    write_csv_rows(oracle_records_path, queue_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    random_baseline_row = supernet_summary.pop("random_baseline_row", None)
    if random_baseline_row:
        random_baseline_queue_row = _queue_row_from_candidate_row(
            random_baseline_row,
            protocol_version=protocol_version,
            batch_id="supernet_random_baseline",
            hamiltonian_terms=list(config.hamiltonian_terms),
            hamiltonian_id=str(config.hamiltonian_id),
            hamiltonian_class=str(config.hamiltonian_class),
        )
        write_csv_rows(random_baseline_path, [random_baseline_queue_row], fieldnames=BENCHMARK_TABLE_FIELDS)
    else:
        write_csv_rows(random_baseline_path, [], fieldnames=BENCHMARK_TABLE_FIELDS)
    summary = {
        "mode": "supernet_native_bootstrap",
        "source": LabelSource.TRACKB_SUPERNET.value,
        "queue": str(queue_path),
        "oracle_records": str(oracle_records_path),
        "random_baseline": str(random_baseline_path),
        "planned_total": len(queue_rows),
        "screening_energy_is_final_label": False,
        "training_free": training_free_summary,
        "supernet_native": supernet_summary,
        "note": "Supernet cheap ranking records are oracle-side candidates; fair VQE labels only verify these top candidates.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return queue_path, oracle_records_path, summary_path


def write_chemistry_excitation_bootstrap_queue(
    config: ClosedLoopConfig,
    *,
    output_dir: Path,
    protocol_version: str = "fair_vqe_protocol_v2",
) -> tuple[Path, Path, Path]:
    """Write chemistry excitation candidates as the P0 fair-label bootstrap queue."""

    if not config.hamiltonian_terms:
        raise ValueError("chemistry excitation bootstrap requires literal hamiltonian_terms")
    if config.active_electrons is None:
        raise ValueError("chemistry excitation bootstrap requires active_electrons")
    if config.active_spatial_orbitals is None:
        raise ValueError("chemistry excitation bootstrap requires active_spatial_orbitals")
    if int(config.active_spatial_orbitals) * 2 != int(config.n_qubits):
        raise ValueError("active_spatial_orbitals must match n_qubits / 2 for chemistry excitation bootstrap")

    from aicir.qas.vqe_loop.p0_chemistry_excitation import build_chemistry_excitation_rows
    from aicir.qas.vqe_loop.training_free import annotate_p0_bootstrap_rows

    output_dir = Path(output_dir)
    queue_path = output_dir / "chemistry_excitation_bootstrap_queue.csv"
    oracle_records_path = output_dir / "chemistry_excitation_bootstrap_oracle_records.csv"
    summary_path = output_dir / "chemistry_excitation_bootstrap_plan_summary.json"
    count = int(config.chemistry_excitation_count)
    if count <= 0:
        count = int(config.initial_labels or 0)
    if count <= 0:
        count = int(config.k_min)
    rows, chemistry_summary = build_chemistry_excitation_rows(
        active_electrons=int(config.active_electrons),
        active_spatial_orbitals=int(config.active_spatial_orbitals),
        hamiltonian_id=str(config.hamiltonian_id),
        hamiltonian_class=str(config.hamiltonian_class),
        count=count,
        max_excitations=int(config.chemistry_excitation_max_excitations),
        seed=int(config.chemistry_excitation_seed),
        excluded_ids=set(),
    )
    rows, training_free_summary = annotate_p0_bootstrap_rows(rows, enabled=bool(config.enable_p0_zero_cost))
    queue_rows = [
        _queue_row_from_candidate_row(
            row,
            protocol_version=protocol_version,
            batch_id="chemistry_excitation_bootstrap",
            hamiltonian_terms=list(config.hamiltonian_terms),
            hamiltonian_id=str(config.hamiltonian_id),
            hamiltonian_class=str(config.hamiltonian_class),
            source=LabelSource.TRACKB_CHEMISTRY_EXCITATION.value,
        )
        for row in rows
    ]
    write_csv_rows(queue_path, queue_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    write_csv_rows(oracle_records_path, queue_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    summary = {
        "mode": "chemistry_excitation_bootstrap",
        "source": LabelSource.TRACKB_CHEMISTRY_EXCITATION.value,
        "queue": str(queue_path),
        "oracle_records": str(oracle_records_path),
        "planned_total": len(queue_rows),
        "screening_energy_is_final_label": False,
        "training_free": training_free_summary,
        "chemistry_excitation": chemistry_summary,
        "note": "Chemistry excitation candidates are screened structurally only; fair VQE labels remain the comparison signal.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return queue_path, oracle_records_path, summary_path

def run_p0_bootstrap_fair(config: ClosedLoopConfig) -> ClosedLoopResult:
    """Run the supported P0 bootstrap plus fair-label step.

    Legacy Stage-0 preparation and Stage-2 trust-region search were removed
    from this one-call API.  Use the bootstrap writers here for P0 and
    ``demos.run_p1_round_demo`` for P1 mutation/oracle/fallback rounds.
    """

    repo_root = _repo_root()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_closed_loop_defaults(
        n_qubits=config.n_qubits,
        initial_labels=config.initial_labels,
        rounds=config.rounds,
        batch_size=config.batch_size,
        local=config.local,
        boundary=config.boundary,
        sparse=config.sparse,
        control=config.control,
    )
    if not p0_bootstrap_enabled(config):
        raise ValueError(
            "closed loop now requires a P0 bootstrap family; configure "
            "use_chemistry_excitation_pool/chemistry_excitation_count or supernet_native_count"
        )
    if int(resolved.initial_labels) > 0:
        raise ValueError(
            "legacy preparation.py initial-label generation was removed; use P0 bootstrap "
            "with initial_labels=0 and a chemistry or supernet bootstrap writer"
        )
    if int(resolved.rounds) > 0:
        raise RuntimeError(
            "The legacy Stage-2 closed-loop runner was removed. Run P0 bootstrap/fair here, "
            "then use aicir.qas.demos.run_p1_round_demo for P1."
        )

    bootstrap_count = effective_supernet_bootstrap_count(config, resolved)
    if bootstrap_count != int(config.supernet_native_count):
        config = replace(config, supernet_native_count=bootstrap_count)

    bootstrap_summary: Path | None = None
    if chemistry_bootstrap_enabled(config):
        initial_queue, oracle_records, bootstrap_summary = write_chemistry_excitation_bootstrap_queue(
            config,
            output_dir=output_dir,
            protocol_version="fair_vqe_protocol_v2",
        )
    else:
        initial_queue, oracle_records, bootstrap_summary = write_supernet_bootstrap_queue(
            config,
            output_dir=output_dir,
            protocol_version="fair_vqe_protocol_v2",
        )

    initial_benchmark = output_dir / f"benchmark_table_{config.n_qubits}q_v2.csv"
    _run_module(
        "aicir.qas.vqe_loop.fair_labeling",
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

    best_energy, best_architecture = _best_completed_label(initial_benchmark)
    loop_summary = {
        "output_dir": str(output_dir),
        "n_qubits": int(config.n_qubits),
        "mode": "p0_bootstrap_fair_only",
        "initial_labels": 0,
        "bootstrap_summary": str(bootstrap_summary) if bootstrap_summary is not None else "",
        "rounds_requested": resolved.rounds,
        "rounds_completed": 0,
        "stop_reason": "p1_requires_run_p1_round_demo",
        "final_best_energy": best_energy,
        "final_best_architecture": best_architecture,
        "final_benchmark_table": str(initial_benchmark),
        "rounds": [],
    }
    summary_path = output_dir / "closed_loop_summary.json"
    summary_path.write_text(json.dumps(loop_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ClosedLoopResult(
        output_dir=output_dir,
        candidates=oracle_records,
        initial_queue=initial_queue,
        initial_benchmark_table=initial_benchmark,
        final_benchmark_table=initial_benchmark,
        round_summaries=(summary_path,),
    )





# Compatibility aliases for older demos/tests that still use the historical
# closed-loop name. The implementation above is intentionally P0/fair only.
P0BootstrapConfig = ClosedLoopConfig
P0BootstrapResult = ClosedLoopResult
run_vqe_qas_closed_loop = run_p0_bootstrap_fair


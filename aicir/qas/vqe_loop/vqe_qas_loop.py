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
class ClosedLoopResolvedDefaults:
    """Concrete loop knobs after resolving qubit-scaled defaults."""

    initial_labels: int
    rounds: int
    local: int
    boundary: int
    sparse: int
    control: int


@dataclass(frozen=True)
class QuotaDecision:
    """Stage-2 quota decision derived from oracle calibration quality."""

    mode: str
    local: int
    boundary: int
    sparse: int
    control: int
    reason: str

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.local, self.boundary, self.sparse, self.control


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Configuration for a small VQE-QAS closed-loop run."""

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
    label_seed: int = 5200
    n_seeds: int = 1
    max_evals: int = 100
    backend: str = "numpy"
    dtype: str = "complex128"
    protocol: Path = Path("aicir/qas/vqe_loop/fair_label_protocol.json")
    include_layerwise: bool = True
    layerwise_count: int | None = None
    layerwise_layers: int = 3


@dataclass(frozen=True)
class ClosedLoopResult:
    """Paths produced by a VQE-QAS closed-loop run."""

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


def default_batch_quotas_for_qubits(n_qubits: int) -> tuple[int, int, int, int]:
    """Return local/boundary/sparse/control quotas for this qubit scale."""

    n = int(n_qubits)
    if n <= 4:
        return 2, 1, 1, 0
    if n <= 8:
        return 2, 2, 2, 0
    if n <= 12:
        return 3, 2, 2, 1
    return 4, 3, 3, 2


def _batch_quotas_from_total(total: int) -> tuple[int, int, int, int]:
    total = max(0, int(total))
    local = min(total, max(0, round(total * 0.4)))
    boundary = min(total - local, max(0, round(total * 0.3)))
    sparse = min(total - local - boundary, max(0, round(total * 0.2)))
    control = max(0, total - local - boundary - sparse)
    return int(local), int(boundary), int(sparse), int(control)


def _quota_from_weights(total: int, weights: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    total = max(0, int(total))
    if total == 0:
        return 0, 0, 0, 0
    raw = [max(0.0, float(weight)) for weight in weights]
    weight_total = sum(raw)
    if weight_total <= 0:
        return 0, 0, 0, total
    scaled = [total * weight / weight_total for weight in raw]
    floors = [int(value) for value in scaled]
    remainder = total - sum(floors)
    order = sorted(range(4), key=lambda index: (scaled[index] - floors[index], -index), reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    return tuple(floors)  # type: ignore[return-value]


def _local_cap_for_qubits(n_qubits: int, total: int, mode: str) -> int:
    n = int(n_qubits)
    total = int(total)
    if mode == "local":
        fraction = 0.55 if n <= 4 else 0.45 if n <= 8 else 0.35 if n <= 12 else 0.30
    elif mode == "balanced_explore":
        fraction = 0.30 if n <= 4 else 0.25 if n <= 8 else 0.20
    else:
        fraction = 0.20 if n <= 8 else 0.15
    return max(0, int(round(total * fraction)))


def _apply_local_cap(quotas: tuple[int, int, int, int], *, n_qubits: int, mode: str) -> tuple[int, int, int, int]:
    local, boundary, sparse, control = quotas
    total = local + boundary + sparse + control
    cap = _local_cap_for_qubits(n_qubits, total, mode)
    if local <= cap:
        return quotas
    extra = local - cap
    local = cap
    # Exploration surplus goes first to sparse, then boundary/control.
    sparse += (extra + 1) // 2
    boundary += extra // 2
    return local, boundary, sparse, control


def _calibration_float(calibration: dict[str, object], key: str) -> float | None:
    raw = calibration.get(key)
    if raw in {"", None}:
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def decide_next_round_quotas(
    *,
    n_qubits: int,
    base_quotas: tuple[int, int, int, int],
    calibration: dict[str, object] | None,
    local_improved: bool | None = None,
) -> QuotaDecision:
    """Choose local/exploration quota mix from trust-region calibration."""

    total = sum(int(value) for value in base_quotas)
    if total <= 0:
        return QuotaDecision("none", 0, 0, 0, 0, "empty_batch")
    if not calibration:
        local, boundary, sparse, control = _apply_local_cap(base_quotas, n_qubits=n_qubits, mode="explore")
        return QuotaDecision("explore", local, boundary, sparse, control, "missing_calibration")

    passes = calibration.get("passes", {})
    if not isinstance(passes, dict):
        passes = {}
    k_min = int(calibration.get("k_min") or 3)
    tr_in_count = int(calibration.get("tr_in_count") or 0)
    tr_in_mae = _calibration_float(calibration, "tr_in_mae")
    tr_out_mae = _calibration_float(calibration, "tr_out_mae")
    sparse_abstain_rate = _calibration_float(calibration, "sparse_abstain_rate")
    mae_ratio = (tr_in_mae / tr_out_mae) if tr_in_mae is not None and tr_out_mae not in {None, 0.0} else None
    sparse_ok = sparse_abstain_rate is not None and sparse_abstain_rate >= 0.8
    weak_signal = tr_in_count > 0 and tr_in_mae is not None and tr_out_mae is not None and tr_in_mae < tr_out_mae and sparse_ok

    if bool(passes.get("overall")) and tr_in_count >= k_min and (mae_ratio is None or mae_ratio <= 0.5):
        mode = "local"
        weights = (0.55, 0.25, 0.10, 0.10) if int(n_qubits) <= 8 else (0.40, 0.25, 0.20, 0.15)
        reason = "oracle_passed"
    elif weak_signal:
        if int(n_qubits) >= 12 and local_improved is False:
            mode = "sparse_explore"
            weights = (0.10, 0.15, 0.50, 0.25)
        else:
            mode = "balanced_explore"
            weights = (0.25, 0.30, 0.30, 0.15) if int(n_qubits) <= 8 else (0.15, 0.25, 0.35, 0.25)
        reason = "weak_tr_signal"
    else:
        mode = "explore"
        weights = (0.15, 0.35, 0.35, 0.15) if int(n_qubits) <= 8 else (0.10, 0.35, 0.35, 0.20)
        reason = "oracle_not_trusted"

    quotas = _apply_local_cap(_quota_from_weights(total, weights), n_qubits=n_qubits, mode=mode)
    if local_improved is False and mode != "explore" and quotas[0] > 1:
        local, boundary, sparse, control = quotas
        local -= 1
        sparse += 1
        quotas = local, boundary, sparse, control
        reason += "_local_failed"
    return QuotaDecision(mode, quotas[0], quotas[1], quotas[2], quotas[3], reason)


def _resolve_batch_quotas(
    *,
    n_qubits: int,
    batch_size: int | None,
    local: int | None,
    boundary: int | None,
    sparse: int | None,
    control: int | None,
) -> tuple[int, int, int, int]:
    if batch_size is None:
        base = default_batch_quotas_for_qubits(n_qubits)
        return (
            int(local) if local is not None else base[0],
            int(boundary) if boundary is not None else base[1],
            int(sparse) if sparse is not None else base[2],
            int(control) if control is not None else base[3],
        )

    total = max(0, int(batch_size))
    base_by_name = dict(zip(("local", "boundary", "sparse", "control"), _batch_quotas_from_total(total)))
    explicit = {
        "local": local,
        "boundary": boundary,
        "sparse": sparse,
        "control": control,
    }
    resolved: dict[str, int] = {
        name: max(0, int(value))
        for name, value in explicit.items()
        if value is not None
    }
    remaining = max(0, total - sum(resolved.values()))
    missing = [name for name in ("local", "boundary", "sparse", "control") if name not in resolved]
    if missing:
        base_total = sum(base_by_name[name] for name in missing)
        for name in missing[:-1]:
            share = round(remaining * (base_by_name[name] / base_total)) if base_total else 0
            share = min(remaining, max(0, int(share)))
            resolved[name] = share
            remaining -= share
        resolved[missing[-1]] = remaining
    return resolved["local"], resolved["boundary"], resolved["sparse"], resolved["control"]


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


def _best_completed_label(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, ""
    _fieldnames, rows = _read_csv(path)
    completed = [
        row for row in rows
        if row.get("label_status") == "completed" and row.get("fair_best_energy") not in {"", None}
    ]
    if not completed:
        return None, ""
    best = min(completed, key=lambda row: float(row["fair_best_energy"]))
    return float(best["fair_best_energy"]), best.get("architecture_id", "")


def _is_improvement(previous_best: float | None, current_best: float | None, min_improvement: float) -> bool:
    if current_best is None:
        return False
    if previous_best is None:
        return True
    return current_best < previous_best - float(min_improvement)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _local_source_improved(labels_path: Path, previous_best: float | None, min_improvement: float) -> bool:
    if previous_best is None or not labels_path.exists():
        return False
    _fieldnames, rows = _read_csv(labels_path)
    for row in rows:
        if row.get("source") != "trackA_local" or row.get("fair_best_energy") in {"", None}:
            continue
        try:
            energy = float(row["fair_best_energy"])
        except (TypeError, ValueError):
            continue
        if energy < previous_best - float(min_improvement):
            return True
    return False


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

    preparation_args = [
        "--scales",
        str(config.n_qubits),
        "--hamiltonian-class",
        str(config.hamiltonian_class),
        "--hamiltonian-id",
        str(config.hamiltonian_id),
        "--initial-labels",
        str(resolved.initial_labels),
        "--holdout-fraction",
        str(config.holdout_fraction),
        "--k-min",
        str(config.k_min),
        "--output-dir",
        str(output_dir),
    ]
    if config.include_layerwise:
        preparation_args.extend(
            [
                "--include-layerwise",
                "--layerwise-count",
                str(config.layerwise_count if config.layerwise_count is not None else resolved.initial_labels),
                "--layerwise-layers",
                str(config.layerwise_layers),
            ]
        )
    _run_module("aicir.qas.vqe_loop.preparation", preparation_args, cwd=repo_root)

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
    round_records: list[dict[str, object]] = []
    best_energy, best_architecture = _best_completed_label(benchmark)
    no_improvement_rounds = 0
    stop_reason = "max_rounds"
    base_quotas = (resolved.local, resolved.boundary, resolved.sparse, resolved.control)
    next_quota_decision = QuotaDecision("default", *base_quotas, reason="initial_round")
    for round_index in range(1, int(resolved.rounds) + 1):
        batch_id = f"round{round_index}"
        round_dir = output_dir / batch_id
        previous_best = best_energy
        quota_decision = next_quota_decision
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
                str(quota_decision.local),
                "--boundary",
                str(quota_decision.boundary),
                "--sparse",
                str(quota_decision.sparse),
                "--control",
                str(quota_decision.control),
                "--ea-population",
                str(config.ea_population),
                "--ea-generations",
                str(config.ea_generations),
                "--ea-seed-count",
                str(config.ea_seed_count),
                "--ea-seed",
                str(config.ea_seed + round_index - 1),
                "--supernet-native-count",
                str(config.supernet_native_count),
                "--supernet-native-layers",
                str(config.supernet_native_layers),
                "--supernet-native-supernet-num",
                str(config.supernet_native_supernet_num),
                "--supernet-native-steps",
                str(config.supernet_native_steps),
                "--supernet-native-ranking-num",
                str(config.supernet_native_ranking_num),
                "--supernet-native-finetune-steps",
                str(config.supernet_native_finetune_steps),
                "--supernet-native-seed",
                str(config.supernet_native_seed + round_index - 1),
                "--supernet-native-device",
                str(config.supernet_native_device),
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
        labels_path = round_dir / f"{batch_id}_labels.csv"
        calibration_path = round_dir / f"{batch_id}_oracle_calibration.json"
        current_best, current_architecture = _best_completed_label(benchmark)
        improved = _is_improvement(previous_best, current_best, config.min_improvement)
        local_improved = _local_source_improved(labels_path, previous_best, config.min_improvement)
        calibration = _load_json(calibration_path)
        next_quota_decision = decide_next_round_quotas(
            n_qubits=config.n_qubits,
            base_quotas=base_quotas,
            calibration=calibration,
            local_improved=local_improved,
        )
        if improved:
            best_energy = current_best
            best_architecture = current_architecture
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1
            if current_best is not None and (best_energy is None or current_best < best_energy):
                best_energy = current_best
                best_architecture = current_architecture
        round_records.append(
            {
                "round": round_index,
                "batch_id": batch_id,
                "benchmark_table": str(benchmark),
                "summary": str(summaries[-1]),
                "best_before": previous_best,
                "best_after": current_best,
                "best_architecture": current_architecture,
                "improved": improved,
                "local_improved": local_improved,
                "quota_mode": quota_decision.mode,
                "quota_reason": quota_decision.reason,
                "batch_quotas": {
                    "local": quota_decision.local,
                    "boundary": quota_decision.boundary,
                    "sparse": quota_decision.sparse,
                    "control": quota_decision.control,
                },
                "next_quota_mode": next_quota_decision.mode,
                "next_quota_reason": next_quota_decision.reason,
                "next_batch_quotas": {
                    "local": next_quota_decision.local,
                    "boundary": next_quota_decision.boundary,
                    "sparse": next_quota_decision.sparse,
                    "control": next_quota_decision.control,
                },
                "no_improvement_rounds": no_improvement_rounds,
            }
        )
        if int(config.patience) > 0 and no_improvement_rounds >= int(config.patience):
            stop_reason = "patience"
            break

    loop_summary = {
        "output_dir": str(output_dir),
        "n_qubits": int(config.n_qubits),
        "initial_labels": resolved.initial_labels,
        "rounds_requested": resolved.rounds,
        "rounds_completed": len(round_records),
        "batch_quotas": {
            "local": resolved.local,
            "boundary": resolved.boundary,
            "sparse": resolved.sparse,
            "control": resolved.control,
        },
        "next_batch_quotas": {
            "mode": next_quota_decision.mode,
            "reason": next_quota_decision.reason,
            "local": next_quota_decision.local,
            "boundary": next_quota_decision.boundary,
            "sparse": next_quota_decision.sparse,
            "control": next_quota_decision.control,
        },
        "patience": int(config.patience),
        "min_improvement": float(config.min_improvement),
        "stop_reason": stop_reason,
        "final_best_energy": best_energy,
        "final_best_architecture": best_architecture,
        "final_benchmark_table": str(benchmark),
        "rounds": round_records,
    }
    (output_dir / "closed_loop_summary.json").write_text(
        json.dumps(loop_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return ClosedLoopResult(
        output_dir=output_dir,
        candidates=candidates,
        initial_queue=initial_queue,
        initial_benchmark_table=initial_benchmark,
        final_benchmark_table=benchmark,
        round_summaries=tuple(summaries),
    )

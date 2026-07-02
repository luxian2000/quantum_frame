"""Run frozen fair-VQE labels for a VQE-QAS benchmark queue.

This runner intentionally does not rank architectures.  It turns pending queue
rows into protocol-versioned labels, including literal-Hamiltonian support,
warm-start parameters, best traces, retry status, and backend/dtype metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.primitives.ansatz import HEAMask, architecture_from_hea_mask
from aicir.qas.primitives.backend_utils import resolve_qas_backend
from aicir.qas.vqe_loop.benchmark_table import architecture_from_candidate_row
from aicir.qas.vqe_loop.benchmark_table import read_csv_rows, write_csv_rows
from aicir.qas.vqe_loop.benchmark_table import problem_from_row_terms, row_hamiltonian_terms
from aicir.qas.vqe_loop.fair_vqe import (
    fair_vqe_final_maxfev,
    optimize_vqe_energy,
)
from aicir.metrics.circuit_structure import parameter_count
from aicir.qas.problems.hamiltonians import (
    VQEProblem,
    tfim_chain_demo_problem,
)
from aicir.qas.vqe_loop.benchmark_table import (
    BENCHMARK_TABLE_FIELDS,
    LabelStatus,
    next_label_status_after_failure,
    load_fair_label_protocol,
)



def _load_warm_start_vector(
    row: dict[str, Any],
    *,
    queue_path: Path,
    n_params: int,
) -> tuple[list[float] | None, str]:
    raw_ref = str(row.get("supernet_init_params_ref", "") or "").strip()
    if not raw_ref:
        return None, "missing"
    ref_path = Path(raw_ref)
    if not ref_path.is_absolute():
        ref_path = queue_path.parent / ref_path
    if not ref_path.exists():
        return None, "missing_file"
    try:
        loaded = json.loads(ref_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid_json"
    if isinstance(loaded, list):
        values = loaded
    elif isinstance(loaded, dict):
        values = list(loaded.values())
    else:
        return None, "invalid_format"
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None, "invalid_value"
    if len(vector) != int(n_params):
        return None, "param_count_mismatch"
    return vector, "loaded"


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if str(protocol.get("protocol_version", "")).strip() != "fair_vqe_protocol_v2":
        raise ValueError("This runner currently implements fair_vqe_protocol_v2 only")
    if protocol.get("frozen") is not True:
        raise ValueError("Fair VQE protocol must be explicitly frozen before labels are generated")
    energy_evaluation = protocol.get("energy_evaluation", {})
    if energy_evaluation.get("state") != "unmeasured_state" or energy_evaluation.get("shots", "missing") is not None:
        raise ValueError("fair_vqe_protocol_v2 requires unmeasured_state energy evaluation with shots=null")
    return protocol


def _validate_queue_protocol_versions(rows: list[dict[str, str]], protocol_version: str) -> None:
    invalid = [
        (index + 1, row.get("architecture_id", ""), row.get("protocol_version", ""))
        for index, row in enumerate(rows)
        if str(row.get("protocol_version", "")).strip() != protocol_version
    ]
    if invalid:
        preview = ", ".join(
            f"line={line} architecture_id={architecture_id!r} protocol_version={version!r}"
            for line, architecture_id, version in invalid[:5]
        )
        raise ValueError(
            f"Queue protocol_version must equal {protocol_version!r} for every row; "
            f"found {len(invalid)} invalid row(s): {preview}"
        )


def _mask_from_row(row: dict[str, str]) -> HEAMask:
    raw = row.get("hea_mask", "")
    if not raw:
        raise ValueError("missing hea_mask")
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f"hea_mask must be a JSON list of length 6, got {raw!r}")
    return HEAMask(
        n_qubits=int(values[0]),
        layers=int(values[1]),
        rotation_block=str(values[2]),
        entangler=str(values[3]),
        final_rotation=str(values[4]),
        entangle_pattern=str(values[5]),
    )


def _architecture_from_row(row: dict[str, str]):
    raw_gene = row.get("ansatz_gene", "")
    if raw_gene and str(raw_gene).strip() not in {"", '""', "null"}:
        return architecture_from_candidate_row(row)
    return architecture_from_hea_mask(_mask_from_row(row))


def _tfim_problem_from_protocol(n_qubits: int, protocol: dict[str, Any]):
    hamiltonian = protocol.get("hamiltonian", {})
    boundary = str(hamiltonian.get("boundary", "OBC")).strip().upper()
    return tfim_chain_demo_problem(
        n_qubits=n_qubits,
        J=float(hamiltonian.get("J", 1.0)),
        h=float(hamiltonian.get("h", 0.5)),
        periodic=boundary in {"PBC", "PERIODIC", "RING"},
    )


def _problem_from_row_or_protocol(row: dict[str, Any], *, n_qubits: int, protocol: dict[str, Any]) -> VQEProblem:
    terms = row_hamiltonian_terms(row)
    if not terms:
        return _tfim_problem_from_protocol(n_qubits, protocol)
    return problem_from_row_terms(row, n_qubits=int(n_qubits), default_name_prefix="custom_pauli")


def _architecture_seed_key(row: dict[str, Any]) -> str:
    for field in ("canonical_arch_hash", "ansatz_gene", "architecture_id"):
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _label_seed_for_row(
    *,
    base_seed: int,
    row_index: int,
    row: dict[str, Any],
    seed_index_offset: int = 0,
    seed_by_architecture_id: bool = False,
) -> int:
    if seed_by_architecture_id:
        key = _architecture_seed_key(row)
        if key:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            return int(base_seed) + int(digest[:12], 16) % 1_000_000_000
    return int(base_seed) + (int(seed_index_offset) + int(row_index)) * 1000


def _label_row(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    seed: int,
    n_seeds: int,
    success_delta_ref: float,
    max_evals_override: int | None,
    backend_kind: str,
    dtype: str,
    queue_path: Path,
) -> dict[str, Any]:
    start = perf_counter()
    architecture = _architecture_from_row(row)
    n_qubits = int(row.get("n_qubits") or architecture.n_qubits)
    problem = _problem_from_row_or_protocol(row, n_qubits=n_qubits, protocol=protocol)
    backend = resolve_qas_backend(kind=backend_kind, fallback_to_cpu=False, dtype=dtype)
    n_params = parameter_count(architecture.circuit)
    max_evals = int(max_evals_override) if max_evals_override is not None else fair_vqe_final_maxfev(n_params)
    warm_start, warm_start_status = _load_warm_start_vector(row, queue_path=queue_path, n_params=n_params)

    energies: list[float] = []
    nfev_total = 0
    best_trace: list[dict[str, Any]] = []
    for seed_index in range(max(1, int(n_seeds))):
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=int(seed) + seed_index,
            n_starts=1,
            evals_per_param=200,
            max_evaluations=max_evals,
            budget_override=max_evals,
            backend=backend,
            init_mode="random_uniform_pi",
            init_scale=float(np.pi),
            initial_parameters=warm_start,
        )
        energies.append(float(result.energy))
        nfev_total += int(result.evaluations)
        best_trace.append(
            {
                "seed": int(seed) + seed_index,
                "energy": float(result.energy),
                "nfev": int(result.evaluations),
                "theta_init_mode": result.metadata.get("theta_init_mode", ""),
                "supernet_warm_start_status": warm_start_status,
                "best_parameters": [float(value) for value in result.best_parameters],
            }
        )

    values = np.asarray(energies, dtype=float)
    reference = float(problem.reference_energy)
    row.update(
        {
            "protocol_version": protocol["protocol_version"],
            "label_status": LabelStatus.COMPLETED.value,
            "failure_reason": "",
            "last_error_digest": "",
            "fair_best_energy": f"{float(np.min(values)):.12f}",
            "fair_mean_energy": f"{float(np.mean(values)):.12f}",
            "fair_std_energy": f"{float(np.std(values)):.12f}",
            "fair_success_rate": f"{float(np.mean(values <= reference + float(success_delta_ref))):.6f}",
            "delta_ref": f"{float(np.min(values) - reference):.12f}",
            "reference_energy": f"{reference:.12f}",
            "optimizer": "COBYLA",
            "n_seeds": int(n_seeds),
            "max_evals": int(max_evals),
            "nfev": int(nfev_total),
            "walltime_s": f"{float(perf_counter() - start):.6f}",
            "success_delta_ref": f"{float(success_delta_ref):.12f}",
            "best_trace": json.dumps(best_trace, ensure_ascii=False),
            "supernet_warm_start_status": warm_start_status,
            "dtype": dtype,
            "backend": backend_kind,
            "created_at": f"{perf_counter():.6f}",
            "hamiltonian_id": row.get("hamiltonian_id") or problem.name,
            "hamiltonian_class": row.get("hamiltonian_class") or "tfim",
        }
    )
    row["hamiltonian_coverage_features"] = row.get("hamiltonian_coverage_features") or row.get("hamiltonian_coverage", "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair VQE labels for a Stage-1.5 queue")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--seed-index-offset",
        type=int,
        default=0,
        help="Offset added to the queue row index when deriving per-row VQE seeds; used by sharded runners.",
    )
    parser.add_argument(
        "--seed-by-architecture-id",
        action="store_true",
        help="Derive per-row seeds from architecture identity instead of queue row index.",
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--success-delta-ref", type=float, default=0.02)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    rows = read_csv_rows(queue_path)
    protocol = load_fair_label_protocol(args.protocol)
    _validate_queue_protocol_versions(rows, str(protocol["protocol_version"]))
    completed = 0
    for index, row in enumerate(rows):
        if row.get("label_status") not in {"", LabelStatus.PENDING.value, LabelStatus.FAILED_RETRYABLE.value}:
            continue
        if args.limit is not None and completed >= int(args.limit):
            break
        if str(row.get("protocol_version", "")).strip() != protocol["protocol_version"]:
            row["label_status"] = LabelStatus.EXCLUDED_PROTOCOL_MISMATCH.value
            continue
        if args.dry_run:
            completed += 1
            continue
        row["label_status"] = LabelStatus.RUNNING.value
        write_csv_rows(Path(args.output), rows, fieldnames=BENCHMARK_TABLE_FIELDS)
        try:
            rows[index] = _label_row(
                row,
                protocol=protocol,
                seed=_label_seed_for_row(
                    base_seed=int(args.seed),
                    row_index=index,
                    row=row,
                    seed_index_offset=int(args.seed_index_offset),
                    seed_by_architecture_id=bool(args.seed_by_architecture_id),
                ),
                n_seeds=int(args.n_seeds),
                success_delta_ref=float(args.success_delta_ref),
                max_evals_override=args.max_evals,
                backend_kind=args.backend,
                dtype=args.dtype,
                queue_path=queue_path,
            )
        except Exception as exc:  # pragma: no cover - exercised by real runners.
            retry_count = int(row.get("retry_count") or 0) + 1
            row["retry_count"] = retry_count
            row["label_status"] = next_label_status_after_failure(retry_count=retry_count).value
            row["failure_reason"] = type(exc).__name__
            row["last_error_digest"] = traceback.format_exc()[-4000:]
        completed += 1
        write_csv_rows(Path(args.output), rows, fieldnames=BENCHMARK_TABLE_FIELDS)

    write_csv_rows(Path(args.output), rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    print(json.dumps({"rows": len(rows), "attempted": completed, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()



"""Benchmark-table schema and label-state helpers for VQE-QAS.

This module is intentionally small: it defines the CSV contract, label states,
row identity rules, default protocol knobs, and retry-state transitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

class LabelStatus(str, Enum):
    """Allowed benchmark-table label states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_NONRETRYABLE = "failed_nonretryable"
    EXCLUDED_PROTOCOL_MISMATCH = "excluded_protocol_mismatch"
    EXCLUDED_INVALID_CANDIDATE = "excluded_invalid_candidate"
    EXCLUDED_DUPLICATE = "excluded_duplicate"


class LabelSource(str, Enum):
    """Allowed sources for fair-VQE labels and queued labels."""

    INITIAL_TRAIN = "initial_train"
    INITIAL_CALIBRATION = "initial_calibration"
    HOLDOUT_ID = "holdout_id"
    HOLDOUT_BOUNDARY = "holdout_boundary"
    HOLDOUT_SPARSE = "holdout_sparse"
    TRACKA_LOCAL = "trackA_local"
    TRACKB_BOUNDARY = "trackB_boundary"
    TRACKB_SPARSE = "trackB_sparse"
    CONTROL_RANDOM = "control_random"
    BASELINE = "baseline"


class ZeroCostStatus(str, Enum):
    """Stage-1b zero-cost feasibility status."""

    PASS = "pass"
    SOFT_FLAG = "soft_flag"
    HARD_REJECT = "hard_reject"


BENCHMARK_TABLE_FIELDS: tuple[str, ...] = (
    "architecture_id",
    "canonical_arch_hash",
    "protocol_version",
    "batch_id",
    "source",
    "label_status",
    "retry_count",
    "failure_reason",
    "last_error_digest",
    "n_qubits",
    "hamiltonian_id",
    "hamiltonian_class",
    "family",
    "depth_group",
    "entangler_type",
    "topology",
    "n_params",
    "two_q_count",
    "hamiltonian_coverage",
    "hamiltonian_coverage_features",
    "hamiltonian_terms",
    "zero_cost_status",
    "zero_cost_reasons",
    "expressibility_score",
    "trainability_score",
    "entanglement_score",
    "zero_cost_feature_score",
    "zero_cost_score_is_ranking_signal",
    "hea_mask",
    "ansatz_gene",
    "supernet_rank_score",
    "supernet_init_params_ref",
    "screening_energy",
    "screening_energy_is_final_label",
    "supernet_warm_start_status",
    "fair_best_energy",
    "fair_mean_energy",
    "fair_std_energy",
    "fair_success_rate",
    "delta_ref",
    "reference_energy",
    "optimizer",
    "n_seeds",
    "max_evals",
    "nfev",
    "walltime_s",
    "success_delta_ref",
    "best_trace",
    "dtype",
    "backend",
    "created_at",
)


def benchmark_row_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the task-local identity for benchmark row replacement."""

    architecture_id = str(row.get("architecture_id", ""))
    hamiltonian_id = str(row.get("hamiltonian_id", ""))
    if not hamiltonian_id:
        hamiltonian_class = str(row.get("hamiltonian_class", ""))
        n_qubits = str(row.get("n_qubits", ""))
        hamiltonian_id = f"{hamiltonian_class}_{n_qubits}q" if hamiltonian_class and n_qubits else ""
    protocol_version = str(row.get("protocol_version", ""))
    return architecture_id, hamiltonian_id, protocol_version


def append_benchmark_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    incoming_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Append label rows, replacing stale rows for the same architecture/task."""

    merged: list[dict[str, Any]] = [dict(row) for row in existing_rows]
    index_by_identity = {
        benchmark_row_identity(row): index
        for index, row in enumerate(merged)
        if benchmark_row_identity(row)[0]
    }
    for row in incoming_rows:
        copied = dict(row)
        identity = benchmark_row_identity(copied)
        if not identity[0]:
            merged.append(copied)
            continue
        if identity in index_by_identity:
            merged[index_by_identity[identity]] = copied
        else:
            index_by_identity[identity] = len(merged)
            merged.append(copied)
    return merged


DEFAULT_RETRY_POLICY: dict[str, float | int] = {
    "max_retry": 2,
    "running_timeout_multiplier": 1.5,
}


DEFAULT_TRUST_REGION_RULES: dict[str, float | int] = {
    "k_min": 5,
    "abstain_rate_max": 0.40,
    "target_sparse_abstain_rate": 0.80,
    "target_tr_coverage": 0.20,
    "target_tr_in_mae_ratio": 0.50,
}


DEFAULT_BATCH_QUOTAS: dict[str, int] = {
    "local": 12,
    "boundary": 10,
    "sparse": 6,
    "control": 4,
}


DEFAULT_SMALL_BATCH_QUOTAS: dict[str, int] = {
    "local": 6,
    "boundary": 5,
    "sparse": 3,
    "control": 2,
}




def next_label_status_after_failure(
    *,
    retry_count: int,
    max_retry: int = int(DEFAULT_RETRY_POLICY["max_retry"]),
) -> LabelStatus:
    """Return retryable/nonretryable state after a failed run."""

    return LabelStatus.FAILED_NONRETRYABLE if int(retry_count) >= int(max_retry) else LabelStatus.FAILED_RETRYABLE

"""Benchmark table schema, row IO, fair-label protocol, and row policies.

This module is the VQE-loop boundary for the unified benchmark table.  It owns
flat CSV rows, label states, fair-label protocol defaults, row-to-object parsing,
and P1 policies that operate on benchmark rows.  Final comparisons still use
``fair_best_energy`` from the shared fair-label path.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

from aicir.core.operators import Hamiltonian
from aicir.qas.core._types import ArchitectureSpec
from aicir.qas.library.ansatz import (
    ChemistryExcitationAnsatzGene,
    ExplicitGateAnsatzGene,
    LayerwiseAnsatzGene,
    OperatorSequenceAnsatzGene,
    SupernetAnsatzGene,
    architecture_from_chemistry_excitation_gene,
    architecture_from_explicit_gate_gene,
    architecture_from_layerwise_gene,
    architecture_from_operator_sequence_gene,
    architecture_from_supernet_gene,
)
from aicir.qas.problems.hamiltonians import VQEProblem, exact_ground_energy


def _raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_size_limit()

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
    TRACKB_SUPERNET = "trackB_supernet"
    TRACKB_CHEMISTRY_EXCITATION = "trackB_chemistry_excitation"
    CONTROL_RANDOM = "control_random"
    P1_ORACLE = "p1_oracle"
    P1_FALLBACK = "p1_fallback"
    P1_CONTROL = "p1_control"
    BASELINE_RANDOM = "baseline_random"
    BASELINE_E2 = "baseline_e2"
    BASELINE_E5 = "baseline_e5"
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
    "parent_architecture_id",
    "crossover_parent_architecture_id",
    "mutation_type",
    "p1_selection_source",
    "predicted_fair_energy",
    "oracle_reason",
    "oracle_neighbor_count",
    "oracle_nearest_distance",
    "oracle_kth_distance",
    "oracle_neighbor_target_std",
    "fallback_selector",
    "fallback_score",
    "VQE_TASK_PROXY",
    "GNN_PROXY",
    "ENSEMBLE",
    "predictor_confidence",
    "task_proxy_hamiltonian_overlap",
    "task_proxy_gradient_sensitivity",
    "task_proxy_adapt_growth_potential",
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

PauliTerm = tuple[float, str]


def is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def as_float(value: Any) -> float | None:
    if is_empty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_with_fieldnames(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    return read_csv_with_fieldnames(path)[1]


def union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(str(field))
    return fieldnames


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else union_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in columns})


def decoded_ansatz_gene_payload(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw_gene = row.get("ansatz_gene", "")
    if raw_gene is None or str(raw_gene).strip() in {"", '""', "null"}:
        return None
    parsed = json.loads(raw_gene) if isinstance(raw_gene, str) else raw_gene
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError("ansatz_gene must decode to a JSON object")
    return parsed


def architecture_from_ansatz_gene_payload(payload: Mapping[str, Any]) -> ArchitectureSpec:
    kind = str(payload.get("kind", "")).lower()
    if kind == "supernet_native":
        return architecture_from_supernet_gene(SupernetAnsatzGene.from_jsonable(payload))
    if kind == "chemistry_excitation":
        return architecture_from_chemistry_excitation_gene(ChemistryExcitationAnsatzGene.from_jsonable(payload))
    if kind == "explicit_gate_sequence":
        return architecture_from_explicit_gate_gene(ExplicitGateAnsatzGene.from_jsonable(payload))
    if kind == "operator_sequence":
        return architecture_from_operator_sequence_gene(OperatorSequenceAnsatzGene.from_jsonable(payload))
    return architecture_from_layerwise_gene(LayerwiseAnsatzGene.from_jsonable(payload))


def architecture_from_candidate_row(row: Mapping[str, Any]) -> ArchitectureSpec:
    cached = row.get("_cached_architecture")
    if isinstance(cached, ArchitectureSpec):
        return cached
    payload = decoded_ansatz_gene_payload(row)
    if payload is None:
        raise ValueError("candidate row requires ansatz_gene")
    architecture = architecture_from_ansatz_gene_payload(payload)
    if isinstance(row, dict):
        row["_cached_architecture"] = architecture
    return architecture


def _term_coeff_and_pauli(term: Any) -> tuple[float, str]:
    if isinstance(term, Mapping):
        coeff = term.get("coefficient", term.get("coeff", term.get("weight", 0.0)))
        pauli = term.get("pauli", term.get("pauli_string", term.get("string", "")))
        return float(coeff), str(pauli).upper()
    if isinstance(term, (list, tuple)) and len(term) >= 2:
        return float(term[0]), str(term[1]).upper()
    raise ValueError(f"Unsupported Hamiltonian term format: {term!r}")


def parse_pauli_hamiltonian_terms(terms: Sequence[Any]) -> tuple[PauliTerm, ...]:
    """Parse literal Pauli-sum terms into canonical ``(coeff, pauli)`` tuples."""

    return tuple(_term_coeff_and_pauli(term) for term in terms)


def row_hamiltonian_terms(row: Mapping[str, Any], *, required: bool = False) -> tuple[PauliTerm, ...]:
    raw = row.get("hamiltonian_terms", "")
    if is_empty(raw):
        if required:
            raise ValueError("hamiltonian_terms must contain at least one Pauli term")
        return ()
    try:
        loaded = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("hamiltonian_terms must be a JSON list of Pauli terms") from exc
    terms = parse_pauli_hamiltonian_terms(loaded)
    if required and not terms:
        raise ValueError("hamiltonian_terms must contain at least one Pauli term")
    return tuple((float(coeff), str(pauli).upper()) for coeff, pauli in terms)


def validate_term_widths(terms: Sequence[PauliTerm], *, n_qubits: int) -> None:
    widths = {len(str(pauli)) for _coeff, pauli in terms}
    if widths != {int(n_qubits)}:
        raise ValueError(
            f"hamiltonian_terms width must match n_qubits={int(n_qubits)}; found widths={sorted(widths)}"
        )


def hamiltonian_from_terms(terms: Sequence[PauliTerm], *, n_qubits: int) -> Hamiltonian:
    validate_term_widths(terms, n_qubits=int(n_qubits))
    return Hamiltonian(n_qubits=int(n_qubits), terms=[(pauli, coeff) for coeff, pauli in terms])


def problem_from_terms(
    terms: Sequence[PauliTerm],
    *,
    n_qubits: int,
    name: str,
    reference_energy: float | None = None,
) -> VQEProblem:
    validate_term_widths(terms, n_qubits=int(n_qubits))
    ref = exact_ground_energy(terms) if reference_energy is None else float(reference_energy)
    return VQEProblem(name=str(name), n_qubits=int(n_qubits), hamiltonian=tuple(terms), reference_energy=ref)


def problem_from_row_terms(
    row: Mapping[str, Any],
    *,
    n_qubits: int | None = None,
    default_problem: VQEProblem | None = None,
    default_name_prefix: str = "row_pauli",
) -> VQEProblem:
    cached = row.get("_cached_problem")
    if isinstance(cached, VQEProblem):
        return cached
    terms = row_hamiltonian_terms(row, required=default_problem is None)
    if not terms:
        if default_problem is None:
            raise ValueError("experiment row requires hamiltonian_terms when no default problem is provided")
        if isinstance(row, dict):
            row["_cached_problem"] = default_problem
        return default_problem
    widths = {len(pauli) for _coeff, pauli in terms}
    resolved_n_qubits = int(n_qubits if n_qubits is not None else (row.get("n_qubits") or max(widths)))
    raw_reference = row.get("reference_energy", "")
    reference = None if is_empty(raw_reference) else float(raw_reference)
    problem = problem_from_terms(
        terms,
        n_qubits=resolved_n_qubits,
        name=str(row.get("hamiltonian_id") or f"{default_name_prefix}_{resolved_n_qubits}q"),
        reference_energy=reference,
    )
    if isinstance(row, dict):
        row["_cached_problem"] = problem
    return problem


DEFAULT_FAIR_LABEL_PROTOCOL: dict[str, Any] = {
    "protocol_version": "fair_vqe_protocol_v2",
    "frozen": True,
    "scope": "QAS fair VQE labels computed from the unmeasured final state",
    "energy_evaluation": {
        "state": "unmeasured_state",
        "shots": None,
    },
    "hamiltonian": {
        "default_class": "tfim",
        "J": 1.0,
        "h": 0.5,
        "boundary": "OBC",
    },
    "theta_initialization": {
        "mode": "random_uniform_pi",
        "range": [-3.141592653589793, 3.141592653589793],
    },
    "optimizer": {
        "name": "COBYLA",
        "rhobeg": 1.0,
        "tol": 1e-6,
        "final_maxfev": "max(1000, 200 * n_params)",
    },
    "required_outputs": [
        "fair_best_energy",
        "fair_mean_energy",
        "fair_std_energy",
        "fair_success_rate",
        "delta_ref",
        "reference_energy",
        "n_seeds",
        "max_evals",
        "nfev",
        "walltime_s",
        "best_trace",
    ],
    "versioning_rules": [
        "Rows from different protocol_version values must not train or validate the same oracle.",
        "Any future change to energy evaluation, optimizer defaults, label budget, initialization, or reference construction requires a new protocol_version.",
    ],
}


def default_fair_label_protocol() -> dict[str, Any]:
    """Return a copy of the built-in frozen fair-label protocol."""

    return copy.deepcopy(DEFAULT_FAIR_LABEL_PROTOCOL)


def validate_fair_label_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a mutable fair-label protocol dictionary."""

    copied = dict(protocol)
    if str(copied.get("protocol_version", "")).strip() != "fair_vqe_protocol_v2":
        raise ValueError("This runner currently implements fair_vqe_protocol_v2 only")
    if copied.get("frozen") is not True:
        raise ValueError("Fair VQE protocol must be explicitly frozen before labels are generated")
    energy_evaluation = copied.get("energy_evaluation", {})
    if not isinstance(energy_evaluation, Mapping):
        raise ValueError("fair_vqe_protocol_v2 requires an energy_evaluation object")
    if energy_evaluation.get("state") != "unmeasured_state" or energy_evaluation.get("shots", "missing") is not None:
        raise ValueError("fair_vqe_protocol_v2 requires unmeasured_state energy evaluation with shots=null")
    return copy.deepcopy(copied)


def load_fair_label_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load a fair-label protocol or return the built-in default.

    ``default`` and the removed legacy ``fair_label_protocol.json`` path both
    resolve to the built-in protocol so older command lines remain usable.
    """

    if path is None:
        return validate_fair_label_protocol(DEFAULT_FAIR_LABEL_PROTOCOL)
    raw_path = str(path).strip()
    if raw_path in {"", "default", "builtin"}:
        return validate_fair_label_protocol(DEFAULT_FAIR_LABEL_PROTOCOL)
    protocol_path = Path(raw_path)
    if not protocol_path.exists() and protocol_path.name == "fair_label_protocol.json":
        return validate_fair_label_protocol(DEFAULT_FAIR_LABEL_PROTOCOL)
    with protocol_path.open(encoding="utf-8") as handle:
        return validate_fair_label_protocol(json.load(handle))

@dataclass
class FairBudgetTracker:
    """Track fair-VQE calls so selector baselines can be compared fairly."""

    rounds: int
    fair_top_k_per_round: int
    round_fair_calls: dict[str, int] = field(default_factory=dict)

    @property
    def expected_total_fair_calls(self) -> int:
        return max(0, int(self.rounds)) * max(0, int(self.fair_top_k_per_round))

    @property
    def total_fair_calls(self) -> int:
        return sum(max(0, int(value)) for value in self.round_fair_calls.values())

    @property
    def remaining_fair_calls(self) -> int:
        return max(0, self.expected_total_fair_calls - self.total_fair_calls)

    def record_round(self, round_id: str, fair_calls: int) -> None:
        self.round_fair_calls[str(round_id)] = max(0, int(fair_calls))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "rounds": int(self.rounds),
            "fair_top_k_per_round": int(self.fair_top_k_per_round),
            "expected_total_fair_calls": self.expected_total_fair_calls,
            "total_fair_calls": self.total_fair_calls,
            "remaining_fair_calls": self.remaining_fair_calls,
            "round_fair_calls": dict(self.round_fair_calls),
        }


@dataclass(frozen=True)
class DeduplicationResult:
    new_children: list[dict[str, Any]]
    reused_labeled: list[dict[str, Any]]
    skipped_duplicate_architecture_ids: list[str]


@dataclass(frozen=True)
class P1Quota:
    q0: int
    q1: int
    qc: int

    @property
    def total(self) -> int:
        return max(0, int(self.q0)) + max(0, int(self.q1)) + max(0, int(self.qc))

    def to_jsonable(self) -> dict[str, int]:
        return {"q0": int(self.q0), "q1": int(self.q1), "qc": int(self.qc)}



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


def _canonical_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
        if isinstance(loaded, str):
            return _canonical_json(loaded)
        value = loaded
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def architecture_key(row: Mapping[str, Any]) -> str:
    """Return a stable architecture identity for deduplication."""

    canonical = row.get("canonical_arch_hash")
    if not is_empty(canonical):
        return str(canonical)
    ansatz_gene = row.get("ansatz_gene")
    if not is_empty(ansatz_gene):
        return _canonical_json(ansatz_gene)
    return str(row.get("architecture_id", "")).strip()


def _architecture_id(row: Mapping[str, Any], fallback: str) -> str:
    value = str(row.get("architecture_id", "")).strip()
    return value if value else fallback


def deduplicate_children(
    children: Sequence[Mapping[str, Any]],
    *,
    labeled_rows: Sequence[Mapping[str, Any]],
    known_unlabeled_rows: Sequence[Mapping[str, Any]] = (),
) -> DeduplicationResult:
    """Deduplicate generated children before L0/L1/L2 evaluation.

    Labeled duplicates are returned for zero-cost reuse.  Known unlabeled and
    current-round duplicates are skipped to avoid repeated evaluation.
    """

    labeled_by_key = {
        architecture_key(row): dict(row)
        for row in labeled_rows
        if architecture_key(row) and not is_empty(row.get("fair_best_energy"))
    }
    blocked_keys = {
        architecture_key(row)
        for row in known_unlabeled_rows
        if architecture_key(row)
    }
    seen_new_keys: set[str] = set()
    reused_keys: set[str] = set()
    new_children: list[dict[str, Any]] = []
    reused_labeled: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index, child in enumerate(children):
        key = architecture_key(child)
        identifier = _architecture_id(child, f"child:{index}")
        if key in labeled_by_key:
            if key not in reused_keys:
                reused_labeled.append(dict(labeled_by_key[key]))
                reused_keys.add(key)
            continue
        if key in blocked_keys or key in seen_new_keys:
            skipped.append(identifier)
            continue
        new_children.append(dict(child))
        seen_new_keys.add(key)
    return DeduplicationResult(
        new_children=new_children,
        reused_labeled=reused_labeled,
        skipped_duplicate_architecture_ids=skipped,
    )


def resolve_p1_selector_fields(selector: str, *, cheap_eval_selector: str = "e2") -> tuple[str, ...]:
    """Resolve P1 fallback selector fields.

    P1 treats ``both`` as a P0-diagnostic mode and delegates it to the configured
    cheap selector so abstain children do not pay for both E2 and E5.
    """

    normalized = str(selector).strip().lower()
    if normalized == "both":
        normalized = str(cheap_eval_selector).strip().lower()
    if normalized == "e2":
        return ("E2",)
    if normalized == "e5":
        return ("E5",)
    if normalized in {"task_proxy", "vqe_task_proxy"}:
        return ("VQE_TASK_PROXY",)
    if normalized in {"gnn_proxy", "graph_predictor"}:
        return ("GNN_PROXY",)
    if normalized == "ensemble":
        return ("ENSEMBLE",)
    raise ValueError(f"unsupported P1 selector: {selector}")


def choose_quota(
    fair_top_k: int,
    *,
    oracle_trusted_count: int,
    previous_oracle_trusted_fair_mean: float | None = None,
    previous_fallback_fair_mean: float | None = None,
    bad_oracle_tolerance: float = 0.0,
) -> P1Quota:
    """Choose q0/q1/qc for quota merge.

    Energies are minimized.  If the previous trusted-oracle slice was worse
    than fallback, shift quota away from q0 without disabling oracle entirely.
    """

    k = max(0, int(fair_top_k))
    if k == 0:
        return P1Quota(q0=0, q1=0, qc=0)
    if int(oracle_trusted_count) <= 0:
        return P1Quota(q0=0, q1=k, qc=0)

    oracle_was_worse = (
        previous_oracle_trusted_fair_mean is not None
        and previous_fallback_fair_mean is not None
        and float(previous_oracle_trusted_fair_mean) > float(previous_fallback_fair_mean) + float(bad_oracle_tolerance)
    )
    q0_fraction = 0.3 if oracle_was_worse else 0.6
    q1_fraction = 0.6 if oracle_was_worse else 0.3
    q0 = min(int(oracle_trusted_count), int(ceil(q0_fraction * k)))
    qc = 1 if k >= 5 else 0
    q1 = max(0, k - q0 - qc)
    target_q1 = int(ceil(q1_fraction * k))
    if q1 < target_q1 and q0 > 0:
        shift = min(q0, target_q1 - q1)
        q0 -= shift
        q1 += shift
    return P1Quota(q0=q0, q1=q1, qc=max(0, k - q0 - q1))


def rank_rows(
    rows: Sequence[Mapping[str, Any]],
    score_field: str | None,
    *,
    descending: bool = False,
    include_missing: bool = True,
) -> list[dict[str, Any]]:
    if score_field is None:
        return [dict(row) for row in rows]
    scored: list[tuple[float, str, dict[str, Any]]] = []
    missing: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        identifier = _architecture_id(row, f"row:{index}")
        value = as_float(row.get(score_field))
        if value is None:
            missing.append((identifier, copied))
        else:
            scored.append((value, identifier, copied))
    ranked = sorted(scored, key=lambda item: (-item[0] if descending else item[0], item[1]))
    output = [row for _value, _identifier, row in ranked]
    if include_missing:
        output.extend(row for _identifier, row in sorted(missing))
    return output


def rank_with_zero_cost_soft_prefilter(
    rows: Sequence[Mapping[str, Any]],
    score_field: str,
    *,
    target_count: int,
    window_multiplier: int = 2,
) -> list[dict[str, Any]]:
    """Rank by score, but prefer non-soft-flag rows within the top window."""

    ranked = rank_rows(rows, score_field)
    count = max(0, int(target_count))
    if count == 0:
        return ranked
    window_size = max(count, count * max(1, int(window_multiplier)))
    window = ranked[:window_size]
    tail = ranked[window_size:]
    pass_rows = [row for row in window if str(row.get("zero_cost_status", "")).strip() != "soft_flag"]
    soft_rows = [row for row in window if str(row.get("zero_cost_status", "")).strip() == "soft_flag"]
    return pass_rows + soft_rows + tail


def take_unique_with_source(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    source: str,
    selected_keys: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if len(selected) >= max(0, int(count)):
            break
        key = architecture_key(row)
        if key in selected_keys:
            continue
        copied = dict(row)
        copied["p1_selection_source"] = source
        selected.append(copied)
        selected_keys.add(key)
    return selected


def merge_quota_candidates(
    oracle_rows: Sequence[Mapping[str, Any]],
    fallback_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    quota: P1Quota,
    fallback_score_field: str,
    fallback_soft_prefilter_multiplier: int | None = None,
    control_score_field: str | None = None,
    control_score_descending: bool = False,
) -> list[dict[str, Any]]:
    """Merge L0/L1/control rows by quota, not by cross-scale score mixing."""

    selected_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    fallback_ranked = (
        rank_with_zero_cost_soft_prefilter(
            fallback_rows,
            fallback_score_field,
            target_count=quota.q1,
            window_multiplier=int(fallback_soft_prefilter_multiplier),
        )
        if fallback_soft_prefilter_multiplier is not None
        else rank_rows(fallback_rows, fallback_score_field)
    )
    merged.extend(
        take_unique_with_source(
            rank_rows(oracle_rows, "predicted_fair_energy"),
            quota.q0,
            "oracle_trusted",
            selected_keys,
        )
    )
    merged.extend(
        take_unique_with_source(
            fallback_ranked,
            quota.q1,
            "fallback_selector",
            selected_keys,
        )
    )
    merged.extend(
        take_unique_with_source(
            rank_rows(control_rows, control_score_field, descending=control_score_descending),
            quota.qc,
            "control",
            selected_keys,
        )
    )
    return merged


__all__ = [
    "BENCHMARK_TABLE_FIELDS",
    "DEFAULT_BATCH_QUOTAS",
    "DEFAULT_FAIR_LABEL_PROTOCOL",
    "DEFAULT_RETRY_POLICY",
    "DEFAULT_SMALL_BATCH_QUOTAS",
    "DEFAULT_TRUST_REGION_RULES",
    "DeduplicationResult",
    "FairBudgetTracker",
    "LabelSource",
    "LabelStatus",
    "P1Quota",
    "PauliTerm",
    "QuotaDecision",
    "ZeroCostStatus",
    "append_benchmark_rows",
    "architecture_from_ansatz_gene_payload",
    "architecture_from_candidate_row",
    "architecture_key",
    "as_float",
    "benchmark_row_identity",
    "choose_quota",
    "decoded_ansatz_gene_payload",
    "deduplicate_children",
    "decide_next_round_quotas",
    "default_batch_quotas_for_qubits",
    "default_fair_label_protocol",
    "hamiltonian_from_terms",
    "is_empty",
    "load_fair_label_protocol",
    "merge_quota_candidates",
    "next_label_status_after_failure",
    "parse_pauli_hamiltonian_terms",
    "problem_from_row_terms",
    "problem_from_terms",
    "rank_with_zero_cost_soft_prefilter",
    "read_csv_rows",
    "read_csv_with_fieldnames",
    "resolve_p1_selector_fields",
    "row_hamiltonian_terms",
    "union_fieldnames",
    "validate_fair_label_protocol",
    "validate_term_widths",
    "write_csv_rows",
]

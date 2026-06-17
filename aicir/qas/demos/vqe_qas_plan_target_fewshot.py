"""Plan target-Hamiltonian few-shot labels for VQE-QAS transfer tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_hea_demo import tfim_chain_hamiltonian
from aicir.qas.vqe_qas_protocol import (
    BENCHMARK_TABLE_FIELDS,
    CandidateRecord,
    LabelSource,
    LabelStatus,
    extract_pauli_hamiltonian_features,
    parse_pauli_hamiltonian_terms,
    select_target_fewshot_batch,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})


def _load_terms(args: argparse.Namespace) -> list[Any]:
    if args.target_terms_json:
        return json.loads(args.target_terms_json)
    if args.target_terms_file:
        return json.loads(Path(args.target_terms_file).read_text(encoding="utf-8"))
    return []


def _terms_for_candidate_row(
    row: dict[str, str],
    args: argparse.Namespace,
    literal_terms: list[Any],
) -> tuple[tuple[float, str], ...]:
    n_qubits = int(row.get("n_qubits") or 0)
    if literal_terms:
        terms = parse_pauli_hamiltonian_terms(literal_terms)
        widths = {len(pauli) for _coeff, pauli in terms}
        return terms if widths == {n_qubits} else ()

    uses_tfim_template = getattr(args, "target_tfim_J", None) is not None or getattr(args, "target_tfim_h", None) is not None
    if uses_tfim_template:
        boundary = str(getattr(args, "target_tfim_boundary", "OBC")).strip().upper()
        return tfim_chain_hamiltonian(
            n_qubits=n_qubits,
            J=float(getattr(args, "target_tfim_J", 1.0) if getattr(args, "target_tfim_J", None) is not None else 1.0),
            h=float(getattr(args, "target_tfim_h", 0.5) if getattr(args, "target_tfim_h", None) is not None else 0.5),
            periodic=boundary in {"PBC", "PERIODIC", "RING"},
        )

    raise ValueError("Target few-shot planning requires --target-terms-json/--target-terms-file or --target-tfim-J/--target-tfim-h")


def _candidate_record_from_row_for_target(
    row: dict[str, str],
    *,
    target_hamiltonian_id: str,
    target_hamiltonian_class: str,
    hamiltonian_features: dict[str, float],
) -> CandidateRecord:
    return CandidateRecord(
        architecture_id=row["architecture_id"],
        canonical_arch_hash=row.get("canonical_arch_hash", row["architecture_id"]),
        family=row.get("family", ""),
        entangler_type=row.get("entangler_type", ""),
        topology=row.get("topology", ""),
        depth_group=row.get("depth_group", ""),
        n_params=float(row.get("n_params") or 0),
        two_q_count=float(row.get("two_q_count") or 0),
        hamiltonian_id=target_hamiltonian_id,
        hamiltonian_class=target_hamiltonian_class,
        hamiltonian_coverage=float(row.get("hamiltonian_coverage") or 0),
        hamiltonian_features=hamiltonian_features,
        metadata={
            "n_qubits": int(row.get("n_qubits") or 0),
            "zero_cost_status": row.get("zero_cost_status", ""),
            "hamiltonian_id": target_hamiltonian_id,
            "hea_mask": row.get("hea_mask", ""),
        },
    )


def _queue_row_for_target(
    candidate_row: dict[str, str],
    *,
    source: LabelSource,
    protocol_version: str,
    batch_id: str,
    target_hamiltonian_id: str,
    target_hamiltonian_class: str,
    hamiltonian_features: dict[str, float],
    hamiltonian_terms: list[Any],
) -> dict[str, Any]:
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update({field: candidate_row.get(field, "") for field in row})
    row.update(
        {
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": source.value,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "hamiltonian_id": target_hamiltonian_id,
            "hamiltonian_class": target_hamiltonian_class,
            "hamiltonian_coverage_features": json.dumps(hamiltonian_features, sort_keys=True),
            "hamiltonian_terms": json.dumps(hamiltonian_terms, ensure_ascii=False),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan target-Hamiltonian few-shot VQE labels")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--target-hamiltonian-id", required=True)
    parser.add_argument("--target-hamiltonian-class", default="tfim")
    parser.add_argument("--target-terms-json", default="")
    parser.add_argument("--target-terms-file", default="")
    parser.add_argument("--target-tfim-J", type=float, default=None)
    parser.add_argument("--target-tfim-h", type=float, default=None)
    parser.add_argument("--target-tfim-boundary", default="OBC")
    parser.add_argument("--n-qubits", type=int, action="append", default=[])
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--batch-id", default="target_fewshot_v1")
    parser.add_argument("--total-labels", type=int, default=32)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--group-key", default="n_qubits")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--trust-d-max", type=float, default=None)
    args = parser.parse_args()

    candidate_rows = _read_csv(Path(args.candidates))
    allowed_n_qubits = {int(value) for value in args.n_qubits}
    if allowed_n_qubits:
        candidate_rows = [
            row for row in candidate_rows
            if int(row.get("n_qubits") or 0) in allowed_n_qubits
        ]
    literal_terms = _load_terms(args)
    records: list[CandidateRecord] = []
    terms_by_architecture_id: dict[str, tuple[tuple[float, str], ...]] = {}
    features_by_architecture_id: dict[str, dict[str, float]] = {}
    skipped_width_mismatch = 0
    for row in candidate_rows:
        if row.get("zero_cost_status") == "hard_reject":
            continue
        row_terms = _terms_for_candidate_row(row, args, literal_terms)
        if not row_terms:
            skipped_width_mismatch += 1
            continue
        features = extract_pauli_hamiltonian_features(row_terms)
        architecture_id = row["architecture_id"]
        terms_by_architecture_id[architecture_id] = row_terms
        features_by_architecture_id[architecture_id] = features
        records.append(
            _candidate_record_from_row_for_target(
                row,
                target_hamiltonian_id=args.target_hamiltonian_id,
                target_hamiltonian_class=args.target_hamiltonian_class,
                hamiltonian_features=features,
            )
        )
    selected = select_target_fewshot_batch(
        records,
        target_hamiltonian_id=args.target_hamiltonian_id,
        total_labels=int(args.total_labels),
        holdout_fraction=float(args.holdout_fraction),
        group_key=args.group_key or None,
        trust_d_max=args.trust_d_max,
        k_min=args.k_min,
    )
    row_by_id = {row["architecture_id"]: row for row in candidate_rows}
    queue_rows = [
        _queue_row_for_target(
            row_by_id[architecture_id],
            source=source,
            protocol_version=args.protocol_version,
            batch_id=args.batch_id,
            target_hamiltonian_id=args.target_hamiltonian_id,
            target_hamiltonian_class=args.target_hamiltonian_class,
            hamiltonian_features=features_by_architecture_id[architecture_id],
            hamiltonian_terms=list(terms_by_architecture_id[architecture_id]),
        )
        for architecture_id, source in selected.items()
    ]
    _write_csv(Path(args.output), queue_rows)

    source_counts: dict[str, int] = {}
    for source in selected.values():
        source_counts[source.value] = source_counts.get(source.value, 0) + 1
    summary = {
        "target_hamiltonian_id": args.target_hamiltonian_id,
        "target_hamiltonian_class": args.target_hamiltonian_class,
        "n_source_candidates": len(candidate_rows),
        "n_eligible_target_candidates": len(records),
        "n_skipped_width_mismatch": skipped_width_mismatch,
        "planned_total": len(queue_rows),
        "source_counts": source_counts,
        "k_min": args.k_min,
        "trust_d_max": args.trust_d_max,
        "hamiltonian_feature_examples": {
            architecture_id: features
            for architecture_id, features in list(features_by_architecture_id.items())[:3]
        },
        "note": "This queue creates target-task labels for few-shot transfer; source labels are not reused as target labels.",
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

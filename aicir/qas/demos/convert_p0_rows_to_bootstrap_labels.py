#!/usr/bin/env python
"""Convert P0 diagnostic rows into P1 bootstrap fair-label rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aicir.qas.library.ansatz import SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import read_csv_rows, write_csv_rows
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, LabelSource, LabelStatus
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float



def _load_ansatz_gene(raw: Any) -> dict[str, Any]:
    if raw is None or str(raw).strip() == "":
        raise ValueError("P0 row is missing ansatz_gene")
    loaded = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(loaded, str):
        loaded = json.loads(loaded)
    if not isinstance(loaded, dict):
        raise ValueError("ansatz_gene must decode to an object")
    return loaded


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _static_counts(gene_payload: Mapping[str, Any]) -> tuple[int, int]:
    if str(gene_payload.get("kind", "")).lower() != "supernet_native":
        return 0, 0
    gene = SupernetAnsatzGene.from_jsonable(dict(gene_payload))
    n_params = 0
    two_q_count = 0
    for layer in gene.single_qubit_layers:
        n_params += sum(1 for gate in layer if str(gate).lower() in {"rx", "ry", "rz"})
    for layer in gene.two_qubit_layers:
        for gate in layer:
            normalized = str(gate).lower()
            if normalized not in {"", "none", "i"}:
                two_q_count += 1
            if normalized == "rzz":
                n_params += 1
    return n_params, two_q_count


def _energy_from_p0_row(row: Mapping[str, Any]) -> float | None:
    return _as_float(row.get("fair_best_energy")) if _as_float(row.get("fair_best_energy")) is not None else _as_float(row.get("fair_high"))


def convert_p0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    preset: str,
    terms: Sequence[tuple[float, str]],
    n_qubits: int,
    reference_energy: float,
    source: str = LabelSource.INITIAL_TRAIN.value,
) -> list[dict[str, Any]]:
    """Return BENCHMARK_TABLE_FIELDS rows usable as P1 bootstrap labels."""

    hamiltonian_terms_json = json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms])
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        energy = _energy_from_p0_row(row)
        if energy is None:
            continue
        gene_payload = _load_ansatz_gene(row.get("ansatz_gene"))
        n_params, two_q_count = _static_counts(gene_payload)
        architecture_id = str(row.get("architecture_id") or f"{preset}_p0_{index:04d}")
        depth = str(row.get("depth") or row.get("depth_group") or "")
        depth_group = depth if depth.startswith("L") else f"L{depth}" if depth else ""
        output = {field: "" for field in BENCHMARK_TABLE_FIELDS}
        output.update(
            {
                "architecture_id": architecture_id,
                "canonical_arch_hash": _canonical_json(gene_payload),
                "protocol_version": "fair_vqe_protocol_v2",
                "batch_id": str(row.get("batch_id") or "p0_bootstrap"),
                "source": str(source),
                "label_status": LabelStatus.COMPLETED.value,
                "retry_count": str(row.get("retry_count") or "0"),
                "failure_reason": "",
                "last_error_digest": "",
                "n_qubits": str(row.get("n_qubits") or int(n_qubits)),
                "hamiltonian_id": str(row.get("hamiltonian_id") or row.get("problem_id") or preset),
                "hamiltonian_class": str(row.get("hamiltonian_class") or "molecular_preset"),
                "family": str(row.get("family") or "supernet_native"),
                "depth_group": depth_group,
                "entangler_type": str(row.get("entangler_type") or "mixed_supernet"),
                "topology": str(row.get("topology") or "supernet_pairs"),
                "n_params": str(row.get("n_params") or n_params),
                "two_q_count": str(row.get("two_q_count") or two_q_count),
                "hamiltonian_terms": hamiltonian_terms_json,
                "ansatz_gene": json.dumps(gene_payload, ensure_ascii=False),
                "fair_best_energy": f"{float(energy):.12f}",
                "reference_energy": f"{float(row.get('reference_energy') or reference_energy):.12f}",
                "optimizer": str(row.get("optimizer") or "COBYLA"),
            }
        )
        converted.append(output)
    return converted


def load_hamiltonian_for_conversion(
    preset: str,
    *,
    hamiltonian_file: str | None = None,
    reference_energy_override: float | None = None,
) -> tuple[tuple[tuple[float, str], ...], int, float]:
    from aicir.qas.demos.run_p0_diagnostic import load_hamiltonian_terms

    return load_hamiltonian_terms(
        preset,
        hamiltonian_file=hamiltonian_file,
        reference_energy_override=reference_energy_override,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert P0 diagnostic rows.csv into P1 bootstrap_labels.csv.")
    parser.add_argument("--input", required=True, help="P0 diagnostic rows.csv")
    parser.add_argument("--output", required=True, help="Output bootstrap_labels.csv")
    parser.add_argument("--preset", required=True, help="Chemistry preset used to reconstruct Hamiltonian terms")
    parser.add_argument("--hamiltonian-file", default=None, help="Optional JSON Pauli-term list for custom/larger molecules")
    parser.add_argument("--reference-energy", type=float, default=None)
    parser.add_argument("--source", default=LabelSource.INITIAL_TRAIN.value)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    hamiltonian_loader=load_hamiltonian_for_conversion,
) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    terms, n_qubits, reference_energy = hamiltonian_loader(
        args.preset,
        hamiltonian_file=args.hamiltonian_file,
        reference_energy_override=args.reference_energy,
    )
    input_rows = read_csv_rows(args.input)
    output_rows = convert_p0_rows(
        input_rows,
        preset=args.preset,
        terms=tuple((float(coeff), str(pauli)) for coeff, pauli in terms),
        n_qubits=int(n_qubits),
        reference_energy=float(reference_energy),
        source=args.source,
    )
    write_csv_rows(args.output, output_rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    result = {
        "input": str(args.input),
        "output": str(args.output),
        "read": len(input_rows),
        "written": len(output_rows),
        "skipped_without_fair_label": len(input_rows) - len(output_rows),
        "preset": str(args.preset),
        "n_qubits": int(n_qubits),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()

"""Export an ADAPT-VQE ansatz as a QAS explicit-gate seed.

The script reruns ADAPT-VQE, converts the final Qiskit circuit into the local
QAS gate-list representation, and writes a pending fair-label queue row.  The
ADAPT energy is stored as diagnostic ``screening_energy`` only; the seed still
needs the normal fair labeler before it is used as a final QAS label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from aicir.ir import circuit_gate_dicts
from aicir.qas.library.ansatz import (
    ExplicitGateAnsatzGene,
    OperatorSequenceAnsatzGene,
    architecture_from_explicit_gate_gene,
    architecture_from_operator_sequence_gene,
)
from aicir.qas.vqe_loop.benchmark_table import parse_pauli_hamiltonian_terms
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, LabelSource, LabelStatus


PauliTerm = tuple[float, str]


def _pauli_on_support(width: int, support: Sequence[int], symbols: Sequence[str]) -> str:
    pauli = ["I"] * int(width)
    for index, symbol in zip(support, symbols):
        pauli[int(index)] = str(symbol).upper()
    return "".join(pauli)


def _qiskit_pauli_from_qas(pauli: str) -> str:
    """Convert QAS Pauli-string qubit order to Qiskit's label order."""

    return str(pauli).upper()[::-1]


def _term_support(pauli: str) -> tuple[int, ...]:
    return tuple(index for index, symbol in enumerate(str(pauli).upper()) if symbol != "I")


def build_operator_pool_paulis(
    terms: Sequence[PauliTerm],
    *,
    pool_kind: str,
) -> list[str]:
    """Build unique Pauli-string operators for ADAPT-style growth."""

    if not terms:
        return []
    width = max(len(str(pauli)) for _coeff, pauli in terms)
    normalized = str(pool_kind).strip().lower()
    pool: set[str] = set()

    if normalized == "edge_two_local":
        edges: set[tuple[int, int]] = set()
        for _coeff, pauli in terms:
            support = _term_support(pauli)
            if len(support) < 2:
                continue
            for left, right in itertools.combinations(support, 2):
                edges.add((min(left, right), max(left, right)))
        for left, right in sorted(edges):
            for axis in "XYZ":
                pool.add(_pauli_on_support(width, (left,), (axis,)))
                pool.add(_pauli_on_support(width, (right,), (axis,)))
            for left_axis in "XYZ":
                for right_axis in "XYZ":
                    pool.add(_pauli_on_support(width, (left, right), (left_axis, right_axis)))
        return sorted(pool)

    if normalized == "support_generic":
        for qubit in range(width):
            for axis in "XYZ":
                pool.add(_pauli_on_support(width, (qubit,), (axis,)))
        supports = {_term_support(pauli) for _coeff, pauli in terms if _term_support(pauli)}
        pairs = sorted({tuple(sorted(pair)) for support in supports for pair in itertools.combinations(support, 2)})
        for left, right in pairs:
            for left_axis in "XYZ":
                for right_axis in "XYZ":
                    pool.add(_pauli_on_support(width, (left, right), (left_axis, right_axis)))
        for _coeff, pauli in terms:
            if len(_term_support(pauli)) >= 3:
                pool.add(str(pauli).upper())
        return sorted(pool)

    if normalized == "hamiltonian_terms":
        return sorted({str(pauli).upper() for _coeff, pauli in terms if _term_support(pauli)})

    raise ValueError(f"unsupported pool_kind: {pool_kind}")


def _load_terms(path: str | Path) -> tuple[PauliTerm, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple((float(coeff), str(pauli).upper()) for coeff, pauli in parse_pauli_hamiltonian_terms(raw))


def _initial_state_circuit(n_qubits: int, kind: str):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(int(n_qubits))
    normalized = str(kind).strip().lower()
    if normalized in {"zero", "zeros", "computational_zero", ""}:
        return circuit
    if normalized == "plus":
        for qubit in range(int(n_qubits)):
            circuit.h(qubit)
        return circuit
    if normalized == "neel":
        for qubit in range(int(n_qubits)):
            if qubit % 2 == 1:
                circuit.x(qubit)
        return circuit
    raise ValueError(f"unsupported initial_state: {kind}")


def _run_adapt_vqe(
    *,
    terms: Sequence[PauliTerm],
    pool_paulis: Sequence[str],
    initial_state: str,
    max_iterations: int,
    optimizer_maxiter: int,
    gradient_threshold: float,
    eigenvalue_threshold: float,
) -> tuple[Any, int, float]:
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import AdaptVQE, VQE
    from qiskit_algorithms.optimizers import COBYLA

    operator = SparsePauliOp.from_list([(_qiskit_pauli_from_qas(pauli), float(coeff)) for coeff, pauli in terms])
    pool = [SparsePauliOp.from_list([(_qiskit_pauli_from_qas(pauli), 1.0)]) for pauli in pool_paulis]
    n_qubits = len(str(terms[0][1]))
    ansatz = _initial_state_circuit(n_qubits, initial_state)
    callback_calls = 0

    def callback(_eval_count, _params, _mean, _metadata):
        nonlocal callback_calls
        callback_calls += 1

    vqe = VQE(
        StatevectorEstimator(),
        ansatz,
        COBYLA(maxiter=int(optimizer_maxiter)),
        callback=callback,
    )
    adapt = AdaptVQE(
        vqe,
        operators=pool,
        max_iterations=int(max_iterations),
        gradient_threshold=float(gradient_threshold),
        eigenvalue_threshold=float(eigenvalue_threshold),
    )
    start = perf_counter()
    result = adapt.compute_minimum_eigenvalue(operator)
    return result, callback_calls, perf_counter() - start


def _explicit_gene_from_qiskit_circuit(qiskit_circuit: Any, *, name: str, decompose_reps: int) -> ExplicitGateAnsatzGene:
    from aicir.core.io.qiskit_io import circuit_from_qiskit

    decomposed = qiskit_circuit.decompose(reps=int(decompose_reps))
    circuit = circuit_from_qiskit(decomposed)
    return ExplicitGateAnsatzGene(
        n_qubits=int(circuit.n_qubits),
        gates=tuple(circuit_gate_dicts(circuit)),
        name=name,
    )


def _queue_row(
    *,
    gene: ExplicitGateAnsatzGene,
    architecture_id: str,
    batch_id: str,
    protocol_version: str,
    hamiltonian_id: str,
    hamiltonian_class: str,
    terms: Sequence[PauliTerm],
    reference_energy: float | None,
    screening_energy: float,
) -> dict[str, Any]:
    architecture = architecture_from_explicit_gate_gene(gene)
    encoded_gene = json.dumps(gene.to_jsonable(), ensure_ascii=False, sort_keys=True)
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": architecture_id,
            "canonical_arch_hash": hashlib.sha256(encoded_gene.encode("utf-8")).hexdigest(),
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": LabelSource.INITIAL_TRAIN.value,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "n_qubits": gene.n_qubits,
            "hamiltonian_id": hamiltonian_id,
            "hamiltonian_class": hamiltonian_class,
            "family": "explicit_gate_sequence",
            "depth_group": f"G{len(gene.gates)}",
            "entangler_type": "explicit",
            "topology": "explicit",
            "n_params": architecture.parameter_count,
            "two_q_count": architecture.two_qubit_gate_count,
            "hamiltonian_terms": json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms]),
            "ansatz_gene": encoded_gene,
            "screening_energy": f"{float(screening_energy):.12f}",
            "screening_energy_is_final_label": "false",
        }
    )
    if reference_energy is not None:
        row["reference_energy"] = f"{float(reference_energy):.12f}"
    return row


def build_operator_sequence_seed_row(
    *,
    selected_paulis: Sequence[str],
    n_qubits: int,
    architecture_id: str,
    batch_id: str,
    protocol_version: str,
    hamiltonian_id: str,
    hamiltonian_class: str,
    terms: Sequence[PauliTerm],
    reference_energy: float | None,
    screening_energy: float,
) -> dict[str, Any]:
    """Build a fair-label queue row for an ADAPT selected-operator sequence."""

    gene = OperatorSequenceAnsatzGene(
        n_qubits=int(n_qubits),
        operators=tuple(str(pauli).upper() for pauli in selected_paulis),
        name=f"adapt_operator_sequence_{hamiltonian_id}",
    )
    architecture = architecture_from_operator_sequence_gene(gene)
    encoded_gene = json.dumps(gene.to_jsonable(), ensure_ascii=False, sort_keys=True)
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": architecture_id,
            "canonical_arch_hash": hashlib.sha256(encoded_gene.encode("utf-8")).hexdigest(),
            "protocol_version": protocol_version,
            "batch_id": batch_id,
            "source": LabelSource.INITIAL_TRAIN.value,
            "label_status": LabelStatus.PENDING.value,
            "retry_count": 0,
            "n_qubits": gene.n_qubits,
            "hamiltonian_id": hamiltonian_id,
            "hamiltonian_class": hamiltonian_class,
            "family": "operator_sequence",
            "depth_group": f"L{gene.layers}",
            "entangler_type": "pauli_operator_sequence",
            "topology": "pauli_operator_sequence",
            "n_params": architecture.parameter_count,
            "two_q_count": architecture.two_qubit_gate_count,
            "hamiltonian_terms": json.dumps([[float(coeff), str(pauli)] for coeff, pauli in terms]),
            "ansatz_gene": encoded_gene,
            "screening_energy": f"{float(screening_energy):.12f}",
            "screening_energy_is_final_label": "false",
        }
    )
    if reference_energy is not None:
        row["reference_energy"] = f"{float(reference_energy):.12f}"
    return row


def _write_queue(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})


def _selected_paulis_from_adapt_result(result: Any, pool_paulis: Sequence[str]) -> tuple[str, ...]:
    for attr in ("operator_indices", "selected_operator_indices"):
        indices = getattr(result, attr, None)
        if indices is None:
            continue
        selected: list[str] = []
        for index in indices:
            try:
                selected.append(str(pool_paulis[int(index)]).upper())
            except (TypeError, ValueError, IndexError):
                continue
        if selected:
            return tuple(selected)
    selected = getattr(result, "selected_paulis", None)
    if selected:
        return tuple(str(pauli).upper() for pauli in selected)
    return ()


def _load_selected_paulis(path: str | Path | None) -> tuple[str, ...]:
    if path is None or str(path).strip() == "":
        return ()
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("selected paulis JSON must be a list of Pauli strings")
    return tuple(str(pauli).upper() for pauli in raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian-terms-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-queue-csv", required=True)
    parser.add_argument("--output-operator-sequence-json", default=None)
    parser.add_argument("--output-operator-sequence-queue-csv", default=None)
    parser.add_argument("--selected-paulis-json", default=None)
    parser.add_argument("--pool-kind", choices=("edge_two_local", "support_generic", "hamiltonian_terms"), required=True)
    parser.add_argument("--initial-state", default="plus", choices=("zero", "plus", "neel"))
    parser.add_argument("--problem-id", default="adapt_seed_problem")
    parser.add_argument("--hamiltonian-class", default="custom_pauli")
    parser.add_argument("--reference-energy", type=float, default=None)
    parser.add_argument("--batch-id", default="adapt_explicit_seed")
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--optimizer-maxiter", type=int, default=200)
    parser.add_argument("--gradient-threshold", type=float, default=1e-5)
    parser.add_argument("--eigenvalue-threshold", type=float, default=1e-5)
    parser.add_argument("--decompose-reps", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    terms = _load_terms(args.hamiltonian_terms_json)
    pool_paulis = build_operator_pool_paulis(terms, pool_kind=args.pool_kind)
    result, callback_calls, elapsed_seconds = _run_adapt_vqe(
        terms=terms,
        pool_paulis=pool_paulis,
        initial_state=args.initial_state,
        max_iterations=args.max_iterations,
        optimizer_maxiter=args.optimizer_maxiter,
        gradient_threshold=args.gradient_threshold,
        eigenvalue_threshold=args.eigenvalue_threshold,
    )
    optimal_circuit = result.optimal_circuit
    if getattr(result, "optimal_parameters", None):
        optimal_circuit = optimal_circuit.assign_parameters(result.optimal_parameters)
    gene = _explicit_gene_from_qiskit_circuit(
        optimal_circuit,
        name=f"adapt_{args.problem_id}_{args.pool_kind}",
        decompose_reps=args.decompose_reps,
    )
    energy = float(result.eigenvalue.real if hasattr(result.eigenvalue, "real") else result.eigenvalue)
    digest = hashlib.sha256(json.dumps(gene.to_jsonable(), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    architecture_id = f"adapt_seed_{args.problem_id}_{args.pool_kind}_{digest}"
    row = _queue_row(
        gene=gene,
        architecture_id=architecture_id,
        batch_id=args.batch_id,
        protocol_version=args.protocol_version,
        hamiltonian_id=args.problem_id,
        hamiltonian_class=args.hamiltonian_class,
        terms=terms,
        reference_energy=args.reference_energy,
        screening_energy=energy,
    )
    _write_queue(args.output_queue_csv, row)
    selected_paulis = _load_selected_paulis(args.selected_paulis_json) or _selected_paulis_from_adapt_result(result, pool_paulis)
    operator_sequence_payload: dict[str, Any] | None = None
    if selected_paulis:
        operator_architecture_id = f"adapt_operator_seed_{args.problem_id}_{args.pool_kind}_{hashlib.sha256(json.dumps(list(selected_paulis)).encode('utf-8')).hexdigest()[:12]}"
        operator_row = build_operator_sequence_seed_row(
            selected_paulis=selected_paulis,
            n_qubits=len(str(terms[0][1])),
            architecture_id=operator_architecture_id,
            batch_id=args.batch_id,
            protocol_version=args.protocol_version,
            hamiltonian_id=args.problem_id,
            hamiltonian_class=args.hamiltonian_class,
            terms=terms,
            reference_energy=args.reference_energy,
            screening_energy=energy,
        )
        if args.output_operator_sequence_queue_csv:
            _write_queue(args.output_operator_sequence_queue_csv, operator_row)
        operator_sequence_payload = {
            "architecture_id": operator_architecture_id,
            "selected_paulis": list(selected_paulis),
            "queue_csv": str(args.output_operator_sequence_queue_csv or ""),
            "ansatz_gene": json.loads(str(operator_row["ansatz_gene"])),
        }
        if args.output_operator_sequence_json:
            Path(args.output_operator_sequence_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_operator_sequence_json).write_text(
                json.dumps(operator_sequence_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    payload = {
        "architecture_id": architecture_id,
        "energy": energy,
        "pool_kind": args.pool_kind,
        "pool_size": len(pool_paulis),
        "initial_state": args.initial_state,
        "callback_calls": callback_calls,
        "elapsed_seconds": elapsed_seconds,
        "num_iterations": getattr(result, "num_iterations", None),
        "termination_criterion": str(getattr(result, "termination_criterion", "")),
        "ansatz_gene": gene.to_jsonable(),
        "queue_csv": str(args.output_queue_csv),
        "operator_sequence_seed": operator_sequence_payload,
        "selected_paulis_inferred": bool(selected_paulis),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()

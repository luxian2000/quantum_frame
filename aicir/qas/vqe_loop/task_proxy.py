"""VQE-specific task-aware cheap proxy evaluators.

These scores are selectors only.  They never replace fair COBYLA labels.
Lower proxy scores are better so they can share the existing P1 ranking path.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene, SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import decoded_ansatz_gene_payload
from aicir.qas.vqe_loop.benchmark_table import row_hamiltonian_terms


def _operator_strings_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        payload = decoded_ansatz_gene_payload(row)
    except (TypeError, ValueError):
        return ()
    if not payload:
        return ()
    kind = str(payload.get("kind", "")).lower()
    if kind == "operator_sequence":
        try:
            return tuple(str(op).upper() for op in OperatorSequenceAnsatzGene.from_jsonable(payload).operators)
        except (TypeError, ValueError, KeyError):
            return ()
    if kind == "supernet_native":
        try:
            gene = SupernetAnsatzGene.from_jsonable(payload)
        except (TypeError, ValueError, KeyError):
            return ()
        operators: list[str] = []
        for layer in gene.single_qubit_layers:
            for qubit, gate in enumerate(layer):
                letter = {"rx": "X", "ry": "Y", "rz": "Z"}.get(str(gate).lower())
                if letter:
                    pauli = ["I"] * int(gene.n_qubits)
                    pauli[qubit] = letter
                    operators.append("".join(pauli))
        return tuple(operators)
    return ()


def _support(pauli: str) -> frozenset[int]:
    return frozenset(index for index, letter in enumerate(str(pauli).upper()) if letter != "I")


def hamiltonian_ansatz_overlap(row: Mapping[str, Any]) -> float:
    """Return fraction of Hamiltonian Pauli terms represented by ansatz operators."""

    try:
        terms = row_hamiltonian_terms(row)
    except (TypeError, ValueError):
        return 0.0
    operators = set(_operator_strings_from_row(row))
    if not terms or not operators:
        return 0.0
    h_terms = {pauli for _coeff, pauli in terms}
    return len(h_terms & operators) / float(len(h_terms))


def ansatz_pauli_support_coverage(row: Mapping[str, Any]) -> float:
    try:
        terms = row_hamiltonian_terms(row)
    except (TypeError, ValueError):
        return 0.0
    operators = _operator_strings_from_row(row)
    if not terms or not operators:
        return 0.0
    op_supports = {_support(operator) for operator in operators}
    covered = 0
    for _coeff, pauli in terms:
        term_support = _support(pauli)
        if term_support in op_supports or any(term_support <= op_support for op_support in op_supports):
            covered += 1
    return covered / float(len(terms))


def gradient_sensitivity_proxy(row: Mapping[str, Any]) -> float:
    """Cheap structural stand-in for initial-gradient sensitivity.

    It rewards task overlap and nonzero parameter count while damping very large
    ansatzes.  Real finite-difference gradients can be plugged into this field
    later without changing P1 selection.
    """

    try:
        payload = dict(decoded_ansatz_gene_payload(row) or {})
    except (TypeError, ValueError):
        payload = {}
    kind = str(payload.get("kind", "")).lower()
    if kind == "operator_sequence":
        n_params = len(payload.get("operators", ()) or ())
    elif kind == "supernet_native":
        layers = payload.get("single_qubit_layers", ()) or ()
        n_params = sum(1 for layer in layers for gate in layer if str(gate).lower() in {"rx", "ry", "rz"})
    else:
        n_params = 0
    overlap = hamiltonian_ansatz_overlap(row)
    support = ansatz_pauli_support_coverage(row)
    return math.sqrt(max(0, n_params)) * (0.5 + overlap + 0.5 * support) / max(1.0, 0.25 * max(0, n_params))


def adapt_growth_potential(row: Mapping[str, Any], operator_pool: Sequence[str] = ()) -> float:
    try:
        terms = row_hamiltonian_terms(row)
    except (TypeError, ValueError):
        return 0.0
    if not terms or not operator_pool:
        return 0.0
    existing = set(_operator_strings_from_row(row))
    weighted_terms = {pauli: abs(float(coeff)) for coeff, pauli in terms}
    current_overlap = hamiltonian_ansatz_overlap(row)
    best = 0.0
    for operator in operator_pool:
        op = str(operator).upper()
        if op in existing:
            continue
        direct = weighted_terms.get(op, 0.0)
        support_bonus = 0.0
        op_support = _support(op)
        for coeff, pauli in terms:
            if _support(pauli) <= op_support:
                support_bonus += abs(float(coeff)) * 0.1
        best = max(best, direct + support_bonus)
    return float(best * (1.0 - current_overlap))


def vqe_task_proxy_score(row: Mapping[str, Any], *, operator_pool: Sequence[str] = ()) -> dict[str, float]:
    overlap = hamiltonian_ansatz_overlap(row)
    support = ansatz_pauli_support_coverage(row)
    gradient = gradient_sensitivity_proxy(row)
    growth = adapt_growth_potential(row, operator_pool=operator_pool)
    quality = 1.5 * overlap + 0.75 * support + 0.25 * gradient + 0.50 * growth
    return {
        "VQE_TASK_PROXY": -float(quality),
        "task_proxy_hamiltonian_overlap": float(overlap),
        "task_proxy_gradient_sensitivity": float(gradient),
        "task_proxy_adapt_growth_potential": float(growth),
    }


def build_vqe_task_proxy_evaluator(*, operator_pool: Sequence[str] = ()):
    def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        scores = vqe_task_proxy_score(row, operator_pool=operator_pool)
        return {
            "VQE_TASK_PROXY": f"{scores['VQE_TASK_PROXY']:.12f}",
            "task_proxy_hamiltonian_overlap": f"{scores['task_proxy_hamiltonian_overlap']:.12f}",
            "task_proxy_gradient_sensitivity": f"{scores['task_proxy_gradient_sensitivity']:.12f}",
            "task_proxy_adapt_growth_potential": f"{scores['task_proxy_adapt_growth_potential']:.12f}",
        }

    return evaluate


__all__ = [
    "adapt_growth_potential",
    "ansatz_pauli_support_coverage",
    "build_vqe_task_proxy_evaluator",
    "gradient_sensitivity_proxy",
    "hamiltonian_ansatz_overlap",
    "vqe_task_proxy_score",
]

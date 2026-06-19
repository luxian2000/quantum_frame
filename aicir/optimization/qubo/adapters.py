"""Adapters from QUBO modeling outputs to aicir Hamiltonians."""

from __future__ import annotations

from collections.abc import Iterable

from ...core.operators import Hamiltonian
from .modeling.backends import IsingModel, QAOATerm


def _infer_n_qubits_from_terms(terms: Iterable[QAOATerm]) -> int:
    max_qubit = -1
    for term in terms:
        if term.qubits:
            max_qubit = max(max_qubit, max(term.qubits))
    if max_qubit < 0:
        raise ValueError("Cannot infer n_qubits from empty QAOA terms.")
    return max_qubit + 1


def qaoa_terms_to_hamiltonian(
    terms: Iterable[QAOATerm],
    *,
    n_qubits: int | None = None,
    offset: float = 0.0,
    include_offset: bool = True,
) -> Hamiltonian:
    """Convert QAOA Z terms into an ``aicir.operators.Hamiltonian``.

    The QUBO modeling layer emits terms such as ``Z_i`` and ``Z_i Z_j``. The
    aicir Hamiltonian accepts local Pauli strings with explicit qubit indices,
    so this adapter preserves sparsity at construction time.
    """

    qaoa_terms = list(terms)
    width = int(n_qubits) if n_qubits is not None else _infer_n_qubits_from_terms(qaoa_terms)
    if width <= 0:
        raise ValueError("n_qubits must be positive.")

    hamiltonian_terms: list[tuple[str, list[int], float] | tuple[str, float]] = []
    for term in qaoa_terms:
        if not term.qubits:
            raise ValueError("QAOA terms must act on at least one qubit.")
        if len(term.qubits) not in {1, 2}:
            raise ValueError("Only one- and two-qubit Z terms are supported.")
        if max(term.qubits) >= width or min(term.qubits) < 0:
            raise IndexError("QAOA term qubit index is outside n_qubits.")
        hamiltonian_terms.append((term.pauli, list(term.qubits), float(term.coefficient)))

    if include_offset and abs(offset) > 1e-12:
        hamiltonian_terms.append(("I" * width, float(offset)))

    if not hamiltonian_terms:
        hamiltonian_terms.append(("I" * width, 0.0))
    return Hamiltonian(n_qubits=width, terms=hamiltonian_terms)


def ising_to_hamiltonian(
    ising: IsingModel,
    *,
    include_offset: bool = True,
) -> Hamiltonian:
    """Convert an indexed Ising model into an aicir Hamiltonian."""

    terms = ising.to_qaoa_terms()
    if ising.variable_names is not None:
        n_qubits = len(ising.variable_names)
    else:
        n_qubits = _infer_n_qubits_from_terms(terms)
    return qaoa_terms_to_hamiltonian(
        terms,
        n_qubits=n_qubits,
        offset=ising.offset,
        include_offset=include_offset,
    )


def model_to_hamiltonian(
    model,
    *,
    compact: bool = True,
    include_offset: bool = True,
) -> Hamiltonian:
    """Export a QUBO ``Model`` as an aicir Hamiltonian."""

    terms, offset, variable_names = model.to_qaoa_terms(compact=compact)
    return qaoa_terms_to_hamiltonian(
        terms,
        n_qubits=len(variable_names),
        offset=offset,
        include_offset=include_offset,
    )


def builder_to_hamiltonian(
    builder,
    *,
    compact: bool = True,
    include_offset: bool = True,
) -> Hamiltonian:
    """Export a ``QuboBuilder`` as an aicir Hamiltonian."""

    terms, offset, variable_names = builder.to_qaoa_terms(compact=compact)
    return qaoa_terms_to_hamiltonian(
        terms,
        n_qubits=len(variable_names),
        offset=offset,
        include_offset=include_offset,
    )


__all__ = [
    "builder_to_hamiltonian",
    "ising_to_hamiltonian",
    "model_to_hamiltonian",
    "qaoa_terms_to_hamiltonian",
]

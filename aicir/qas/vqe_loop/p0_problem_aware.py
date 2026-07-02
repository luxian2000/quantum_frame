"""Problem-aware samplers for diagnostic P0 candidate generation."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Mapping, Sequence

from aicir.qas.primitives.ansatz import SupernetAnsatzGene

PauliTerm = tuple[float, str]

DEFAULT_SINGLE_QUBIT_GATES: tuple[str, ...] = ("i", "h", "rx", "ry", "rz")
DEFAULT_TWO_QUBIT_GATES: tuple[str, ...] = ("cx", "rzz")


def _pair_key(pair: Sequence[int]) -> tuple[int, int]:
    left, right = int(pair[0]), int(pair[1])
    return (left, right) if left <= right else (right, left)


def _as_terms(terms: Sequence[Sequence[object] | PauliTerm]) -> tuple[PauliTerm, ...]:
    parsed: list[PauliTerm] = []
    for term in terms:
        if len(term) != 2:
            raise ValueError("Pauli terms must be (coefficient, pauli_label) pairs")
        coeff, pauli = term
        parsed.append((float(coeff), str(pauli).upper()))
    return tuple(parsed)


def pauli_coupling_weights(terms: Sequence[Sequence[object] | PauliTerm]) -> dict[tuple[int, int], float]:
    """Return absolute coefficient mass for qubit pairs co-active in Pauli terms."""

    weights: dict[tuple[int, int], float] = {}
    for coeff, pauli in _as_terms(terms):
        active = [index for index, char in enumerate(pauli) if char != "I"]
        for left, right in combinations(active, 2):
            key = _pair_key((left, right))
            weights[key] = weights.get(key, 0.0) + abs(float(coeff))
    return weights


def prioritized_two_qubit_pairs(
    terms: Sequence[Sequence[object] | PauliTerm],
    *,
    candidate_pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Sort candidate entangler pairs by Pauli coupling mass."""

    weights = pauli_coupling_weights(terms)
    pairs = tuple((int(left), int(right)) for left, right in candidate_pairs)
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (-weights.get(_pair_key(pair), 0.0), abs(pair[0] - pair[1]), pair[0], pair[1]),
        )
    )


def _normalized_pair_weights(
    terms: Sequence[Sequence[object] | PauliTerm],
    pairs: Sequence[Sequence[int]],
) -> dict[tuple[int, int], float]:
    raw = pauli_coupling_weights(terms)
    pair_weights = {tuple((int(left), int(right))): raw.get(_pair_key((left, right)), 0.0) for left, right in pairs}
    max_weight = max(pair_weights.values(), default=0.0)
    if max_weight <= 0.0:
        return {pair: 0.0 for pair in pair_weights}
    return {pair: float(weight) / float(max_weight) for pair, weight in pair_weights.items()}


def _rng(seed: int | None = None, rng: random.Random | None = None) -> random.Random:
    return rng if rng is not None else random.Random(seed)


def _choose_two_qubit_gate(
    pair: tuple[int, int],
    *,
    pair_weights: Mapping[tuple[int, int], float],
    entangler_probability_floor: float,
    two_qubit_gates: Sequence[str],
    rng: random.Random,
) -> str:
    normalized = float(pair_weights.get(pair, 0.0))
    floor = max(0.0, min(1.0, float(entangler_probability_floor)))
    probability = floor + (1.0 - floor) * normalized
    if rng.random() >= probability:
        return "none"
    return str(rng.choice(tuple(two_qubit_gates) or DEFAULT_TWO_QUBIT_GATES)).lower()


def sample_problem_aware_supernet_gene(
    *,
    n_qubits: int,
    depth: int,
    pairs: Sequence[Sequence[int]],
    hamiltonian_terms: Sequence[Sequence[object] | PauliTerm],
    seed: int | None = None,
    rng: random.Random | None = None,
    single_qubit_gates: Sequence[str] = DEFAULT_SINGLE_QUBIT_GATES,
    two_qubit_gates: Sequence[str] = DEFAULT_TWO_QUBIT_GATES,
    entangler_probability_floor: float = 0.10,
) -> SupernetAnsatzGene:
    """Sample a native supernet gene biased toward Pauli-coupled entanglers."""

    local_rng = _rng(seed, rng)
    normalized = _normalized_pair_weights(hamiltonian_terms, pairs)
    normalized_pairs = tuple((int(left), int(right)) for left, right in pairs)
    single_choices = tuple(str(gate).lower() for gate in single_qubit_gates) or DEFAULT_SINGLE_QUBIT_GATES
    single_layers: list[tuple[str, ...]] = []
    two_layers: list[tuple[str, ...]] = []

    for _ in range(max(0, int(depth))):
        single_layers.append(tuple(local_rng.choice(single_choices) for _ in range(int(n_qubits))))
        two_layers.append(
            tuple(
                _choose_two_qubit_gate(
                    pair,
                    pair_weights=normalized,
                    entangler_probability_floor=entangler_probability_floor,
                    two_qubit_gates=two_qubit_gates,
                    rng=local_rng,
                )
                for pair in normalized_pairs
            )
        )

    return SupernetAnsatzGene(
        n_qubits=int(n_qubits),
        single_qubit_layers=tuple(single_layers),
        two_qubit_layers=tuple(two_layers),
        two_qubit_pairs=normalized_pairs,
    )


__all__ = [
    "pauli_coupling_weights",
    "prioritized_two_qubit_pairs",
    "sample_problem_aware_supernet_gene",
]

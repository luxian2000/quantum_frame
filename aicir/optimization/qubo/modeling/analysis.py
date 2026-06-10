from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .integer import EncodedInteger
from .registry import VariableRegistry
from .solution import DecodedSolution, decode_solution

Assignment = Mapping[str | int, int | bool | float] | Sequence[int | bool | float]


@dataclass(frozen=True)
class BruteForceResult:
    best_energy: float
    best_assignments: list[tuple[int, ...]]
    energies: dict[tuple[int, ...], float]


def qubo_energy(
    qubo: Mapping[tuple[int, int], float],
    assignment: Assignment,
    offset: float = 0.0,
) -> float:
    bits = _assignment_to_indices(assignment)
    energy = float(offset)
    for (left, right), coeff in qubo.items():
        energy += float(coeff) * bits.get(left, 0) * bits.get(right, 0)
    return energy


def brute_force_qubo(
    qubo: Mapping[tuple[int, int], float],
    offset: float = 0.0,
    variable_count: int | None = None,
    max_variables: int = 20,
    atol: float = 1e-9,
) -> BruteForceResult:
    if variable_count is None:
        variable_count = _infer_variable_count(qubo)
    if variable_count < 0:
        raise ValueError("variable_count must be non-negative.")
    if variable_count > max_variables:
        raise ValueError("brute_force_qubo refuses to enumerate more than max_variables variables.")

    best_energy: float | None = None
    best_assignments: list[tuple[int, ...]] = []
    energies: dict[tuple[int, ...], float] = {}
    for assignment in product((0, 1), repeat=variable_count):
        energy = qubo_energy(qubo, assignment, offset=offset)
        energies[assignment] = energy
        if best_energy is None or energy < best_energy - atol:
            best_energy = energy
            best_assignments = [assignment]
        elif abs(energy - best_energy) <= atol:
            best_assignments.append(assignment)

    return BruteForceResult(
        best_energy=0.0 if best_energy is None else best_energy,
        best_assignments=best_assignments,
        energies=energies,
    )


def brute_force_builder(
    builder,
    max_variables: int = 20,
    clean: bool = True,
    atol: float = 1e-9,
) -> BruteForceResult:
    qubo, offset = builder.to_qubo_indices(clean=clean)
    return brute_force_qubo(
        qubo,
        offset=offset,
        variable_count=len(builder.registry.names()),
        max_variables=max_variables,
        atol=atol,
    )


def brute_force_model(
    model,
    max_variables: int = 20,
    clean: bool = True,
    atol: float = 1e-9,
) -> BruteForceResult:
    builder = model.to_qubo_builder()
    return brute_force_builder(builder, max_variables=max_variables, clean=clean, atol=atol)


def decode_best_solutions(
    result: BruteForceResult,
    registry: VariableRegistry,
    integers: Sequence[EncodedInteger] | None = None,
    include_auxiliary: bool = False,
) -> list[DecodedSolution]:
    return [
        decode_solution(
            assignment,
            registry=registry,
            integers=integers,
            include_auxiliary=include_auxiliary,
        )
        for assignment in result.best_assignments
    ]


def _assignment_to_indices(assignment: Assignment) -> dict[int, int]:
    if isinstance(assignment, Mapping):
        return {int(key): _normalize_bit(value) for key, value in assignment.items()}
    return {index: _normalize_bit(value) for index, value in enumerate(assignment)}


def _normalize_bit(value: int | bool | float) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError("QUBO assignments must contain binary 0/1 values.")


def _infer_variable_count(qubo: Mapping[tuple[int, int], float]) -> int:
    if not qubo:
        return 0
    return max(max(left, right) for left, right in qubo) + 1


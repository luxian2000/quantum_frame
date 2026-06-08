from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .integer import EncodedInteger
from .registry import VariableRegistry

Assignment = Mapping[str | int, int | bool | float] | Sequence[int | bool | float]


@dataclass(frozen=True)
class DecodedSolution:
    binary: dict[str, int]
    integers: dict[str, int]
    auxiliary: dict[str, int]

    def decisions(self) -> dict[str, int]:
        result = dict(self.binary)
        result.update(self.integers)
        return result


def decode_solution(
    assignment: Assignment,
    registry: VariableRegistry,
    integers: Sequence[EncodedInteger] | None = None,
    include_auxiliary: bool = False,
) -> DecodedSolution:
    bits_by_name = _assignment_to_names(assignment, registry)
    integer_bit_names = {name for integer in integers or [] for name in integer.bit_names()}

    binary: dict[str, int] = {}
    auxiliary: dict[str, int] = {}
    for var_id, name in enumerate(registry.names()):
        value = bits_by_name.get(name, 0)
        metadata = registry.metadata(var_id)
        if metadata.role == "auxiliary":
            auxiliary[name] = value
        elif name not in integer_bit_names:
            binary[name] = value

    decoded_integers = {
        integer.name: decode_integer(integer, bits_by_name)
        for integer in integers or []
    }

    if include_auxiliary:
        return DecodedSolution(binary=binary, integers=decoded_integers, auxiliary=auxiliary)
    return DecodedSolution(binary=binary, integers=decoded_integers, auxiliary={})


def decode_integer(integer: EncodedInteger, assignment: Assignment) -> int:
    bits_by_name = _assignment_to_names(assignment, integer.registry)
    value = integer.lower_bound
    for weight, bit_name in zip(integer.weights, integer.bit_names()):
        value += weight * bits_by_name.get(bit_name, 0)
    return int(value)


def _assignment_to_names(assignment: Assignment, registry: VariableRegistry) -> dict[str, int]:
    if isinstance(assignment, Mapping):
        result: dict[str, int] = {}
        for key, value in assignment.items():
            bit = _normalize_bit(value)
            if isinstance(key, int):
                result[registry.name(key)] = bit
            else:
                result[str(key)] = bit
        return result

    return {registry.name(index): _normalize_bit(value) for index, value in enumerate(assignment)}


def _normalize_bit(value: int | bool | float) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError("Solution assignments must contain binary 0/1 values.")


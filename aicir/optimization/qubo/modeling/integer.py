from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Literal

from .polynomial import Polynomial, Sum
from .registry import GLOBAL_REGISTRY, VariableRegistry

Encoding = Literal["log", "unary"]


@dataclass(frozen=True)
class EncodedInteger:
    name: str
    lower_bound: int
    upper_bound: int
    bits: list[Polynomial]
    weights: list[int]
    registry: VariableRegistry

    def expression(self) -> Polynomial:
        if not self.bits:
            return Polynomial.constant(self.lower_bound, registry=self.registry)
        return self.lower_bound + Sum(weight * bit for weight, bit in zip(self.weights, self.bits))

    def weighted_terms(self, scale: float = 1.0) -> list[tuple[float, Polynomial]]:
        return [(scale * weight, bit) for weight, bit in zip(self.weights, self.bits)]

    def linear_expression(self, scale: float = 1.0):
        from .linear import LinearExpression

        return LinearExpression.from_integer(self, scale=scale)

    def bit_names(self) -> list[str]:
        return [bit.variables()[0] for bit in self.bits]


def Integer(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    encoding: Encoding = "log",
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger:
    if upper_bound is None:
        raise ValueError("Integer upper_bound must be provided.")
    if lower_bound < 0:
        raise ValueError("Integer lower_bound must be non-negative in this version.")
    if upper_bound < lower_bound:
        raise ValueError("Integer upper_bound must be >= lower_bound.")
    if encoding == "log":
        return LogEncodedInteger(
            name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            registry=registry,
            role=role,
            source=source,
        )
    if encoding == "unary":
        return UnaryEncodedInteger(
            name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            registry=registry,
            role=role,
            source=source,
        )
    raise ValueError(f"Unknown integer encoding: {encoding}")


def LogEncodedInteger(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger:
    if upper_bound is None:
        raise ValueError("LogEncodedInteger upper_bound must be provided.")
    _validate_bounds(lower_bound, upper_bound)
    span = upper_bound - lower_bound
    if span == 0:
        return EncodedInteger(name, lower_bound, upper_bound, [], [], registry)

    weights = bounded_log_weights(span)
    bits = [
        Polynomial.variable(f"{name}[{bit}]", registry=registry, role=role, source=source)
        for bit in range(len(weights))
    ]
    return EncodedInteger(name, lower_bound, upper_bound, bits, weights, registry)


def UnaryEncodedInteger(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger:
    if upper_bound is None:
        raise ValueError("UnaryEncodedInteger upper_bound must be provided.")
    _validate_bounds(lower_bound, upper_bound)
    span = upper_bound - lower_bound
    bits = [
        Polynomial.variable(f"{name}[{index}]", registry=registry, role=role, source=source)
        for index in range(span)
    ]
    return EncodedInteger(name, lower_bound, upper_bound, bits, [1] * span, registry)


def _validate_bounds(lower_bound: int, upper_bound: int) -> None:
    if lower_bound < 0:
        raise ValueError("Integer lower_bound must be non-negative in this version.")
    if upper_bound < lower_bound:
        raise ValueError("Integer upper_bound must be >= lower_bound.")


def bounded_log_weights(span: int) -> list[int]:
    if span < 0:
        raise ValueError("Integer span must be non-negative.")
    if span == 0:
        return []
    bit_count = max(1, ceil(log2(span + 1)))
    weights: list[int] = []
    remaining = span
    for bit in range(bit_count):
        weight = min(2**bit, remaining)
        weights.append(weight)
        remaining -= weight
    return weights


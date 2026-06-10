from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable

from .integer import EncodedInteger
from .polynomial import Polynomial, Sum
from .registry import GLOBAL_REGISTRY, VariableRegistry


@dataclass(frozen=True)
class LinearExpression:
    terms: dict[int, float]
    offset: float = 0.0
    registry: VariableRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)

    @staticmethod
    def from_terms(
        weighted_terms: Iterable[tuple[float, Polynomial]],
        offset: float = 0.0,
    ) -> LinearExpression:
        weighted_terms = list(weighted_terms)
        if not weighted_terms:
            return LinearExpression({}, float(offset))
        registry = weighted_terms[0][1].registry
        terms: dict[int, float] = {}
        for weight, variable in weighted_terms:
            if not isinstance(weight, Real):
                raise ValueError("Linear expression weights must be real numbers.")
            if variable.registry is not registry:
                raise ValueError("Linear expression terms must share the same registry.")
            if len(variable.terms) != 1:
                raise ValueError("Linear expression terms require plain binary variables.")
            ((term, coeff),) = variable.terms.items()
            if len(term) != 1 or coeff != 1.0:
                raise ValueError("Linear expression terms require plain binary variables.")
            var_id = term[0]
            terms[var_id] = terms.get(var_id, 0.0) + float(weight)
        return LinearExpression({key: value for key, value in terms.items() if abs(value) > 1e-12}, float(offset), registry)

    @staticmethod
    def from_integer(value: EncodedInteger, scale: float = 1.0) -> LinearExpression:
        weighted_terms = [(scale * weight, bit) for weight, bit in zip(value.weights, value.bits)]
        return LinearExpression.from_terms(weighted_terms, offset=scale * value.lower_bound)

    def weighted_terms(self) -> list[tuple[float, Polynomial]]:
        return [(weight, Polynomial({(var_id,): 1.0}, self.registry)) for var_id, weight in self.terms.items()]

    def expression(self) -> Polynomial:
        return self.offset + Sum(weight * variable for weight, variable in self.weighted_terms())

    def __add__(self, other: LinearExpression | EncodedInteger | Polynomial | int | float) -> LinearExpression:
        other_linear = _as_linear(other, self.registry)
        terms = dict(self.terms)
        for var_id, coeff in other_linear.terms.items():
            terms[var_id] = terms.get(var_id, 0.0) + coeff
        return LinearExpression({key: value for key, value in terms.items() if abs(value) > 1e-12}, self.offset + other_linear.offset, self.registry)

    def __radd__(self, other: LinearExpression | EncodedInteger | Polynomial | int | float) -> LinearExpression:
        return self + other

    def __sub__(self, other: LinearExpression | EncodedInteger | Polynomial | int | float) -> LinearExpression:
        return self + (-_as_linear(other, self.registry))

    def __rsub__(self, other: LinearExpression | EncodedInteger | Polynomial | int | float) -> LinearExpression:
        return _as_linear(other, self.registry) + (-self)

    def __neg__(self) -> LinearExpression:
        return LinearExpression({var_id: -coeff for var_id, coeff in self.terms.items()}, -self.offset, self.registry)

    def __mul__(self, scale: float) -> LinearExpression:
        if not isinstance(scale, Real):
            return NotImplemented
        return LinearExpression({var_id: float(scale) * coeff for var_id, coeff in self.terms.items()}, float(scale) * self.offset, self.registry)

    def __rmul__(self, scale: float) -> LinearExpression:
        return self * scale


def Linear(
    weighted_terms: Iterable[tuple[float, Polynomial]] | EncodedInteger,
    offset: float = 0.0,
) -> LinearExpression:
    if isinstance(weighted_terms, EncodedInteger):
        return LinearExpression.from_integer(weighted_terms)
    return LinearExpression.from_terms(weighted_terms, offset=offset)


def _as_linear(value, registry: VariableRegistry) -> LinearExpression:
    if isinstance(value, LinearExpression):
        if value.registry is not registry:
            raise ValueError("Linear expression registries must match.")
        return value
    if isinstance(value, EncodedInteger):
        linear = LinearExpression.from_integer(value)
        if linear.registry is not registry:
            raise ValueError("Linear expression registries must match.")
        return linear
    if isinstance(value, Polynomial):
        return LinearExpression.from_terms([(1.0, value)])
    if isinstance(value, Real):
        return LinearExpression({}, float(value), registry)
    return NotImplemented


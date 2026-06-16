from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from numbers import Real
from typing import Iterable

from .registry import GLOBAL_REGISTRY, VariableRegistry

Number = int | float
Term = tuple[int, ...]


def _canonical_term(term: Iterable[int]) -> Term:
    return tuple(sorted(set(term)))


def _as_poly(value: Polynomial | Number) -> Polynomial:
    if isinstance(value, Polynomial):
        return value
    if isinstance(value, Real):
        return Polynomial.constant(float(value))
    return NotImplemented


@dataclass(frozen=True)
class Polynomial:
    """Sparse polynomial over binary variables.

    Terms are represented by tuples of integer variable IDs. Since variables are
    binary, repeated variables inside a product are collapsed: x_i * x_i = x_i.
    """

    terms: dict[Term, float]
    registry: VariableRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)

    @staticmethod
    def constant(value: Number, registry: VariableRegistry = GLOBAL_REGISTRY) -> Polynomial:
        if value == 0:
            return Polynomial({}, registry)
        return Polynomial({(): float(value)}, registry)

    @staticmethod
    def variable(
        name: str,
        registry: VariableRegistry = GLOBAL_REGISTRY,
        role: str = "decision",
        source: str | None = None,
    ) -> Polynomial:
        var_id = registry.get_or_create(name, kind="binary", role=role, source=source)
        return Polynomial({(var_id,): 1.0}, registry)

    def clean(self, eps: float = 1e-12) -> Polynomial:
        return Polynomial({term: coeff for term, coeff in self.terms.items() if abs(coeff) > eps}, self.registry)

    def degree(self) -> int:
        if not self.terms:
            return 0
        return max(len(term) for term in self.terms)

    def variables(self) -> list[str]:
        ids = sorted({var_id for term in self.terms for var_id in term})
        return [self.registry.name(var_id) for var_id in ids]

    def __add__(self, other: Polynomial | Number) -> Polynomial:
        other_poly = _as_poly(other)
        if other_poly is NotImplemented:
            return NotImplemented
        result = dict(self.terms)
        for term, coeff in other_poly.terms.items():
            result[term] = result.get(term, 0.0) + coeff
        return Polynomial(result, self.registry).clean()

    def __radd__(self, other: Polynomial | Number) -> Polynomial:
        return self + other

    def __sub__(self, other: Polynomial | Number) -> Polynomial:
        return self + (-_as_poly(other))

    def __rsub__(self, other: Polynomial | Number) -> Polynomial:
        return _as_poly(other) + (-self)

    def __neg__(self) -> Polynomial:
        return Polynomial({term: -coeff for term, coeff in self.terms.items()}, self.registry)

    def __mul__(self, other: Polynomial | Number) -> Polynomial:
        other_poly = _as_poly(other)
        if other_poly is NotImplemented:
            return NotImplemented
        result: dict[Term, float] = {}
        for left_term, left_coeff in self.terms.items():
            for right_term, right_coeff in other_poly.terms.items():
                term = _canonical_term((*left_term, *right_term))
                result[term] = result.get(term, 0.0) + left_coeff * right_coeff
        return Polynomial(result, self.registry).clean()

    def __rmul__(self, other: Polynomial | Number) -> Polynomial:
        return self * other

    def __pow__(self, power: int) -> Polynomial:
        if not isinstance(power, int) or power < 0:
            raise ValueError("Polynomial powers must be non-negative integers.")
        result = Polynomial.constant(1.0, self.registry)
        for _ in range(power):
            result = result * self
        return result

    def to_qubo_indices(self) -> tuple[dict[tuple[int, int], float], float]:
        qubo: dict[tuple[int, int], float] = {}
        offset = 0.0
        for term, coeff in self.terms.items():
            if len(term) == 0:
                offset += coeff
            elif len(term) == 1:
                var_id = term[0]
                qubo[(var_id, var_id)] = qubo.get((var_id, var_id), 0.0) + coeff
            elif len(term) == 2:
                i, j = term
                key = (i, j)
                qubo[key] = qubo.get(key, 0.0) + coeff
            else:
                raise ValueError("QUBO output only supports degree <= 2 polynomials.")
        return ({key: value for key, value in qubo.items() if abs(value) > 1e-12}, offset)

    def to_qubo(self) -> tuple[dict[tuple[str, str], float], float]:
        qubo_ids, offset = self.to_qubo_indices()
        names = self.registry.id_to_name
        return ({(names[i], names[j]): coeff for (i, j), coeff in qubo_ids.items()}, offset)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for term, coeff in sorted(self.terms.items(), key=lambda item: (len(item[0]), item[0])):
            if len(term) == 0:
                parts.append(f"{coeff:g}")
                continue
            names = "*".join(self.registry.name(var_id) for var_id in term)
            parts.append(f"{coeff:g}*{names}")
        return " + ".join(parts)


def Binary(
    name: str,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> Polynomial:
    return Polynomial.variable(name, registry=registry, role=role, source=source)


def Sum(values: Iterable[Polynomial | Number]) -> Polynomial:
    result: dict[Term, float] = {}
    registry = GLOBAL_REGISTRY
    for value in values:
        poly = _as_poly(value)
        if poly is NotImplemented:
            return NotImplemented
        registry = poly.registry
        for term, coeff in poly.terms.items():
            result[term] = result.get(term, 0.0) + coeff
    return Polynomial(result, registry).clean()


def binary_array(
    prefix: str,
    shape: int | tuple[int, ...],
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> list:
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        return [Binary(f"{prefix}[{i}]", registry=registry, role=role, source=source) for i in range(shape[0])]
    return [
        binary_array(f"{prefix}[{i}]", shape[1:], registry=registry, role=role, source=source)
        for i in range(shape[0])
    ]


def iter_indices(shape: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    return product(*(range(size) for size in shape))


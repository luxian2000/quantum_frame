from __future__ import annotations

from dataclasses import dataclass, field

from .polynomial import Polynomial
from .registry import GLOBAL_REGISTRY, VariableRegistry

# Literal encoding: variable id v -> positive literal node 2v, complement node 2v+1.
# complement(node) == node ^ 1.


@dataclass(frozen=True)
class Posiform:
    """A quadratic pseudo-Boolean function as a posiform.

    Terms are products of one or two literals with strictly positive
    coefficients, plus a separate constant. A literal is a variable or its
    Boolean complement, encoded as an integer node (see module header).
    """

    terms: dict[tuple[int, ...], float]
    constant: float
    registry: VariableRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)


def to_posiform(poly: Polynomial) -> Posiform:
    """Rewrite a degree <= 2 polynomial as a posiform via x = 1 - x_bar.

    Negative coefficients are moved onto complemented literals so that every
    monomial coefficient becomes nonnegative, at the cost of an additive
    constant. The choice x_i x_j = x_i - x_i x_bar_j is fixed (deterministic);
    a stronger posiform would require roof duality, which is intentionally not
    used here.
    """
    if poly.degree() > 2:
        raise ValueError("to_posiform only supports degree <= 2 polynomials.")

    terms: dict[tuple[int, ...], float] = {}
    constant = 0.0

    def add(key: tuple[int, ...], coeff: float) -> None:
        terms[key] = terms.get(key, 0.0) + coeff

    for term, coeff in poly.terms.items():
        if len(term) == 0:
            constant += coeff
        elif len(term) == 1:
            v = term[0]
            if coeff > 0:
                add((2 * v,), coeff)
            elif coeff < 0:
                constant += coeff
                add((2 * v + 1,), -coeff)
        else:
            i, j = term
            if coeff > 0:
                add((2 * i, 2 * j), coeff)
            elif coeff < 0:
                # coeff < 0: c*xi*xj = c + (-c)*x_bar_i + (-c)*xi*x_bar_j
                constant += coeff
                add((2 * i + 1,), -coeff)
                add((2 * i, 2 * j + 1), -coeff)

    cleaned = {key: value for key, value in terms.items() if abs(value) > 1e-12}
    return Posiform(cleaned, constant, poly.registry)

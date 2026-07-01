import itertools
import random

from aicir.optimization.qubo import Binary, split_cores
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.polynomial import Polynomial


def _eval(poly, bits):
    total = 0.0
    for term, coeff in poly.terms.items():
        product = 1
        for var_id in term:
            product *= bits[var_id]
        total += coeff * product
    return total


def _min_energy(poly):
    var_ids = sorted({var_id for term in poly.terms for var_id in term})
    if not var_ids:
        return _eval(poly, {})
    return min(
        _eval(poly, dict(zip(var_ids, combo)))
        for combo in itertools.product((0, 1), repeat=len(var_ids))
    )


def test_split_cores_single_frustrated_core():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    f = a * b - 2 * a - 2 * b + 5   # global minimum is 2

    pieces = split_cores(f)

    assert _min_energy(f) == 2
    assert abs(_min_energy(f) - sum(_min_energy(p) for p in pieces)) < 1e-9


def test_split_cores_preserves_minimum_random():
    rng = random.Random(0)
    for _ in range(50):
        reg = VariableRegistry()
        xs = [Binary(f"x{i}", registry=reg) for i in range(4)]
        f = Polynomial.constant(0.0, reg)
        for i in range(4):
            if rng.random() < 0.5:
                f = f + rng.randint(-3, 3) * xs[i]
        for i in range(4):
            for j in range(i + 1, 4):
                if rng.random() < 0.5:
                    f = f + rng.randint(-3, 3) * xs[i] * xs[j]
        pieces = split_cores(f)
        assert abs(_min_energy(f) - sum(_min_energy(p) for p in pieces)) < 1e-9


def test_split_cores_produces_disjoint_cores():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    d = Binary("d", registry=reg)
    f = (a * b - 2 * a - 2 * b + 5) + (c * d - 2 * c - 2 * d + 5)

    non_constant = [p for p in split_cores(f) if any(term for term in p.terms)]
    var_sets = [{v for term in p.terms for v in term} for p in non_constant]

    for i in range(len(var_sets)):
        for j in range(i + 1, len(var_sets)):
            assert var_sets[i].isdisjoint(var_sets[j])


def test_split_cores_rejects_cubic():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    try:
        split_cores(a * b * c)
    except ValueError as exc:
        assert "degree <= 2" in str(exc)
    else:
        raise AssertionError("Expected cubic input to be rejected.")


def test_split_cores_omits_independently_satisfiable_terms():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    # a*b - 2a - 2b + 5 is a frustrated hard core (min 2); the loose "+ c" is a
    # positive literal that is independently satisfiable (c = 0) and must NOT be
    # returned as its own core. This guards the self-dual filter: dropping it
    # would emit c as a spurious second core (min still preserved, but not lean).
    f = (a * b - 2 * a - 2 * b + 5) + c
    c_id = next(iter(c.terms))[0]

    pieces = split_cores(f)
    non_constant = [p for p in pieces if any(term for term in p.terms)]

    assert len(non_constant) == 1
    core_vars = {v for term in non_constant[0].terms for v in term}
    assert c_id not in core_vars
    assert abs(_min_energy(f) - sum(_min_energy(p) for p in pieces)) < 1e-9

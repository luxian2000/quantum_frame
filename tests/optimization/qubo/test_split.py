import itertools

from aicir.optimization.qubo import Binary, Sum, split_polynomial
from aicir.optimization.qubo.modeling import VariableRegistry


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


def _var_ids(poly):
    return {var_id for term in poly.terms for var_id in term}


def test_split_partitions_into_disjoint_components() -> None:
    reg = VariableRegistry()
    x = [Binary(f"x{i}", registry=reg) for i in range(5)]
    f = 3 * x[0] * x[1] + x[1] + 5 * x[2] * x[3] - x[3] + 2 * x[4] + 7

    pieces = split_polynomial(f)

    # 3 variable components {x0,x1}, {x2,x3}, {x4} plus the constant part.
    assert len(pieces) == 4
    # Parts partition the terms exactly.
    assert Sum(pieces).terms == f.terms
    # Non-constant parts share no variables.
    non_constant = [piece for piece in pieces if _var_ids(piece)]
    for i, left in enumerate(non_constant):
        for right in non_constant[i + 1 :]:
            assert _var_ids(left).isdisjoint(_var_ids(right))
    # The decomposition preserves the global minimum.
    assert _min_energy(f) == sum(_min_energy(piece) for piece in pieces)


def test_split_keeps_fully_connected_block_together() -> None:
    reg = VariableRegistry()
    x = [Binary(f"c{i}", registry=reg) for i in range(3)]
    f = x[0] * x[1] + x[1] * x[2] + x[0] * x[2]

    pieces = split_polynomial(f)

    assert len(pieces) == 1
    assert pieces[0].terms == f.terms


def test_split_returns_constant_as_its_own_part() -> None:
    reg = VariableRegistry()
    f = 5 + 0 * Binary("const_x", registry=reg)

    pieces = split_polynomial(f)

    assert len(pieces) == 1
    assert pieces[0].terms == {(): 5.0}


def test_split_empty_polynomial_returns_empty_list() -> None:
    assert split_polynomial(Sum([])) == []

from aicir.optimization.qubo import Binary, Integer, Linear, ModelContext


def test_linear_expression_combines_duplicate_terms() -> None:
    ctx = ModelContext()
    x = ctx.binary("x")

    expression = Linear([(2.0, x), (3.0, x)], offset=1.0)

    assert expression.terms == {0: 5.0}
    assert expression.offset == 1.0
    assert expression.expression().terms == {(): 1.0, (0,): 5.0}


def test_linear_expression_arithmetic() -> None:
    ctx = ModelContext()
    x = ctx.binary("x")
    y = ctx.binary("y")

    expression = Linear([(2.0, x)]) + Linear([(3.0, y)], offset=4.0) - 1.0

    assert expression.offset == 3.0
    assert expression.terms == {0: 2.0, 1: 3.0}


def test_linear_expression_from_integer() -> None:
    ctx = ModelContext()
    value = Integer("z", upper_bound=5, registry=ctx.registry)

    expression = value.linear_expression(scale=2.0)

    assert expression.terms == {0: 2.0, 1: 4.0, 2: 4.0}
    assert expression.offset == 0.0


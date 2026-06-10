from aicir.optimization.qubo import Integer, LogEncodedInteger, ModelContext, Sum, UnaryEncodedInteger, integer_equality, integer_less_equal


def test_log_encoded_integer_uses_bounded_weights() -> None:
    ctx = ModelContext()

    value = LogEncodedInteger("z", lower_bound=0, upper_bound=5, registry=ctx.registry)

    assert value.weights == [1, 2, 2]
    assert value.bit_names() == ["z[0]", "z[1]", "z[2]"]
    assert value.expression().variables() == ["z[0]", "z[1]", "z[2]"]


def test_unary_encoded_integer() -> None:
    ctx = ModelContext()

    value = UnaryEncodedInteger("u", lower_bound=0, upper_bound=3, registry=ctx.registry)

    assert value.weights == [1, 1, 1]
    assert value.bit_names() == ["u[0]", "u[1]", "u[2]"]


def test_integer_dispatch_and_context_method() -> None:
    ctx = ModelContext()

    log_value = Integer("a", upper_bound=4, encoding="log", registry=ctx.registry)
    unary_value = ctx.integer("b", upper_bound=2, encoding="unary")

    assert log_value.weights == [1, 2, 1]
    assert unary_value.weights == [1, 1]
    assert log_value.registry is ctx.registry
    assert unary_value.registry is ctx.registry


def test_constant_integer_expression_uses_registry() -> None:
    ctx = ModelContext()

    value = Integer("c", lower_bound=3, upper_bound=3, registry=ctx.registry)
    expression = value.expression()

    assert value.bits == []
    assert expression.terms == {(): 3.0}
    assert expression.registry is ctx.registry


def test_integer_equality_matches_expression_square() -> None:
    ctx = ModelContext()
    value = Integer("eq", upper_bound=5, registry=ctx.registry)

    constraint = integer_equality(value, target=3, penalty=2.0)
    generic = (value.expression() - 3) ** 2

    assert constraint.expression.terms == generic.terms


def test_integer_less_equal_matches_linear_inequality_form() -> None:
    ctx = ModelContext()
    value = Integer("le", upper_bound=5, registry=ctx.registry)

    constraint, slack = integer_less_equal(value, upper_bound=4, slack_prefix="le_s", penalty=3.0)
    generic = (value.expression() + Sum(weight * bit for weight, bit in zip([1, 2, 1], slack)) - 4) ** 2

    assert constraint.expression.terms == generic.terms
    assert [bit.variables()[0] for bit in slack] == ["le_s[0]", "le_s[1]", "le_s[2]"]


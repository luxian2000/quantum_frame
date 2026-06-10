from aicir.optimization.qubo import ModelContext, decode_integer, decode_solution, linear_inequality


def test_decode_solution_filters_auxiliary_by_default() -> None:
    ctx = ModelContext()
    x = ctx.binary("x")
    constraint, slack = linear_inequality([(2.0, x)], upper_bound=3, slack_prefix="s")
    assert constraint.expression.terms

    decoded = decode_solution({"x": 1, "s[0]": 1, "s[1]": 0}, registry=ctx.registry)

    assert decoded.binary == {"x": 1}
    assert decoded.auxiliary == {}
    assert decoded.decisions() == {"x": 1}


def test_decode_solution_can_include_auxiliary() -> None:
    ctx = ModelContext()
    x = ctx.binary("x")
    linear_inequality([(2.0, x)], upper_bound=3, slack_prefix="s")

    decoded = ctx.decode_solution({"x": 1, "s[0]": 1, "s[1]": 0}, include_auxiliary=True)

    assert decoded.auxiliary == {"s[0]": 1, "s[1]": 0}


def test_decode_integer_from_named_assignment() -> None:
    ctx = ModelContext()
    value = ctx.integer("z", upper_bound=5)

    decoded = decode_integer(value, {"z[0]": 1, "z[1]": 0, "z[2]": 1})

    assert decoded == 3


def test_decode_solution_decodes_integer_and_hides_bits() -> None:
    ctx = ModelContext()
    x = ctx.binary("x")
    value = ctx.integer("z", upper_bound=5)

    decoded = ctx.decode_solution(
        {"x": 1, "z[0]": 1, "z[1]": 1, "z[2]": 0},
        integers=[value],
    )

    assert decoded.binary == {"x": 1}
    assert decoded.integers == {"z": 3}
    assert decoded.decisions() == {"x": 1, "z": 3}


def test_decode_solution_from_bitstring() -> None:
    ctx = ModelContext()
    ctx.binary("x")
    value = ctx.integer("z", upper_bound=3)

    decoded = ctx.decode_solution([1, 0, 1], integers=[value])

    assert decoded.binary == {"x": 1}
    assert decoded.integers == {"z": 2}


def test_decode_solution_rejects_non_binary_values() -> None:
    ctx = ModelContext()
    ctx.binary("x")

    try:
        ctx.decode_solution({"x": 0.5})
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected non-binary solution value to be rejected.")


from aicir.optimization.qubo import (
    Binary,
    Model,
    ModelContext,
    ObjectiveFragment,
    Polynomial,
    QuboBuilder,
    Sum,
    linear_objective,
    quadratic_objective,
)


def test_linear_objective_matches_polynomial_objective() -> None:
    ctx = ModelContext()
    x = [ctx.binary(f"obj_lin_x{i}") for i in range(3)]

    structured = Model(ctx.zero())
    structured.add_objective(linear_objective((weight, variable) for weight, variable in zip([1.0, -2.0, 3.0], x)))

    generic = Model(Sum(weight * variable for weight, variable in zip([1.0, -2.0, 3.0], x)))

    assert structured.to_qubo_indices() == generic.to_qubo_indices()


def test_quadratic_objective_matches_polynomial_objective() -> None:
    ctx = ModelContext()
    x = [ctx.binary(f"obj_quad_x{i}") for i in range(3)]

    structured = Model(ctx.zero())
    structured.add_objective(
        quadratic_objective(
            [
                (2.0, x[0], x[1]),
                (-3.0, x[1], x[2]),
                (4.0, x[0], x[0]),
            ]
        )
    )

    generic = Model(2.0 * x[0] * x[1] - 3.0 * x[1] * x[2] + 4.0 * x[0])

    assert structured.to_qubo_indices() == generic.to_qubo_indices()


def test_objective_fragment_does_not_materialize_lazy_expression_during_builder_export() -> None:
    x = Binary("obj_lazy_x")
    x_id = _var_id(x)

    def fail_if_materialized():
        raise AssertionError("Lazy objective expression should not be materialized during builder export.")

    def add_linear_x(builder: QuboBuilder) -> None:
        builder.add_linear(x_id, 5.0)

    model = Model(0 * x)
    model.add_objective(
        ObjectiveFragment(
            registry=x.registry,
            expression_factory=fail_if_materialized,
            builder_action=add_linear_x,
        )
    )

    qubo, offset = model.to_qubo_indices()

    assert qubo == {(x_id, x_id): 5.0}
    assert offset == 0.0


def test_model_polynomial_materializes_objective_fragments() -> None:
    ctx = ModelContext()
    x = [ctx.binary(f"obj_poly_x{i}") for i in range(2)]
    model = Model(ctx.zero())
    model.add_objective(quadratic_objective([(7.0, x[0], x[1])]))

    polynomial = model.polynomial()

    assert polynomial.terms == {(0, 1): 7.0}


def test_objective_fragment_rejects_registry_mismatch() -> None:
    ctx = ModelContext()
    other_ctx = ModelContext()
    x = ctx.binary("obj_reg_x")
    other = other_ctx.binary("obj_reg_y")
    model = Model(0 * x)

    try:
        model.add_objective(other)
    except ValueError as exc:
        assert "registry" in str(exc)
    else:
        raise AssertionError("Expected registry mismatch to be rejected.")


def _var_id(variable):
    ((term, coeff),) = variable.terms.items()
    assert coeff == 1.0
    return term[0]


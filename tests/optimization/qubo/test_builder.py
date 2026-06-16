from aicir.optimization.qubo import (
    Binary,
    Constraint,
    Model,
    QuboBuilder,
    Sum,
    at_least_one,
    at_most_one,
    cardinality,
    linear_inequality,
    weighted_equality,
)


def _var_id(variable):
    ((term, coeff),) = variable.terms.items()
    assert coeff == 1.0
    return term[0]


def test_builder_cardinality_matches_constraint_builder() -> None:
    x = [Binary(f"builder_card_x{i}") for i in range(4)]
    builder = QuboBuilder(registry=x[0].registry)

    builder.add_cardinality_penalty([_var_id(variable) for variable in x], count=2, penalty=3.0)
    expected = (3.0 * cardinality(x, count=2).expression).to_qubo_indices()

    assert builder.to_qubo_indices() == expected


def test_builder_weighted_equality_matches_constraint_builder() -> None:
    x = [Binary(f"builder_we_x{i}") for i in range(3)]
    builder = QuboBuilder(registry=x[0].registry)
    weighted_ids = [(2.0, _var_id(x[0])), (-1.0, _var_id(x[1])), (4.0, _var_id(x[2]))]

    builder.add_weighted_equality_penalty(weighted_ids, target=3.0, penalty=5.0)
    expected = weighted_equality(
        [(2.0, x[0]), (-1.0, x[1]), (4.0, x[2])],
        target=3.0,
        penalty=5.0,
    ).as_penalty().to_qubo_indices()

    assert builder.to_qubo_indices() == expected


def test_builder_at_most_one_matches_constraint_builder() -> None:
    x = [Binary(f"builder_amo_x{i}") for i in range(4)]
    builder = QuboBuilder(registry=x[0].registry)

    builder.add_at_most_one_penalty([_var_id(variable) for variable in x], penalty=6.0)
    expected = at_most_one(x, penalty=6.0).as_penalty().to_qubo_indices()

    assert builder.to_qubo_indices() == expected


def test_builder_at_least_one_matches_constraint_builder() -> None:
    x = [Binary(f"builder_alo_x{i}") for i in range(3)]
    builder = QuboBuilder(registry=x[0].registry)

    slack_ids = builder.add_at_least_one_penalty(
        [_var_id(variable) for variable in x],
        slack_prefix="builder_alo_s",
        penalty=5.0,
    )
    constraint, slack = at_least_one(x, slack_prefix="builder_alo_s", penalty=5.0)

    assert slack_ids == [_var_id(variable) for variable in slack]
    assert builder.to_qubo_indices() == constraint.as_penalty().to_qubo_indices()


def test_builder_linear_inequality_matches_constraint_builder() -> None:
    x = [Binary(f"builder_li_x{i}") for i in range(2)]
    builder = QuboBuilder(registry=x[0].registry)

    slack_ids = builder.add_linear_inequality_penalty(
        [(2.0, _var_id(x[0])), (3.0, _var_id(x[1]))],
        upper_bound=4,
        slack_prefix="builder_li_s",
        penalty=7.0,
    )
    constraint, slack = linear_inequality(
        [(2.0, x[0]), (3.0, x[1])],
        upper_bound=4,
        slack_prefix="builder_li_s",
        penalty=7.0,
    )

    assert slack_ids == [_var_id(variable) for variable in slack]
    assert builder.to_qubo_indices() == constraint.as_penalty().to_qubo_indices()


def test_builder_add_polynomial() -> None:
    x = Binary("builder_poly_x")
    y = Binary("builder_poly_y")
    polynomial = 2.0 + 3.0 * x - 4.0 * x * y + Sum([y])
    builder = QuboBuilder(registry=x.registry)

    builder.add_polynomial(polynomial)

    assert builder.to_qubo_indices() == polynomial.to_qubo_indices()


def test_builder_bulk_linear_terms_match_single_writes() -> None:
    x = [Binary(f"builder_bulk_lin_x{i}") for i in range(3)]
    ids = [_var_id(variable) for variable in x]
    bulk = QuboBuilder(registry=x[0].registry)
    single = QuboBuilder(registry=x[0].registry)

    weighted_ids = [(2.0, ids[0]), (-3.0, ids[1]), (0.0, ids[2]), (5.0, ids[0])]
    bulk.add_linear_terms(weighted_ids)
    for weight, var_id in weighted_ids:
        single.add_linear(var_id, weight)

    assert bulk.to_qubo_indices() == single.to_qubo_indices()


def test_builder_bulk_quadratic_terms_match_single_writes() -> None:
    x = [Binary(f"builder_bulk_quad_x{i}") for i in range(3)]
    ids = [_var_id(variable) for variable in x]
    bulk = QuboBuilder(registry=x[0].registry)
    single = QuboBuilder(registry=x[0].registry)

    weighted_ids = [(2.0, ids[0], ids[1]), (-3.0, ids[2], ids[1]), (0.0, ids[0], ids[2])]
    bulk.add_quadratic_terms(weighted_ids)
    for weight, left_id, right_id in weighted_ids:
        single.add_quadratic(left_id, right_id, weight)

    assert bulk.to_qubo_indices() == single.to_qubo_indices()


def test_builder_to_qubo_indices_clean_and_raw_modes() -> None:
    x = Binary("builder_clean_x")
    x_id = _var_id(x)
    builder = QuboBuilder(registry=x.registry)
    builder.add_linear(x_id, 1.0)
    builder.add_linear(x_id, -1.0)

    clean_qubo, clean_offset = builder.to_qubo_indices()
    raw_qubo, raw_offset = builder.to_qubo_indices(clean=False)
    raw_view, _ = builder.to_qubo_indices(clean=False, copy=False)

    assert clean_qubo == {}
    assert clean_offset == 0.0
    assert raw_qubo == {(x_id, x_id): 0.0}
    assert raw_offset == 0.0
    assert raw_view is builder.qubo


def test_model_uses_constraint_builder_action() -> None:
    x = Binary("builder_hook_x")
    y = Binary("builder_hook_y")
    z = Binary("builder_hook_z")
    x_id = _var_id(x)

    def add_linear_x(builder: QuboBuilder, penalty: float) -> None:
        builder.add_linear(x_id, penalty)

    model = Model(0 * x)
    model.add_constraint(Constraint(x * y * z, penalty=7.0, builder_action=add_linear_x))

    qubo, offset = model.to_qubo_indices()

    assert qubo == {(x_id, x_id): 7.0}
    assert offset == 0.0


def test_model_builder_action_does_not_materialize_lazy_expression() -> None:
    x = Binary("builder_lazy_x")
    x_id = _var_id(x)

    def fail_if_materialized():
        raise AssertionError("Lazy expression should not be materialized during builder export.")

    def add_linear_x(builder: QuboBuilder, penalty: float) -> None:
        builder.add_linear(x_id, penalty)

    model = Model(0 * x)
    model.add_constraint(
        Constraint(
            penalty=3.0,
            registry=x.registry,
            expression_factory=fail_if_materialized,
            builder_action=add_linear_x,
        )
    )

    qubo, offset = model.to_qubo_indices()

    assert qubo == {(x_id, x_id): 3.0}
    assert offset == 0.0


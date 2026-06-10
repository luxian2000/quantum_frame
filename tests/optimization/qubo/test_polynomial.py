from aicir.optimization.qubo import (
    Binary,
    Model,
    Sum,
    assignment_matrix,
    at_least_one,
    at_most_one,
    binary_array,
    cardinality,
    linear_inequality,
    one_hot,
    one_hot_columns,
    one_hot_rows,
    permutation,
    weighted_equality,
)
from aicir.optimization.qubo.modeling.integer import bounded_log_weights


def test_binary_square_collapses_to_linear_term() -> None:
    x = Binary("square_x")

    qubo, offset = (x * x).to_qubo()

    assert qubo == {("square_x", "square_x"): 1.0}
    assert offset == 0.0


def test_one_hot_qubo_coefficients() -> None:
    x0 = Binary("oh_x0")
    x1 = Binary("oh_x1")
    model = Model(0 * x0)
    model.add_constraint(one_hot([x0, x1], penalty=2.0))

    qubo, offset = model.to_qubo()

    assert qubo == {
        ("oh_x0", "oh_x0"): -2.0,
        ("oh_x1", "oh_x1"): -2.0,
        ("oh_x0", "oh_x1"): 4.0,
    }
    assert offset == 2.0


def test_cardinality_qubo_coefficients() -> None:
    x = [Binary(f"card_x{i}") for i in range(3)]
    model = Model(Sum([]))
    model.add_constraint(cardinality(x, count=2, penalty=1.5))

    qubo, offset = model.to_qubo()

    assert qubo[("card_x0", "card_x0")] == -4.5
    assert qubo[("card_x1", "card_x1")] == -4.5
    assert qubo[("card_x2", "card_x2")] == -4.5
    assert qubo[("card_x0", "card_x1")] == 3.0
    assert qubo[("card_x0", "card_x2")] == 3.0
    assert qubo[("card_x1", "card_x2")] == 3.0
    assert offset == 6.0


def test_cardinality_fast_path_matches_generic_expansion() -> None:
    x = [Binary(f"fast_x{i}") for i in range(4)]
    fast = cardinality(x, count=2).expression
    generic = (Sum(x) - 2) ** 2

    assert fast.terms == generic.terms


def test_cardinality_rejects_non_variable_expression() -> None:
    x = Binary("bad_expr_x")
    y = Binary("bad_expr_y")

    try:
        cardinality([x + y], count=1)
    except ValueError as exc:
        assert "plain binary variables" in str(exc)
    else:
        raise AssertionError("Expected non-variable cardinality input to be rejected.")


def test_cardinality_rejects_duplicate_variables() -> None:
    x = Binary("dup_x")

    try:
        cardinality([x, x], count=1)
    except ValueError as exc:
        assert "distinct" in str(exc)
    else:
        raise AssertionError("Expected duplicate cardinality input to be rejected.")


def test_at_most_one_qubo_coefficients() -> None:
    x = [Binary(f"amo_x{i}") for i in range(3)]
    model = Model(Sum([]))
    model.add_constraint(at_most_one(x, penalty=4.0))

    qubo, offset = model.to_qubo()

    assert qubo == {
        ("amo_x0", "amo_x1"): 4.0,
        ("amo_x0", "amo_x2"): 4.0,
        ("amo_x1", "amo_x2"): 4.0,
    }
    assert offset == 0.0


def test_at_least_one_matches_generic_slack_expansion() -> None:
    x = [Binary(f"alo_x{i}") for i in range(3)]

    constraint, slack = at_least_one(x, slack_prefix="alo_s", penalty=2.0)
    generic = (Sum(x) - Sum((2**bit) * slack[bit] for bit in range(len(slack))) - 1) ** 2

    assert constraint.expression.terms == generic.terms
    assert [var.variables()[0] for var in slack] == ["alo_s[0]", "alo_s[1]"]


def test_weighted_equality_fast_path_matches_generic_expansion() -> None:
    x = Binary("we_x")
    y = Binary("we_y")
    z = Binary("we_z")

    fast = weighted_equality([(2.0, x), (-3.0, y), (4.0, z)], target=5.0).expression
    generic = (2.0 * x - 3.0 * y + 4.0 * z - 5.0) ** 2

    assert fast.terms == generic.terms


def test_linear_inequality_matches_generic_slack_expansion() -> None:
    x = Binary("li_x")
    y = Binary("li_y")

    constraint, slack = linear_inequality(
        [(2.0, x), (3.0, y)],
        upper_bound=4,
        slack_prefix="li_s",
        penalty=7.0,
    )
    weights = bounded_log_weights(4)
    generic = (2.0 * x + 3.0 * y + Sum(weight * bit for weight, bit in zip(weights, slack)) - 4.0) ** 2

    assert constraint.expression.terms == generic.terms
    assert len(slack) == 3
    ((term, _),) = slack[0].terms.items()
    assert slack[0].registry.metadata(term[0]).source == "linear_inequality"


def test_weighted_equality_combines_duplicate_variables() -> None:
    x = Binary("we_dup_x")
    y = Binary("we_dup_y")

    fast = weighted_equality([(2.0, x), (3.0, x), (1.0, y)], target=2.0).expression
    generic = (5.0 * x + y - 2.0) ** 2

    assert fast.terms == generic.terms


def test_one_hot_rows_and_columns_match_manual_constraints() -> None:
    x = binary_array("matrix_x", (2, 3))

    matrix_model = Model(Sum([]))
    matrix_model.add_constraints(one_hot_rows(x, penalty=2.0, label="rows"))
    matrix_model.add_constraints(one_hot_columns(x, penalty=2.0, label="cols"))

    manual_model = Model(Sum([]))
    for row in x:
        manual_model.add_constraint(one_hot(row, penalty=2.0))
    for col_index in range(3):
        manual_model.add_constraint(one_hot([x[row_index][col_index] for row_index in range(2)], penalty=2.0))

    assert matrix_model.to_qubo_indices() == manual_model.to_qubo_indices()


def test_assignment_matrix_matches_manual_row_and_column_one_hot() -> None:
    x = binary_array("assign_x", (3, 2))

    model = Model(Sum([]))
    model.add_constraints(assignment_matrix(x, penalty=3.0, label="assign"))

    manual_model = Model(Sum([]))
    for row in x:
        manual_model.add_constraint(one_hot(row, penalty=3.0))
    for col_index in range(2):
        manual_model.add_constraint(one_hot([x[row_index][col_index] for row_index in range(3)], penalty=3.0))

    assert model.to_qubo_indices() == manual_model.to_qubo_indices()


def test_permutation_requires_square_matrix() -> None:
    x = binary_array("perm_bad_x", (2, 3))

    try:
        permutation(x)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("Expected permutation to reject non-square matrices.")


def test_matrix_constraints_reject_ragged_matrix() -> None:
    x0 = Binary("ragged_x0")
    x1 = Binary("ragged_x1")
    x2 = Binary("ragged_x2")

    try:
        assignment_matrix([[x0, x1], [x2]])
    except ValueError as exc:
        assert "rectangular" in str(exc)
    else:
        raise AssertionError("Expected assignment matrix to reject ragged input.")


def test_to_ising_single_variable() -> None:
    x = Binary("ising_x")
    model = Model(2 * x)

    ising = model.to_ising()

    assert ising == {"h": {"ising_x": 1.0}, "J": {}, "offset": 1.0}


def test_to_qubo_indices_matches_named_qubo() -> None:
    x = Binary("idx_x")
    y = Binary("idx_y")
    model = Model(3 * x + 4 * x * y)

    qubo, offset = model.to_qubo()
    qubo_ids, offset_ids = model.to_qubo_indices()
    id_to_name = model.objective.registry.id_to_name
    named_from_ids = {(id_to_name[i], id_to_name[j]): coeff for (i, j), coeff in qubo_ids.items()}

    assert named_from_ids == qubo
    assert offset_ids == offset


def test_high_order_expression_rejected_for_qubo() -> None:
    x = Binary("ho_x")
    y = Binary("ho_y")
    z = Binary("ho_z")

    try:
        (x * y * z).to_qubo()
    except ValueError as exc:
        assert "degree <= 2" in str(exc)
    else:
        raise AssertionError("Expected high-order expression to be rejected.")


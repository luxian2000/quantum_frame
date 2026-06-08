from aicir.optimization.qubo import (
    Binary,
    Model,
    ModelContext,
    QuboBuilder,
    Sum,
    brute_force_builder,
    brute_force_model,
    brute_force_qubo,
    cardinality,
    decode_best_solutions,
    integer_equality,
    linear_inequality,
    permutation,
    qubo_energy,
)


def test_qubo_energy_evaluates_offset_linear_and_quadratic_terms() -> None:
    qubo = {(0, 0): -1.0, (1, 1): 2.0, (0, 1): 4.0}

    assert qubo_energy(qubo, [0, 0], offset=3.0) == 3.0
    assert qubo_energy(qubo, [1, 0], offset=3.0) == 2.0
    assert qubo_energy(qubo, [1, 1], offset=3.0) == 8.0


def test_brute_force_qubo_finds_all_minimizers() -> None:
    result = brute_force_qubo({(0, 1): 1.0}, variable_count=2)

    assert result.best_energy == 0.0
    assert set(result.best_assignments) == {(0, 0), (0, 1), (1, 0)}


def test_brute_force_model_uses_model_registry_size() -> None:
    ctx = ModelContext()
    ctx.binary("bf_unused")
    x = ctx.binary("bf_model_x")
    model = Model(-1.0 * x)

    result = brute_force_model(model)

    assert result.best_energy == -1.0
    assert set(result.best_assignments) == {(0, 1), (1, 1)}


def test_brute_force_builder_uses_builder_registry_size() -> None:
    ctx = ModelContext()
    ctx.binary("bf_builder_unused")
    x = ctx.binary("bf_builder_x")
    builder = ctx.qubo_builder()
    builder.add_linear(_var_id(x), -1.0)

    result = brute_force_builder(builder)

    assert result.best_energy == -1.0
    assert set(result.best_assignments) == {(0, 1), (1, 1)}


def test_cardinality_fast_path_and_generic_have_same_energy_landscape() -> None:
    ctx = ModelContext()
    x = [ctx.binary(f"bf_card_x{i}") for i in range(4)]

    fast_model = Model(0 * x[0])
    fast_model.add_constraint(cardinality(x, count=2, penalty=3.0))

    generic_model = Model(3.0 * (Sum(x) - 2) ** 2)

    fast_qubo, fast_offset = fast_model.to_qubo_indices()
    generic_qubo, generic_offset = generic_model.to_qubo_indices()
    fast = brute_force_qubo(fast_qubo, fast_offset, variable_count=4)
    generic = brute_force_qubo(generic_qubo, generic_offset, variable_count=4)

    assert fast.best_energy == generic.best_energy
    assert set(fast.best_assignments) == set(generic.best_assignments)
    assert fast.energies == generic.energies


def test_linear_inequality_fast_path_and_generic_have_same_energy_landscape() -> None:
    ctx = ModelContext()
    x = [ctx.binary(f"bf_li_x{i}") for i in range(2)]
    constraint, slack = linear_inequality([(2.0, x[0]), (3.0, x[1])], upper_bound=4, penalty=5.0)

    fast_model = Model(0 * x[0])
    fast_model.add_constraint(constraint)
    generic_model = Model(5.0 * constraint.expression)

    variable_count = len(ctx.registry.names())
    fast_qubo, fast_offset = fast_model.to_qubo_indices()
    generic_qubo, generic_offset = generic_model.to_qubo_indices()
    fast = brute_force_qubo(fast_qubo, fast_offset, variable_count=variable_count)
    generic = brute_force_qubo(generic_qubo, generic_offset, variable_count=variable_count)

    assert [bit.variables()[0] for bit in slack] == ["slack[0]", "slack[1]", "slack[2]"]
    assert fast.best_energy == generic.best_energy
    assert set(fast.best_assignments) == set(generic.best_assignments)
    assert fast.energies == generic.energies


def test_permutation_model_and_builder_have_same_energy_landscape() -> None:
    ctx = ModelContext()
    x = ctx.binary_array("bf_perm_x", (3, 3))

    model = Model(0 * x[0][0])
    model.add_constraints(permutation(x, penalty=4.0))

    builder = QuboBuilder(registry=ctx.registry)
    ids = [[_var_id(x[row][col]) for col in range(3)] for row in range(3)]
    for row in range(3):
        builder.add_cardinality_penalty(ids[row], count=1, penalty=4.0)
    for col in range(3):
        builder.add_cardinality_penalty([ids[row][col] for row in range(3)], count=1, penalty=4.0)

    model_qubo, model_offset = model.to_qubo_indices()
    builder_qubo, builder_offset = builder.to_qubo_indices()
    model_result = brute_force_qubo(model_qubo, model_offset, variable_count=9)
    builder_result = brute_force_qubo(builder_qubo, builder_offset, variable_count=9)

    assert model_result.best_energy == builder_result.best_energy == 0.0
    assert len(model_result.best_assignments) == 6
    assert set(model_result.best_assignments) == set(builder_result.best_assignments)
    assert model_result.energies == builder_result.energies


def test_decode_best_solutions_hides_auxiliary_slack_by_default() -> None:
    ctx = ModelContext()
    x = ctx.binary("bf_decode_x")
    constraint, _ = linear_inequality([(2.0, x)], upper_bound=3, slack_prefix="bf_decode_s", penalty=10.0)
    model = Model(-1.0 * x)
    model.add_constraint(constraint)

    qubo, offset = model.to_qubo_indices()
    result = brute_force_qubo(qubo, offset, variable_count=len(ctx.registry.names()))
    decoded = decode_best_solutions(result, ctx.registry)

    assert result.best_energy == -1.0
    assert [solution.decisions() for solution in decoded] == [{"bf_decode_x": 1}]
    assert all(solution.auxiliary == {} for solution in decoded)

    decoded_with_aux = decode_best_solutions(result, ctx.registry, include_auxiliary=True)
    assert decoded_with_aux[0].auxiliary == {"bf_decode_s[0]": 1, "bf_decode_s[1]": 0}


def test_decode_best_solutions_decodes_integer_variables() -> None:
    ctx = ModelContext()
    value = ctx.integer("bf_decode_z", upper_bound=3)
    model = Model(ctx.zero())
    model.add_constraint(integer_equality(value, target=2, penalty=5.0))

    qubo, offset = model.to_qubo_indices()
    result = brute_force_qubo(qubo, offset, variable_count=len(ctx.registry.names()))
    decoded = decode_best_solutions(result, ctx.registry, integers=[value])

    assert result.best_energy == 0.0
    assert [solution.decisions() for solution in decoded] == [{"bf_decode_z": 2}]


def test_brute_force_rejects_large_enumerations_by_default() -> None:
    try:
        brute_force_qubo({}, variable_count=21)
    except ValueError as exc:
        assert "max_variables" in str(exc)
    else:
        raise AssertionError("Expected brute force enumeration guard to reject large problems.")


def _var_id(variable):
    ((term, coeff),) = variable.terms.items()
    assert coeff == 1.0
    return term[0]


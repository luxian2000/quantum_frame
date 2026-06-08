from aicir.optimization.qubo import (
    ModelContext,
    brute_force_builder,
    brute_force_model,
    brute_force_qubo,
    decode_best_solutions,
    graph_coloring_model,
    graph_coloring_qubo_builder,
    knapsack_model,
    knapsack_qubo_builder,
    tsp_model,
    tsp_qubo_builder,
)


def test_tsp_model_builds_qubo() -> None:
    distances = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ]

    qubo, offset = tsp_model(distances, penalty=5.0).to_qubo()

    assert offset == 30.0
    assert ("x[0][0]", "x[0][0]") in qubo
    assert ("x[0][0]", "x[1][1]") in qubo


def test_tsp_builder_matches_model() -> None:
    distances = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ]

    model_qubo = tsp_model(distances, penalty=5.0, prefix="tb").to_qubo()
    builder_qubo = tsp_qubo_builder(distances, penalty=5.0, prefix="tb").to_qubo()

    assert builder_qubo == model_qubo


def test_tsp_model_and_builder_have_same_energy_landscape() -> None:
    distances = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ]
    registry = ModelContext().registry

    model = tsp_model(distances, penalty=10.0, prefix="tsp_bf", registry=registry)
    builder = tsp_qubo_builder(distances, penalty=10.0, prefix="tsp_bf", registry=registry)

    model_result = brute_force_model(model)
    builder_result = brute_force_builder(builder)

    assert model_result.best_energy == builder_result.best_energy == 6.0
    assert set(model_result.best_assignments) == set(builder_result.best_assignments)
    assert model_result.energies == builder_result.energies


def test_graph_coloring_model_builds_conflict_terms() -> None:
    model = graph_coloring_model(node_count=2, edges=[(0, 1)], color_count=3, penalty=7.0)

    qubo, offset = model.to_qubo()

    assert offset == 14.0
    assert qubo[("x[0][0]", "x[1][0]")] == 7.0
    assert qubo[("x[0][1]", "x[1][1]")] == 7.0
    assert qubo[("x[0][2]", "x[1][2]")] == 7.0


def test_graph_coloring_builder_matches_model() -> None:
    edges = [(0, 1), (1, 2)]

    model_qubo = graph_coloring_model(3, edges, color_count=2, penalty=4.0, prefix="gb").to_qubo()
    builder_qubo = graph_coloring_qubo_builder(3, edges, color_count=2, penalty=4.0, prefix="gb").to_qubo()

    assert builder_qubo == model_qubo


def test_graph_coloring_model_and_builder_have_same_energy_landscape() -> None:
    edges = [(0, 1), (1, 2)]
    registry = ModelContext().registry

    model = graph_coloring_model(3, edges, color_count=2, penalty=6.0, prefix="gc_bf", registry=registry)
    builder = graph_coloring_qubo_builder(3, edges, color_count=2, penalty=6.0, prefix="gc_bf", registry=registry)

    model_result = brute_force_model(model)
    builder_result = brute_force_builder(builder)

    assert model_result.best_energy == builder_result.best_energy == 0.0
    assert len(model_result.best_assignments) == 2
    assert set(model_result.best_assignments) == set(builder_result.best_assignments)
    assert model_result.energies == builder_result.energies


def test_knapsack_model_builds_capacity_terms() -> None:
    model = knapsack_model(values=[4.0, 3.0], weights=[2, 1], capacity=2, penalty=5.0)

    qubo, offset = model.to_qubo()

    assert offset == 20.0
    assert ("x[0]", "x[0]") in qubo
    assert ("s[0]", "s[0]") in qubo
    assert ("x[0]", "s[0]") in qubo or ("s[0]", "x[0]") in qubo


def test_knapsack_builder_matches_model() -> None:
    model_qubo = knapsack_model(
        values=[4.0, 3.0, 2.0],
        weights=[2, 1, 3],
        capacity=3,
        penalty=5.0,
        item_prefix="kb",
        slack_prefix="ks",
    ).to_qubo()
    builder_qubo = knapsack_qubo_builder(
        values=[4.0, 3.0, 2.0],
        weights=[2, 1, 3],
        capacity=3,
        penalty=5.0,
        item_prefix="kb",
        slack_prefix="ks",
    ).to_qubo()

    assert builder_qubo == model_qubo


def test_knapsack_model_and_builder_have_same_energy_landscape_and_decoding() -> None:
    registry = ModelContext().registry
    model = knapsack_model(
        values=[4.0, 3.0],
        weights=[2, 1],
        capacity=2,
        penalty=10.0,
        item_prefix="ks_bf_x",
        slack_prefix="ks_bf_s",
        registry=registry,
    )
    builder = knapsack_qubo_builder(
        values=[4.0, 3.0],
        weights=[2, 1],
        capacity=2,
        penalty=10.0,
        item_prefix="ks_bf_x",
        slack_prefix="ks_bf_s",
        registry=registry,
    )
    model_qubo, model_offset = model.to_qubo_indices()
    builder_qubo, builder_offset = builder.to_qubo_indices()

    model_result = brute_force_qubo(model_qubo, model_offset, variable_count=len(registry.names()))
    builder_result = brute_force_qubo(builder_qubo, builder_offset, variable_count=len(registry.names()))
    decoded = decode_best_solutions(model_result, registry)

    assert model_result.best_energy == builder_result.best_energy == -4.0
    assert set(model_result.best_assignments) == set(builder_result.best_assignments)
    assert model_result.energies == builder_result.energies
    assert [solution.decisions() for solution in decoded] == [{"ks_bf_x[0]": 1, "ks_bf_x[1]": 0}]
    assert decoded[0].auxiliary == {}


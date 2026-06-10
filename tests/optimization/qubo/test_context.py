from aicir.optimization.qubo import Binary, ModelContext, VariableRegistry, binary_array, knapsack_qubo_builder


def test_model_context_isolates_variable_registries() -> None:
    left = ModelContext()
    right = ModelContext()

    left_x = left.binary("x")
    right_x = right.binary("x")

    assert left_x.registry is not right_x.registry
    assert left.registry.names() == ["x"]
    assert right.registry.names() == ["x"]


def test_binary_accepts_explicit_registry() -> None:
    registry = VariableRegistry()

    x = Binary("x", registry=registry)
    y = Binary("y", registry=registry)

    assert x.registry is registry
    assert y.registry is registry
    assert registry.names() == ["x", "y"]
    assert registry.metadata(0).role == "decision"


def test_binary_array_accepts_explicit_registry() -> None:
    registry = VariableRegistry()

    x = binary_array("x", (2, 2), registry=registry)

    assert x[0][0].registry is registry
    assert registry.names() == ["x[0][0]", "x[0][1]", "x[1][0]", "x[1][1]"]


def test_problem_builder_accepts_explicit_registry() -> None:
    registry = VariableRegistry()

    builder = knapsack_qubo_builder([1.0, 2.0], [1, 2], capacity=2, registry=registry)

    assert builder.registry is registry
    assert registry.names() == ["x[0]", "x[1]", "s[0]", "s[1]"]


def test_model_context_zero_uses_context_registry() -> None:
    ctx = ModelContext()

    zero = ctx.zero()

    assert zero.registry is ctx.registry
    assert zero.terms == {}


def test_auxiliary_variable_metadata() -> None:
    ctx = ModelContext()

    aux = ctx.auxiliary_binary("s[0]", source="capacity")

    assert aux.variables() == ["s[0]"]
    assert ctx.registry.auxiliary_names() == ["s[0]"]
    assert ctx.registry.metadata(0).source == "capacity"


def test_auxiliary_integer_metadata() -> None:
    ctx = ModelContext()

    value = ctx.auxiliary_integer("slack", upper_bound=3, source="capacity")

    assert value.bit_names() == ["slack[0]", "slack[1]"]
    assert ctx.registry.auxiliary_names() == ["slack[0]", "slack[1]"]
    assert ctx.registry.metadata(0).role == "auxiliary"


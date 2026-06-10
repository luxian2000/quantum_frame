from aicir.optimization.qubo import Binary, Model, QuboBuilder, qubo_to_ising_indices


def test_qubo_to_ising_indices() -> None:
    ising = qubo_to_ising_indices({(0, 0): 2.0, (0, 1): 4.0}, offset=0.0, variable_names=["x", "y"])

    assert ising.h == {0: 2.0, 1: 1.0}
    assert ising.J == {(0, 1): 1.0}
    assert ising.offset == 2.0
    assert ising.named() == {"h": {"x": 2.0, "y": 1.0}, "J": {("x", "y"): 1.0}, "offset": 2.0}


def test_builder_qaoa_terms() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("qaoa_x")
    y = builder.registry.get_or_create("qaoa_y")
    builder.add_linear(x, 2.0)
    builder.add_quadratic(x, y, 4.0)

    terms, offset, names = builder.to_qaoa_terms()

    assert terms[0].qubits == (0,)
    assert terms[0].pauli == "Z"
    assert terms[-1].qubits == (0, 1)
    assert terms[-1].pauli == "ZZ"
    assert offset == 2.0
    assert names == ["qaoa_x", "qaoa_y"]


def test_model_ising_indices_matches_named_ising() -> None:
    x = Binary("mi_x")
    y = Binary("mi_y")
    model = Model(2.0 * x + 4.0 * x * y)

    ising = model.to_ising_indices()

    assert ising.named() == model.to_ising()


def test_backend_exports_include_variable_metadata() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("x")
    s = builder.registry.get_or_create("s[0]", role="auxiliary", source="test_slack")
    builder.add_quadratic(x, s, 2.0)

    matrix = builder.to_sparse_matrix()
    ising = builder.to_ising_indices()

    assert [metadata.role for metadata in matrix.variable_metadata] == ["decision", "auxiliary"]
    assert [metadata.source for metadata in ising.variable_metadata] == [None, "test_slack"]


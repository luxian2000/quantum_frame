from aicir.optimization.qubo import Binary, Model


def test_sparse_matrix_compact_output() -> None:
    x = Binary("mat_x")
    y = Binary("mat_y")
    model = Model(2 * x + 6 * x * y)

    matrix = model.to_sparse_matrix()

    assert matrix.shape == (2, 2)
    assert matrix.variable_names == ["mat_x", "mat_y"]
    assert matrix.to_dense() == [[2.0, 6.0], [0.0, 0.0]]
    assert matrix.offset == 0.0


def test_sparse_matrix_symmetric_output_splits_quadratic_terms() -> None:
    x = Binary("sym_x")
    y = Binary("sym_y")
    model = Model(6 * x * y)

    matrix = model.to_sparse_matrix(symmetric=True)

    assert matrix.to_dense() == [[0.0, 3.0], [3.0, 0.0]]


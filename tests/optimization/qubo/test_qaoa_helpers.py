import numpy as np

from aicir.core.operators import Hamiltonian
from aicir.optimization.qubo import (
    Binary,
    Model,
    QuboBuilder,
    builder_to_basic_qaoa,
    builder_to_qaoa_matrix,
    bitstring_to_qubo_assignment,
    model_to_basic_qaoa,
    model_to_qaoa_matrix,
    most_likely_bitstring,
    most_likely_qaoa_assignment,
    run_model_qaoa,
    run_qubo_qaoa,
)


def test_model_to_qaoa_matrix_returns_basic_qaoa_input() -> None:
    x = Binary("qaoa_helper_x")
    y = Binary("qaoa_helper_y")
    model = Model(2.0 * x + 4.0 * x * y)

    matrix, n_qubits = model_to_qaoa_matrix(model)

    assert n_qubits == 2
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix).real, [6.0, 2.0, 0.0, 0.0])


def test_builder_to_qaoa_matrix_can_drop_offset() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("x")
    builder.add_linear(x, 2.0)

    matrix, n_qubits = builder_to_qaoa_matrix(builder, include_offset=False)

    assert n_qubits == 1
    assert np.allclose(np.diag(matrix).real, [1.0, -1.0])


def test_model_to_basic_qaoa_builds_solver() -> None:
    x = Binary("qaoa_solver_x")
    model = Model(2.0 * x)

    solver = model_to_basic_qaoa(model, p=1, seed=123)

    assert solver.n_qubits == 1
    assert solver.n_params == 2
    assert isinstance(solver.problem_hamiltonian, Hamiltonian)
    assert solver.build_circuit(np.array([0.1, 0.2])).n_qubits == 1


def test_builder_to_basic_qaoa_builds_solver() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("x")
    builder.add_linear(x, 2.0)

    solver = builder_to_basic_qaoa(builder, p=1, seed=123)

    assert solver.n_qubits == 1
    assert solver.n_params == 2
    assert isinstance(solver.problem_hamiltonian, Hamiltonian)


def test_run_model_qaoa_runs_short_optimization() -> None:
    x = Binary("qaoa_run_x")
    model = Model(2.0 * x)

    result = run_model_qaoa(model, p=1, max_iters=2, lr=0.01, seed=123)

    assert len(result.energy_history) == 2
    assert result.statevector.shape == (2,)


def test_run_qubo_qaoa_alias() -> None:
    x = Binary("qaoa_alias_x")
    model = Model(2.0 * x)

    result = run_qubo_qaoa(model, p=1, max_iters=1, lr=0.01, seed=123)

    assert len(result.energy_history) == 1


def test_most_likely_bitstring() -> None:
    statevector = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.complex128)

    bitstring, probability = most_likely_bitstring(statevector)

    assert bitstring == "10"
    assert np.isclose(probability, 0.81)


def test_bitstring_to_qubo_assignment_uses_ising_convention() -> None:
    assignment = bitstring_to_qubo_assignment("01", ["x0", "x1"])

    assert assignment == {"x0": 1, "x1": 0}


def test_most_likely_qaoa_assignment() -> None:
    statevector = np.array([0.1, 0.8, 0.2, 0.3], dtype=np.complex128)

    decoded = most_likely_qaoa_assignment(statevector, ["x0", "x1"])

    assert decoded.bitstring == "01"
    assert np.isclose(decoded.probability, 0.64)
    assert decoded.assignment == {"x0": 1, "x1": 0}

import numpy as np

from aicir import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.vqc.QAOA import BasicQAOA


def _dense_reference(qaoa, params, backend):
    result = qaoa.measure(params, backend=backend, return_state=True)
    operator = qaoa.problem_hamiltonian.to_matrix(backend)
    return float(result.final_state.expectation(operator))


def test_sparse_cost_expectation_matches_dense_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=7)
    params = qaoa.initial_params()

    result = qaoa.measure(params, backend=backend, return_state=True)
    sparse = qaoa._sparse_cost_expectation(result.final_state.data, backend)
    dense = _dense_reference(qaoa, params, backend)

    assert np.isclose(sparse, dense, atol=1e-6)


def test_sparse_cost_expectation_matches_dense_non_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("X", [0], 0.5), ("YZ", [0, 1], -0.25), ("ZZ", 1.0)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, seed=3)
    params = qaoa.initial_params()

    result = qaoa.measure(params, backend=backend, return_state=True)
    sparse = qaoa._sparse_cost_expectation(result.final_state.data, backend)
    dense = _dense_reference(qaoa, params, backend)

    assert np.isclose(sparse, dense, atol=1e-6)


def test_energy_exact_does_not_build_dense_matrix(monkeypatch):
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("XIX", 0.3)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, seed=5)
    params = qaoa.initial_params()

    def forbidden_to_matrix(*args, **kwargs):
        raise AssertionError("exact gate-level energy must not build a dense Hamiltonian matrix")

    monkeypatch.setattr(qaoa.problem_hamiltonian, "to_matrix", forbidden_to_matrix)

    energy = qaoa.energy(params, backend=backend)
    assert np.isfinite(energy)


def test_energy_exact_matches_previous_dense_values_diagonal_and_non_diagonal():
    backend = NumpyBackend()
    for terms in (
        [("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)],
        [("XIX", 0.3), ("ZZI", -1.0), ("IYY", 0.4)],
    ):
        hamiltonian = Hamiltonian(n_qubits=3, terms=terms)
        qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=11)
        params = qaoa.initial_params()

        result = qaoa.measure(params, backend=backend, return_state=True)
        dense = float(result.final_state.expectation(qaoa.problem_hamiltonian.to_matrix(backend)))
        actual = qaoa.energy(params, backend=backend)

        assert np.isclose(actual, dense, atol=1e-6)

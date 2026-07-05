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

import numpy as np

from aicir import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.vqc.QAOA import BasicQAOA, Circuit, _append_trotter_slice, hadamard, rx


def _unitary(circuit, backend):
    return np.asarray(circuit.unitary(backend=backend), dtype=np.complex128)


def _legacy_build_circuit_reference(qaoa, params, backend):
    gammas, betas = qaoa.split_params(params)
    circuit = Circuit(n_qubits=qaoa.n_qubits, backend=backend)
    for qubit in range(qaoa.n_qubits):
        circuit.append(hadamard(qubit))

    for layer in range(qaoa.p):
        gamma = float(gammas[layer])
        gamma_step = gamma / qaoa.trotter_steps
        for _ in range(qaoa.trotter_steps):
            _append_trotter_slice(circuit, qaoa._cost_terms, gamma_step, qaoa.trotter_order)
        beta = float(betas[layer])
        for qubit in range(qaoa.n_qubits):
            circuit.append(rx(2.0 * beta, qubit))
    return circuit


def test_build_circuit_tape_matches_legacy_reference_unitary():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(
        n_qubits=3,
        terms=[("XYZ", 0.3), ("YZX", -0.2), ("ZZI", 0.5), ("IXY", -0.4)],
    )
    for order in (1, 2):
        qaoa = BasicQAOA(
            problem_hamiltonian=hamiltonian, p=2, trotter_steps=2, trotter_order=order, seed=13
        )
        params = qaoa.initial_params()
        actual = qaoa.build_circuit(params, backend=backend)
        legacy = _legacy_build_circuit_reference(qaoa, params, backend)
        np.testing.assert_allclose(_unitary(actual, backend), _unitary(legacy, backend), atol=1e-6)


def test_qaoa_tape_owner_indices_cover_all_parameters():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("X", [0], 0.4)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=1)
    params = qaoa.initial_params()
    owners = {rec.owner for rec in qaoa._qaoa_tape(params) if rec.owner is not None}
    assert owners == set(range(qaoa.n_params))


def _fd_reference(qaoa, params, backend):
    return qaoa.finite_difference_gradient(params, eps=1e-5, backend=backend)


def test_analytic_gradient_matches_fd_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=17)
    params = qaoa.initial_params()

    analytic = qaoa.analytic_gradient(params, backend=backend)
    fd = _fd_reference(qaoa, params, backend)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_analytic_gradient_matches_fd_non_diagonal_and_trotterized():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("XX", 0.6), ("YZ", [0, 1], -0.3), ("ZZ", 1.0)])
    for order in (1, 2):
        qaoa = BasicQAOA(
            problem_hamiltonian=hamiltonian, p=2, trotter_steps=3, trotter_order=order, seed=19
        )
        params = qaoa.initial_params()

        analytic = qaoa.analytic_gradient(params, backend=backend)
        fd = _fd_reference(qaoa, params, backend)

        np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_run_grad_method_analytic_reaches_same_optimum_as_fd():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", -1.0)])
    init = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).initial_params()

    fd_run = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).run(
        max_iters=40, lr=0.1, init_params=init, backend=backend
    )
    an_run = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).run(
        max_iters=40, lr=0.1, init_params=init, backend=backend, grad_method="analytic"
    )

    assert np.isclose(fd_run.energy, an_run.energy, atol=1e-3)

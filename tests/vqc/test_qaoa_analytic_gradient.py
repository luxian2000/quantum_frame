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

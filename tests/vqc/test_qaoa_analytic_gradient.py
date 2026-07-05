import numpy as np

from aicir import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.vqc.QAOA import BasicQAOA


def _unitary(circuit, backend):
    return np.asarray(circuit.unitary(backend=backend), dtype=np.complex128)


def test_build_circuit_tape_matches_reference_unitary():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("XIX", 0.3)])
    for order in (1, 2):
        qaoa = BasicQAOA(
            problem_hamiltonian=hamiltonian, p=2, trotter_steps=2, trotter_order=order, seed=13
        )
        params = qaoa.initial_params()
        circuit = qaoa.build_circuit(params, backend=backend)
        # 从磁带重建的线路应与 build_circuit 一致（build_circuit 本身即走磁带）
        from aicir.vqc.QAOA import _circuit_from_tape

        rebuilt = _circuit_from_tape(qaoa._qaoa_tape(params), qaoa.n_qubits, backend)
        np.testing.assert_allclose(_unitary(circuit, backend), _unitary(rebuilt, backend), atol=1e-6)


def test_qaoa_tape_owner_indices_cover_all_parameters():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("X", [0], 0.4)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=1)
    params = qaoa.initial_params()
    owners = {rec.owner for rec in qaoa._qaoa_tape(params) if rec.owner is not None}
    assert owners == set(range(qaoa.n_params))

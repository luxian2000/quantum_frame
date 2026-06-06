from demos.H2O.H2O import build_h2o_hamiltonian, exact_ground_energy
from aicir.measure import hamiltonian_pauli_terms


def test_h2o_demo_hamiltonian_builds_expected_pauli_terms():
    hamiltonian = build_h2o_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    assert hamiltonian.n_qubits == 6
    assert len(terms) == 62
    assert terms[0].pauli == "IIIIII"
    assert terms[0].coefficient == -4.5241061234101245
    assert any(term.pauli == "ZIIZII" for term in terms)
    assert any(term.pauli == "YZYYZY" for term in terms)


def test_h2o_demo_exact_ground_energy_is_finite():
    energy = exact_ground_energy(build_h2o_hamiltonian())

    assert energy < 0.0


def test_h2o_recorded_supernet_circuit_approximates_ground_energy():
    # The 6-qubit H2O search takes ~1 minute, so instead of re-running it we
    # verify the recorded circuit (demos/H2O/H2O_cir.py, produced by the
    # supernet method) reproduces an energy close to exact.
    from demos.H2O.H2O_cir import build_h2o_qas_circuit
    from aicir.channel.backends.torch_backend import TorchBackend

    hamiltonian = build_h2o_hamiltonian()
    exact = exact_ground_energy(hamiltonian)

    backend = TorchBackend(device="cpu")
    hamiltonian_matrix = hamiltonian.to_matrix(backend)
    circuit = build_h2o_qas_circuit()
    state = backend.apply_unitary(
        backend.zeros_state(circuit.n_qubits), circuit.unitary(backend=backend)
    )
    energy = float(backend.expectation_sv(state, hamiltonian_matrix).real)

    assert circuit.n_qubits == 6
    # Rayleigh-Ritz lower bound, and the supernet lands within ~5 mHa of exact.
    assert energy >= exact - 1e-6
    assert energy <= exact + 1e-2

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

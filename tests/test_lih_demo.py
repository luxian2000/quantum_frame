from demos.LiH import build_lih_hamiltonian, exact_ground_energy
from aicir.measure import hamiltonian_pauli_terms


def test_lih_demo_hamiltonian_builds_expected_pauli_terms():
    hamiltonian = build_lih_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    assert hamiltonian.n_qubits == 4
    assert len(terms) == 27
    assert terms[0].pauli == "IIII"
    assert terms[0].coefficient == -0.7059409881285760
    assert any(term.pauli == "ZIIZ" for term in terms)
    assert any(term.pauli == "XXXX" for term in terms)


def test_lih_demo_exact_ground_energy_is_finite():
    energy = exact_ground_energy(build_lih_hamiltonian())

    assert energy < 0.0

from demos.LiH.LiH import (
    build_lih_hamiltonian,
    exact_ground_energy,
    search_ground_state_qas,
)
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


def test_lih_vqa_qas_search_approaches_exact_ground_energy():
    hamiltonian = build_lih_hamiltonian()
    exact = exact_ground_energy(hamiltonian)

    # Default config is deterministic (fixed seed) and runs in ~1.5s.
    result = search_ground_state_qas(hamiltonian)
    metrics = result.final_metrics

    assert result.best_circuit.n_qubits == 4
    # Variational energy obeys the Rayleigh-Ritz lower bound and gets close.
    assert metrics["fine_tuned_energy"] >= exact - 1e-6
    assert metrics["fine_tuned_energy"] <= exact + 2e-2
    assert "baseline_vqe_energy" in metrics

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, NumpyBackend, PauliEstimator, hadamard, s_gate
from aicir.measure import (
    PauliTerm,
    allocate_group_shots,
    group_pauli_terms,
    hamiltonian_pauli_terms,
    pauli_eigenvalue_from_bits,
    pauli_expectation_from_counts,
)


def test_hamiltonian_pauli_terms_decomposes_hamiltonian():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[
        ("ZI", 0.5),
        ("XX", -1.25),
        ("II", 0.75),
    ])

    terms = hamiltonian_pauli_terms(hamiltonian)

    assert terms == (
        PauliTerm(0.5, "ZI"),
        PauliTerm(-1.25, "XX"),
        PauliTerm(0.75, "II"),
    )


def test_hamiltonian_constructor_infers_width_and_has_no_term_api():
    hamiltonian = Hamiltonian([
        ("ZI", 0.5),
        ("XX", -1.25),
    ])

    assert hamiltonian.n_qubits == 2
    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(0.5, "ZI"),
        PauliTerm(-1.25, "XX"),
    )
    assert not hasattr(hamiltonian, "term")


def test_hamiltonian_from_list_uses_pauli_first_order():
    hamiltonian = Hamiltonian.from_list([("ZI", 0.5), ("XX", -1.25)])

    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(0.5, "ZI"),
        PauliTerm(-1.25, "XX"),
    )


def test_hamiltonian_supports_local_pauli_string_with_qubit_indices():
    hamiltonian = Hamiltonian(n_qubits=4, terms=[
        ("ZZ", -1.0, [0, 3]),
        ("X", 0.25, [2]),
        ("YY", [1, 3], 2.0),
    ])

    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(-1.0, "ZIIZ"),
        PauliTerm(0.25, "IIXI"),
        PauliTerm(2.0, "IYIY"),
    )


def test_hamiltonian_local_terms_infer_width_from_largest_qubit_index():
    hamiltonian = Hamiltonian([
        ("ZZ", -1.0, [0, 3]),
        ("X", 0.25, [2]),
    ])

    assert hamiltonian.n_qubits == 4
    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(-1.0, "ZIIZ"),
        PauliTerm(0.25, "IIXI"),
    )


def test_hamiltonian_terms_default_coefficient_to_one():
    hamiltonian = Hamiltonian(n_qubits=4, terms=[
        "ZZ",
        ("X",),
        ("YY", [1, 3]),
    ])

    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(1.0, "ZZII"),
        PauliTerm(1.0, "XIII"),
        PauliTerm(1.0, "IYIY"),
    )


def test_hamiltonian_terms_default_to_leading_qubit_indices():
    hamiltonian = Hamiltonian(n_qubits=4, terms=[
        ("ZZ", -1.0),
        ("X", 0.25),
    ])

    assert hamiltonian_pauli_terms(hamiltonian) == (
        PauliTerm(-1.0, "ZZII"),
        PauliTerm(0.25, "XIII"),
    )


def test_hamiltonian_local_terms_validate_qubit_indices():
    with pytest.raises(ValueError, match="Pauli 字符串长度必须与 qubits 长度一致"):
        Hamiltonian(n_qubits=4, terms=[("ZZ", -1.0, [0])])

    with pytest.raises(ValueError, match="qubits 不能包含重复下标"):
        Hamiltonian(n_qubits=4, terms=[("ZZ", -1.0, [0, 0])])

    with pytest.raises(IndexError, match="超出范围"):
        Hamiltonian(n_qubits=4, terms=[("ZZ", -1.0, [0, 4])])


def test_hamiltonian_rejects_coefficient_first_order():
    with pytest.raises((TypeError, ValueError)):
        Hamiltonian(n_qubits=4, terms=[(-1.0, "ZZ", [0, 3])])


def test_group_pauli_terms_qubit_wise_commuting_and_shot_allocation():
    groups = group_pauli_terms(
        [
            PauliTerm(1.0, "ZI"),
            PauliTerm(2.0, "ZZ"),
            PauliTerm(3.0, "XX"),
        ]
    )

    assert [(group.basis, group.terms) for group in groups] == [
        ("ZZ", (PauliTerm(1.0, "ZI"), PauliTerm(2.0, "ZZ"))),
        ("XX", (PauliTerm(3.0, "XX"),)),
    ]
    assert allocate_group_shots(groups, 11) == (6, 5)
    assert allocate_group_shots(groups, 10, strategy="coefficient") == (5, 5)


def test_pauli_expectation_from_counts_uses_msb_qubit_order():
    counts = {"|00>": 2, "|01>": 1, "|10>": 1}

    assert pauli_eigenvalue_from_bits("ZI", "10") == -1
    expectation, variance = pauli_expectation_from_counts("ZI", counts)

    assert expectation == pytest.approx(0.5)
    assert variance == pytest.approx((1.0 - 0.25) / 4)


def test_pauli_estimator_estimates_z_terms_with_group_covariance():
    circuit = Circuit(n_qubits=2)
    hamiltonian = Hamiltonian(n_qubits=2, terms=[
        ("ZI", 1.0),
        ("IZ", 2.0),
        ("ZZ", 3.0),
    ])

    result = PauliEstimator(NumpyBackend(), shots=64).estimate(circuit, hamiltonian)

    assert result.energy == pytest.approx(6.0)
    assert result.variance == pytest.approx(0.0)
    assert result.std_error == pytest.approx(0.0)
    assert result.shots == 64
    assert len(result.groups) == 1
    assert result.groups[0].basis == "ZZ"
    assert result.groups[0].counts == {"|00>": 64}


def test_pauli_estimator_applies_x_basis_change():
    circuit = Circuit(hadamard(0), n_qubits=1)
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("X", 1.0)])

    result = PauliEstimator(NumpyBackend(), shots=32).estimate(circuit, hamiltonian)

    assert result.energy == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.0)
    assert result.groups[0].basis == "X"
    assert result.groups[0].counts == {"|0>": 32}


def test_pauli_estimator_applies_y_basis_change():
    circuit = Circuit(hadamard(0), s_gate(0), n_qubits=1)
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Y", 1.0)])

    result = PauliEstimator(NumpyBackend(), shots=32).estimate(circuit, hamiltonian)

    assert result.energy == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.0)
    assert result.groups[0].basis == "Y"
    assert result.groups[0].counts == {"|0>": 32}


def test_pauli_estimator_handles_identity_terms_without_shots():
    circuit = Circuit(n_qubits=1)
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("I", 2.5), ("Z", 1.0)])

    result = PauliEstimator(NumpyBackend(), shots=16).estimate(circuit, hamiltonian)

    assert result.energy == pytest.approx(3.5)
    assert result.shots == 16
    assert [(group.basis, group.shots) for group in result.groups] == [("I", 0), ("Z", 16)]
    assert result.term_results[0].shots == 0


def test_pauli_estimator_rejects_dense_matrix_hamiltonian():
    circuit = Circuit(n_qubits=1)
    estimator = PauliEstimator(NumpyBackend(), shots=16)

    with pytest.raises(TypeError):
        estimator.estimate(circuit, np.diag([1.0, -1.0]))

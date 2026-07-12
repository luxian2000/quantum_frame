"""aicir.qas.core.problem：QAS 问题输入归一化。"""

import numpy as np
import pytest

from aicir.chemistry import get_molecule
from aicir.core.operators import Hamiltonian
from aicir.core.state import State
from aicir.qas.core.problem import (
    QASProblem,
    normalize_problem,
    normalize_terms,
    terms_coeff_first,
    terms_label_first,
)

# ──────────────────────────────────────────────────────────────────────────────
# normalize_terms / 顺序转换
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_terms_accepts_label_first_order():
    assert normalize_terms([("ZZ", -1.0), ("XI", 0.5)]) == [("ZZ", -1.0), ("XI", 0.5)]


def test_normalize_terms_accepts_coeff_first_order():
    assert normalize_terms([(-1.0, "ZZ"), (0.5, "XI")]) == [("ZZ", -1.0), ("XI", 0.5)]


def test_normalize_terms_accepts_mixed_orders_per_item():
    assert normalize_terms([("ZZ", -1.0), (0.5, "XI")]) == [("ZZ", -1.0), ("XI", 0.5)]


def test_normalize_terms_rejects_ambiguous_both_strings():
    with pytest.raises(ValueError):
        normalize_terms([("ZZ", "XI")])


def test_normalize_terms_rejects_ambiguous_both_numbers():
    with pytest.raises(ValueError):
        normalize_terms([(1.0, 2.0)])


def test_normalize_terms_rejects_wrong_arity():
    with pytest.raises(ValueError):
        normalize_terms([("ZZ", -1.0, 3)])


def test_terms_label_first_and_coeff_first_round_trip():
    coeff_first = [(-1.0, "ZZ"), (0.5, "XI")]
    label_first = terms_label_first(coeff_first)
    assert label_first == [("ZZ", -1.0), ("XI", 0.5)]
    assert terms_coeff_first(label_first) == coeff_first


# ──────────────────────────────────────────────────────────────────────────────
# normalize_problem
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_problem_hamiltonian_instance():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.5)])

    problem = normalize_problem(hamiltonian)

    assert isinstance(problem, QASProblem)
    assert problem.kind == "hamiltonian"
    assert problem.hamiltonian is hamiltonian
    assert problem.n_qubits == 2


def test_normalize_problem_terms_label_first():
    problem = normalize_problem([("ZZ", -1.0), ("XI", 0.5)])

    assert problem.kind == "hamiltonian"
    assert problem.n_qubits == 2
    assert problem.hamiltonian is not None
    assert len(problem.hamiltonian.terms) == 2


def test_normalize_problem_terms_coeff_first():
    problem = normalize_problem([(-1.0, "ZZ"), (0.5, "XI")])

    assert problem.kind == "hamiltonian"
    assert problem.n_qubits == 2


def test_normalize_problem_matrix_default_is_hamiltonian():
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    problem = normalize_problem(matrix)

    assert problem.kind == "hamiltonian"
    assert problem.matrix is matrix
    assert problem.n_qubits == 1


def test_normalize_problem_matrix_trace_one_psd_is_density_matrix():
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

    problem = normalize_problem(rho)

    assert problem.kind == "density_matrix"
    assert problem.matrix is rho


def test_normalize_problem_matrix_explicit_kind_overrides_heuristic():
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)

    problem = normalize_problem(rho, kind="hamiltonian")

    assert problem.kind == "hamiltonian"


def test_normalize_problem_statevector_column():
    vector = np.array([[0.0], [1.0]], dtype=np.complex64)

    problem = normalize_problem(vector)

    assert problem.kind == "state"
    assert isinstance(problem.state, State)
    assert problem.n_qubits == 1


def test_normalize_problem_state_instance():
    state = State.zero_state(2)

    problem = normalize_problem(state)

    assert problem.kind == "state"
    assert problem.state is state
    assert problem.n_qubits == 2


def test_normalize_problem_chemistry_molecule_hamiltonian_via_duck_typing():
    h2 = get_molecule("h2")

    problem = normalize_problem(h2)

    assert problem.kind == "hamiltonian"
    assert problem.n_qubits == h2.n_qubits
    assert problem.hamiltonian is not None
    assert problem.metadata.get("source_type") == "MoleculeHamiltonian"


def test_normalize_problem_rejects_unrecognized_input():
    with pytest.raises(ValueError):
        normalize_problem(object())


def test_normalize_problem_n_qubits_mismatch_raises():
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    with pytest.raises(ValueError):
        normalize_problem(matrix, n_qubits=5)

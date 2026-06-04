import numpy as np
import pytest

from aicir.channel.operators import Hamiltonian
from aicir.chemistry import (
    MOLECULES,
    available_molecules,
    get_molecule,
    molecule_hamiltonian,
    molecule_matrix,
)


def test_available_molecule_presets_are_limited_to_canonical_names():
    names = available_molecules()

    assert len(names) <= 10
    assert names == ("h2", "h2_jw", "h2_tapered")
    assert get_molecule("h2").name == "h2"


def test_molecule_hamiltonian_returns_fresh_hamiltonian():
    h1 = molecule_hamiltonian("h2")
    h2 = molecule_hamiltonian("h2")

    assert isinstance(h1, Hamiltonian)
    assert h1 is not h2
    assert h1.n_qubits == 2


def test_molecule_matrices_are_hermitian_with_expected_shapes():
    expected_shapes = {
        "h2": (4, 4),
        "h2_jw": (16, 16),
        "h2_tapered": (2, 2),
    }

    for name in available_molecules():
        matrix = molecule_matrix(name)
        assert matrix.shape == expected_shapes[name]
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-6)


def test_h2_default_terms_are_the_compact_parity_hamiltonian():
    preset = get_molecule("h2")

    assert preset.formula == "H2"
    assert preset.n_qubits == 2
    assert preset.terms == (
        (-1.05237325, "II"),
        (0.39793742, "IZ"),
        (-0.39793742, "ZI"),
        (-0.01128010, "ZZ"),
        (0.18093120, "XX"),
    )


def test_unknown_molecule_and_old_names_are_not_silently_created():
    for name in ("lih", "h2-jw", "h2_sto3g_parity_2q"):
        with pytest.raises(KeyError):
            get_molecule(name)

    assert "lih" not in MOLECULES
    assert "h2_sto3g_parity_2q" not in MOLECULES

import numpy as np
import pytest

from aicir.core.operators import Hamiltonian
from aicir.chemistry import (
    MOLECULES,
    MoleculeHamiltonian,
    available_molecules,
    get_molecule,
    molecule_hamiltonian,
    molecule_matrix,
    register_molecule,
)


def test_available_molecule_presets_are_limited_to_canonical_names():
    names = available_molecules()

    assert len(names) <= 12  # 保持"仅少量已验证 preset"的纪律
    assert names == ("beh2", "h2", "h2_jw", "h2_tapered", "h2o", "lih", "n2", "nh3")
    assert get_molecule("h2").name == "h2"


def test_molecule_hamiltonian_returns_fresh_hamiltonian():
    h1 = molecule_hamiltonian("h2")
    h2 = molecule_hamiltonian("h2")

    assert isinstance(h1, Hamiltonian)
    assert h1 is not h2
    assert h1.n_qubits == 2


def test_molecule_matrices_are_hermitian_with_expected_shapes():
    for name in available_molecules():
        preset = get_molecule(name)
        if preset.n_qubits > 6:  # 12–16 qubit 分子 dense 构造过慢/过大，跳过（见结构守卫）
            continue
        matrix = molecule_matrix(name)
        dim = 1 << preset.n_qubits
        assert matrix.shape == (dim, dim)
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


# 已验证的基态能量（dense-matrix 最小本征值）——preset 系数正确性的守卫。
# 参考值取自各分子 demo 的 exact_ground_energy / result.md。仅覆盖 dense 对角化
# 快速可行的小分子（<=6 qubit）；nh3/n2/beh2 项数多、维度大，dense 构造过慢，走结构守卫。
_GROUND_ENERGIES = {
    "h2": -1.8572750302,
    "lih": -1.0593119860,
    "h2o": -6.1596636772,
}

# dense 构造过慢/过大的 preset（12–16 qubit）：只做结构一致性检查；系数由上游
# PySCF/Qiskit Nature 保证（同各自 demo），本仓不再重复 dense 对角化验证。
_STRUCTURAL_ONLY = ("nh3", "n2", "beh2")


@pytest.mark.parametrize("name,expected", sorted(_GROUND_ENERGIES.items()))
def test_preset_ground_energy_matches_reference(name, expected):
    energy = float(np.linalg.eigvalsh(molecule_matrix(name))[0])
    assert energy == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", _STRUCTURAL_ONLY)
def test_large_preset_builds_with_consistent_structure(name):
    preset = get_molecule(name)
    hamiltonian = preset.to_hamiltonian()  # 不做 dense 对角化（2^14/2^16 过大）
    assert hamiltonian.n_qubits == preset.n_qubits
    assert len(preset.terms) > 0
    assert all(len(pauli) == preset.n_qubits for _, pauli in preset.terms)


def test_metadata_is_present_for_every_preset():
    for name in available_molecules():
        preset = get_molecule(name)
        assert isinstance(preset, MoleculeHamiltonian)
        assert preset.basis and preset.mapping and preset.geometry and preset.source
        # 每个 Pauli 串长度都等于 n_qubits
        assert all(len(pauli) == preset.n_qubits for _, pauli in preset.terms)


def test_register_molecule_rejects_duplicates():
    with pytest.raises(ValueError):
        register_molecule(get_molecule("h2"))


def test_unknown_molecule_and_old_names_are_not_silently_created():
    for name in ("nonesuch", "h2-jw", "h2_sto3g_parity_2q"):
        with pytest.raises(KeyError):
            get_molecule(name)

    assert "h2_sto3g_parity_2q" not in MOLECULES

import pytest

pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from aicir.chemistry import build_molecule, get_molecule

_H2_GEOMETRY = "H 0 0 0; H 0 0 0.735"  # PySCF 默认 H2 几何，与预置一致


def _term_map(mol):
    return {pauli: complex(coeff) for coeff, pauli in mol.terms}


def test_h2_jw_reproduces_preset_terms():
    preset = get_molecule("h2_jw")
    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner", name="h2_jw")
    assert built.n_qubits == preset.n_qubits
    pm, bm = _term_map(preset), _term_map(built)
    assert set(pm) == set(bm)
    for pauli in pm:
        assert abs(pm[pauli] - bm[pauli]) < 1e-4


def test_jw_populates_metadata():
    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner")
    assert built.n_electrons == 2
    assert built.hf_occupation is not None
    assert len(built.hf_occupation) == built.n_qubits
    assert sum(built.hf_occupation) == 2
    assert built.excitations is not None
    assert all(kind in ("single", "double") for kind, _ in built.excitations)


def test_parity_mapping_leaves_metadata_none():
    built = build_molecule(
        _H2_GEOMETRY, basis="sto-3g", mapping="parity", two_qubit_reduction=True
    )
    assert built.hf_occupation is None
    assert built.excitations is None
    assert built.terms  # Hamiltonian 仍可用

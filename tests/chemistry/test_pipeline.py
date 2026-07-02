import numpy as np
import pytest

pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from aicir.backends import NumpyBackend
from aicir.chemistry import build_molecule, get_molecule
from aicir.core.circuit import Circuit, pauli_x
from aicir.core.state import State

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


def test_hf_occupation_matches_terms_bit_order():
    """回归测试：hf_occupation/excitations 须与 terms 采用同一比特序。

    用 hf_occupation 摆 HF 态，再用 build_molecule 出的 Hamiltonian 算
    ⟨HF|H|HF⟩，须逼近真实电子基态 HF 能量。若 hf_occupation 的比特序与
    terms 不一致，会在错误的 qubit 上摆 X 门，得到明显偏离的能量。
    """

    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner", name="h2_jw")

    backend = NumpyBackend()
    circuit = Circuit(
        *[pauli_x(i) for i, bit in enumerate(built.hf_occupation) if bit == 1],
        n_qubits=built.n_qubits,
    )
    state = State.zero_state(built.n_qubits, backend).evolve(circuit.unitary(backend))
    psi = state.to_numpy()

    h_matrix = built.to_hamiltonian().to_matrix(backend)
    h_matrix = backend.to_numpy(h_matrix)

    energy = complex(np.conj(psi) @ h_matrix @ psi).real

    # 独立参照：qiskit problem 自带的电子基态 HF 能量（不含核排斥能）。
    from qiskit_nature.second_q.drivers import PySCFDriver

    problem = PySCFDriver(atom=_H2_GEOMETRY, basis="sto-3g", charge=0, spin=0).run()
    reference_energy = problem.reference_energy - problem.hamiltonian.nuclear_repulsion_energy

    assert energy == pytest.approx(reference_energy, abs=1e-4)


def test_parity_mapping_leaves_metadata_none():
    built = build_molecule(
        _H2_GEOMETRY, basis="sto-3g", mapping="parity", two_qubit_reduction=True
    )
    assert built.hf_occupation is None
    assert built.excitations is None
    assert built.terms  # Hamiltonian 仍可用

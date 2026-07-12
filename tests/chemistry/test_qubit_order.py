"""比特序回归测试：钉住 ``pipeline.build_molecule`` 与 ``spec.generate_hamiltonian``

两条 Qiskit Nature 生成路径必须共享同一比特序（``qiskit_label``，即 Qiskit
``SparsePauliOp.to_list()`` 原样标签，不翻转），与已冻结的分子预置一致。
Phase 2 之前 ``spec.py`` 在 ``_sparse_pauli_terms`` 里对 label 做了
``[::-1]`` 镜像翻转，与 ``pipeline.py``/预置的比特序相反——本文件即钉住修复后的
正确行为。
"""

import pytest

from aicir.chemistry import get_molecule
from aicir.chemistry.spec import GeneratedHamiltonian, MolecularSpec, PresetSpec, generate_hamiltonian


def _term_set(terms, ndigits=10):
    return {(pauli, round(complex(coefficient).real, ndigits)) for coefficient, pauli in terms}


def test_generate_hamiltonian_preset_matches_frozen_preset_no_deps():
    """无 pyscf/qiskit-nature 依赖：preset 路径直接透传预置 terms，不做任何翻转。"""

    generated = generate_hamiltonian(PresetSpec("h2"))
    preset = get_molecule("h2")

    assert generated.n_qubits == preset.n_qubits
    assert _term_set(generated.terms) == _term_set(preset.terms)


_H2_GEOMETRY = "H 0 0 0; H 0 0 0.735"  # PySCF 默认 H2 几何，与 h2_jw 预置一致


pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from aicir.chemistry import build_molecule  # noqa: E402


def test_build_molecule_matches_frozen_preset_term_for_term():
    """钉住 bridge 重构：build_molecule 输出须与冻结预置逐项吻合（比特序不变）。"""

    preset = get_molecule("h2_jw")
    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner", name="h2_jw")

    assert built.n_qubits == preset.n_qubits
    # preset 系数存的是 8 位小数的四舍五入值，built 是全精度现算值；按 preset 精度截断比较。
    assert _term_set(built.terms, ndigits=8) == _term_set(preset.terms, ndigits=8)


def test_generate_hamiltonian_molecular_spec_matches_build_molecule():
    """真正的修复点：两条路径对同一分子须产出同一比特序（不再镜像相反）。"""

    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner", name="h2_jw")
    generated = generate_hamiltonian(
        MolecularSpec(geometry=_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner")
    )

    assert generated.n_qubits == built.n_qubits
    assert _term_set(generated.terms) == _term_set(built.terms)


def test_generated_hamiltonian_metadata_declares_qubit_order():
    generated = generate_hamiltonian(
        MolecularSpec(geometry=_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner")
    )

    assert generated.metadata["qubit_order"] == "qiskit_label"


def test_generated_hamiltonian_to_hamiltonian_matches_terms():
    generated = generate_hamiltonian(
        MolecularSpec(geometry=_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner")
    )

    hamiltonian = generated.to_hamiltonian()
    assert hamiltonian.n_qubits == generated.n_qubits
    got = {
        ("".join(term.qubit_labels), round(term.coefficient.real, 10))
        for term in hamiltonian.terms
    }
    assert got == _term_set(generated.terms)

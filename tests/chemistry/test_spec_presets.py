"""Preset Hamiltonian 生成路径的回归测试（覆盖 spec.py 的 preset 导入修复）。"""

from aicir.chemistry import get_molecule
from aicir.chemistry.spec import PresetSpec, generate_hamiltonian, spec_from_mapping


def _term_set(terms):
    return {(pauli, round(float(coefficient), 12)) for coefficient, pauli in terms}


def test_generate_hamiltonian_preset_spec_matches_frozen_preset():
    generated = generate_hamiltonian(PresetSpec("h2"))
    preset = get_molecule("h2")

    assert generated.n_qubits == preset.n_qubits
    assert _term_set(generated.terms) == _term_set(preset.terms)


def test_generate_hamiltonian_preset_via_mapping_matches_frozen_preset():
    generated = generate_hamiltonian(spec_from_mapping({"preset": "h2"}))
    preset = get_molecule("h2")

    assert generated.n_qubits == preset.n_qubits
    assert _term_set(generated.terms) == _term_set(preset.terms)

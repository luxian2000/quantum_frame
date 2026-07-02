import json
from pathlib import Path

import pytest


def test_pauli_terms_spec_generates_stable_metadata():
    from aicir.chemistry.spec import PauliTermsSpec, generate_hamiltonian

    generated = generate_hamiltonian(
        PauliTermsSpec(
            terms=((1.0, "ZI"), (-0.5, "IX")),
            source="manual",
            metadata={"basis": "custom"},
        )
    )

    assert generated.n_qubits == 2
    assert generated.hamiltonian_class == "pauli_terms"
    assert generated.terms == ((1.0, "ZI"), (-0.5, "IX"))
    assert generated.hamiltonian_id.startswith("pauli_terms_2q_")
    assert generated.metadata["source"] == "manual"
    assert generated.metadata["basis"] == "custom"


def test_load_hamiltonian_input_supports_legacy_term_list(tmp_path):
    from aicir.chemistry.spec import load_hamiltonian_input

    path = tmp_path / "terms.json"
    path.write_text(json.dumps([[1.0, "ZI"], [-0.5, "IX"]]), encoding="utf-8")

    generated = load_hamiltonian_input(path)

    assert generated.n_qubits == 2
    assert generated.hamiltonian_class == "pauli_terms"
    assert generated.terms == ((1.0, "ZI"), (-0.5, "IX"))


def test_molecular_spec_requires_optional_qiskit_nature_when_not_installed(monkeypatch):
    from aicir.chemistry.spec import MolecularSpec, generate_hamiltonian

    def fake_import(name, *args, **kwargs):
        if name.startswith("qiskit_nature"):
            raise ImportError("missing qiskit_nature")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="qiskit-nature.*pyscf"):
        generate_hamiltonian(
            MolecularSpec(
                geometry=(("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))),
                basis="sto3g",
                mapping="jordan_wigner",
            )
        )


def test_qas_cli_loads_generated_hamiltonian_spec(tmp_path):
    from aicir.qas.vqe_loop.__main__ import _load_generated_hamiltonian

    path = tmp_path / "hamiltonian.json"
    path.write_text(
        json.dumps(
            {
                "kind": "pauli_terms",
                "terms": [[1.0, "ZI"], [-0.5, "IX"]],
                "source": "manual",
            }
        ),
        encoding="utf-8",
    )

    generated = _load_generated_hamiltonian(str(path))

    assert generated is not None
    assert generated.n_qubits == 2
    assert generated.hamiltonian_id.startswith("pauli_terms_2q_")


def test_qas_cli_uses_default_runtime_options_for_hamiltonian_only():
    from aicir.qas.vqe_loop.__main__ import _default_output_dir, build_parser

    args = build_parser().parse_args(["--hamiltonian", "lih_r01_sto3g.json"])

    assert args.output_dir is None
    assert args.rounds == "auto"
    assert args.batch_size == "auto"
    assert args.backend == "npu"
    assert args.dtype == "complex64"
    assert _default_output_dir("lih_r01_sto3g.json") == "outputs/qas_lih_r01_sto3g_loop"


def test_molecular_mapping_accepts_defaults_when_kind_is_omitted():
    from aicir.chemistry.spec import MolecularSpec, spec_from_mapping

    spec = spec_from_mapping(
        {
            "molecule": "LiH",
            "geometry": [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.1]]],
        }
    )

    assert isinstance(spec, MolecularSpec)
    assert spec.basis == "sto3g"
    assert spec.charge == 0
    assert spec.spin == 0
    assert spec.unit == "angstrom"
    assert spec.driver == "pyscf"
    assert spec.mapping == "jordan_wigner"


def test_molecular_mapping_expands_diatomic_distance_shorthand():
    from aicir.chemistry.spec import MolecularSpec, spec_from_mapping

    spec = spec_from_mapping({"molecule": "LiH", "distance": 0.1})

    assert isinstance(spec, MolecularSpec)
    assert spec.geometry == [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.1]]]
    assert spec.basis == "sto3g"

    h2 = spec_from_mapping({"molecule": "H2", "distance": 0.735})
    assert h2.geometry == [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]]


def test_preset_mapping_loads_canonical_h2_jw_r0735_library_terms():
    from aicir.chemistry.spec import generate_hamiltonian, spec_from_mapping

    generated = generate_hamiltonian({"preset": "h2_sto3g_jw_r0735_4q"})

    assert generated.n_qubits == 4
    assert generated.hamiltonian_class == "molecular_preset"
    assert generated.hamiltonian_id == "h2_sto3g_jw_r0735_4q"
    assert len(generated.terms) == 19
    assert generated.terms[0] == (-0.09706626816762543, "IIII")
    assert generated.terms[-1] == (-0.17391653067620093, "YYYY")

    spec = spec_from_mapping({"preset": "h2_sto3g_jw_r0735_4q"})
    assert generate_hamiltonian(spec).terms == generated.terms

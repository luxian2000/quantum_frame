from aicir.chemistry import MoleculeHamiltonian, get_molecule


def test_existing_preset_metadata_defaults_to_none():
    mol = get_molecule("h2")
    assert mol.n_electrons is None
    assert mol.hf_occupation is None
    assert mol.excitations is None


def test_metadata_fields_are_settable():
    mol = MoleculeHamiltonian(
        name="toy",
        formula="H2",
        n_qubits=4,
        terms=((-1.0, "IIII"),),
        basis="STO-3G",
        mapping="JordanWignerMapper",
        geometry="toy",
        source="toy",
        n_electrons=2,
        hf_occupation=(1, 1, 0, 0),
        excitations=(("single", (0, 2)), ("double", (0, 1, 2, 3))),
    )
    assert mol.n_electrons == 2
    assert mol.hf_occupation == (1, 1, 0, 0)
    assert mol.excitations == (("single", (0, 2)), ("double", (0, 1, 2, 3)))

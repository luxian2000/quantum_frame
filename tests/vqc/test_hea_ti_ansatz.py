import numpy as np
import pytest

from aicir import NumpyBackend
from aicir.vqc.ansatz import (
    global_evolution_unitary,
    hea_ti,
    hea_ti_ansatz,
    hea_ti_parameter_count,
    power_law_couplings,
    trapped_ion_hamiltonian,
)


def test_power_law_couplings_match_trapped_ion_default_shape():
    couplings = power_law_couplings(4, j0=2.0, alpha=1.0)

    assert couplings.shape == (4, 4)
    assert np.allclose(np.diag(couplings), 0.0)
    assert couplings[0, 1] == pytest.approx(2.0)
    assert couplings[0, 2] == pytest.approx(1.0)
    assert couplings[0, 3] == pytest.approx(2.0 / 3.0)
    assert np.allclose(couplings, couplings.T)


def test_tfim_hamiltonian_matches_two_qubit_formula():
    hamiltonian = trapped_ion_hamiltonian(
        2,
        kind="tfim",
        couplings=[[0.0, 1.0], [1.0, 0.0]],
        transverse_field=0.25,
        dtype=np.complex128,
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    eye = np.eye(2, dtype=np.complex128)
    expected = np.kron(x, x) + 0.25 * (np.kron(z, eye) + np.kron(eye, z))

    assert np.allclose(hamiltonian, expected)


def test_xy_hamiltonian_matches_charge_conserving_formula():
    hamiltonian = trapped_ion_hamiltonian(
        2,
        kind="xy",
        couplings=[[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    expected = 0.5 * (np.kron(x, x) + np.kron(y, y))

    assert np.allclose(hamiltonian, expected)


def test_global_evolution_unitary_is_unitary():
    hamiltonian = trapped_ion_hamiltonian(2, kind="xy", dtype=np.complex128)
    unitary = global_evolution_unitary(hamiltonian, 0.4, dtype=np.complex128)

    assert unitary.shape == (4, 4)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-10)


def test_general_hea_ti_layout_and_parameter_count():
    circuit = hea_ti_ansatz(2, layers=2, variant="general")

    assert circuit.n_qubits == 2
    assert len(circuit.parameters) == hea_ti_parameter_count(2, layers=2, variant="general")
    assert [parameter.name for parameter in circuit.parameters] == [f"theta_{index}" for index in range(12)]
    assert [gate["type"] for gate in circuit.gates[:7]] == ["rx", "ry", "rx", "rx", "ry", "rx", "unitary"]
    assert sum(gate["type"] == "unitary" for gate in circuit.gates) == 2
    assert hea_ti_parameter_count(2, layers=2, variant="general", include_evolution_times=True) == 14


def test_symmetry_hea_ti_uses_rz_and_xy_global_evolution():
    circuit = hea_ti_ansatz(3, layers=1, variant="chemistry", evolution_time=0.4)

    assert len(circuit.parameters) == 3
    assert [gate["type"] for gate in circuit.gates] == ["rz", "rz", "rz", "unitary"]
    assert circuit.gates[-1]["label"] == "HEA-TI-symmetry"
    assert circuit.gates[-1]["parameter"].shape == (8, 8)


def test_hea_ti_accepts_numeric_rotation_parameters_and_builds_unitary():
    count = hea_ti_parameter_count(2, layers=1, variant="general")
    circuit = hea_ti(
        2,
        layers=1,
        variant="general",
        parameters=np.linspace(0.1, 0.6, count),
        evolution_times=[0.2],
    )

    assert circuit.parameters == ()
    assert circuit.unitary(backend=NumpyBackend()).shape == (4, 4)
    assert circuit.gates[-1]["evolution_time"] == pytest.approx(0.2)


def test_hea_ti_can_place_global_evolution_before_rotations():
    circuit = hea_ti_ansatz(2, layers=1, variant="xy", rotation_first=False)

    assert circuit.gates[0]["type"] == "unitary"
    assert [gate["type"] for gate in circuit.gates[1:]] == ["rz", "rz"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_qubits": 0}, "n_qubits"),
        ({"n_qubits": 2, "layers": -1}, "layers"),
        ({"n_qubits": 2, "variant": "bad"}, "Unsupported HEA-TI variant"),
        ({"n_qubits": 2, "evolution_times": [0.1, 0.2]}, "Expected 1 evolution time"),
        ({"n_qubits": 2, "couplings": [[0.0, 1.0, 2.0]]}, "couplings"),
        ({"n_qubits": 2, "parameters": [0.1]}, "Expected at least"),
    ],
)
def test_hea_ti_validates_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        hea_ti_ansatz(**kwargs)

import pytest
from aicir import Circuit, measure, reset, hadamard


def test_measure_basis_id_flow_to_dict():
    cir = Circuit(hadamard(0), measure(0, 1, basis="X", id="m0"), n_qubits=2)
    d = cir.gates[-1]
    assert d["type"] == "measure"
    assert d["qubits"] == [0, 1]
    assert d["basis"] == "X"
    assert d["id"] == "m0"


def test_measure_default_basis_z_no_id():
    cir = Circuit(measure(0), n_qubits=1)
    d = cir.gates[-1]
    assert d.get("basis", "Z") == "Z"
    assert d.get("id") is None


def test_measure_iterable_form_with_basis():
    cir = Circuit(measure([0, 1], basis="y"), n_qubits=2)
    assert cir.gates[-1]["basis"] == "Y"


def test_reset_factory_still_works():
    cir = Circuit(reset(0, 1), n_qubits=2)
    assert cir.gates[-1]["type"] == "reset"
    assert cir.gates[-1]["qubits"] == [0, 1]

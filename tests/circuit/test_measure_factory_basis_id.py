import pytest
from aicir import Circuit, measure, reset, hadamard


def test_measure_basis_id_flow_to_dict():
    cir = Circuit(hadamard(0), measure([0, 1], basis="X", id="m0"), n_qubits=2)
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
    cir = Circuit(reset([0, 1]), n_qubits=2)
    assert cir.gates[-1]["type"] == "reset"
    assert cir.gates[-1]["qubits"] == [0, 1]


def test_multi_positional_form_rejected():
    with pytest.raises(TypeError):
        measure(0, 1)          # 多个比特须用列表 measure([0, 1])
    with pytest.raises(TypeError):
        reset(0, 1)


def test_single_int_and_list_forms_equivalent():
    assert measure(0).qubits == measure([0]).qubits == (0,)
    assert measure([0, 1]).qubits == (0, 1)
    assert reset([0, 1]).qubits == (0, 1)
    assert measure().qubits == ()

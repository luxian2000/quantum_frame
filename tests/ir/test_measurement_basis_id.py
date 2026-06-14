import pytest

from aicir.ir.measurement import Measurement


def test_defaults_basis_z_id_none():
    m = Measurement((0, 1))
    assert m.basis == "Z"
    assert m.id is None


def test_basis_normalized_uppercase_and_validated():
    assert Measurement((0,), basis="x").basis == "X"
    with pytest.raises(ValueError):
        Measurement((0,), basis="W")


def test_to_dict_round_trip_includes_basis_and_id():
    m = Measurement((0, 2), basis="Y", id="m0")
    d = m.to_dict()
    assert d["type"] == "measure"
    assert d["qubits"] == [0, 2]
    assert d["basis"] == "Y"
    assert d["id"] == "m0"
    assert Measurement.from_dict(d) == m


def test_from_dict_backward_compatible_without_basis_id():
    m = Measurement.from_dict({"type": "measure", "qubits": [1]})
    assert m.basis == "Z"
    assert m.id is None


def test_reset_keeps_basis_default_and_no_id_field_emitted():
    r = Measurement((0,), measurement_type="reset")
    d = r.to_dict()
    assert d["type"] == "reset"
    assert "id" not in d
    assert "basis" not in d

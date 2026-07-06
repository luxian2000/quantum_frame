import pytest
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister


def test_measure_creg_all_bits():
    reg = ClassicalRegister(2, "c")
    m = measure([0, 1], creg=reg)
    d = m.to_dict()
    assert d["classical_bits"] == [0, 1]
    assert d["classical_register"] == "c"


def test_measure_cbits_explicit():
    reg = ClassicalRegister(3, "c")
    m = measure([0, 1], cbits=[reg[2], reg[0]])
    d = m.to_dict()
    assert d["classical_bits"] == [2, 0]
    assert d["classical_register"] == "c"


def test_measure_no_classical_target_unchanged():
    d = measure([0, 1]).to_dict()
    assert "classical_register" not in d
    assert d.get("classical_bits", []) == []


def test_measure_creg_rejects_nonz_basis():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError, match="Z"):
        measure(0, basis="X", creg=reg)


def test_measure_creg_and_cbits_mutually_exclusive():
    reg = ClassicalRegister(2, "c")
    with pytest.raises(ValueError):
        measure([0, 1], creg=reg, cbits=[reg[0], reg[1]])


def test_measure_cbits_cross_register_rejected():
    a, b = ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
    with pytest.raises(ValueError, match="同一"):
        measure([0, 1], cbits=[a[0], b[0]])


def test_measure_creg_length_mismatch_rejected():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError):
        measure([0, 1], creg=reg)  # 2 qubits, reg 只有 1 位

import pytest
from aicir.core.classical import Bit, ClassicalRegister, Condition


def test_register_indexing_and_len():
    reg = ClassicalRegister(3, "c")
    assert reg.name == "c" and len(reg) == 3
    b = reg[2]
    assert isinstance(b, Bit) and b.register_name == "c" and b.index == 2
    with pytest.raises(IndexError):
        reg[3]


def test_bit_condition_sugar():
    reg = ClassicalRegister(2, "c")
    cond = reg[0] == 1
    assert isinstance(cond, Condition)
    assert (cond.register_name, cond.index, cond.op, cond.value) == ("c", 0, "==", 1)
    ne = reg[1] != 0
    assert (ne.index, ne.op, ne.value) == (1, "!=", 0)
    with pytest.raises(ValueError):
        reg[0] == 2  # 位只能 0/1


def test_register_int_condition():
    reg = ClassicalRegister(2, "c")
    cond = reg == 3
    assert (cond.register_name, cond.index, cond.op, cond.value) == ("c", None, "==", 3)


def test_evaluate_lsb_convention():
    reg = ClassicalRegister(2, "c")
    store = {"c": [1, 0]}  # bit0=1,bit1=0 -> int 1 (LSB=bit0)
    assert (reg[0] == 1).evaluate(store) is True
    assert (reg[1] == 0).evaluate(store) is True
    assert (reg == 1).evaluate(store) is True
    assert (reg == 2).evaluate(store) is False
    assert (reg != 2).evaluate(store) is True


def test_evaluate_missing_register_defaults_zero():
    assert (ClassicalRegister(1, "x")[0] == 0).evaluate({}) is True


def test_condition_roundtrip():
    c = ClassicalRegister(2, "c") == 3
    assert Condition.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_hashable():
    reg = ClassicalRegister(2, "c")
    {reg, reg[0]}  # 不应抛错

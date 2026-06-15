"""DiffMethod 策略注册表测试（NEXT.md §6 第一片）。"""

import pytest

from aicir.qml.diff import DiffMethod


def test_diff_method_normalizes_name_and_aliases():
    m = DiffMethod(name="  psr ", fn=lambda fn, p: p, aliases=["multipsr"])
    assert m.name == "psr"
    assert m.aliases == ("multipsr",)


def test_diff_method_empty_name_raises():
    with pytest.raises(ValueError):
        DiffMethod(name="  ", fn=lambda fn, p: p)


def test_diff_method_non_callable_fn_raises():
    with pytest.raises(TypeError):
        DiffMethod(name="psr", fn=123)


def test_diff_method_is_frozen():
    m = DiffMethod(name="psr", fn=lambda fn, p: p)
    with pytest.raises(Exception):
        m.name = "fd"

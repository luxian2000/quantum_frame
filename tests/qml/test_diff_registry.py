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


from aicir.qml import deriv
from aicir.qml.diff import (
    canonical_diff_name,
    get_diff_method,
    register_diff_method,
    registered_diff_methods,
    resolve_diff_method,
    unregister_diff_method,
)


def test_builtin_methods_registered():
    assert set(registered_diff_methods()) == {"psr", "fd", "auto", "spsa", "spsr"}


def test_resolve_returns_bound_function():
    assert resolve_diff_method("psr") is deriv.psr
    assert resolve_diff_method("auto") is deriv.auto


def test_resolve_unknown_raises_with_listing():
    with pytest.raises(ValueError) as exc:
        resolve_diff_method("nope")
    assert "psr" in str(exc.value)


def test_mpsr_not_registered():
    assert "mpsr" not in registered_diff_methods()
    assert get_diff_method("mpsr") is None
    with pytest.raises(ValueError):
        resolve_diff_method("mpsr")


def test_register_and_unregister_roundtrip():
    spec = DiffMethod("dummy", lambda fn, p: p)
    register_diff_method(spec)
    try:
        assert "dummy" in registered_diff_methods()
        assert resolve_diff_method("dummy") is spec.fn
    finally:
        unregister_diff_method("dummy")
    assert "dummy" not in registered_diff_methods()


def test_duplicate_register_raises():
    with pytest.raises(ValueError):
        register_diff_method(DiffMethod("psr", lambda fn, p: p))


def test_alias_resolution():
    spec = DiffMethod("dummy2", lambda fn, p: p, aliases=("dummy_alias",))
    register_diff_method(spec)
    try:
        assert canonical_diff_name("dummy_alias") == "dummy2"
        assert get_diff_method("dummy_alias") is spec
    finally:
        unregister_diff_method("dummy2")


def test_canonical_unknown_passthrough():
    assert canonical_diff_name("unknown_xyz") == "unknown_xyz"

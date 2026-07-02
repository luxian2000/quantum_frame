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
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    unregister_diff,
)


def test_builtin_fn_gradients_registered():
    assert set(registered_diffs(category="fn_gradient")) == {"psr", "fd", "auto", "spsa", "spsr"}


def test_builtin_circuit_gradient_and_preconditioners_registered():
    assert set(registered_diffs(category="circuit_gradient")) == {"ad"}
    assert set(registered_diffs(category="preconditioner")) == {"qng", "bdqng", "kqng", "dqng"}


def test_registered_diffs_no_filter_is_all_categories():
    assert set(registered_diffs()) == {
        "psr", "fd", "auto", "spsa", "spsr",
        "ad", "qng", "bdqng", "kqng", "dqng",
    }


def test_resolve_returns_bound_function():
    assert resolve_diff("psr") is deriv.psr
    assert resolve_diff("auto") is deriv.auto


def test_resolve_rejects_non_fn_gradient():
    # ad/qng 已注册可被 get_diff 发现，但 resolve_diff 只解析 fn_gradient
    assert get_diff("ad") is not None
    assert get_diff("qng") is not None
    for name in ("ad", "qng", "bdqng", "kqng", "dqng"):
        with pytest.raises(ValueError):
            resolve_diff(name)


def test_resolve_unknown_raises_with_listing():
    with pytest.raises(ValueError) as exc:
        resolve_diff("nope")
    assert "psr" in str(exc.value)


def test_mpsr_not_registered():
    assert "mpsr" not in registered_diffs()
    assert get_diff("mpsr") is None
    with pytest.raises(ValueError):
        resolve_diff("mpsr")


def test_register_and_unregister_roundtrip():
    spec = DiffMethod("dummy", lambda fn, p: p)
    register_diff(spec)
    try:
        assert "dummy" in registered_diffs()
        assert resolve_diff("dummy") is spec.fn
    finally:
        unregister_diff("dummy")
    assert "dummy" not in registered_diffs()


def test_duplicate_register_raises():
    with pytest.raises(ValueError):
        register_diff(DiffMethod("psr", lambda fn, p: p))


def test_alias_resolution():
    spec = DiffMethod("dummy2", lambda fn, p: p, aliases=("dummy_alias",))
    register_diff(spec)
    try:
        assert canonical_diff("dummy_alias") == "dummy2"
        assert get_diff("dummy_alias") is spec
    finally:
        unregister_diff("dummy2")


def test_canonical_unknown_passthrough():
    assert canonical_diff("unknown_xyz") == "unknown_xyz"


from aicir.qml.diff import select_diff


class GPUBackend:  # noqa: N801 - 模拟 Torch 系后端类名
    pass


def test_select_prefers_auto_on_torch_noiseless_no_shots():
    assert select_diff(backend=GPUBackend()) == "auto"


def test_select_falls_back_to_psr_with_shots():
    assert select_diff(backend=GPUBackend(), shots=1024) == "psr"


def test_select_falls_back_to_psr_when_noisy():
    assert select_diff(backend=GPUBackend(), noisy=True) == "psr"


def test_select_psr_on_non_torch_backend():
    assert select_diff(backend=None) == "psr"


def test_select_never_returns_stochastic():
    for kwargs in ({}, {"shots": 1000}, {"noisy": True}, {"backend": GPUBackend()}):
        assert select_diff(**kwargs) not in {"spsa", "spsr"}


def test_diff_api_reexported_from_qml():
    import aicir.qml as qml

    assert hasattr(qml, "DiffMethod")
    assert hasattr(qml, "resolve_diff")
    assert hasattr(qml, "select_diff")
    assert qml.resolve_diff("psr") is qml.psr

"""SearchStrategy 协议 + 策略注册表（QAS README §2.1）。"""

import importlib

import pytest

from aicir.qas import run
from aicir.qas.core.strategies import (
    SearchStrategy,
    StrategySpec,
    get_spec,
    get_strategy,
    register_strategy,
    registered_strategies,
    unregister_strategy,
)


def test_supernet_strategy_registered_as_import_side_effect():
    assert "supernet" in registered_strategies()
    spec = get_spec("supernet")
    assert spec is not None and spec.requires_torch is True
    assert isinstance(get_strategy("supernet"), SearchStrategy)


def test_dqas_strategy_registered_as_import_side_effect():
    assert "dqas" in registered_strategies()
    spec = get_spec("dqas")
    assert spec is not None and spec.requires_torch is True
    assert isinstance(get_strategy("dqas"), SearchStrategy)


def test_nonmigrated_methods_fall_back_to_table():
    for method in (
        "crlqas",
        "pprdql",
        "pporb",
        "supernet_classification",
        "supernet_h2",
        "qdrats",
    ):
        assert get_strategy(method) is None


def test_register_get_unregister_with_alias():
    class _S(SearchStrategy):
        def run(self, request):
            return 1

    register_strategy(StrategySpec("xx", _S(), aliases=("x_alias",)))
    try:
        assert get_strategy("xx") is get_strategy("x_alias")
        assert "xx" in registered_strategies()
        assert get_spec("x_alias").name == "xx"
    finally:
        unregister_strategy("xx")
    assert get_strategy("xx") is None and get_strategy("x_alias") is None


def test_duplicate_register_rejected_unless_overwrite():
    class _S(SearchStrategy):
        def run(self, request):
            return 1

    register_strategy(StrategySpec("yy", _S()))
    try:
        with pytest.raises(ValueError):
            register_strategy(StrategySpec("yy", _S()))
        register_strategy(StrategySpec("yy", _S()), overwrite=True)  # 不抛
    finally:
        unregister_strategy("yy")


def test_run_routes_supernet_through_registry():
    captured = {}

    class _Dummy(SearchStrategy):
        def run(self, request):
            captured["method"] = request.method
            return "DUMMY"

    register_strategy(StrategySpec("supernet", _Dummy()), overwrite=True)
    try:
        out = run("supernet", config=None)
        assert out == "DUMMY"
        assert captured["method"] == "supernet"
    finally:
        # 复原真实 SupernetStrategy（重载注册模块重新注册）。
        unregister_strategy("supernet")
        importlib.reload(importlib.import_module("aicir.qas.core.strategies"))

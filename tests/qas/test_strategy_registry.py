"""QAS SearchStrategy 协议 + 注册表测试（模块化第一片）。"""

import pytest

pytest.importorskip("torch")

from aicir.qas.core.registry import (
    StrategySpec,
    get_spec,
    get_strategy,
    register_strategy,
    registered_strategies,
    unregister_strategy,
)
from aicir.qas.core.runner import run
from aicir.qas.core.strategy import SearchStrategy


class _Dummy(SearchStrategy):
    name = "dummy"

    def __init__(self) -> None:
        self.calls: list = []

    def run(self, request):
        self.calls.append(request)
        return "ok"


def test_base_strategy_is_abstract():
    with pytest.raises(TypeError):
        SearchStrategy()  # type: ignore[abstract]


def test_register_get_list_roundtrip():
    d = _Dummy()
    register_strategy(StrategySpec("dummy", d))
    try:
        assert get_strategy("dummy") is d
        assert "dummy" in registered_strategies()
        assert get_spec("dummy").requires_torch is False
    finally:
        unregister_strategy("dummy")
    assert get_strategy("dummy") is None


def test_duplicate_register_raises():
    d = _Dummy()
    register_strategy(StrategySpec("dummy", d))
    try:
        with pytest.raises(ValueError):
            register_strategy(StrategySpec("dummy", _Dummy()))
    finally:
        unregister_strategy("dummy")


def test_spec_rejects_non_strategy():
    with pytest.raises(TypeError):
        StrategySpec("bad", object())  # type: ignore[arg-type]


def test_supernet_registered_as_strategy():
    strat = get_strategy("supernet")
    assert strat is not None
    assert isinstance(strat, SearchStrategy)
    assert "supernet" in registered_strategies()
    assert get_spec("supernet").requires_torch is True


def test_run_supernet_routes_through_strategy(monkeypatch):
    import aicir.qas.algorithms.supernet as supernet

    captured: dict = {}

    def fake_train_supernet(**kwargs):
        captured.update(kwargs)
        return "RESULT"

    monkeypatch.setattr(supernet, "train_supernet", fake_train_supernet)

    out = run("supernet", objective="OBJ", config="CFG", dataset="DS", hamiltonian="H")

    assert out == "RESULT"
    assert captured == {"objective": "OBJ", "config": "CFG", "dataset": "DS", "hamiltonian": "H"}

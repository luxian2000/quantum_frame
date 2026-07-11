"""aicir.protocols 在 optimizer 层的落地测试：HistoryRecord 兼容旧 dict 键访问、
Optimizer protocol 与 minimize() 调度器的类型检查（Phase 1 item 4）。
"""

import numpy as np
import pytest

from aicir.optimizer import GD, Adam, minimize
from aicir.protocols import HistoryRecord, Optimizer


def test_history_record_supports_dict_style_access():
    rec = HistoryRecord(step=2, fun=0.5, grad_norm=0.3, learning_rate=0.1, extras={"perturbation": 0.02})

    assert rec["step"] == 2
    assert rec["fun"] == 0.5
    assert rec.get("grad_norm") == 0.3
    assert rec.get("perturbation") == 0.02
    assert rec.get("missing", "default") == "default"
    assert "learning_rate" in rec
    assert "perturbation" in rec
    assert "missing" not in rec
    with pytest.raises(KeyError):
        rec["missing"]


def test_history_record_named_fields_shadow_extras_lookup_order():
    # 具名字段优先于 extras（即便 extras 中恰好放了同名 key）
    rec = HistoryRecord(step=1, fun=1.0, extras={"fun": "should-not-be-seen"})

    assert rec["fun"] == 1.0
    assert rec.get("fun") == 1.0


def test_gd_minimize_history_entries_are_history_records_with_dict_access():
    def fn(params):
        return float(np.dot(params, params))

    def grad(params):
        return 2.0 * np.asarray(params, dtype=float)

    optimizer = GD(max_iters=5, learning_rate=0.1)
    result = optimizer.minimize(fn, np.array([1.0]), gradient_fn=grad)

    assert result.history
    first = result.history[0]
    assert isinstance(first, HistoryRecord)
    assert first["step"] == 0
    assert first.get("learning_rate") == pytest.approx(0.1)
    assert first.get("grad_norm") is not None


def test_gd_and_adam_satisfy_optimizer_protocol():
    assert isinstance(GD(), Optimizer)
    assert isinstance(Adam(), Optimizer)
    assert not isinstance(object(), Optimizer)
    assert not isinstance(42, Optimizer)


def test_minimize_rejects_non_optimizer_with_clear_type_error():
    with pytest.raises(TypeError, match="minimize"):
        minimize(lambda params: float(np.sum(params)), np.array([1.0]), optimizer=object())

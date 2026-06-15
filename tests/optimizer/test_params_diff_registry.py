"""params.py 通过 DiffMethod 注册表分发梯度方法。"""

import numpy as np
import pytest

from aicir.optimizer.params import Adam


def _cos_sum(params):
    # 参数移位规则对 cos 目标精确（梯度 = -sin），最小值在每个分量 = π。
    # 必须用 PSR 兼容目标，普通二次式会让 psr/spsr 算出错误"梯度"。
    return float(np.sum(np.cos(np.asarray(params, dtype=float))))


def test_adam_can_use_spsr_via_registry():
    # spsr 此前无法从 params.py 触达；现在应可运行并下降。
    init = np.array([1.0, 2.0])
    rng = np.random.default_rng(0)
    opt = Adam(
        gradient_method="spsr",
        gradient_kwargs={"rng": rng},
        learning_rate=0.15,
        max_iters=100,
    )
    result = opt.minimize(_cos_sum, init)
    assert result.best_fun < _cos_sum(init)


def test_adam_unknown_method_lists_registered():
    opt = Adam(gradient_method="bogus", max_iters=5)
    with pytest.raises(ValueError) as exc:
        opt.minimize(_cos_sum, np.array([1.0]))
    assert "psr" in str(exc.value)
    assert "bogus" in str(exc.value)


def test_adam_psr_via_registry_descends():
    init = np.array([1.0, 2.0])
    opt = Adam(gradient_method="psr", learning_rate=0.2, max_iters=80)
    result = opt.minimize(_cos_sum, init)
    assert result.best_fun < _cos_sum(init)


def test_adam_rejects_auto_in_classical_path():
    # auto 需 Torch 后端与 autograd 图，经经典优化器黑盒目标应及早报清晰错误。
    opt = Adam(gradient_method="auto", max_iters=5)
    with pytest.raises(ValueError) as exc:
        opt.minimize(_cos_sum, np.array([1.0]))
    assert "auto" in str(exc.value)

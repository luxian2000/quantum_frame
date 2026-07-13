"""Array-in/array-out 返回值契约（``aicir/qml/deriv/__init__.py`` 文档化的现状）。

- NumPy 入参 → ``psr``/``fd`` 等一律返回 ``np.ndarray``。
- torch 入参/backend → ``rotosolve``/``qng`` 族返回同设备的 torch 张量。

本文件只钉住契约本身（返回类型/设备），不重复各方法自己的数值正确性测试
（那些在 ``tests/qml/test_gradient.py``/``tests/qml/test_grad_free.py``/
``tests/qml/test_psr4.py`` 里）。线路保持最小（1 qubit）。
"""

import numpy as np
import pytest

from aicir.qml import dqng, fd, psr, qng, rotosolve


def test_psr_numpy_params_returns_numpy_array():
    def objective(theta):
        return np.cos(theta[0])

    grad = psr(objective, np.array([0.3]))

    assert isinstance(grad, np.ndarray)
    assert not hasattr(grad, "requires_grad")


def test_fd_numpy_params_returns_numpy_array():
    def objective(theta):
        return np.cos(theta[0])

    grad = fd(objective, np.array([0.3]))

    assert isinstance(grad, np.ndarray)
    assert not hasattr(grad, "requires_grad")


def test_rotosolve_torch_params_returns_torch_tensor_on_same_device():
    torch = pytest.importorskip("torch")

    def objective(theta):
        return torch.sin(theta[0] + 0.3)

    params = torch.tensor([0.7], dtype=torch.float64)
    optimized, value = rotosolve(objective, params, return_value=True)

    assert isinstance(optimized, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert optimized.device == params.device


def test_qng_always_returns_numpy_even_with_torch_backend_state():
    """qng（稠密 QFIM 版本）没有 torch 设备驻留分支——始终返回 NumPy。

    只有 kqng/dqng 实现了 NPU/torch 设备驻留路径（见 __init__.py 顶部的契约
    文档）；qng/bdqng 内部一律通过 _as_scalar/_as_state_vector 把中间结果转回
    host，即便 grad/qfim 是 torch 张量也会被 np.asarray(...) 转成 NumPy。
    """
    torch = pytest.importorskip("torch")

    natural_grad = qng(
        None,
        None,
        np.array([0.1, 0.2]),
        grad=torch.tensor([2.0, 4.0], dtype=torch.float64),
        qfim=np.diag([2.0, 4.0]),
        damping=0.0,
    )

    assert isinstance(natural_grad, np.ndarray)
    assert np.allclose(natural_grad, [1.0, 1.0])


def test_dqng_torch_grad_input_returns_torch_tensor_on_same_device():
    """dqng 是真正实现 torch 设备驻留分支的 QNG 变体：torch grad/qfim_diag 入参 → torch 出参。"""
    torch = pytest.importorskip("torch")

    grad = torch.tensor([2.0, 4.0], dtype=torch.float64)
    qfim_diag = torch.tensor([2.0, 4.0], dtype=torch.float64)
    natural_grad = dqng(
        None,
        None,
        np.array([0.1, 0.2]),
        grad=grad,
        qfim_diag=qfim_diag,
        damping=0.0,
    )

    assert isinstance(natural_grad, torch.Tensor)
    assert natural_grad.device == grad.device
    assert torch.allclose(natural_grad, torch.tensor([1.0, 1.0], dtype=natural_grad.dtype))

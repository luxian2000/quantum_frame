"""标量目标函数的 Hessian 矩阵：psr 对角线自检 + fd 降级路径。"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from ._coerce import _as_scalar
from .fn_gradient import mpsr


def _psr_second_at_index(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    index: tuple[int, ...],
    shift: float,
    coefficient: float,
) -> float:
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += 2.0 * shift
    minus[index] -= 2.0 * shift
    center = _as_scalar(fn(theta), label="fn(params)")
    forward = _as_scalar(fn(plus), label="fn(params + 2*shift)")
    backward = _as_scalar(fn(minus), label="fn(params - 2*shift)")
    return float((coefficient ** 2) * (forward - 2.0 * center + backward))


def _fd_second_at_indices(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    left: tuple[int, ...],
    right: tuple[int, ...],
    eps: float,
) -> float:
    if left == right:
        plus = theta.copy()
        minus = theta.copy()
        plus[left] += eps
        minus[left] -= eps
        center = _as_scalar(fn(theta), label="fn(params)")
        forward = _as_scalar(fn(plus), label="fn(params + eps)")
        backward = _as_scalar(fn(minus), label="fn(params - eps)")
        return float((forward - 2.0 * center + backward) / (eps ** 2))

    pp = theta.copy()
    pm = theta.copy()
    mp = theta.copy()
    mm = theta.copy()
    pp[left] += eps
    pp[right] += eps
    pm[left] += eps
    pm[right] -= eps
    mp[left] -= eps
    mp[right] += eps
    mm[left] -= eps
    mm[right] -= eps
    return float((
        _as_scalar(fn(pp), label="fn(params + eps_i + eps_j)")
        - _as_scalar(fn(pm), label="fn(params + eps_i - eps_j)")
        - _as_scalar(fn(mp), label="fn(params - eps_i + eps_j)")
        + _as_scalar(fn(mm), label="fn(params - eps_i - eps_j)")
    ) / (4.0 * eps ** 2))


def _fd_second_with_uncertainty(
    fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    index: tuple[int, ...],
    eps: float,
) -> tuple[float, float]:
    """返回 fd 二阶对角元及其**自身**的误差估计。

    二阶中心差分的舍入误差约为 ``δf / eps**2``：目标函数只要走 complex64 态矢量
    （δf≈1e-7），默认 ``eps=1e-3`` 就能放大出 0.1 量级的误差——此时 fd 自己都不准，
    不能拿它去判定精确的 psr 是否"不一致"。这里用 ``eps`` 与 ``2*eps`` 两个步长的
    差值作为 fd 的不确定度，供调用方判断 fd 是否有资格做裁判。
    """
    fine = _fd_second_at_indices(fn, theta, index, index, eps)
    coarse = _fd_second_at_indices(fn, theta, index, index, 2.0 * eps)
    return fine, abs(fine - coarse)


def _psr_agrees_with_fd(psr_value: float, fd_value: float, fd_uncertainty: float) -> bool:
    """psr 与 fd 是否一致——容差取 fd 自身不确定度与固定容差中的较大者。

    只有当 fd 足够可信（不确定度小）且仍与 psr 显著不符时，才认定生成元谱不是标准
    Pauli 旋转的 {-1,0,1}。
    """
    tolerance = max(1e-3 * abs(psr_value), 1e-5, 4.0 * fd_uncertainty)
    return abs(psr_value - fd_value) <= tolerance


def hessian(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    method: str = "auto",
    shift: float = np.pi / 2.0,
    coefficient: float = 0.5,
    eps: float = 1e-3,
) -> np.ndarray:
    """计算标量目标函数的完整 Hessian 矩阵。

    ``method``：

    - ``"auto"``（默认）：先用 psr 二阶公式估计每个参数的对角元，并与 fd 对照；
      若某个对角元不一致（生成元谱不是标准 Pauli 旋转的 {-1,0,1}），整体降级为
      fd 并通过 ``warnings.warn(..., RuntimeWarning)`` 提示——这是拆包前的默认
      行为（当时唯一可选项就是这套逻辑，只是没有名字也不报警），拆包只补上了
      警告，数值结果不变。
    - ``"psr"``：纯 psr 二阶公式，不做 fd 对照/降级；若检测到对角元与 fd 不一致
      （生成元谱非 {-1,0,1}），直接抛 ``ValueError`` 而不是静默换算法。
    - ``"fd"``：纯有限差分，不做 psr 计算。
    """
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("params must contain at least one parameter")

    method_norm = str(method).strip().lower()
    if method_norm not in {"auto", "psr", "fd"}:
        raise ValueError("method must be 'auto', 'psr', or 'fd'")
    eps_value = float(eps)
    if not eps_value > 0.0:
        raise ValueError("eps must be a positive number")

    indices = list(np.ndindex(theta.shape))
    psr_diag_cache: dict[tuple[int, ...], float] = {}
    fd_diag_cache: dict[tuple[int, ...], float] = {}

    active_method = method_norm
    if method_norm == "auto":
        active_method = "psr"
        for index in indices:
            shifted_value = _psr_second_at_index(fn, theta, index, float(shift), float(coefficient))
            fd_value, fd_uncertainty = _fd_second_with_uncertainty(fn, theta, index, eps_value)
            psr_diag_cache[index] = shifted_value
            fd_diag_cache[index] = fd_value
            if not _psr_agrees_with_fd(shifted_value, fd_value, fd_uncertainty):
                warnings.warn(
                    "hessian(method='auto') 检测到 psr 二阶对角元与 fd 不一致"
                    f"（index={index}），生成元谱可能不是标准 Pauli 旋转的 "
                    "{-1,0,1}；已自动降级为 fd。如需固定算法，请显式传入 "
                    "method='psr'（不一致时报错）或 method='fd'。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                active_method = "fd"
                break
    elif method_norm == "psr":
        for index in indices:
            shifted_value = _psr_second_at_index(fn, theta, index, float(shift), float(coefficient))
            fd_value, fd_uncertainty = _fd_second_with_uncertainty(fn, theta, index, eps_value)
            psr_diag_cache[index] = shifted_value
            if not _psr_agrees_with_fd(shifted_value, fd_value, fd_uncertainty):
                raise ValueError(
                    "hessian(method='psr') 检测到 psr 二阶对角元与 fd 不一致"
                    f"（index={index}）；生成元谱可能不是标准 Pauli 旋转的 "
                    "{-1,0,1}，psr 二阶公式不适用。请改用 method='fd' 或 "
                    "method='auto'（自动降级并给出 RuntimeWarning）。"
                )

    matrix = np.zeros((theta.size, theta.size), dtype=float)
    for row, left in enumerate(indices):
        for col in range(row, theta.size):
            right = indices[col]
            if active_method == "psr":
                if left == right:
                    value = psr_diag_cache.get(left)
                    if value is None:
                        value = _psr_second_at_index(fn, theta, left, float(shift), float(coefficient))
                else:
                    value = mpsr(
                        fn,
                        theta,
                        parameter_indices=[left, right],
                        shift=shift,
                        coefficient=coefficient,
                    )
            else:
                if left == right and left in fd_diag_cache:
                    value = fd_diag_cache[left]
                else:
                    value = _fd_second_at_indices(fn, theta, left, right, eps_value)
            matrix[row, col] = value
            matrix[col, row] = value
    return 0.5 * (matrix + matrix.T)

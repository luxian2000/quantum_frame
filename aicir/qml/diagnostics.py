"""训练期贫瘠高原（barren plateau）诊断（qml 成熟化 #7）。

把梯度方差信号 + QFIM 谱接到真实训练目标（:class:`~aicir.qml.qfun.QFun`），
而非 ``metrics.trainability`` 的固定 local-probe。核心信号：

- **梯度方差**：随机采参数、算真实目标梯度，方差随比特数指数衰减 = 贫瘠高原。
- **QFIM 谱**：量子 Fisher 信息矩阵的近零特征值 = 平坦方向（过参数化/不可辨识）。

梯度经 ``QFun.grad``（走 ``aicir.qml.deriv`` 单一事实来源）；QFIM 经 ``qml.qfim``。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .deriv import qfim

_TWO_PI = 2.0 * np.pi


def gradient_variance(
    qfun: Any,
    n_parameters: int,
    *,
    n_samples: int = 30,
    seed: int = 0,
    param_scale: float = _TWO_PI,
    vanish_threshold: float = 1e-6,
) -> dict[str, Any]:
    """随机采参数下真实目标梯度的方差统计（贫瘠高原信号）。

    Args:
        qfun: 训练目标（expval 返回的 :class:`QFun`）。
        n_parameters: 参数向量长度。
        n_samples: 随机参数采样数。
        seed: 随机种子。
        param_scale: 参数在 ``[-scale, scale]`` 均匀采样。
        vanish_threshold: 梯度方差低于此值判为消失（``vanishing=True``）。

    Returns:
        dict：``n_parameters``/``n_samples``/``gradient_variance``（逐参数方差均值）/
        ``mean_gradient_norm``/``max_abs_gradient``/``vanishing``。
    """
    if n_parameters < 1:
        raise ValueError("n_parameters 须 ≥1")
    rng = np.random.default_rng(seed)
    grads = np.zeros((int(n_samples), int(n_parameters)), dtype=float)
    for i in range(int(n_samples)):
        theta = rng.uniform(-param_scale, param_scale, size=int(n_parameters))
        grads[i] = np.asarray(qfun.grad(theta), dtype=float).reshape(-1)
    var = float(np.mean(np.var(grads, axis=0))) if n_samples > 1 else 0.0
    return {
        "n_parameters": int(n_parameters),
        "n_samples": int(n_samples),
        "gradient_variance": var,
        "mean_gradient_norm": float(np.mean(np.linalg.norm(grads, axis=1))),
        "max_abs_gradient": float(np.max(np.abs(grads))) if grads.size else 0.0,
        "vanishing": bool(var < vanish_threshold),
    }


def _fit_decay_rate(n_qubits, variances) -> tuple[float, bool]:
    """拟合 log2(方差) 对比特数的斜率；显著负斜率判为贫瘠高原。

    返回 ``(decay_rate, is_barren)``。``decay_rate`` 为每比特 log2 方差变化率
    （负 = 随 n 衰减）；``is_barren`` 当斜率 < -0.5（约每比特方差减半以上）。
    """
    n = np.asarray(n_qubits, dtype=float)
    var = np.asarray(variances, dtype=float)
    mask = var > 0
    if mask.sum() < 2:
        return 0.0, False
    slope = float(np.polyfit(n[mask], np.log2(var[mask]), 1)[0])
    return slope, bool(slope < -0.5)


def barren_plateau_scan(
    make_qfun: Callable[[int], tuple[Any, int]],
    n_qubits: list[int],
    *,
    n_samples: int = 20,
    seed: int = 0,
    param_scale: float = _TWO_PI,
) -> dict[str, Any]:
    """跨比特数扫描梯度方差，拟合衰减率判定贫瘠高原。

    Args:
        make_qfun: ``n -> (qfun, n_parameters)``，为 n 比特构造目标与参数数。
        n_qubits: 待扫描比特数列表。
        n_samples/seed/param_scale: 转发给 :func:`gradient_variance`。

    Returns:
        dict：``n_qubits``/``variances``/``mean_gradient_norms``/``decay_rate``/
        ``is_barren``。
    """
    variances, norms = [], []
    for n in n_qubits:
        qf, n_params = make_qfun(int(n))
        rep = gradient_variance(qf, n_params, n_samples=n_samples, seed=seed,
                                param_scale=param_scale)
        variances.append(rep["gradient_variance"])
        norms.append(rep["mean_gradient_norm"])
    decay_rate, is_barren = _fit_decay_rate(n_qubits, variances)
    return {
        "n_qubits": list(n_qubits),
        "variances": variances,
        "mean_gradient_norms": norms,
        "decay_rate": decay_rate,
        "is_barren": is_barren,
    }


def qfim_spectrum(
    qfun_or_state_fn: Any,
    params: Any,
    *,
    backend: Any = None,
    metric_eps: float = 1e-3,
    rank_tol: float = 1e-6,
) -> dict[str, Any]:
    """QFIM 特征谱与有效秩（平坦方向诊断）。

    ``qfun_or_state_fn`` 可为 :class:`QFun`（用其 ``statevector``）或
    ``state_fn(params)->态``。近零特征值 = 平坦方向（过参数化/不可辨识）。

    Returns:
        dict：``eigenvalues``（降序）/``effective_rank``（> ``rank_tol·max`` 的个数）/
        ``max_eigenvalue``/``min_eigenvalue``。
    """
    state_fn = getattr(qfun_or_state_fn, "statevector", qfun_or_state_fn)
    matrix = np.asarray(qfim(state_fn, np.asarray(params, dtype=float),
                             metric_eps=metric_eps, backend=backend))
    eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[::-1]  # 对称化后升序→降序
    max_eig = float(eig[0]) if eig.size else 0.0
    rank = int(np.sum(eig > rank_tol * max(max_eig, 1e-30)))
    return {
        "eigenvalues": eig,
        "effective_rank": max(rank, 0),
        "max_eigenvalue": max_eig,
        "min_eigenvalue": float(eig[-1]) if eig.size else 0.0,
    }

"""训练期贫瘠高原诊断（qml 成熟化 #7）。

把 metrics.trainability 的梯度方差信号 + qfim 谱接到真实训练目标（QFun），
而非固定 local-probe。
"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, PauliString, cx, ry
from aicir.qml import qfun
from aicir.qml import gradient_variance, barren_plateau_scan, qfim_spectrum
from aicir.qml.diagnostics import _fit_decay_rate


def _hea_qfun(n, seed=0):
    obs = Hamiltonian([PauliString("Z", n_qubits=n, qubits=[0])])

    @qfun(device="numpy", differential="psr", observable=obs)
    def f(t):
        gates = [ry(t[q], q) for q in range(n)]
        for q in range(n - 1):
            gates.append(cx(q + 1, [q]))
        gates += [ry(t[n + q], q) for q in range(n)]
        return Circuit(*gates, n_qubits=n)

    return f, 2 * n


def test_gradient_variance_report_structure():
    f, n_params = _hea_qfun(3)
    rep = gradient_variance(f, n_params, n_samples=12, seed=1)
    assert rep["n_parameters"] == n_params
    assert rep["n_samples"] == 12
    assert rep["gradient_variance"] > 0.0
    assert np.isfinite(rep["mean_gradient_norm"])
    assert isinstance(rep["vanishing"], bool)


def test_fit_decay_rate_detects_exponential_decay():
    # 合成指数衰减方差 var(n) = C·2^-n → log2 斜率 ≈ -1，判为 barren
    ns = np.array([2, 3, 4, 5, 6])
    var = 0.5 * 2.0 ** (-ns.astype(float))
    rate, is_barren = _fit_decay_rate(ns, var)
    assert rate < -0.5      # 明显负斜率（每比特方差减半）
    assert is_barren is True


def test_fit_decay_rate_flat_not_barren():
    # 方差随 n 基本不变 → 非 barren
    ns = np.array([2, 3, 4, 5])
    var = np.array([0.3, 0.31, 0.29, 0.30])
    rate, is_barren = _fit_decay_rate(ns, var)
    assert is_barren is False


def test_barren_plateau_scan_runs():
    def make(n):
        return _hea_qfun(n)

    scan = barren_plateau_scan(make, [2, 3, 4], n_samples=8, seed=2)
    assert scan["n_qubits"] == [2, 3, 4]
    assert len(scan["variances"]) == 3
    assert all(v >= 0 for v in scan["variances"])
    assert np.isfinite(scan["decay_rate"])
    assert isinstance(scan["is_barren"], bool)


def test_qfim_spectrum_psd_and_effective_rank():
    f, n_params = _hea_qfun(2)
    x = np.random.default_rng(0).uniform(-1, 1, size=n_params)
    spec = qfim_spectrum(f, x)
    eig = np.asarray(spec["eigenvalues"])
    assert eig.shape == (n_params,)
    assert np.all(eig >= -1e-6)                       # QFIM 半正定
    assert 1 <= spec["effective_rank"] <= n_params
    assert np.all(np.diff(eig) <= 1e-9)               # 降序


def test_qfun_statevector_helper():
    f, n_params = _hea_qfun(2)
    psi = f.statevector(np.zeros(n_params))
    assert psi.shape == (4,)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-6

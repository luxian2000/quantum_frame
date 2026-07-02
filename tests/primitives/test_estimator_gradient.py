"""Estimator 梯度 primitives 测试（NEXT.md §6 / QML todo 2.2）。

把 ``select_diff`` 接入 Estimator：``estimator.gradient(...)`` 以自身后端/shots/
噪声能力为依据自动选择梯度方法（不支持 Torch 时降级到 ``psr``/``fd``），
对模板电路的可训练参数求期望值的梯度。

基准：``ry(theta, 0)`` 下 ``<Z> = cos(theta)``，故梯度 ``= -sin(theta)``。
"""

import numpy as np

from aicir import Circuit, Hamiltonian, NumpyBackend, Parameter, ry
from aicir.primitives import (
    GradientResult,
    NoisyEstimator,
    ShotEstimator,
    StatevectorEstimator,
)
from aicir.noise import BitFlipChannel, NoiseModel

H_Z = Hamiltonian([("Z", 1.0)])


def _ry_template():
    return Circuit(ry(Parameter("t"), 0), n_qubits=1)


def test_statevector_gradient_matches_analytic():
    est = StatevectorEstimator(NumpyBackend())
    result = est.gradient(_ry_template(), H_Z, parameter_values=[0.3])
    assert isinstance(result, GradientResult)
    assert np.allclose(result.gradient, [-np.sin(0.3)])


def test_statevector_gradient_auto_degrades_to_psr_on_numpy():
    # numpy 后端不支持 Torch 自动微分 → auto 应降级到 psr（精确）。
    est = StatevectorEstimator(NumpyBackend())
    result = est.gradient(_ry_template(), H_Z, parameter_values=[0.3])
    assert result.method == "psr"


def test_explicit_method_overrides_auto():
    est = StatevectorEstimator(NumpyBackend())
    result = est.gradient(_ry_template(), H_Z, parameter_values=[0.3], method="fd")
    assert result.method == "fd"
    assert np.allclose(result.gradient, [-np.sin(0.3)], atol=1e-3)


def test_shot_estimator_gradient_selects_psr():
    # 有 shots 时，auto 仍降级到支持 shots 的 psr。
    est = ShotEstimator(NumpyBackend(), shots=8192)
    result = est.gradient(_ry_template(), H_Z, parameter_values=[0.3])
    assert result.method == "psr"
    assert np.allclose(result.gradient, [-np.sin(0.3)], atol=0.15)


def test_noisy_estimator_gradient_selects_noise_capable_psr():
    noise = NoiseModel().add_channel(BitFlipChannel(0, 0.0), after_gates=["ry"])
    est = NoisyEstimator(noise, NumpyBackend())
    result = est.gradient(_ry_template(), H_Z, parameter_values=[0.3])
    # 噪声路径下 auto 不可用 → psr（supports_noise=True）。
    assert result.method == "psr"
    # p=0 的 bit-flip 等于无噪声，梯度应回到解析值。
    assert np.allclose(result.gradient, [-np.sin(0.3)], atol=1e-6)

"""Target 下游集成测试（NEXT.md §3）。

把 ``Target`` 接入 primitives（按能力选 Estimator 执行路径）与 metrics
（从 Target 构造硬件 profile），减少散落的执行路径判断与硬编码门集。
"""

import numpy as np
import pytest

from aicir import BitFlipChannel, NoiseModel
from aicir.devices import Target
from aicir.metrics import DEFAULT_NATIVE_GATES, HardwareProfile
from aicir.primitives import (
    NoisyEstimator,
    ShotEstimator,
    StatevectorEstimator,
    estimator_for_target,
)


# --- estimator_for_target：按 Target 能力选择执行路径 ---

def test_statevector_target_gives_statevector_estimator():
    target = Target(n_qubits=2, supports_statevector=True)
    assert isinstance(estimator_for_target(target), StatevectorEstimator)


def test_shots_request_gives_shot_estimator():
    target = Target(n_qubits=2, supports_shots=True, supports_statevector=True)
    est = estimator_for_target(target, shots=512)
    assert isinstance(est, ShotEstimator)
    assert est.shots == 512


def test_sampling_only_target_defaults_to_shot_estimator():
    target = Target(n_qubits=2, supports_statevector=False, supports_shots=True)
    assert isinstance(estimator_for_target(target), ShotEstimator)


def test_noise_model_gives_noisy_estimator():
    target = Target(n_qubits=1, supports_density_matrix=True)
    noise = NoiseModel().add_channel(BitFlipChannel(0, 0.01), after_gates=["ry"])
    assert isinstance(estimator_for_target(target, noise_model=noise), NoisyEstimator)


def test_shots_on_non_shot_target_raises():
    target = Target(n_qubits=2, supports_shots=False, supports_statevector=True)
    with pytest.raises(ValueError):
        estimator_for_target(target, shots=100)


def test_noise_on_non_density_matrix_target_raises():
    target = Target(n_qubits=1, supports_density_matrix=False)
    noise = NoiseModel().add_channel(BitFlipChannel(0, 0.01), after_gates=["ry"])
    with pytest.raises(ValueError):
        estimator_for_target(target, noise_model=noise)


def test_target_with_no_execution_capability_raises():
    target = Target(
        n_qubits=1,
        supports_statevector=False,
        supports_shots=False,
        supports_density_matrix=False,
    )
    with pytest.raises(ValueError):
        estimator_for_target(target)


# --- HardwareProfile.from_target：metrics 从 Target 取门集与拓扑 ---

def test_hardware_profile_from_target():
    target = Target(n_qubits=3, basis_gates=("rx", "ry", "rz", "cx"), coupling_map=[(0, 1), (1, 2)])
    profile = HardwareProfile.from_target(target)
    assert set(profile.native_gates) == {"rx", "ry", "rz", "cx"}
    assert tuple(profile.coupling_map) == ((0, 1), (1, 2))


def test_hardware_profile_from_unrestricted_target_uses_defaults():
    target = Target(n_qubits=2)  # 空门集 + 全连接
    profile = HardwareProfile.from_target(target)
    assert tuple(profile.native_gates) == tuple(DEFAULT_NATIVE_GATES)
    assert tuple(profile.coupling_map) == ()

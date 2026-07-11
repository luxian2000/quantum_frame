"""AlgorithmResult 协议一致性测试（Phase 1：跨层结果词汇表统一）。

直接构造各 Result 数据类以规避 ``aicir.vqc.__init__`` 的 torch 门控与真实求解
开销；``aicir.protocols`` 本身零重依赖，isinstance 检查只验证成员是否存在。
"""

import numpy as np

from aicir.optimizer import OptimizationResult
from aicir.primitives import EstimateResult
from aicir.protocols import AlgorithmResult
from aicir.vqc.QAOA import QAOAResult
from aicir.vqc.SSVQE import SSVQEResult
from aicir.vqc.VQD import VQDResult
from aicir.vqc.VQE import VQEResult


def test_optimization_result_satisfies_algorithm_result():
    result = OptimizationResult(x=np.array([1.0]), fun=0.5, nit=1, nfev=1, success=True, message="ok")

    assert isinstance(result, AlgorithmResult)
    assert result.value == result.fun
    assert result.parameters is result.x
    assert result.metadata == {}


def test_vqe_result_satisfies_algorithm_result():
    result = VQEResult(
        energy=-1.0,
        parameters=np.array([0.1]),
        statevector=None,
        energy_history=[-0.5, -1.0],
    )

    assert isinstance(result, AlgorithmResult)
    assert result.value == -1.0
    assert result.history == [-0.5, -1.0]
    assert result.metadata == {}


def test_qaoa_result_satisfies_algorithm_result_and_parameters_is_gammas_betas_concat():
    gammas = np.array([0.1, 0.2])
    betas = np.array([0.3, 0.4])
    result = QAOAResult(
        energy=-2.0,
        gammas=gammas,
        betas=betas,
        statevector=None,
        energy_history=[-1.0, -2.0],
        parameters=np.concatenate([gammas, betas]),
    )

    assert isinstance(result, AlgorithmResult)
    assert result.value == -2.0
    assert result.history == [-1.0, -2.0]
    assert np.array_equal(result.parameters, np.concatenate([gammas, betas]))


def test_vqd_result_satisfies_algorithm_result_value_is_ground_state():
    result = VQDResult(
        energies=np.array([-1.0, -0.5]),
        parameters=np.zeros((2, 1)),
        statevectors=np.zeros((2, 2)),
        objective_histories=[[-0.8, -1.0], [-0.3, -0.5]],
    )

    assert isinstance(result, AlgorithmResult)
    # value 取基态（level 0，无重叠罚项）能量
    assert result.value == -1.0
    assert result.history == [-0.8, -1.0]
    assert result.metadata == {}


def test_ssvqe_result_satisfies_algorithm_result_value_is_weighted_cost():
    result = SSVQEResult(
        weighted_cost=-3.0,
        energies=np.array([-2.0, -1.0]),
        parameters=np.zeros((1, 1)),
        statevectors=np.zeros((2, 2)),
        cost_history=[-2.5, -3.0],
    )

    assert isinstance(result, AlgorithmResult)
    # value 取实际优化的加权代价（联合子空间目标），而非某单一本征态能量
    assert result.value == -3.0
    assert result.history == [-2.5, -3.0]
    assert result.metadata == {}


def test_estimate_result_satisfies_algorithm_result_and_energy_alias():
    result = EstimateResult(value=-1.0, variance=0.0, shots=None)

    assert isinstance(result, AlgorithmResult)
    assert result.energy == result.value
    assert result.parameters is None
    assert result.history is None

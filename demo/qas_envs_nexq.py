"""
预定义的量子架构搜索环境 (基于nexq库)

提供常用的环境配置：2量子比特、3量子比特等，支持有噪声和无噪声版本。
"""

import numpy as np
from qas_env_nexq import QuantumArchSearchEnv
from utils_nexq import (
    get_default_gates,
    get_default_observables,
    get_bell_state,
    get_ghz_state,
    get_w_state,
)


class BasicNQubitEnv(QuantumArchSearchEnv):
    """
    基础N量子比特环境。
    
    可以接受任意大小的目标态。
    """
    
    def __init__(
        self,
        target: np.ndarray,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
    ):
        """
        初始化N量子比特环境。
        
        Args:
            target: 目标量子态向量
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
        """
        n_qubits = int(np.log2(len(target)))
        assert 2**n_qubits == len(target), \
            f"目标态长度必须是2的幂，得到 {len(target)}"
        
        state_observables = get_default_observables(n_qubits)
        action_gates = get_default_gates(n_qubits)
        
        super().__init__(
            target=target,
            n_qubits=n_qubits,
            state_observables=state_observables,
            action_gates=action_gates,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
        )


class BasicTwoQubitEnv(BasicNQubitEnv):
    """
    基础2量子比特环境。
    
    默认目标态为Bell态 |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    
    def __init__(
        self,
        target: np.ndarray = None,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
    ):
        """
        初始化2量子比特环境。
        
        Args:
            target: 目标量子态向量，默认为Bell态
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
        """
        if target is None:
            target = get_bell_state()
        
        assert len(target) == 4, f"2量子比特目标态长度应为4，得到 {len(target)}"
        
        super().__init__(
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
        )


class BasicThreeQubitEnv(BasicNQubitEnv):
    """
    基础3量子比特环境。
    
    默认目标态为GHZ态 |GHZ⟩ = (|000⟩ + |111⟩)/√2
    """
    
    def __init__(
        self,
        target: np.ndarray = None,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
    ):
        """
        初始化3量子比特环境。
        
        Args:
            target: 目标量子态向量，默认为GHZ态
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
        """
        if target is None:
            target = get_ghz_state(3)
        
        assert len(target) == 8, f"3量子比特目标态长度应为8，得到 {len(target)}"
        
        super().__init__(
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
        )


class BasicFourQubitEnv(BasicNQubitEnv):
    """
    基础4量子比特环境。
    
    默认目标态为GHZ态 |GHZ⟩ = (|0000⟩ + |1111⟩)/√2
    """
    
    def __init__(
        self,
        target: np.ndarray = None,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 30,
    ):
        """
        初始化4量子比特环境。
        
        Args:
            target: 目标量子态向量，默认为GHZ态
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
        """
        if target is None:
            target = get_ghz_state(4)
        
        assert len(target) == 16, f"4量子比特目标态长度应为16，得到 {len(target)}"
        
        super().__init__(
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
        )


# 有噪声版本的环境

class NoisyNQubitEnv(QuantumArchSearchEnv):
    """
    有噪声的N量子比特环境。
    
    包含量子门和观测的噪声模拟。
    """
    
    def __init__(
        self,
        target: np.ndarray,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        """
        初始化有噪声的N量子比特环境。
        
        Args:
            target: 目标量子态向量
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
            error_rate: 噪声错误率（0到1之间）
        """
        n_qubits = int(np.log2(len(target)))
        assert 2**n_qubits == len(target), \
            f"目标态长度必须是2的幂，得到 {len(target)}"
        
        state_observables = get_default_observables(n_qubits)
        action_gates = get_default_gates(n_qubits)
        
        super().__init__(
            target=target,
            n_qubits=n_qubits,
            state_observables=state_observables,
            action_gates=action_gates,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
            error_observables=error_rate,
            error_gates=error_rate,
        )


class NoisyTwoQubitEnv(NoisyNQubitEnv):
    """
    有噪声的2量子比特环境。
    
    默认目标态为Bell态，包含噪声模拟。
    """
    
    def __init__(
        self,
        target: np.ndarray = None,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        """
        初始化有噪声的2量子比特环境。
        
        Args:
            target: 目标量子态向量，默认为Bell态
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
            error_rate: 噪声错误率
        """
        if target is None:
            target = get_bell_state()
        
        assert len(target) == 4, f"2量子比特目标态长度应为4，得到 {len(target)}"
        
        super().__init__(
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
            error_rate=error_rate,
        )


class NoisyThreeQubitEnv(NoisyNQubitEnv):
    """
    有噪声的3量子比特环境。
    
    默认目标态为GHZ态，包含噪声模拟。
    """
    
    def __init__(
        self,
        target: np.ndarray = None,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        """
        初始化有噪声的3量子比特环境。
        
        Args:
            target: 目标量子态向量，默认为GHZ态
            fidelity_threshold: 保真度阈值
            reward_penalty: 奖励惩罚系数
            max_timesteps: 最大时步数
            error_rate: 噪声错误率
        """
        if target is None:
            target = get_ghz_state(3)
        
        assert len(target) == 8, f"3量子比特目标态长度应为8，得到 {len(target)}"
        
        super().__init__(
            target=target,
            fidelity_threshold=fidelity_threshold,
            reward_penalty=reward_penalty,
            max_timesteps=max_timesteps,
            error_rate=error_rate,
        )

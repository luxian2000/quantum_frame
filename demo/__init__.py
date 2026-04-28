"""
量子架构搜索模块 (nexq库版本)

包含：
- qas_env_nexq: 核心环境类
- qas_envs_nexq: 预定义的环境
- utils_nexq: 工具函数
- qas_demo: 简单演示脚本
- qas_sb3_train: Stable-Baselines3 训练脚本
"""

from qas_env_nexq import QuantumArchSearchEnv
from qas_envs_nexq import (
    BasicNQubitEnv,
    BasicTwoQubitEnv,
    BasicThreeQubitEnv,
    BasicFourQubitEnv,
    NoisyNQubitEnv,
    NoisyTwoQubitEnv,
    NoisyThreeQubitEnv,
)
from utils_nexq import (
    get_default_gates,
    get_default_observables,
    get_bell_state,
    get_ghz_state,
    get_w_state,
)

__all__ = [
    # 环境
    'QuantumArchSearchEnv',
    'BasicNQubitEnv',
    'BasicTwoQubitEnv',
    'BasicThreeQubitEnv',
    'BasicFourQubitEnv',
    'NoisyNQubitEnv',
    'NoisyTwoQubitEnv',
    'NoisyThreeQubitEnv',
    # 工具
    'get_default_gates',
    'get_default_observables',
    'get_bell_state',
    'get_ghz_state',
    'get_w_state',
]

__version__ = '1.0.0'
__author__ = 'Quantum Computing Team'

"""
使用 Stable-Baselines3 进行量子架构搜索 (基于nexq库)

提供与原始项目类似的强化学习训练方案。
需要安装 stable-baselines3: pip install stable-baselines3
"""

import numpy as np
import sys
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# 导入环境
from qas_env_nexq import QuantumArchSearchEnv
from qas_envs_nexq import BasicTwoQubitEnv, BasicThreeQubitEnv, NoisyTwoQubitEnv, NoisyThreeQubitEnv
from utils_nexq import get_bell_state, get_ghz_state

try:
    from stable_baselines3 import A2C, PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("警告: stable-baselines3 未安装。请运行: pip install stable-baselines3")


class QuantumEnvWrapper:
    """
    将nexq环境包装为Gym兼容的接口。
    
    stable-baselines3 期望标准的gym接口，这个包装器提供该接口。
    """
    
    def __init__(self, env: QuantumArchSearchEnv):
        """
        初始化包装器。
        
        Args:
            env: 量子架构搜索环境
        """
        self.env = env
        
        # 定义观测空间和动作空间（为与gym兼容）
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        
        class Discrete:
            def __init__(self, n):
                self.n = n
        
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=self.env.observation_space_shape,
            dtype=np.float32
        )
        self.action_space = Discrete(len(self.env.action_gates))
    
    def reset(self):
        """重置环境"""
        return self.env.reset()
    
    def step(self, action):
        """执行动作"""
        return self.env.step(action)
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.env.render(mode=mode)
    
    def seed(self, seed=None):
        """设置随机种子"""
        return self.env.seed(seed)
    
    def close(self):
        """关闭环境"""
        pass


def create_wrapped_env(env_name: str = 'BasicTwoQubit') -> Optional[QuantumEnvWrapper]:
    """
    创建并包装环境。
    
    Args:
        env_name: 环境名称
                 'BasicTwoQubit', 'BasicThreeQubit',
                 'NoisyTwoQubit', 'NoisyThreeQubit'
    
    Returns:
        包装后的环境，或None（如果环境名称无效）
    """
    env_map = {
        'BasicTwoQubit': BasicTwoQubitEnv,
        'BasicThreeQubit': BasicThreeQubitEnv,
        'NoisyTwoQubit': NoisyTwoQubitEnv,
        'NoisyThreeQubit': NoisyThreeQubitEnv,
    }
    
    if env_name not in env_map:
        print(f"未知环境: {env_name}")
        return None
    
    env_class = env_map[env_name]
    env = env_class()
    return QuantumEnvWrapper(env)


def train_a2c(env_name: str = 'BasicTwoQubit', total_timesteps: int = 20000):
    """
    使用 A2C (Advantage Actor-Critic) 训练。
    
    Args:
        env_name: 环境名称
        total_timesteps: 总训练时步数
    """
    if not STABLE_BASELINES_AVAILABLE:
        print("错误: stable-baselines3 未安装")
        return
    
    print(f"\n{'='*60}")
    print(f"使用 A2C 训练 {env_name} 环境")
    print(f"{'='*60}")
    
    # 创建环境
    env = create_wrapped_env(env_name)
    if env is None:
        return
    
    print(f"环境信息:")
    print(f"  观测空间形状: {env.observation_space.shape}")
    print(f"  动作空间大小: {env.action_space.n}")
    
    # 创建模型
    print(f"\n创建 A2C 模型...")
    model = A2C(
        'MlpPolicy',
        env,
        gamma=0.99,
        learning_rate=0.0001,
        verbose=1,
        device='cpu',
    )
    
    # 训练
    print(f"\n开始训练 ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model_path = f'./models/a2c_{env_name}'
    print(f"\n保存模型到 {model_path}")
    model.save(model_path)
    
    # 评估
    print(f"\n评估模型...")
    evaluate_agent(model, env, num_episodes=10, render=False)


def train_ppo(env_name: str = 'BasicTwoQubit', total_timesteps: int = 20000):
    """
    使用 PPO (Proximal Policy Optimization) 训练。
    
    Args:
        env_name: 环境名称
        total_timesteps: 总训练时步数
    """
    if not STABLE_BASELINES_AVAILABLE:
        print("错误: stable-baselines3 未安装")
        return
    
    print(f"\n{'='*60}")
    print(f"使用 PPO 训练 {env_name} 环境")
    print(f"{'='*60}")
    
    # 创建环境
    env = create_wrapped_env(env_name)
    if env is None:
        return
    
    print(f"环境信息:")
    print(f"  观测空间形状: {env.observation_space.shape}")
    print(f"  动作空间大小: {env.action_space.n}")
    
    # 创建模型
    print(f"\n创建 PPO 模型...")
    model = PPO(
        'MlpPolicy',
        env,
        gamma=0.99,
        n_epochs=4,
        clip_range=0.2,
        learning_rate=0.0001,
        verbose=1,
        device='cpu',
    )
    
    # 训练
    print(f"\n开始训练 ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model_path = f'./models/ppo_{env_name}'
    print(f"\n保存模型到 {model_path}")
    model.save(model_path)
    
    # 评估
    print(f"\n评估模型...")
    evaluate_agent(model, env, num_episodes=10, render=False)


def evaluate_agent(model, env: QuantumEnvWrapper, num_episodes: int = 10, render: bool = True):
    """
    评估训练好的代理。
    
    Args:
        model: 训练好的模型
        env: 环境
        num_episodes: 评估的剧集数
        render: 是否渲染
    """
    print(f"\n评估 {num_episodes} 个剧集...")
    
    episode_rewards = []
    episode_fidelities = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_fidelity = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_fidelity = info['fidelity']
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_fidelities.append(episode_fidelity)
        
        print(f"  Episode {episode+1}: reward={episode_reward:.4f}, fidelity={episode_fidelity:.4f}")
    
    print(f"\n评估结果:")
    print(f"  平均奖励: {np.mean(episode_rewards):.4f}")
    print(f"  平均保真度: {np.mean(episode_fidelities):.4f}")
    print(f"  最大保真度: {np.max(episode_fidelities):.4f}")


def main():
    """
    主函数 - 演示不同的训练方法。
    """
    print("\n╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "量子架构搜索 - Stable-Baselines3 集成".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    if not STABLE_BASELINES_AVAILABLE:
        print("请先安装 stable-baselines3:")
        print("  pip install stable-baselines3")
        return
    
    # 演示 1: A2C 训练 (2量子比特)
    print("\n[演示1] 使用 A2C 训练 2量子比特环境")
    train_a2c('BasicTwoQubit', total_timesteps=10000)
    
    # 演示 2: PPO 训练 (2量子比特)
    print("\n[演示2] 使用 PPO 训练 2量子比特环境")
    train_ppo('BasicTwoQubit', total_timesteps=10000)
    
    # 演示 3: PPO 训练 (3量子比特，可选)
    print("\n[演示3] 使用 PPO 训练 3量子比特环境")
    print("(这可能需要更长时间...)")
    train_ppo('BasicThreeQubit', total_timesteps=15000)
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

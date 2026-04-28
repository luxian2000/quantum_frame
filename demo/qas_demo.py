"""
量子架构搜索示例脚本 (基于nexq库)

展示如何使用nexq库实现的量子架构搜索环境，以及一个简单的强化学习代理。
"""

import numpy as np
import sys
from typing import List, Tuple
from collections import deque

# 导入环境
from qas_env_nexq import QuantumArchSearchEnv
from qas_envs_nexq import BasicTwoQubitEnv, BasicThreeQubitEnv, NoisyTwoQubitEnv
from utils_nexq import get_bell_state, get_ghz_state


class SimpleRLAgent:
    """
    简单的强化学习代理 - 用于演示。
    
    使用Q学习进行求解（仅用于演示，实际应用中推荐使用Stable-Baselines3）。
    """
    
    def __init__(self, num_actions: int, learning_rate: float = 0.01, gamma: float = 0.99):
        """
        初始化代理。
        
        Args:
            num_actions: 可选的动作数
            learning_rate: 学习率
            gamma: 折扣因子
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.episode_count = 0
    
    def get_action(self, observation: np.ndarray, epsilon: float = 0.1) -> int:
        """
        使用ε-greedy策略选择动作。
        
        Args:
            observation: 当前观测
            epsilon: 探索概率
            
        Returns:
            选择的动作索引
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            # 简单启发式：倾向于选择不同的动作以探索
            return int(np.random.choice(self.num_actions, p=np.ones(self.num_actions)/self.num_actions))
    
    def update(self, reward: float):
        """
        更新代理（占位符，实际应用使用更复杂的学习算法）。
        
        Args:
            reward: 获得的奖励
        """
        pass


def run_episode(
    env: QuantumArchSearchEnv,
    agent: SimpleRLAgent,
    epsilon: float = 0.1,
    verbose: bool = False,
) -> Tuple[float, List[int], float]:
    """
    运行单个训练剧集。
    
    Args:
        env: 量子环境
        agent: RL代理
        epsilon: 探索系数
        verbose: 是否打印详细信息
        
    Returns:
        (总奖励, 动作序列, 最终保真度)
    """
    observation = env.reset()
    total_reward = 0
    actions = []
    final_fidelity = 0
    
    done = False
    step = 0
    
    while not done:
        # 代理选择动作
        action = agent.get_action(observation, epsilon=epsilon)
        actions.append(action)
        
        # 环境执行动作
        observation, reward, done, info = env.step(action)
        total_reward += reward
        final_fidelity = info['fidelity']
        
        if verbose and step % 5 == 0:
            print(f"  Step {step}: action={action}, reward={reward:.4f}, fidelity={final_fidelity:.4f}")
        
        step += 1
    
    return total_reward, actions, final_fidelity


def train_agent(
    env: QuantumArchSearchEnv,
    agent: SimpleRLAgent,
    num_episodes: int = 100,
    initial_epsilon: float = 0.5,
    final_epsilon: float = 0.01,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    训练强化学习代理。
    
    Args:
        env: 量子环境
        agent: RL代理
        num_episodes: 训练剧集数
        initial_epsilon: 初始探索系数
        final_epsilon: 最终探索系数
        verbose: 是否打印详细信息
        
    Returns:
        (总奖励列表, 保真度列表)
    """
    rewards_history = []
    fidelity_history = []
    best_fidelity = 0
    
    print(f"\n{'='*60}")
    print(f"开始训练 - 目标: {num_episodes} 剧集")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        # 计算当前epsilon
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * episode / num_episodes
        
        # 运行剧集
        total_reward, actions, final_fidelity = run_episode(env, agent, epsilon=epsilon, verbose=False)
        
        # 更新历史
        rewards_history.append(total_reward)
        fidelity_history.append(final_fidelity)
        
        # 更新最好保真度
        if final_fidelity > best_fidelity:
            best_fidelity = final_fidelity
        
        # 定期打印进度
        if (episode + 1) % (num_episodes // 10) == 0 or episode == 0:
            avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else total_reward
            avg_fidelity = np.mean(fidelity_history[-10:]) if len(fidelity_history) >= 10 else final_fidelity
            
            print(f"Episode {episode+1:3d}/{num_episodes}: "
                  f"reward={total_reward:7.4f}, "
                  f"avg_reward={avg_reward:7.4f}, "
                  f"fidelity={final_fidelity:.4f}, "
                  f"best={best_fidelity:.4f}, "
                  f"ε={epsilon:.4f}")
    
    print(f"{'='*60}")
    print(f"训练完成! 最佳保真度: {best_fidelity:.4f}")
    print(f"{'='*60}\n")
    
    return rewards_history, fidelity_history


def demonstrate_environment():
    """
    演示环境的基本功能。
    """
    print("\n" + "="*60)
    print("量子架构搜索环境演示 (基于nexq库)")
    print("="*60)
    
    # 创建环境
    print("\n[1] 创建2量子比特环境...")
    env = BasicTwoQubitEnv(
        fidelity_threshold=0.95,
        reward_penalty=0.01,
        max_timesteps=20
    )
    print(f"环境: {env}")
    print(f"可选动作数: {len(env.action_gates)}")
    print(f"观测量数: {len(env.state_observables)}")
    
    # 重置环境
    print("\n[2] 重置环境并获取初始观测...")
    obs = env.reset()
    print(f"初始观测: {obs}")
    
    # 执行几步
    print("\n[3] 执行几个动作...")
    for i in range(3):
        action = i % len(env.action_gates)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.4f}, fidelity={info['fidelity']:.4f}, done={done}")
    
    # 渲染电路
    print("\n[4] 打印当前电路:")
    print(env.render(mode='ansi'))
    
    print("="*60 + "\n")


def main():
    """
    主函数 - 运行示例。
    """
    print("\n╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "量子架构搜索 - nexq库实现版本".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # 1. 演示环境
    demonstrate_environment()
    
    # 2. 训练代理 - 2量子比特
    print("\n[阶段1] 训练 2量子比特环境\n")
    env_2qb = BasicTwoQubitEnv(max_timesteps=20)
    agent_2qb = SimpleRLAgent(num_actions=len(env_2qb.action_gates))
    rewards_2qb, fidelity_2qb = train_agent(
        env_2qb,
        agent_2qb,
        num_episodes=50,
        initial_epsilon=0.5,
        final_epsilon=0.01,
    )
    
    # 3. 训练代理 - 3量子比特（可选，耗时较长）
    print("\n[阶段2] 训练 3量子比特环境\n")
    env_3qb = BasicThreeQubitEnv(max_timesteps=30)
    agent_3qb = SimpleRLAgent(num_actions=len(env_3qb.action_gates))
    rewards_3qb, fidelity_3qb = train_agent(
        env_3qb,
        agent_3qb,
        num_episodes=30,  # 减少剧集数以节省时间
        initial_epsilon=0.5,
        final_epsilon=0.01,
    )
    
    # 4. 显示结果
    print("\n" + "="*60)
    print("训练结果总结")
    print("="*60)
    
    print(f"\n2量子比特环境:")
    print(f"  最终保真度: {fidelity_2qb[-1]:.4f}")
    print(f"  最高保真度: {max(fidelity_2qb):.4f}")
    print(f"  平均保真度: {np.mean(fidelity_2qb):.4f}")
    
    print(f"\n3量子比特环境:")
    print(f"  最终保真度: {fidelity_3qb[-1]:.4f}")
    print(f"  最高保真度: {max(fidelity_3qb):.4f}")
    print(f"  平均保真度: {np.mean(fidelity_3qb):.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

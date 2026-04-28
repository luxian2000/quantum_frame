# 量子架构搜索 - nexq库实现

基于 **nexq 量子模拟器库**实现的 **强化学习量子架构搜索系统**，将 [quantum-arch-search](https://github.com/qdevpsi3/quantum-arch-search) 项目从 Cirq 迁移到 nexq。

## 项目结构

```
demo/
├── README.md                    # 本文件
├── utils_nexq.py               # 工具函数（门、观测量、目标态）
├── qas_env_nexq.py            # 核心环境类
├── qas_envs_nexq.py           # 预定义环境（2量子比特、3量子比特等）
├── qas_demo.py                # 简单演示脚本
└── qas_sb3_train.py           # Stable-Baselines3 集成训练脚本
```

## 功能特性

### 1. **量子架构搜索环境** (`qas_env_nexq.py`)

- 基于 nexq 库的量子电路模拟
- 标准的 Gym 环境接口（`reset()`, `step()`, `render()`）
- 支持可配置的：
  - 目标量子态
  - 可选的量子门
  - 观测量集合
  - 保真度阈值
  - 奖励结构

### 2. **工具函数** (`utils_nexq.py`)

提供常用的量子态和门操作：

- `get_default_gates(n_qubits)`: 默认量子门集合
  - 单量子门：RZ, X, Y, Z, H
  - 双量子门：CNOT

- `get_default_observables(n_qubits)`: 默认观测量（X, Y, Z 泡利算子）

- `get_bell_state()`: Bell 态 |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

- `get_ghz_state(n_qubits)`: GHZ 态（多量子比特纠缠态）

- `get_w_state(n_qubits)`: W 态（多量子比特叠加态）

### 3. **预定义环境** (`qas_envs_nexq.py`)

#### 基础环境

- **BasicTwoQubitEnv**: 2量子比特环境（默认目标：Bell态）
- **BasicThreeQubitEnv**: 3量子比特环境（默认目标：GHZ态）
- **BasicFourQubitEnv**: 4量子比特环境
- **BasicNQubitEnv**: 任意N量子比特环境

#### 有噪声环境

- **NoisyTwoQubitEnv**: 2量子比特环境 + 噪声
- **NoisyThreeQubitEnv**: 3量子比特环境 + 噪声
- **NoisyNQubitEnv**: 任意N量子比特环境 + 噪声

### 4. **训练脚本**

#### 简单演示 (`qas_demo.py`)

使用简单的 Q学习代理进行演示：

```bash
python qas_demo.py
```

- 展示环境的基本功能
- 演示代理如何与环境交互
- 训练 2量子比特和 3量子比特环境

#### Stable-Baselines3 集成 (`qas_sb3_train.py`)

使用专业的强化学习算法：

```bash
# 安装依赖
pip install stable-baselines3

# 运行训练
python qas_sb3_train.py
```

支持的算法：
- **A2C** (Advantage Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **DQN** (Deep Q-Network)

## 使用示例

### 1. 基本使用

```python
from qas_envs_nexq import BasicTwoQubitEnv
from qas_demo import SimpleRLAgent

# 创建环境
env = BasicTwoQubitEnv(
    fidelity_threshold=0.95,
    reward_penalty=0.01,
    max_timesteps=20
)

# 创建代理
agent = SimpleRLAgent(num_actions=len(env.action_gates))

# 训练
for episode in range(100):
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        print(f"Action: {action}, Reward: {reward:.4f}, Fidelity: {info['fidelity']:.4f}")
```

### 2. 使用自定义目标态

```python
import numpy as np
from qas_envs_nexq import BasicNQubitEnv

# 自定义目标态 (例如：3量子比特)
target_state = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)  # GHZ 态

# 创建环境
env = BasicNQubitEnv(target=target_state, max_timesteps=30)

# 使用环境...
```

### 3. 有噪声环境

```python
from qas_envs_nexq import NoisyTwoQubitEnv

# 创建有噪声的环境
env = NoisyTwoQubitEnv(
    error_rate=0.001,  # 1% 错误率
    max_timesteps=20
)

# 使用环境...
```

### 4. 使用 Stable-Baselines3

```python
from stable_baselines3 import PPO
from qas_sb3_train import QuantumEnvWrapper
from qas_envs_nexq import BasicTwoQubitEnv

# 创建并包装环境
base_env = BasicTwoQubitEnv()
env = QuantumEnvWrapper(base_env)

# 创建 PPO 模型
model = PPO('MlpPolicy', env, learning_rate=0.0001)

# 训练
model.learn(total_timesteps=20000)

# 使用
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## 环境 API

### 环境初始化

```python
env = QuantumArchSearchEnv(
    target=target_state,           # 目标量子态
    n_qubits=2,                    # 量子比特数
    state_observables=observables, # 观测量列表
    action_gates=gates,            # 可选的量子门列表
    fidelity_threshold=0.95,       # 保真度阈值
    reward_penalty=0.01,           # 奖励惩罚
    max_timesteps=20,              # 最大时步数
)
```

### 环境交互

```python
# 重置环境
observation = env.reset()  # 返回初始观测值

# 执行动作
observation, reward, done, info = env.step(action)
# observation: 新的观测值 (numpy 数组)
# reward: 获得的奖励 (float)
# done: 是否终止 (bool)
# info: 额外信息 (dict)
#   - 'fidelity': 当前保真度
#   - 'circuit': 当前电路对象
#   - 'timestep': 当前时步

# 渲染电路
env.render()  # 打印当前电路

# 获取电路的幺正矩阵
U = env.get_circuit_unitary()

# 获取电路的最终态
state = env.get_circuit_state()
```

## 关键概念

### 保真度 (Fidelity)

代理生成的量子态与目标态的相似度：

$$F = |\langle \psi_{target} | \psi_{generated} \rangle|^2$$

- 范围：0 到 1
- 值为 1：完全相同
- 值为 0：正交

### 奖励结构

```python
if fidelity > threshold:
    reward = fidelity - penalty  # 正奖励
else:
    reward = -penalty             # 负奖励
```

### 观测值

环境状态由所有观测量在当前量子态上的期望值组成：

$$\langle O_i \rangle = \langle \psi | O_i | \psi \rangle$$

## 性能对比

与原始 Cirq 版本相比：

| 特性 | nexq 版本 | Cirq 版本 |
|------|----------|---------|
| 后端选择 | 支持 (NumPy, Torch, NPU) | 仅 NumPy |
| 性能 | 优化的矩阵乘法 | 标准实现 |
| 易用性 | 简洁的 API | 更冗长 |
| 扩展性 | 支持自定义后端 | 固定后端 |

## 依赖

- `numpy`: 数值计算
- `nexq`: 量子模拟（位于 `quantum_frame` 中）
- `stable-baselines3` (可选): 专业 RL 算法

## 安装与运行

### 1. 安装依赖

```bash
# 基础依赖
pip install numpy

# 可选：安装 Stable-Baselines3
pip install stable-baselines3
```

### 2. 运行简单演示

```bash
cd /Users/luxian/GitSpace/quantum_frame/demo
python qas_demo.py
```

### 3. 运行 Stable-Baselines3 训练

```bash
cd /Users/luxian/GitSpace/quantum_frame/demo
python qas_sb3_train.py
```

## 扩展与定制

### 添加新的环境

```python
from qas_env_nexq import QuantumArchSearchEnv
from utils_nexq import get_default_gates, get_default_observables

class MyCustomEnv(QuantumArchSearchEnv):
    def __init__(self, target):
        n_qubits = int(np.log2(len(target)))
        gates = get_default_gates(n_qubits)
        observables = get_default_observables(n_qubits)
        
        super().__init__(
            target=target,
            n_qubits=n_qubits,
            state_observables=observables,
            action_gates=gates,
        )
```

### 添加噪声模型

修改 `_add_noise_to_circuit()` 方法以实现更复杂的噪声模型。

### 使用不同的计算后端

```python
from nexq.channel.backends.torch_backend import TorchBackend

env = BasicTwoQubitEnv()
env.backend = TorchBackend()  # 使用 Torch 后端
```

## 原始项目参考

- **论文**: Quantum Architecture Search via Deep Reinforcement Learning
- **原始项目**: https://github.com/qdevpsi3/quantum-arch-search
- **原始库**: Google Cirq (https://quantumai.google/cirq)

## 许可证

与 quantum_frame 项目保持一致。

## 联系与支持

有关 nexq 库的问题，请参考 quantum_frame 项目文档。

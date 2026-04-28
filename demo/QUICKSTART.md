# 快速开始指南

## 什么是这个项目？

这是将 **quantum-arch-search** 项目从 Cirq 迁移到 **nexq** 库的实现。

- **原始项目**: 使用强化学习自动设计量子电路以制备目标量子态
- **新实现**: 基于高性能的 nexq 量子模拟器库
- **优势**: 更好的性能、更灵活的后端支持、更简洁的 API

## 核心文件说明

| 文件 | 说明 |
|------|------|
| `utils_nexq.py` | 工具函数（量子门、观测量、目标态等） |
| `qas_env_nexq.py` | 核心环境类 - 实现量子架构搜索环境 |
| `qas_envs_nexq.py` | 预定义环境（2量子比特、3量子比特等） |
| `qas_demo.py` | 简单演示脚本（无需依赖库）|
| `qas_sb3_train.py` | 专业训练脚本（需要 Stable-Baselines3） |
| `test_qas.py` | 自动化测试脚本 |
| `README.md` | 详细文档 |

## 5分钟快速开始

### 1. 验证安装

```bash
cd /Users/luxian/GitSpace/quantum_frame/demo
python test_qas.py
```

✅ 所有测试应该通过

### 2. 运行演示

```bash
python qas_demo.py
```

这会：
- 展示环境的基本功能
- 训练 2量子比特环境 (50 剧集)
- 训练 3量子比特环境 (30 剧集)

### 3. 简单的代码示例

```python
from qas_envs_nexq import BasicTwoQubitEnv
from qas_demo import SimpleRLAgent

# 创建环境
env = BasicTwoQubitEnv()

# 创建代理
agent = SimpleRLAgent(num_actions=len(env.action_gates))

# 训练循环
for episode in range(100):
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        if episode % 10 == 0:
            print(f"Fidelity: {info['fidelity']:.4f}")
```

## 进阶：使用 Stable-Baselines3

### 安装

```bash
pip install stable-baselines3
```

### 训练

```bash
python qas_sb3_train.py
```

这会使用 PPO 和 A2C 算法进行训练。

## 创建自定义环境

```python
import numpy as np
from qas_envs_nexq import BasicNQubitEnv

# 定义目标态 (例如：3量子比特的 |000⟩ 态)
target = np.zeros(8)
target[0] = 1.0

# 创建环境
env = BasicNQubitEnv(target=target, max_timesteps=30)

# 使用环境
obs = env.reset()
for i in range(10):
    action = i % len(env.action_gates)
    obs, reward, done, info = env.step(action)
    print(f"Step {i}: Fidelity = {info['fidelity']:.4f}")
```

## 理解输出

### 观测值 (Observation)
- 一个向量，包含所有可观测量的期望值
- 长度 = 观测量数量 (默认为 3*n_qubits)

### 奖励 (Reward)
```
if 保真度 > 阈值:
    奖励 = 保真度 - 惩罚系数
else:
    奖励 = -惩罚系数
```

### 信息 (Info)
包含：
- `fidelity`: 当前保真度 (0-1)
- `circuit`: 当前电路对象
- `timestep`: 当前时步

## 支持的环境

### 基础环境（无噪声）
- `BasicTwoQubitEnv` - 2量子比特（默认：Bell态）
- `BasicThreeQubitEnv` - 3量子比特（默认：GHZ态）
- `BasicFourQubitEnv` - 4量子比特（默认：GHZ态）
- `BasicNQubitEnv` - N量子比特（自定义）

### 有噪声环境
- `NoisyTwoQubitEnv` - 2量子比特 + 噪声
- `NoisyThreeQubitEnv` - 3量子比特 + 噪声
- `NoisyNQubitEnv` - N量子比特 + 噪声

## 关键 API

### 环境初始化
```python
env = QuantumArchSearchEnv(
    target=target_state,           # 目标态向量
    n_qubits=2,                    # 量子比特数
    state_observables=observables, # 观测量列表
    action_gates=gates,            # 可选的量子门
    fidelity_threshold=0.95,       # 保真度阈值
    reward_penalty=0.01,           # 奖励惩罚
    max_timesteps=20,              # 最大时步数
)
```

### 环境交互
```python
# 重置
obs = env.reset()

# 执行动作
obs, reward, done, info = env.step(action)

# 获取当前态的幺正矩阵
U = env.get_circuit_unitary()

# 获取当前态的量子态向量
psi = env.get_circuit_state()

# 打印当前电路
env.render()
```

## 常见问题

### Q: 保真度总是很低？
A: 这是正常的。强化学习需要时间学习。运行更多剧集或调整超参数。

### Q: 如何改变目标态？
A: 使用 `BasicNQubitEnv` 并传入自定义的 `target` 参数。

### Q: 如何使用不同的计算后端？
A: 
```python
from nexq.channel.backends.torch_backend import TorchBackend

env = BasicTwoQubitEnv()
env.backend = TorchBackend()  # 使用 Torch GPU 后端
```

### Q: 支持多少个量子比特？
A: 理论上任意数量，但实际受限于计算资源。
- 2-3 量子比特：非常快
- 4-5 量子比特：较快
- 6+ 量子比特：慢（指数增长）

## 项目结构

```
quantum_frame/demo/
├── __init__.py                 # 包初始化
├── utils_nexq.py               # 工具函数
├── qas_env_nexq.py            # 核心环境
├── qas_envs_nexq.py           # 预定义环境
├── qas_demo.py                # 简单演示
├── qas_sb3_train.py           # SB3 训练
├── test_qas.py                # 测试套件
├── QUICKSTART.md              # 本文件
├── README.md                  # 详细文档
└── models/                    # 保存的模型（自动创建）
```

## 下一步

1. ✅ 阅读本文档 (QUICKSTART.md)
2. ✅ 运行 `test_qas.py` 验证安装
3. ✅ 运行 `qas_demo.py` 查看演示
4. 📖 阅读 `README.md` 了解详细信息
5. 🚀 开始使用自己的环境和代理！

## 许可证与参考

- 原始论文: *Quantum Architecture Search via Deep Reinforcement Learning*
- 原始项目: https://github.com/qdevpsi3/quantum-arch-search
- 迁移实现: 使用 nexq 量子库

---

**提示**: 有问题？查看 `README.md` 中的更多详细信息！

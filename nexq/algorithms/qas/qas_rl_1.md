# nexq.algorithms.qas.qas_rl_1.py

本目录包含量子架构搜索（Quantum Architecture Search，QAS）相关工具，核心训练脚本为 `qas_rl_1.py`。

---

## 依赖

| 必需          | 可选（RL 训练需要）   |
| ------------- | --------------------- |
| `numpy`     | `gymnasium`         |
| nexq 内部模块 | `stable-baselines3` |

安装可选依赖：

```bash
pip install gymnasium stable-baselines3
```

---

## 模块功能概览

`qas_rl_1.py` 实现了以下组件：

| 类 / 函数                      | 说明                                                          |
| ------------------------------ | ------------------------------------------------------------- |
| `QuantumArchSearchEnvCore`   | 核心量子环境（无框架依赖），支持 reset / step / build_circuit |
| `QuantumArchSearchGymEnv`    | Gymnasium 封装，兼容 stable-baselines3                        |
| `TrainConfig`                | 训练配置 dataclass                                            |
| `run_training(config)`       | 完整的训练 + 评估流程入口                                     |
| `create_core_env(config)`    | 根据预设名称构造核心环境                                      |
| `get_default_gates(n)`       | 返回默认动作门集合（RZ / Pauli / H / CNOT）                   |
| `get_default_observables(n)` | 返回默认观测量（每比特 X/Y/Z）                                |
| `get_bell_state()`           | Bell 态\|Φ+⟩                                                |
| `get_ghz_state(n)`           | n 比特 GHZ 态                                                 |

---

## 作为 Python 库使用

### 最简示例

```python
from nexq.algorithms.qas import TrainConfig, run_training

config = TrainConfig(
    env_name="basic2",      # 'basic2'=Bell(2 qubits), 'basic3'=GHZ(3 qubits)
    algo="ppo",             # 'ppo' | 'a2c' | 'dqn'
    total_timesteps=10000,
    fidelity_threshold=0.95,
    reward_penalty=0.01,
    max_timesteps=20,
    seed=42,
    eval_episodes=10,
    save_path=None,         # 若不为 None 则保存模型，如 "outputs/qas_ppo"
)

metrics = run_training(config)
print(metrics)
# {'reward_mean': ..., 'reward_max': ..., 'fidelity_mean': ..., 'fidelity_max': ...}
```

### 只使用核心环境（无 Gymnasium）

```python
from nexq.algorithms.qas.qas_rl_1 import (
    QuantumArchSearchEnvCore,
    get_bell_state,
    get_default_gates,
    get_default_observables,
)

n_qubits = 2
core = QuantumArchSearchEnvCore(
    target=get_bell_state(),
    n_qubits=n_qubits,
    state_observables=get_default_observables(n_qubits),
    action_gates=get_default_gates(n_qubits),
)

obs = core.reset(seed=0)
obs, reward, done, info = core.step(4)   # action index 4 = Hadamard on qubit 0
print(f"fidelity={info['fidelity']:.4f}, reward={reward:.4f}")
```

### 保存训练好的模型

```python
config = TrainConfig(
    env_name="basic3",
    algo="ppo",
    total_timesteps=50000,
    save_path="outputs/qas_ghz_ppo",
)
metrics = run_training(config)
# 模型保存至 outputs/qas_ghz_ppo.zip（stable-baselines3 格式）
```

---

## 命令行使用

```bash
# 在 quantum_frame/ 目录下运行
python -m nexq.algorithms.qas.qas_rl_1 --env basic2 --algo ppo --timesteps 10000
```

全部参数：

```
--env             {basic2, basic3}       环境预设（默认 basic2）
--algo            {ppo, a2c, dqn}        RL 算法（默认 ppo）
--timesteps       INT                    总训练步数（默认 10000）
--fidelity-threshold FLOAT              成功保真度阈值（默认 0.95）
--reward-penalty  FLOAT                 每步惩罚（默认 0.01）
--max-timesteps   INT                   每 episode 最大步数（默认 20）
--seed            INT                   随机种子（默认 42）
--eval-episodes   INT                   评估 episode 数（默认 10）
--save-path       PATH                  模型保存路径（可选）
```

示例：

```bash
# 使用 A2C 搜索 GHZ 线路，保存模型
python -m nexq.algorithms.qas.qas_rl_1 \
    --env basic3 \
    --algo a2c \
    --timesteps 30000 \
    --fidelity-threshold 0.99 \
    --save-path outputs/qas_a2c_ghz
```

---

## 环境预设说明

| 预设名     | 目标态         | 量子比特数 | 动作空间大小        |
| ---------- | -------------- | ---------- | ------------------- |
| `basic2` | Bell 态\|Φ+⟩ | 2          | 12（每比特 6 种门） |
| `basic3` | GHZ 态         | 3          | 18（每比特 6 种门） |

每个动作对应以下门之一（每比特）：

- `RZ(π/4)` — 绕 Z 轴旋转 45°
- `Pauli-X / Y / Z` — Pauli 门
- `Hadamard (H)` — 超位置门
- `CNOT` — 受控非门（目标为下一比特，循环）

---

## 参考

- 原始论文：[arXiv:2104.07715](https://arxiv.org/abs/2104.07715)
- 原始项目：[quantum-arch-search](https://github.com/qdevpsi3/quantum-arch-search)

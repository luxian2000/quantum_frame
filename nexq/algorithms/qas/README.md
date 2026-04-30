# QAS 模块说明

`nexq.algorithms.qas` 是量子构架搜索（Quantum Architecture Search, QAS）模块。

该模块用于在给定目标量子态时，自动搜索量子线路结构。当前仓库中可直接使用的强化学习量子态制备实现主要包括：

- `CRLQAS.py`：课程学习 + DDQN + Adam-SPSA 的量子架构搜索（面向哈密顿量能量最小化）
- `PPR_DQL.py`：基于 nexq 状态演化实现的 PPR-DQL（Probabilistic Policy Reuse with Deep Q-Learning）
- `PPO_RB.py`：Trust Region-based PPO with Rollback 版本的量子架构搜索

注意：`qas` 目录中的 `state_qas.py` 已删除，当前文档仅覆盖现有可用实现。

## 1. 已提供能力

- 目标：输入 `State`，输出 `Circuit`
- 方法：基于强化学习逐步向空线路追加量子门
- 奖励：当保真度超过阈值时给正奖励，否则每步惩罚
- 约束：禁止 `unitary` 门，动作门需要是 `nexq/core/gates.py` 支持的门
- 导出：搜索得到的 `Circuit` 可继续导出为 OpenQASM 2.0 / 3.0

## 2. 主要接口

- `PPORollbackConfig`：PPO-RB 超参数配置
- `ppo_rb_qas(target_density_matrix, epsilon, config=None)`：训练并返回 `(theta, circuit)`
- `AdamSPSAConfig`：CRLQAS 中 Adam-SPSA 参数优化器配置
- `CRLQASConfig`：CRLQAS 超参数配置
- `train_crlqas(hamiltonian, config=None)`：训练并返回 `CRLQASResult`
- `crlqas(hamiltonian, config=None)`：返回 `(circuit, minimum_energy)`
- `PPRDQLConfig`：PPR-DQL 超参数配置
- `train_ppr_dql(state, config=None, policy_library=None)`：训练并返回包含 `Circuit`、策略和保真度信息的结果对象
- `ppr_dql_state_to_circuit(state, config=None, policy_library=None)`：使用 PPR-DQL 搜索并直接返回 `Circuit`

可通过以下方式导入：

```python
from nexq.algorithms.qas import PPRDQLConfig, ppr_dql_state_to_circuit, train_ppr_dql
from nexq.algorithms.qas import AdamSPSAConfig, CRLQASConfig, crlqas, train_crlqas
from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
```

## 3. 依赖

运行 `CRLQAS.py`、`PPR_DQL.py` 和 `PPO_RB.py` 都需要可用的 `torch`。

## 4. 使用方法：PPO_RB

`PPO_RB` 的输入是目标密度矩阵，输出是策略参数 `theta` 与搜索得到的 `Circuit`。

### 4.1 输入参数（`ppo_rb_qas`）

函数签名：`ppo_rb_qas(target_density_matrix, epsilon, config=None)`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `target_density_matrix` | `np.ndarray` | 是 | 目标态密度矩阵，必须是方阵，维度为 `2^n × 2^n`。例如 3 比特时为 `(8, 8)`。内部会转为 `complex64`。 |
| `epsilon` | `float` | 是 | 保真度阈值。环境在 `fidelity >= epsilon` 时结束当前 episode，并叠加 `terminal_bonus`。常用 `0.9~0.99`。 |
| `config` | `PPORollbackConfig \\| None` | 否 | 训练配置；传 `None` 使用默认超参数。 |

返回值：

- `theta: Dict[str, torch.Tensor]`：策略网络参数快照。
- `circuit: Circuit`：训练过程中发现的最优线路（若未记录到，则回退到当前策略贪婪推演得到的线路）。

### 4.2 超参数（`PPORollbackConfig`）

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `episode_num` | `200` | 训练总 episode 数。增大通常可提升收敛概率，但耗时增加。 |
| `max_steps_per_episode` | `20` | 单个 episode 最大门数（最大步数）。过小可能无法达到目标态。 |
| `update_timestep` | `64` | 每收集多少步轨迹后执行一次 PPO 更新。 |
| `epoch_num` | `4` | 每次更新时对同一批轨迹迭代优化的轮数（PPO epoch）。 |
| `epsilon_clip` | `0.2` | PPO clip 系数，控制策略更新步长。常用 `0.1~0.3`。 |
| `rollback_alpha` | `-0.3` | 回滚项系数（KL 超阈值时启用）。保持负值可形成论文中的 rollback 形式。 |
| `kl_threshold` | `0.03` | KL 散度阈值，超过时使用 rollback surrogate。 |
| `gamma` | `0.99` | 奖励折扣因子。越接近 1 越重视长期回报。 |
| `value_loss_coef` | `0.5` | 价值函数损失权重。 |
| `entropy_coef` | `0.01` | 熵正则权重，用于鼓励探索。 |
| `learning_rate` | `0.002` | Adam 学习率。 |
| `hidden_dim` | `256` | 策略/价值网络隐藏层宽度。 |
| `gate_penalty` | `0.01` | 每追加一个门的惩罚项，抑制过长线路。 |
| `seed` | `42` | 随机种子（NumPy 与 PyTorch）。 |
| `action_gates` | `None` | 可选动作集合。`None` 时自动生成并校验支持的门集。建议任务化时显式收窄动作空间。 |
| `terminal_bonus` | `0.0` | 达到 `epsilon` 时附加奖励。 |
| `log_interval` | `0` | 日志打印间隔（按 episode，`0` 为关闭）。 |
| `init_theta` | `None` | 热启动参数字典。键名和张量形状匹配时才会加载。 |

实践建议：

- GHZ/W 等结构化目标建议显式提供 `action_gates`，显著降低搜索难度。
- 若训练不稳定，可先减小 `learning_rate`，再调 `epsilon_clip` 与 `entropy_coef`。
- 若线路过长，可增大 `gate_penalty` 或减小 `max_steps_per_episode`。

```python
import numpy as np

from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas

# 3 比特 GHZ: (|000> + |111>) / sqrt(2)
target = np.zeros((8, 1), dtype=np.complex64)
target[0, 0] = 1 / np.sqrt(2)
target[7, 0] = 1 / np.sqrt(2)
rho_target = target @ target.conj().T

config = PPORollbackConfig(
    episode_num=800,
    max_steps_per_episode=8,
    gate_penalty=0.005,
    seed=42,
)

theta, circuit = ppo_rb_qas(rho_target, epsilon=0.95, config=config)

print(f"参数张量数量: {len(theta)}")
print(f"线路门数: {len(circuit.gates)}")
print(circuit.show())
```

## 5. 使用方法：PPR_DQL

`PPR_DQL` 的输入是目标 `State`，可直接返回 `Circuit`，也可以返回包含训练信息的结果对象。

### 5.1 输入参数（`train_ppr_dql` / `ppr_dql_state_to_circuit`）

函数签名：

- `train_ppr_dql(target_state, config=None, policy_library=None)`
- `ppr_dql_state_to_circuit(target_state, config=None, policy_library=None)`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `target_state` | `State` | 是 | 目标量子态对象。算法会从该对象读取 `n_qubits` 和 `backend`。 |
| `config` | `PPRDQLConfig \\| None` | 否 | 训练超参数；传 `None` 使用默认值。 |
| `policy_library` | `Sequence[PPRDQLPolicy] \\| None` | 否 | 旧策略库（可选）。用于 PPR 的策略重用。每个策略需与当前任务具有相同 `n_qubits` 与动作空间长度。 |

返回值：

- `train_ppr_dql(...) -> PPRDQLResult`
    - `circuit: Circuit`：最优线路
    - `policy: PPRDQLPolicy`：训练后的策略
    - `best_fidelity: float`：最优保真度
    - `episode_rewards: List[float]`：每个 episode 的总奖励
    - `selected_policy_indices: List[int]`：每个 episode 选择的策略索引（`0` 表示当前新策略）
- `ppr_dql_state_to_circuit(...) -> Circuit`：仅返回线路，便于快速调用

### 5.2 超参数（`PPRDQLConfig`）

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `episode_num` | `200` | 训练总 episode 数。代码要求 `> 0`。 |
| `max_steps_per_episode` | `20` | 单 episode 最大步数。代码要求 `> 0`。 |
| `gamma` | `0.99` | DQN 目标中的折扣因子。 |
| `learning_rate` | `1e-3` | Adam 学习率。 |
| `batch_size` | `32` | 回放采样 batch 大小。代码要求 `> 0`。 |
| `replay_capacity` | `10000` | 经验回放容量。代码要求 `> 0`。 |
| `warmup_transitions` | `64` | 至少积累多少转移后再开始训练；与 `batch_size` 共同决定启动时机。 |
| `hidden_dim` | `128` | Q 网络隐藏层宽度。 |
| `target_update_interval` | `10` | 每多少个 episode 同步一次目标网络。 |
| `fidelity_threshold` | `0.95` | 成功阈值。代码要求在 `(0, 1]`。 |
| `gate_penalty` | `0.01` | 每步门惩罚。 |
| `terminal_bonus` | `1.0` | 达到阈值时附加奖励。 |
| `epsilon_start` | `1.0` | ε-greedy 初始探索率。 |
| `epsilon_end` | `0.05` | ε-greedy 下限。 |
| `epsilon_decay` | `0.985` | 每 episode 后 `epsilon *= epsilon_decay`。 |
| `policy_reuse_probability` | `1.0` | 若当前 episode 选中了旧策略，初始重用概率 `ψ`。 |
| `policy_reuse_decay` | `0.95` | 旧策略重用概率步内衰减因子（`ψ <- ψ * decay`）。 |
| `temperature_init` | `0.0` | 策略选择 softmax 温度系数初值。 |
| `temperature_step` | `0.01` | 每 episode 温度增量。 |
| `action_gates` | `None` | 自定义动作集合；`None` 时使用默认门集（每比特 `rz(pi/4), X, Y, Z, H` 与环形 `cx`）。 |
| `seed` | `42` | 随机种子（NumPy / random / PyTorch）。 |
| `log_interval` | `0` | 日志间隔（按 episode，`0` 为关闭）。 |

实践建议：

- 优先保证 `action_gates` 与任务匹配；动作空间越贴近目标，训练越快越稳。
- 如果早期完全学不动，可先提高 `terminal_bonus` 或降低 `gate_penalty`。
- 如果后期抖动明显，可降低 `learning_rate`，并减小 `epsilon_end`。
- 使用 `policy_library` 时，应保证旧策略动作定义与当前任务完全一致。

```python
import numpy as np

from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.state import State
from nexq.algorithms.qas import PPRDQLConfig, ppr_dql_state_to_circuit

backend = NumpyBackend()
target = np.zeros(8, dtype=np.complex64)
target[0] = 1 / np.sqrt(2)
target[7] = 1 / np.sqrt(2)
state = State.from_array(target, n_qubits=3, backend=backend)

config = PPRDQLConfig(
    episode_num=800,
    max_steps_per_episode=3,
    fidelity_threshold=0.99,
    gate_penalty=0.0,
    action_gates=[
        {"type": "hadamard", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        {"type": "cx", "target_qubit": 2, "control_qubits": [0], "control_states": [1]},
    ],
    seed=42,
)

circuit = ppr_dql_state_to_circuit(state, config=config)

print(circuit)
print(circuit.show())
```

## 6. 使用方法：CRLQAS

`CRLQAS` 的目标是最小化给定哈密顿量的能量。结构搜索由 DDQN 决策，参数优化由 Adam-SPSA 执行。

### 6.1 输入参数（`train_crlqas` / `crlqas`）

函数签名：

- `train_crlqas(hamiltonian, config=None)`
- `crlqas(hamiltonian, config=None)`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `hamiltonian` | `np.ndarray \| Hamiltonian` | 是 | 目标哈密顿量，支持直接传矩阵，或传 `nexq.channel.operators.Hamiltonian` 对象。 |
| `config` | `CRLQASConfig \| None` | 否 | 训练超参数；传 `None` 使用默认值。 |

返回值：

- `train_crlqas(...) -> CRLQASResult`：包含最优 `circuit`、`minimum_energy`、课程阈值和训练轨迹。
- `crlqas(...) -> Tuple[Circuit, float]`：快捷接口，仅返回 `(circuit, minimum_energy)`。

### 6.2 超参数（`CRLQASConfig`）

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `max_episodes` | `300` | 训练总 episode 数。 |
| `n_act` | `12` | 每个 episode 的最大动作数（门数上限）。 |
| `gamma` | `0.99` | DDQN 目标中的折扣因子。 |
| `epsilon_start` | `1.0` | ε-greedy 初始探索率。 |
| `epsilon_min` | `0.05` | ε-greedy 下限。 |
| `epsilon_decay` | `0.9995` | 每个 episode 后的探索率衰减。 |
| `replay_capacity` | `20000` | 回放缓冲区容量。 |
| `batch_size` | `64` | 每次 DDQN 训练的采样 batch。 |
| `q_hidden_dim` | `256` | Q 网络隐藏层维度。 |
| `q_learning_rate` | `1e-3` | Q 网络优化器学习率。 |
| `train_interval` | `10` | 每多少环境步执行一次 DDQN 更新。 |
| `target_update_interval` | `200` | 每多少环境步同步一次目标网络。 |
| `success_reward` | `5.0` | 达到课程阈值时奖励。 |
| `failure_reward` | `-5.0` | 触发失败终止时奖励。 |
| `reward_floor` | `-1.0` | 形状奖励下限。 |
| `curriculum_initial_threshold` | `0.2` | 初始课程阈值。 |
| `curriculum_mu` | `-2.0` | 课程更新中的参考最小能量。 |
| `curriculum_adjust_period` | `500` | 每隔多少 episode 做一次贪心课程阈值调整。 |
| `curriculum_delta` | `0.2` | 课程阈值重置/微调幅度。 |
| `curriculum_kappa` | `100.0` | 成功后阈值下降步长分母（`delta / kappa`）。 |
| `curriculum_reset_patience` | `40` | 连续失败多少个 episode 后触发阈值重置。 |
| `chemical_accuracy` | `1.6e-3` | 课程阈值下限（化学精度）。 |
| `random_halt_p` | `0.5` | 随机停止负二项采样参数。 |
| `action_gates` | `None` | 动作集合；`None` 时默认使用 `nexq/core/gates.py` 可校验通过的全门集；传列表则使用自定义门集。 |
| `adam_spsa` | `AdamSPSAConfig()` | 参数优化器配置。 |
| `seed` | `42` | 随机种子（NumPy / random / PyTorch）。 |
| `log_interval` | `0` | 日志打印间隔（按 episode，`0` 为关闭）。 |

`adam_spsa` 子配置（`AdamSPSAConfig`）：

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `iterations` | `30` | 每次结构更新后 SPSA 的优化迭代次数。 |
| `a` | `0.08` | 学习率序列基值。 |
| `alpha` | `0.602` | 学习率衰减指数。 |
| `c` | `0.12` | 扰动幅度序列基值。 |
| `gamma_sp` | `0.101` | 扰动幅度衰减指数。 |
| `beta_1` | `0.9` | 一阶动量系数。 |
| `beta_2` | `0.999` | 二阶动量系数。 |
| `lam` | `0.0` | `beta_1` 的衰减指数。 |
| `epsilon` | `1e-8` | 数值稳定项。 |

实践建议：

- 先用较小 `n_act` 和较小 `adam_spsa.iterations` 做可行性验证，再逐步放大。
- 若收敛慢，可适度提高 `success_reward` 或减小 `curriculum_initial_threshold`。
- 若训练震荡，可降低 `q_learning_rate`，并增大 `target_update_interval`。
- 对结构化任务建议手动提供 `action_gates`，可显著减少搜索空间。

### 6.3 最小示例（H2）

```python
from nexq.algorithms.qas import AdamSPSAConfig, CRLQASConfig, train_crlqas
from nexq.channel.operators import Hamiltonian

h2 = Hamiltonian(n_qubits=2)
h2.term(-1.052373245772859, {"I": [0, 1]})
h2.term(0.39793742484318045, {"Z": [0]})
h2.term(-0.39793742484318045, {"Z": [1]})
h2.term(-0.01128010425623538, {"Z": [0, 1]})
h2.term(0.18093119978423156, {"X": [0, 1]})

cfg = CRLQASConfig(
        max_episodes=300,
        n_act=10,
        adam_spsa=AdamSPSAConfig(iterations=20),
        seed=42,
)

result = train_crlqas(h2, config=cfg)
print(result.minimum_energy)
print(result.circuit.show())
```

## 7. 可选：自定义动作门集合

`PPRDQLConfig.action_gates` 支持自定义动作门集合。每个动作是一个门字典，格式与 `Circuit` 门定义一致。

注意：

- 不允许包含 `{"type": "unitary", ...}`
- 建议只使用 `gate_to_matrix` 可解析的门

示例：

```python
custom_actions = [
    {"type": "hadamard", "target_qubit": 0},
    {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
]

config = PPRDQLConfig(action_gates=custom_actions)
circuit = ppr_dql_state_to_circuit(state, config=config)
```

## 8. 示例脚本

- `PPO_RB_demo_ghz4.py`：使用 PPO-RB 搜索 4 比特 GHZ 线路
- `PPR_DQL_demo_ghz3.py`：使用 PPR-DQL 搜索 3 比特 GHZ 线路，并导出 OpenQASM 3.0 到 `demo/ppr_dql_ghz3_circuit.qasm`
- `CRLQAS_demo_h2.py`：使用 CRLQAS 搜索小分子 H2 的低能量线路，并导出 OpenQASM 3.0 到 `demo/crlqas_h2_circuit.qasm`

运行示例：

```bash
python nexq/algorithms/qas/demo/PPR_DQL_demo_ghz3.py
python nexq/algorithms/qas/demo/CRLQAS_demo_h2.py
```

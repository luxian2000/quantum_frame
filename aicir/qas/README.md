# QAS 模块说明

`aicir.qas` 是量子构架搜索（Quantum Architecture Search, QAS）模块。

该模块用于自动搜索量子线路结构。当前仓库中包含两类 QAS 实现：一类面向变分量子算法（VQA）的 ansatz 架构搜索，另一类面向给定目标态或哈密顿量任务的强化学习式量子线路搜索。运行下列实现需要可用的 `torch`。

- `VQA_QAS.py`：基于超网络和权重共享的 VQA ansatz 架构搜索，支持分类任务和 H2 VQE 示例
- `CRLQAS.py`：课程学习 + DDQN + Adam-SPSA 的量子架构搜索（面向哈密顿量能量最小化）
- `PPR_DQL.py`：基于 aicir 状态演化实现的 PPR-DQL（Probabilistic Policy Reuse with Deep Q-Learning）
- `PPO_RB.py`：Trust Region-based PPO with Rollback 版本的量子架构搜索

## 1. 已提供能力

- VQA ansatz 搜索：`VQA_QAS.py` 使用超网络、权重共享、架构排序和微调，为分类、VQE 或自定义 VQA 目标选择 ansatz。
- 多超网络训练：`VQA_QAS.py` 支持 `supernet_num`，每次采样架构后选择损失最小的超网络并只更新该超网络的活跃参数。
- 内置 VQA 示例：`classification_vqa_qas` 提供 3 量子比特合成二分类任务，`h2_vqe_qas` 提供 4 量子比特 H2 VQE 任务。
- 强化学习线路搜索：`PPO_RB.py`、`PPR_DQL.py` 和 `CRLQAS.py` 从空线路出发逐步追加 aicir 支持的量子门，搜索目标态制备或低能量线路。
- 参数优化：VQA QAS 使用 PyTorch autograd 或参数位移法；CRLQAS 使用 Adam-SPSA 微调变分参数。
- 输出对象：各方法最终都返回或包含 aicir `Circuit`，可继续使用 `Circuit.show()` 展示，并可导出为 OpenQASM 2.0 / 3.0。
- 约束：核心线路构建和仿真使用 aicir；动作门或搜索门需要是 `aicir/core/gates.py` 支持的门。

## 2. 接口

使用统一入口 `run_qas` 和统一配置工厂 `config`。用户只需要记住 QAS 方法名，例如 `VQA_QAS` 对应 `config.vqa_qas(...)`，不需要导入 `VQAQASConfig`、`PPRDQLConfig` 等具体配置类。

推荐入口：

- `run_qas(method, **kwargs)`：按方法名运行对应 QAS 实现
- `config.<method>(**kwargs)`：按方法名创建对应配置对象，例如 `config.vqa_qas(...)`、`config.ppr_dql(...)`
- `config.create(method, **kwargs)`：当方法名来自字符串或外部配置文件时，按方法名创建配置对象
- `QASRunConfig`：把方法名、配置和任务输入封装成请求对象后传给 `run_qas`
- `default_qas_config(method, **kwargs)`：兼容旧入口，内部等价于 `config.create(method, **kwargs)`
- `available_qas_methods()`：返回当前支持的统一入口方法名

统一入口支持的方法和参数：

| `method`               | 必要参数                                        | 可选参数                                 | 返回值               |
| ------------------------ | ----------------------------------------------- | ---------------------------------------- | -------------------- |
| `"vqa_qas"`            | `objective_fn` 或在 `config` 中指定内置任务 | `config`、`dataset`、`hamiltonian` | `VQAQASResult`     |
| `"vqa_classification"` | 无                                              | `config`                               | `VQAQASResult`     |
| `"vqa_h2"`             | 无                                              | `config`                               | `VQAQASResult`     |
| `"ppo_rb"`             | `target_density_matrix`、`epsilon`          | `config`                               | `(theta, circuit)` |
| `"ppr_dql"`            | `target_state`                                | `config`、`policy_library`           | `PPRDQLResult`     |
| `"crlqas"`             | `hamiltonian`                                 | `config`                               | `CRLQASResult`     |

方法名大小写不敏感，也支持常见别名，例如 `"VQA_QAS"`、`"h2_vqe"`、`"ppr"`。

示例：

```python
from aicir.qas import config, run_qas

cfg = config.vqa_qas(task="classification", supernet_steps=20, ranking_num=20, finetune_steps=5)
result = run_qas("VQA_QAS", config=cfg)

# 如果方法名来自外部配置文件，可以用字符串创建配置对象。
cfg = config.create("VQA_QAS", task="classification", supernet_steps=20)
result = run_qas("VQA_QAS", config=cfg)
```

各方法的配置函数：

- `config.vqa_qas(...)`：VQA_QAS 搜索空间和训练超参数配置
- `config.vqa_classification(...)`：内置分类任务的 VQA_QAS 配置
- `config.vqa_h2(...)`：内置 H2 VQE 任务的 VQA_QAS 配置
- `config.ppo_rb(...)`：PPO-RB 超参数配置
- `config.ppr_dql(...)`：PPR-DQL 超参数配置
- `config.crlqas(...)`：CRLQAS 超参数配置；`adam_spsa` 可传字典，例如 `config.crlqas(adam_spsa={"iterations": 10})`

底层专用接口仍然保留，适合需要直接控制某个算法实现的用户：

- `train_vqa_qas(objective_fn, config=None, dataset=None, hamiltonian=None)` / `vqa_qas(...)`
- `classification_vqa_qas(config=None)` / `h2_vqe_qas(config=None)`
- `ppo_rb_qas(target_density_matrix, epsilon, config=None)`
- `train_ppr_dql(state, config=None, policy_library=None)` / `ppr_dql_state_to_circuit(...)`
- `train_crlqas(hamiltonian, config=None)` / `crlqas(...)`

可通过以下方式导入：

```python
from aicir.qas import config, run_qas
```

## 3. VQA_QAS：面向变分量子算法的超网络架构搜索

`VQA_QAS.py` 面向 VQA ansatz 结构搜索，使用一阶段超网络、按层权重共享、架构排序和选中 ansatz 微调。

论文依据：

- Yuxuan Du, Tao Huang, Shan You, Min-Hsiu Hsieh, Dacheng Tao, *Quantum circuit architecture search for variational quantum algorithms*, npj Quantum Information, 2022.

当前实现使用 `aicir.core.Circuit`、aicir 门构造器、`TorchBackend` 和 aicir 态向量/矩阵演化能力，不依赖 PennyLane、Qiskit、Cirq 或其他量子 SDK。当前实现以无噪声仿真为主；`NoiseConfig` 仅作为 noisy QAS 的占位配置，启用时会抛出 `NotImplementedError`。

### 3.1 输入参数（`train_vqa_qas` / `vqa_qas`）

函数签名：

- `train_vqa_qas(objective_fn, config=None, dataset=None, hamiltonian=None)`
- `vqa_qas(objective_fn, config=None, dataset=None, hamiltonian=None)`
- `classification_vqa_qas(config=None)`
- `h2_vqe_qas(config=None)`

| 参数             | 类型                                               | 必填 | 说明                                                                                            |
| ---------------- | -------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------- |
| `objective_fn` | `Callable \| str \| None`                          | 否   | 自定义目标函数，或内置任务名。`classification_vqa_qas` 和 `h2_vqe_qas` 会自动选择内置目标。 |
| `config`       | `VQAQASConfig \| None`                            | 否   | 搜索空间、训练步数、学习率和随机种子等配置；传 `None` 使用默认值。                            |
| `dataset`      | `Mapping \| None`                                 | 否   | 分类任务数据集。为 `None` 时，内置分类任务会生成 3 维合成二分类数据。                         |
| `hamiltonian`  | `Hamiltonian \| np.ndarray \| torch.Tensor \| None` | 否   | VQE 任务哈密顿量。为 `None` 时，`h2_vqe_qas` 使用内置 4 量子比特 H2 哈密顿量。              |

返回值：

- `VQAQASResult`
  - `best_architecture`：排序和微调后选中的 `Architecture`
  - `best_circuit`：选中架构对应的 aicir `Circuit`
  - `best_score`：分类任务为验证损失，H2 VQE 任务为微调后能量
  - `ranking_records`：候选架构排序记录
  - `supernet_log`：超网络训练日志
  - `finetune_log`：固定架构微调日志
  - `final_metrics`：任务相关最终指标

### 3.2 超参数（`VQAQASConfig`）

| 字段                          |                       默认值 | 说明                                                                           |
| ----------------------------- | ---------------------------: | ------------------------------------------------------------------------------ |
| `n_qubits`                  |                        `3` | 量子比特数。                                                                   |
| `layers`                    |                        `3` | ansatz 层数。                                                                  |
| `single_qubit_gates`        |                  `("ry",)` | 单量子比特候选旋转门，目前支持 `rx`、`ry`、`rz`。                        |
| `two_qubit_pairs`           | `((0, 1), (0, 2), (1, 2))` | 允许搜索 CNOT/无 CNOT 的连接对，格式为 `(control, target)`。                 |
| `search_single_qubit_gates` |                     `True` | 是否搜索单量子比特门布局；关闭时每层使用第一个候选门。                         |
| `search_two_qubit_gates`    |                     `True` | 是否搜索双量子比特门 mask；关闭时默认启用所有给定连接对。                      |
| `supernet_num`              |                        `1` | 超网络数量 `W`。每个采样架构会在所有超网络上评估，并只更新损失最小的超网络。 |
| `supernet_steps`            |                      `100` | 一阶段超网络优化步数。                                                         |
| `ranking_num`               |                       `50` | 排序阶段采样的候选架构数量。                                                   |
| `finetune_steps`            |                       `20` | 对选中固定架构执行独立参数微调的步数。                                         |
| `learning_rate`             |                     `0.05` | 超网络共享参数学习率。                                                         |
| `finetune_learning_rate`    |                     `0.03` | 固定架构微调学习率。                                                           |
| `seed`                      |                       `42` | 随机种子。                                                                     |
| `device`                    |                    `"cpu"` | `TorchBackend` 使用的设备。                                                  |
| `task`                      |         `"classification"` | 内置任务类型，如 `classification` 或 `h2_vqe`。                            |
| `log_interval`              |                        `0` | 日志打印间隔；`0` 表示关闭。                                                 |
| `use_parameter_shift`       |                    `False` | 是否使用参数位移法更新梯度；默认使用 PyTorch autograd。                        |

实践建议：

- 分类任务默认搜索空间为 3 量子比特、3 层、固定 `RY`，并在 `(0, 1)`、`(0, 2)`、`(1, 2)` 上搜索 CNOT/无 CNOT；当 `layers=3` 时逻辑搜索空间大小为 `8^3`。
- H2 VQE 默认搜索空间为 4 量子比特、3 层、`RY/RZ`，并在链式连接 `(0, 1)`、`(1, 2)`、`(2, 3)` 上搜索 CNOT/无 CNOT；当 `layers=3` 时逻辑搜索空间大小为 `128^3`。
- 权重共享规则与论文一致：两个 ansatz 只要在第 `l` 层具有相同单量子比特门布局，就共享该层可训练参数；共享关系不受双量子比特门选择和其他层布局影响。

### 3.3 分类任务示例

```python
from aicir.qas import config, run_qas

cfg = config.vqa_qas(
    n_qubits=3,
    layers=3,
    single_qubit_gates=("ry",),
    two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
    supernet_num=5,
    supernet_steps=400,
    ranking_num=500,
    finetune_steps=20,
    seed=42,
)

result = run_qas("vqa_classification", config=cfg)
print(result.best_score)
print(result.best_circuit.show())
```

内置分类任务使用 3 维特征的合成二分类数据集，并划分为训练集、验证集和测试集。默认搜索空间固定单量子比特门为 `RY`，并在 `(0, 1)`、`(0, 2)`、`(1, 2)` 上搜索是否添加 CNOT；当 `layers=3` 时，逻辑搜索空间大小为 `8^3`。

### 3.4 H2 VQE 示例

```python
from aicir.qas import config, run_qas

cfg = config.vqa_h2(
    n_qubits=4,
    layers=3,
    single_qubit_gates=("ry", "rz"),
    two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
    supernet_num=5,
    supernet_steps=500,
    ranking_num=500,
    finetune_steps=50,
    seed=42,
)

result = run_qas("vqa_h2", config=cfg)
print(result.final_metrics)
print(result.best_circuit.show())
```

内置 H2 任务使用一组固定 H-H 键长下的 4 量子比特 Pauli 哈密顿量系数，搜索 `RY`/`RZ` 单量子比特门布局，以及链式连接对上的 CNOT/无 CNOT 选择。结果会报告 QAS 排序阶段最优能量、微调后能量、固定硬件高效 VQE baseline、选中线路和 CNOT 数量。当 `layers=3` 时，逻辑搜索空间大小为 `128^3`。

## 4. PPO_RB：基于真正近端策略优化的量子架构搜索

`PPO_RB` 的输入是目标密度矩阵，输出是策略参数 `theta` 与搜索得到的 `Circuit`。

论文依据：

- X. Zhu and X. Hou, *Quantum architecture search via truly proximal policy optimization*, Scientific Reports, 2023, doi: `10.1038/s41598-023-32349-2`.

### 4.1 输入参数（`ppo_rb_qas`）

函数签名：`ppo_rb_qas(target_density_matrix, epsilon, config=None)`

| 参数                      | 类型                         | 必填 | 说明                                                                                                          |
| ------------------------- | ---------------------------- | ---- | ------------------------------------------------------------------------------------------------------------- |
| `target_density_matrix` | `np.ndarray`               | 是   | 目标态密度矩阵，必须是方阵，维度为 `2^n × 2^n`。例如 3 比特时为 `(8, 8)`。内部会转为 `complex64`。     |
| `epsilon`               | `float`                    | 是   | 保真度阈值。环境在 `fidelity >= epsilon` 时结束当前 episode，并叠加 `terminal_bonus`。常用 `0.9~0.99`。 |
| `config`                | `PPORollbackConfig \| None` | 否   | 训练配置；传 `None` 使用默认超参数。                                                                        |

返回值：

- `theta: Dict[str, torch.Tensor]`：策略网络参数快照。
- `circuit: Circuit`：训练过程中发现的最优线路（若未记录到，则回退到当前策略贪婪推演得到的线路）。

### 4.2 超参数（`PPORollbackConfig`）

| 字段                      |    默认值 | 说明                                                                              |
| ------------------------- | --------: | --------------------------------------------------------------------------------- |
| `episode_num`           |   `200` | 训练总 episode 数。增大通常可提升收敛概率，但耗时增加。                           |
| `max_steps_per_episode` |    `20` | 单个 episode 最大门数（最大步数）。过小可能无法达到目标态。                       |
| `update_timestep`       |    `64` | 每收集多少步轨迹后执行一次 PPO 更新。                                             |
| `epoch_num`             |     `4` | 每次更新时对同一批轨迹迭代优化的轮数（PPO epoch）。                               |
| `epsilon_clip`          |   `0.2` | PPO clip 系数，控制策略更新步长。常用 `0.1~0.3`。                               |
| `rollback_alpha`        |  `-0.3` | 回滚项系数（KL 超阈值时启用）。保持负值可形成论文中的 rollback 形式。             |
| `kl_threshold`          |  `0.03` | KL 散度阈值，超过时使用 rollback surrogate。                                      |
| `gamma`                 |  `0.99` | 奖励折扣因子。越接近 1 越重视长期回报。                                           |
| `value_loss_coef`       |   `0.5` | 价值函数损失权重。                                                                |
| `entropy_coef`          |  `0.01` | 熵正则权重，用于鼓励探索。                                                        |
| `learning_rate`         | `0.002` | Adam 学习率。                                                                     |
| `hidden_dim`            |   `256` | 策略/价值网络隐藏层宽度。                                                         |
| `gate_penalty`          |  `0.01` | 每追加一个门的惩罚项，抑制过长线路。                                              |
| `seed`                  |    `42` | 随机种子（NumPy 与 PyTorch）。                                                    |
| `action_gates`          |  `None` | 可选动作集合。`None` 时自动生成并校验支持的门集。建议任务化时显式收窄动作空间。 |
| `terminal_bonus`        |   `0.0` | 达到 `epsilon` 时附加奖励。                                                     |
| `log_interval`          |     `0` | 日志打印间隔（按 episode，`0` 为关闭）。                                        |
| `init_theta`            |  `None` | 热启动参数字典。键名和张量形状匹配时才会加载。                                    |

实践建议：

- GHZ/W 等结构化目标建议显式提供 `action_gates`，显著降低搜索难度。
- 若训练不稳定，可先减小 `learning_rate`，再调 `epsilon_clip` 与 `entropy_coef`。
- 若线路过长，可增大 `gate_penalty` 或减小 `max_steps_per_episode`。

### 4.3 最小示例（GHZ）

```python
import numpy as np

from aicir.qas import config, run_qas

# 3 比特 GHZ: (|000> + |111>) / sqrt(2)
target = np.zeros((8, 1), dtype=np.complex64)
target[0, 0] = 1 / np.sqrt(2)
target[7, 0] = 1 / np.sqrt(2)
rho_target = target @ target.conj().T

cfg = config.ppo_rb(
    episode_num=800,
    max_steps_per_episode=8,
    gate_penalty=0.005,
    seed=42,
)

theta, circuit = run_qas("ppo_rb", target_density_matrix=rho_target, epsilon=0.95, config=cfg)

print(f"参数张量数量: {len(theta)}")
print(f"线路门数: {len(circuit.gates)}")
print(circuit.show())
```

## 5. PPR_DQL：基于持续强化学习和策略复用的量子架构搜索

`PPR_DQL` 的输入是目标 `State`，可直接返回 `Circuit`，也可以返回包含训练信息的结果对象。

论文依据：

- *Quantum Architecture Search via Continual Reinforcement Learning*, arXiv:`2112.05779v1`.

### 5.1 输入参数（`train_ppr_dql` / `ppr_dql_state_to_circuit`）

函数签名：

- `train_ppr_dql(target_state, config=None, policy_library=None)`
- `ppr_dql_state_to_circuit(target_state, config=None, policy_library=None)`

| 参数               | 类型                              | 必填 | 说明                                                                                              |
| ------------------ | --------------------------------- | ---- | ------------------------------------------------------------------------------------------------- |
| `target_state`   | `State`                         | 是   | 目标量子态对象。算法会从该对象读取 `n_qubits` 和 `backend`。                                  |
| `config`         | `PPRDQLConfig \| None`           | 否   | 训练超参数；传 `None` 使用默认值。                                                              |
| `policy_library` | `Sequence[PPRDQLPolicy] \| None` | 否   | 旧策略库（可选）。用于 PPR 的策略重用。每个策略需与当前任务具有相同 `n_qubits` 与动作空间长度。 |

返回值：

- `train_ppr_dql(...) -> PPRDQLResult`
  - `circuit: Circuit`：最优线路
  - `policy: PPRDQLPolicy`：训练后的策略
  - `best_fidelity: float`：最优保真度
  - `episode_rewards: List[float]`：每个 episode 的总奖励
  - `selected_policy_indices: List[int]`：每个 episode 选择的策略索引（`0` 表示当前新策略）
- `ppr_dql_state_to_circuit(...) -> Circuit`：仅返回线路，便于快速调用

### 5.2 超参数（`PPRDQLConfig`）

| 字段                         |    默认值 | 说明                                                                                       |
| ---------------------------- | --------: | ------------------------------------------------------------------------------------------ |
| `episode_num`              |   `200` | 训练总 episode 数。代码要求 `> 0`。                                                      |
| `max_steps_per_episode`    |    `20` | 单 episode 最大步数。代码要求 `> 0`。                                                    |
| `gamma`                    |  `0.99` | DQN 目标中的折扣因子。                                                                     |
| `learning_rate`            |  `1e-3` | Adam 学习率。                                                                              |
| `batch_size`               |    `32` | 回放采样 batch 大小。代码要求 `> 0`。                                                    |
| `replay_capacity`          | `10000` | 经验回放容量。代码要求 `> 0`。                                                           |
| `warmup_transitions`       |    `64` | 至少积累多少转移后再开始训练；与 `batch_size` 共同决定启动时机。                         |
| `hidden_dim`               |   `128` | Q 网络隐藏层宽度。                                                                         |
| `target_update_interval`   |    `10` | 每多少个 episode 同步一次目标网络。                                                        |
| `fidelity_threshold`       |  `0.95` | 成功阈值。代码要求在 `(0, 1]`。                                                          |
| `gate_penalty`             |  `0.01` | 每步门惩罚。                                                                               |
| `terminal_bonus`           |   `1.0` | 达到阈值时附加奖励。                                                                       |
| `epsilon_start`            |   `1.0` | ε-greedy 初始探索率。                                                                     |
| `epsilon_end`              |  `0.05` | ε-greedy 下限。                                                                           |
| `epsilon_decay`            | `0.985` | 每 episode 后 `epsilon *= epsilon_decay`。                                               |
| `policy_reuse_probability` |   `1.0` | 若当前 episode 选中了旧策略，初始重用概率 `ψ`。                                         |
| `policy_reuse_decay`       |  `0.95` | 旧策略重用概率步内衰减因子（`ψ <- ψ * decay`）。                                       |
| `temperature_init`         |   `0.0` | 策略选择 softmax 温度系数初值。                                                            |
| `temperature_step`         |  `0.01` | 每 episode 温度增量。                                                                      |
| `action_gates`             |  `None` | 自定义动作集合；`None` 时使用默认门集（每比特 `rz(pi/4), X, Y, Z, H` 与环形 `cx`）。 |
| `seed`                     |    `42` | 随机种子（NumPy / random / PyTorch）。                                                     |
| `log_interval`             |     `0` | 日志间隔（按 episode，`0` 为关闭）。                                                     |

实践建议：

- 优先保证 `action_gates` 与任务匹配；动作空间越贴近目标，训练越快越稳。
- 如果早期完全学不动，可先提高 `terminal_bonus` 或降低 `gate_penalty`。
- 如果后期抖动明显，可降低 `learning_rate`，并减小 `epsilon_end`。
- 使用 `policy_library` 时，应保证旧策略动作定义与当前任务完全一致。

### 5.3 自定义动作门集合

`config.ppr_dql(action_gates=...)` 支持自定义动作门集合。每个动作是一个门字典，格式与 `Circuit` 门定义一致。

注意：

- 不允许包含 `{"type": "unitary", ...}`
- 建议只使用 `gate_to_matrix` 可解析的门

示例：

```python
from aicir.qas import config, run_qas

custom_actions = [
    {"type": "hadamard", "target_qubit": 0},
    {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
]

cfg = config.ppr_dql(action_gates=custom_actions)
result = run_qas("ppr_dql", target_state=state, config=cfg)
circuit = result.circuit
```

### 5.4 最小示例（GHZ）

```python
import numpy as np

from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.core.state import State
from aicir.qas import config, run_qas

backend = NumpyBackend()
target = np.zeros(8, dtype=np.complex64)
target[0] = 1 / np.sqrt(2)
target[7] = 1 / np.sqrt(2)
state = State.from_array(target, n_qubits=3, backend=backend)

cfg = config.ppr_dql(
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

result = run_qas("ppr_dql", target_state=state, config=cfg)
circuit = result.circuit

print(circuit)
print(circuit.show())
```

## 6. CRLQAS：面向硬件误差的课程强化学习量子架构搜索

`CRLQAS` 的目标是最小化给定哈密顿量的能量。结构搜索由 DDQN 决策，参数优化由 Adam-SPSA 执行。

论文依据：

- *Curriculum reinforcement learning for quantum architecture search under hardware errors*, arXiv:`2402.03500`.

### 6.1 输入参数（`train_crlqas` / `crlqas`）

函数签名：

- `train_crlqas(hamiltonian, config=None)`
- `crlqas(hamiltonian, config=None)`

| 参数            | 类型                         | 必填 | 说明                                                                              |
| --------------- | ---------------------------- | ---- | --------------------------------------------------------------------------------- |
| `hamiltonian` | `np.ndarray \| Hamiltonian` | 是   | 目标哈密顿量，支持直接传矩阵，或传 `aicir.channel.operators.Hamiltonian` 对象。 |
| `config`      | `CRLQASConfig \| None`      | 否   | 训练超参数；传 `None` 使用默认值。                                              |

返回值：

- `train_crlqas(...) -> CRLQASResult`：包含最优 `circuit`、`minimum_energy`、课程阈值和训练轨迹。
- `crlqas(...) -> Tuple[Circuit, float]`：快捷接口，仅返回 `(circuit, minimum_energy)`。

### 6.2 超参数（`CRLQASConfig`）

| 字段                             |               默认值 | 说明                                                                                               |
| -------------------------------- | -------------------: | -------------------------------------------------------------------------------------------------- |
| `max_episodes`                 |              `300` | 训练总 episode 数。                                                                                |
| `n_act`                        |               `12` | 每个 episode 的最大动作数（门数上限）。                                                            |
| `gamma`                        |             `0.99` | DDQN 目标中的折扣因子。                                                                            |
| `epsilon_start`                |              `1.0` | ε-greedy 初始探索率。                                                                             |
| `epsilon_min`                  |             `0.05` | ε-greedy 下限。                                                                                   |
| `epsilon_decay`                |           `0.9995` | 每个 episode 后的探索率衰减。                                                                      |
| `replay_capacity`              |            `20000` | 回放缓冲区容量。                                                                                   |
| `batch_size`                   |               `64` | 每次 DDQN 训练的采样 batch。                                                                       |
| `q_hidden_dim`                 |              `256` | Q 网络隐藏层维度。                                                                                 |
| `q_learning_rate`              |             `1e-3` | Q 网络优化器学习率。                                                                               |
| `train_interval`               |               `10` | 每多少环境步执行一次 DDQN 更新。                                                                   |
| `target_update_interval`       |              `200` | 每多少环境步同步一次目标网络。                                                                     |
| `success_reward`               |              `5.0` | 达到课程阈值时奖励。                                                                               |
| `failure_reward`               |             `-5.0` | 触发失败终止时奖励。                                                                               |
| `reward_floor`                 |             `-1.0` | 形状奖励下限。                                                                                     |
| `curriculum_initial_threshold` |              `0.2` | 初始课程阈值。                                                                                     |
| `curriculum_mu`                |             `-2.0` | 课程更新中的参考最小能量。                                                                         |
| `curriculum_adjust_period`     |              `500` | 每隔多少 episode 做一次贪心课程阈值调整。                                                          |
| `curriculum_delta`             |              `0.2` | 课程阈值重置/微调幅度。                                                                            |
| `curriculum_kappa`             |            `100.0` | 成功后阈值下降步长分母（`delta / kappa`）。                                                      |
| `curriculum_reset_patience`    |               `40` | 连续失败多少个 episode 后触发阈值重置。                                                            |
| `chemical_accuracy`            |           `1.6e-3` | 课程阈值下限（化学精度）。                                                                         |
| `random_halt_p`                |              `0.5` | 随机停止负二项采样参数。                                                                           |
| `action_gates`                 |             `None` | 动作集合；`None` 时默认使用 `aicir/core/gates.py` 可校验通过的全门集；传列表则使用自定义门集。 |
| `adam_spsa`                    | `AdamSPSAConfig()` | 参数优化器配置。                                                                                   |
| `seed`                         |               `42` | 随机种子（NumPy / random / PyTorch）。                                                             |
| `log_interval`                 |                `0` | 日志打印间隔（按 episode，`0` 为关闭）。                                                         |

`adam_spsa` 子配置（`AdamSPSAConfig`）：

| 字段           |    默认值 | 说明                                 |
| -------------- | --------: | ------------------------------------ |
| `iterations` |    `30` | 每次结构更新后 SPSA 的优化迭代次数。 |
| `a`          |  `0.08` | 学习率序列基值。                     |
| `alpha`      | `0.602` | 学习率衰减指数。                     |
| `c`          |  `0.12` | 扰动幅度序列基值。                   |
| `gamma_sp`   | `0.101` | 扰动幅度衰减指数。                   |
| `beta_1`     |   `0.9` | 一阶动量系数。                       |
| `beta_2`     | `0.999` | 二阶动量系数。                       |
| `lam`        |   `0.0` | `beta_1` 的衰减指数。              |
| `epsilon`    |  `1e-8` | 数值稳定项。                         |

实践建议：

- 先用较小 `n_act` 和较小 `adam_spsa.iterations` 做可行性验证，再逐步放大。
- 若收敛慢，可适度提高 `success_reward` 或减小 `curriculum_initial_threshold`。
- 若训练震荡，可降低 `q_learning_rate`，并增大 `target_update_interval`。
- 对结构化任务建议手动提供 `action_gates`，可显著减少搜索空间。

### 6.3 最小示例（H2）

```python
from aicir.channel.operators import Hamiltonian
from aicir.qas import config, run_qas

h2 = Hamiltonian(n_qubits=2)
h2.term(-1.052373245772859, {"I": [0, 1]})
h2.term(0.39793742484318045, {"Z": [0]})
h2.term(-0.39793742484318045, {"Z": [1]})
h2.term(-0.01128010425623538, {"Z": [0, 1]})
h2.term(0.18093119978423156, {"X": [0, 1]})

cfg = config.crlqas(
    max_episodes=300,
    n_act=10,
    adam_spsa={"iterations": 20},
    seed=42,
)

result = run_qas("crlqas", hamiltonian=h2, config=cfg)
print(result.minimum_energy)
print(result.circuit.show())
```

## 7. 示例脚本

- `PPO_RB_demo_ghz4.py`：使用 PPO-RB 搜索 4 比特 GHZ 线路
- `PPR_DQL_demo_ghz3.py`：使用 PPR-DQL 搜索 3 比特 GHZ 线路，并导出 OpenQASM 3.0 到 `demo/ppr_dql_ghz3_circuit.qasm`
- `CRLQAS_demo_h2.py`：使用 CRLQAS 搜索小分子 H2 的低能量线路，并导出 OpenQASM 3.0 到 `demo/crlqas_h2_circuit.qasm`

运行示例：

```bash
python aicir/qas/demo/PPR_DQL_demo_ghz3.py
python aicir/qas/demo/CRLQAS_demo_h2.py
```

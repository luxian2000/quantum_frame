# QAS 模块说明

`nexq.qas` 是量子构架搜索（Quantum Architecture Search, QAS）模块。

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
from nexq.qas import PPRDQLConfig, ppr_dql_state_to_circuit, train_ppr_dql
from nexq.qas import AdamSPSAConfig, CRLQASConfig, crlqas, train_crlqas
from nexq.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
```

## 3. 依赖

运行 `CRLQAS.py`、`PPR_DQL.py` 和 `PPO_RB.py` 都需要可用的 `torch`。
架构先验评分、候选库、MaxCut/资源分配问题、任务级验证 runner 可以只依赖 NumPy 后端运行。

当前本地无 `torch` 环境下，`TorchBackend`、`NPUBackend` 以及 RL 搜索接口不会从顶层包导出；对应测试会跳过。云服务器安装 `torch` 后，这些接口会自动恢复导出。

## 4. Task-Level Validation

Roadmap 中的验证目标是比较“参数优化后的任务目标”，不是只比较架构先验分数。当前已提供一条轻量验证链路：

- `nexq.qas.problems`：`ProblemInstance`、`MaxCutInstance`、`ResourceAllocationInstance`
- `nexq.qas.task_evaluation`：统一参数绑定、理想/含噪目标评估、固定预算随机优化
- `nexq.qas.experiments.runner`：构建 QAOA/HEA/RealAmplitudes baselines，运行 QAS prior ranking，再做任务级比较
- `nexq.qas.demo.qas_vs_baselines`：4 比特 MaxCut smoke demo

本地 smoke test：

```bash
python -m nexq.qas.demo.qas_vs_baselines
```

核心接口示例：

```python
from nexq.qas import OptimizerConfig, SearchConfig, maxcut_line, run_validation_experiment

problem = maxcut_line(n_qubits=4)
report = run_validation_experiment(
    problem,
    search_config=SearchConfig(n_qubits=4, candidate_layers=1, n_samples=8),
    optimizer_config=OptimizerConfig(max_evaluations=16, seed=2026),
    qas_top_k=3,
)

print("\n".join(report.summary_lines()))
```

多 seed 汇总接口用于判断结果是否稳定，而不是只看一次随机优化结果：

```python
from nexq.qas import run_multi_seed_validation_experiment

multi_seed_report = run_multi_seed_validation_experiment(
    problem,
    seeds=[2026, 2027, 2028],
    search_config=SearchConfig(n_qubits=4, candidate_layers=1, n_samples=8),
    optimizer_config=OptimizerConfig(max_evaluations=16),
    qas_top_k=3,
)

print("\n".join(multi_seed_report.summary_lines()))
```

当前 roadmap 状态：

- 已完成：MaxCut / resource allocation 问题抽象、任务级验证、同预算 baseline/QAS 对比、单 seed demo、NPU backend smoke 验证。
- 进行中：多 seed 汇总、固定报告 schema、更多实际问题与噪声设置。
- P0.5 进行中：已加入 QAS zero-cost trainability 第一版 `gradient_norm` / `gradient_variance`，基于任务无关 local-probe parameter-shift；已加入 `HardwareProfile` 与 `topology_mapping_efficiency`，输入硬件拓扑、native gate、联通度、routing/depth 和可选映射边质量，但不重复计算 noise fidelity。`expressibility` 保持 KL-Haar / MMD；`noise_robustness` 后续单独升级为噪声暴露/敏感度 profile。调研与实现建议见 `docs/qas_metric_research_report.md`。
- P1.1 已完成第一版：`supercircuit_progressive` 按 Training-Free QAS / zero-cost NAS 的粗到细流程，先随机采样 SuperCircuit masks，用 DAG input-output path count、两比特门平衡、硬件拓扑可映射性等便宜结构代理预筛，再交给四个 zero-cost metrics 排序。
- P1.2 已完成第一版：`supercircuit_evolution` 在 SuperCircuit mask 上做多代 elite selection、mutation、crossover，每一代都用四个 zero-cost metrics 评分。
- P1.3 已完成第一版：`run_task_feedback_validation_experiment` 用 zero-cost 搜索得到父代，再以小预算任务调参后的分数作为反馈，突变出下一代。它是任务相关精搜路线，不属于 zero-cost 主评估。
- P2/P3 待做：更多实例、噪声/硬件 profile、统计胜率与消融。

当前 demo 优先级已调整为 VQE 最小闭环：

- D0 已完成：2-qubit H2 toy Hamiltonian + HEAMask 搜索空间 + zero-cost guardrail + final multi-start validation；H2 只作为 smoke demo。
- D1 待做：在云端 NPU/torch 环境运行 `python -m nexq.qas.demo.vqe_h2_hea_search`，记录输出表和耗时。
- D2 待做：把 H2 demo 从单 seed 扩展为 3-5 seed 汇总，报告 `sa_best`、Stage 1 top、baselines 的平均能量和胜率。
- D3 待做：扩大 HEA 搜索空间到更多 qubit/layer 或更真实的 VQE Hamiltonian，验证 Stage 1 guardrail 是否能稳定压缩候选池。
- D4 待做：若 SA 结果稳定，再加入 GA/reflective mutation；ADAPT-VQE / operator-pool 搜索先作为 future work，不进入当前最小 demo 主线。

SuperCircuit/SubCircuit zero-cost 搜索已提供第一版，不训练 SuperCircuit、不做参数继承，只把 SuperCircuit 作为结构搜索空间，每个 SubCircuit mask 作为一个候选架构：

```python
from nexq.qas import ArchitectureSearch, SearchConfig

search = ArchitectureSearch()
result = search.run(
    SearchConfig(
        n_qubits=4,
        candidate_layers=2,
        search_strategy="supercircuit",
        include_common_candidates=False,
        population_size=16,
        top_k=5,
    )
)
```

Progressive 版本先做便宜结构预筛，适合把候选池开大，再只对保留的子集计算更贵的 expressibility / trainability：

```python
from nexq.metrics.hardware import HardwareProfile

search = ArchitectureSearch(hardware_profile=HardwareProfile(coupling_map=[(0, 1), (1, 2), (2, 3)]))
result = search.run(
    SearchConfig(
        n_qubits=4,
        candidate_layers=2,
        search_strategy="supercircuit_progressive",
        include_common_candidates=False,
        population_size=64,
        progressive_keep=12,
        top_k=5,
    )
)
```

Evolution 版本会多代迭代，适合正式搜索：

```python
result = search.run(
    SearchConfig(
        n_qubits=4,
        candidate_layers=2,
        search_strategy="supercircuit_evolution",
        include_common_candidates=False,
        population_size=16,
        search_generations=3,
        beam_width=4,
        mutation_rate=0.35,
        top_k=5,
    )
)
```

任务反馈版本会在 zero-cost 之后引入小预算调参反馈：

```python
from nexq.qas import run_task_feedback_validation_experiment

report = run_task_feedback_validation_experiment(
    maxcut_line(n_qubits=4),
    search_config=SearchConfig(
        n_qubits=4,
        candidate_layers=2,
        search_strategy="supercircuit_evolution",
        population_size=12,
        search_generations=2,
        beam_width=3,
    ),
    optimizer_config=OptimizerConfig(max_evaluations=8, seed=2026),
    qas_top_k=3,
    feedback_generations=2,
)
```

三种搜索策略可以用统一 demo 横向比较：

```bash
python -m nexq.qas.demo.qas_strategy_comparison
```

正式组合链路是：

```text
progressive 粗筛 -> reflective evolution zero-cost 结构搜索 -> task_feedback 小预算任务精修
```

对应接口：

```python
from nexq.qas import run_hybrid_qas_validation_experiment
```

在把 random-parameter objective 用作 VQE/任务搜索 fitness 之前，先运行 proxy validation，检查随机参数分布是否和短步优化结果有区分度：

```bash
python -m nexq.qas.demo.random_proxy_validation
```

## 5. Minimal VQE-QAS Demo

当前最小可跑 VQE-QAS demo 使用硬编码 Pauli Hamiltonian 和 HEA 搜索空间，目标是展示两阶段闭环，而不是追求化学精度：

```text
Stage 1: 四个 zero-cost metrics 做 guardrail/filter
Stage 2: 用当前验证更可靠的 zero-cost signal 做候选排序/选种
Final: Stage 1/2 选出的 candidates + baselines 做 fair multi-start VQE validation
```

运行：

```bash
python -m nexq.qas.demo.vqe_h2_hea_search
```

H2 toy 问题太小，很多合理 HEA baseline 都能接近参考能量，因此它主要证明 pipeline 可运行。当前 4-qubit transverse-field Ising/TFIM 主 demo 已切换为 trainability-prior Stage1：先用四项 zero-cost metrics 做 guardrail，再用 `trainability` 主导排序，最后直接对 trainability top candidates 做 fair final VQE validation。

```bash
python -m nexq.qas.demo.vqe_ising4_hea_search
```

等价的显式入口是：

```bash
python -m nexq.qas.demo.vqe_ising4_trainability_prior
```

这样改主线的原因是云端诊断显示 short-step VQE 在小预算下会误导搜索方向：`short_40` 和 fair final VQE 的 Spearman 约为 `-0.806`，即使增加到 `short_400` 也只有约 `0.333`；相反，`trainability` 是当前 top candidate 集合里最稳定的正相关 zero-cost signal，Spearman 约为 `0.491`、top-5 overlap 为 `3/5`。

以下 runner 保留为诊断工具，不再作为当前主线。若要判断 SA 步数是否足够、以及 final VQE 是否因为固定 cap 对多参数结构不公平，可以运行 budget sweep：

```bash
python -m nexq.qas.demo.vqe_ising4_budget_sweep
```

该 runner 固定 `T_init=0.30`、`T_final=0.001`，只改变 `steps=36/80/120`，并同时输出两套 final validation：

- capped budget：沿用 smoke demo 的固定上限，便于和已有结果对齐。
- per-param fair budget：按 `max(40, n_params * 20)` 设置每个 start 的评估预算，用于检查参数更多的搜索结构是否被固定 cap 低估。

如果 budget sweep 显示单链 SA 很快陷入同一个局部区域，可以运行 diverse multi-start SA。它不是简单取 Stage 1 top-5，而是从 top-1 开始，用 mask 汉明距离贪心选择结构差异更大的起点；但该 runner 仍依赖 short-step VQE fitness，因此只用于诊断：

```bash
python -m nexq.qas.demo.vqe_ising4_multistart_sa
```

在继续 GA / beam search 之前，先验证 short-step VQE fitness 是否能预测 fair final VQE 排序：

```bash
python -m nexq.qas.demo.vqe_ising4_fitness_correlation
```

如果 `spearman_short_vs_fair` 较低，说明搜索失败主要来自 fitness 信号不可靠，而不是 SA 步数或起点不足。

进一步定位最小可用 short-step 预算，并比较 zero-cost 分数和 fair VQE 的相关性：

```bash
python -m nexq.qas.demo.vqe_ising4_fitness_budget_sweep
```

该 runner 固定同一批 Stage1 top candidates 和同一套 fair final VQE energy，只改变 short-step budget（默认 `40/100/200/400`），同时报告 `weighted / expressibility / trainability / noise / hardware` 与 fair VQE 的相关性。

`vqe_ising4_trainability_prior` 会用 trainability 主导 Stage1 排序，一方面直接对 trainability top candidates 跑 fair final VQE，另一方面保留 trainability top-1 seeded SA trace 作为反例/诊断：如果 SA 被 short-step fitness 拉向更差结构，报告会把它和 baselines 一起列出来。

HEA mask 的搜索维度固定为：

- `layers`: `1 / 2 / 3`
- `rotation_block`: `ry / ry_rz / rx_ry_rz`
- `entangler`: `cx / cz / rzz`
- `final_rotation`: `ry / ry_rz`
- `entangle_pattern`: `linear / ring`

注意：

- 2-qubit H2 的候选空间很小，Stage 1 在这个 demo 中主要展示流程；更大 qubit/layer 搜索空间下才会体现明显压缩价值。
- Stage 1 summary 会打印 `candidates / kept / filtered` 和四项指标的 `min / p25 / max`，用于判断 guardrail 是否真的产生过滤，而不是只看前几条 top rows。
- 当前 4q Ising 主线不再把 short-step VQE/SA 作为核心搜索结论；short-step VQE 只作为诊断，因为小预算下它和 fair VQE 排序负相关。
- 若后续重新启用 SA/GA/beam search，必须先证明所用 fitness 与 fair final VQE 至少有稳定正相关和可接受的 top-k overlap。

## 6. 使用方法：PPO_RB

`PPO_RB` 的输入是目标密度矩阵，输出是策略参数 `theta` 与搜索得到的 `Circuit`。

### 6.1 输入参数（`ppo_rb_qas`）

函数签名：`ppo_rb_qas(target_density_matrix, epsilon, config=None)`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `target_density_matrix` | `np.ndarray` | 是 | 目标态密度矩阵，必须是方阵，维度为 `2^n × 2^n`。例如 3 比特时为 `(8, 8)`。内部会转为 `complex64`。 |
| `epsilon` | `float` | 是 | 保真度阈值。环境在 `fidelity >= epsilon` 时结束当前 episode，并叠加 `terminal_bonus`。常用 `0.9~0.99`。 |
| `config` | `PPORollbackConfig \\| None` | 否 | 训练配置；传 `None` 使用默认超参数。 |

返回值：

- `theta: Dict[str, torch.Tensor]`：策略网络参数快照。
- `circuit: Circuit`：训练过程中发现的最优线路（若未记录到，则回退到当前策略贪婪推演得到的线路）。

### 6.2 超参数（`PPORollbackConfig`）

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

from nexq.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas

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

## 6. 使用方法：PPR_DQL

`PPR_DQL` 的输入是目标 `State`，可直接返回 `Circuit`，也可以返回包含训练信息的结果对象。

### 6.1 输入参数（`train_ppr_dql` / `ppr_dql_state_to_circuit`）

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

### 6.2 超参数（`PPRDQLConfig`）

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
from nexq.qas import PPRDQLConfig, ppr_dql_state_to_circuit

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

## 7. 使用方法：CRLQAS

`CRLQAS` 的目标是最小化给定哈密顿量的能量。结构搜索由 DDQN 决策，参数优化由 Adam-SPSA 执行。

### 7.1 输入参数（`train_crlqas` / `crlqas`）

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

### 7.2 超参数（`CRLQASConfig`）

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

### 7.3 最小示例（H2）

```python
from nexq.qas import AdamSPSAConfig, CRLQASConfig, train_crlqas
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

## 8. 可选：自定义动作门集合

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

## 9. 示例脚本

- `PPO_RB_demo_ghz4.py`：使用 PPO-RB 搜索 4 比特 GHZ 线路
- `PPR_DQL_demo_ghz3.py`：使用 PPR-DQL 搜索 3 比特 GHZ 线路，并导出 OpenQASM 3.0 到 `demo/ppr_dql_ghz3_circuit.qasm`
- `CRLQAS_demo_h2.py`：使用 CRLQAS 搜索小分子 H2 的低能量线路，并导出 OpenQASM 3.0 到 `demo/crlqas_h2_circuit.qasm`

运行示例：

```bash
python nexq/qas/demo/PPR_DQL_demo_ghz3.py
python nexq/qas/demo/CRLQAS_demo_h2.py
```


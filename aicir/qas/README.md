# QAS 模块说明

`aicir.qas` 是量子构架搜索（Quantum Architecture Search, QAS）模块。

该模块用于自动搜索量子线路结构。当前仓库中包含多类 QAS 实现：一类面向变分量子算法（VQA）的 ansatz 架构搜索，一类面向给定目标态或哈密顿量任务的强化学习式量子线路搜索，另有面向 VQE ansatz 拓扑压缩的多目标遗传搜索。算法实现位于 `aicir/qas/algorithms/`；运行 `supernet.py`、`CRLQAS.py`、`PPR_DQL.py` 和 `PPO_RB.py` 需要可用的 `torch`，`MoG_VQE.py` 的默认路径不依赖 `torch`。

- `algorithms/supernet.py`：基于超网络和权重共享的 VQA ansatz 架构搜索，支持分类任务和 H2 VQE 示例
- `algorithms/MoG_VQE.py`：MoG-VQE（Multiobjective Genetic VQE），输入 block-based hardware-efficient ansatz，使用 NSGA-II 修改线路拓扑，并输出修改后的 aicir `Circuit`
- `algorithms/CRLQAS.py`：课程学习 + DDQN + Adam-SPSA 的量子架构搜索（面向哈密顿量能量最小化）
- `algorithms/PPR_DQL.py`：基于 aicir 状态演化实现的 PPR-DQL（Probabilistic Policy Reuse with Deep Q-Learning）
- `algorithms/PPO_RB.py`：Trust Region-based PPO with Rollback 版本的量子架构搜索

当前重点主线是 `vqe_loop/`：面向 VQE/QAS 的“局域 oracle + EA 闭环搜索”。流程从候选 ansatz 空间生成开始，经过 hard filter 和 zero-cost soft flag，用少量 fair VQE 标签建立 benchmark table，再通过 trust-region oracle 判断新候选是否位于可信邻域内；可信候选进入排序，不可信候选 abstain，并将 boundary、sparse、control 候选送回 VQE 扩表。对于 LiH 这类 12 qubits、复杂 Pauli 项 Hamiltonian，当前策略默认降低 `trackA_local` 比例，把 `trackB_sparse` 作为主探索方向，并保留少量 `boundary` 与 `control_random` 防止 oracle 或 zero-cost 偏置。

Hamiltonian 输入已经从手写 `--hamiltonian-id` 迁移到规范化输入：`--hamiltonian` 可接收 legacy `[[coeff, pauli], ...]` 加权 Pauli 项，也可接收 `kind=molecular` 的分子构型、基组、active space 和 mapping 规格。分子输入由 `aicir.chemistry.spec` 通过可选 PySCF/Qiskit Nature 依赖一次性生成 qubit Hamiltonian；后续 fair VQE 标签仍走 aicir 后端，Ascend NPU 只用于 VQE 执行，不要求 PySCF 或 Qiskit 支持 NPU。

### QAS 子项目 README 索引

| 子项目 README | 说明 |
| --- | --- |
| `vqe_loop/README.md` | VQE-QAS 闭环入口、fair-label 协议、trust-region geometry、P1/Stage-2 planning、多 NPU 分片和 Hamiltonian 输入格式。 |

目前 `qas/` 下只有 `vqe_loop/` 维护独立子 README；其他子目录的使用说明集中在本文件中。

当前代码分层：

| 子目录 | 独立 README | 说明 |
| --- | --- | --- |
| `core/` | 无 | 传统 QAS 的类型、评估器、reward、search env、统一 runner/config。 |
| `primitives/` | 无 | 可复用 ansatz 构造和 backend 解析工具。 |
| `algorithms/` | 无 | MoG-VQE、PPO-RB、PPR-DQL、CRLQAS、supernet 等具体算法实现。 |
| `library/` | 无 | 可复用候选架构库。 |
| `problems/` | 无 | VQE/QAS 问题和 Hamiltonian 构造。 |
| `demos/` | 无 | 旧版或研究型演示脚本；新闭环优先使用 `vqe_loop/` 包入口。 |
| `tests/` | 无 | QAS 相关测试和 smoke checks。 |
| `vqe_loop/` | `vqe_loop/README.md` | VQE-QAS 闭环，包括一键入口、fair-label 协议、trust-region geometry、P1/Stage-2 planning、fair VQE 和多 NPU 分片。 |

## 1. 已提供能力

- VQA ansatz 搜索：`supernet.py` 使用超网络、权重共享、架构排序和微调，为分类、VQE 或自定义 VQA 目标选择 ansatz。
- 多目标遗传 VQE 拓扑搜索：`MoG_VQE.py` 将线路表示为二量子比特 block 序列，通过 NSGA-II 同时最小化能量和 CNOT 数量，适合从 block-based HEA 出发压缩 VQE ansatz。
- 多超网络训练：`supernet.py` 支持 `supernet_num`，每次采样架构后选择损失最小的超网络并只更新该超网络的活跃参数。
- 内置 VQA 示例：`classification_supernet` 提供 3 量子比特合成二分类任务，`h2_vqe_supernet` 提供 4 量子比特 H2 VQE 任务。
- 强化学习线路搜索：`PPO_RB.py`、`PPR_DQL.py` 和 `CRLQAS.py` 从空线路出发逐步追加 aicir 支持的量子门，搜索目标态制备或低能量线路。
- 参数优化：VQA QAS 使用 PyTorch autograd 或参数位移法；CRLQAS 使用 Adam-SPSA 微调变分参数。
- 输出对象：各方法最终都返回或包含 aicir `Circuit`，可继续使用 `Circuit.show()` 展示，并可导出为 OpenQASM 2.0 / 3.0。
- 约束：核心线路构建和仿真使用 aicir；动作门或搜索门需要是 `aicir/core/gates.py` 支持的门。

## 2. 接口

使用统一入口 `run` 和统一配置工厂 `config`。用户只需要记住 QAS 方法名，例如 `supernet` 对应 `config.supernet(...)`，不需要导入 `SupernetConfig`、`PPRDQLConfig` 等具体配置类。

推荐入口：

- `run(method, **kwargs)`：按方法名运行对应 QAS 实现
- `config.<method>(**kwargs)`：按方法名创建对应配置对象，例如 `config.supernet(...)`、`config.ppr_dql(...)`
- `config.create(method, **kwargs)`：当方法名来自字符串或外部配置文件时，按方法名创建配置对象
- `QASRunConfig`：把方法名、配置和任务输入封装成请求对象后传给 `run`
- `default_qas_config(method, **kwargs)`：兼容旧入口，内部等价于 `config.create(method, **kwargs)`
- `available_qas_methods()`：返回当前支持的统一入口方法名

统一入口支持的方法和参数：

| `method`               | 必要参数                                     | 可选参数                                 | 返回值               |
| ------------------------ | -------------------------------------------- | ---------------------------------------- | -------------------- |
| `"supernet"`            | `objective` 或在 `config` 中指定内置任务 | `config`、`dataset`、`hamiltonian` | `SupernetResult`     |
| `"supernet_classification"` | 无                                           | `config`                               | `SupernetResult`     |
| `"supernet_h2"`             | 无                                           | `config`                               | `SupernetResult`     |
| `"ppo_rb"`             | `target_density_matrix`、`epsilon`       | `config`                               | `(theta, circuit)` |
| `"ppr_dql"`            | `target_state`                             | `config`、`policy_library`           | `PPRDQLResult`     |
| `"crlqas"`             | `hamiltonian`                              | `config`                               | `CRLQASResult`     |

方法名大小写不敏感，也支持常见别名，例如 `"h2"`、`"h2_vqe"`、`"ppr"`、`"crl"`。

示例：

```python
from aicir.qas import config, run

cfg = config.supernet(task="classification", supernet_steps=20, ranking_num=20, finetune_steps=5)
result = run("supernet", config=cfg)

# 如果方法名来自外部配置文件，可以用字符串创建配置对象。
cfg = config.create("supernet", task="classification", supernet_steps=20)
result = run("supernet", config=cfg)
```

各方法的配置函数：

- `config.supernet(...)`：supernet 搜索空间和训练超参数配置
- `config.supernet_classification(...)`：内置分类任务的 supernet 配置
- `config.supernet_h2(...)`：内置 H2 VQE 任务的 supernet 配置
- `config.ppo_rb(...)`：PPO-RB 超参数配置
- `config.ppr_dql(...)`：PPR-DQL 超参数配置
- `config.crlqas(...)`：CRLQAS 超参数配置；`adam_spsa` 可传字典，例如 `config.crlqas(adam_spsa={"iterations": 10})`

底层专用接口仍然保留，适合需要直接控制某个算法实现的用户：

- `train_supernet(objective, config=None, dataset=None, hamiltonian=None)`
- `block_hardware_efficient_ansatz(n_qubits, layers=1, topology="linear")` / `run_mog_vqe(initial_ansatz, hamiltonian=None, energy_evaluator=None, config=None, backend=None)`
- `classification_supernet(config=None)` / `h2_vqe_supernet(config=None)`
- `ppo_rb_qas(target_density_matrix, epsilon, config=None)`
- `train_ppr_dql(state, config=None, policy_library=None)` / `ppr_dql_state_to_circuit(...)`
- `train_crlqas(hamiltonian, config=None)` / `crlqas(...)`

可通过以下方式导入：

```python
from aicir.qas import config, run
```

## 3. supernet：面向变分量子算法的超网络架构搜索

`supernet.py` 面向 VQA ansatz 结构搜索，使用一阶段超网络、按层权重共享、架构排序和选中 ansatz 微调。

论文依据：

- Yuxuan Du, Tao Huang, Shan You, Min-Hsiu Hsieh, Dacheng Tao, *Quantum circuit architecture search for variational quantum algorithms*, npj Quantum Information, 2022.

当前实现使用 `aicir.core.Circuit`、aicir 门构造器、`GPUBackend` 和 aicir 态向量/矩阵演化能力，不依赖 PennyLane、Qiskit、Cirq 或其他量子 SDK。当前实现以无噪声仿真为主；`NoiseConfig` 仅作为 noisy QAS 的占位配置，启用时会抛出 `NotImplementedError`。

Supernet assignment 规则作为默认实现：每个采样 ansatz 会先在全部 `W=supernet_num` 个 supernet 上计算 objective，然后分配给 loss 最小的 supernet，并只更新该 supernet 的活跃参数。补充材料中的 evolutionary ranking 和 noisy differentiable QAS 目前保留为显式扩展点，开启时会抛出 `NotImplementedError`。

### 3.1 输入参数（`train_supernet`）

函数签名：

- `train_supernet(objective, config=None, dataset=None, hamiltonian=None)`
- `classification_supernet(config=None)`
- `h2_vqe_supernet(config=None)`

| 参数            | 类型                                               | 必填 | 说明                                                                                            |
| --------------- | -------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------- |
| `objective`   | `Callable \| str \| None`                          | 否   | 自定义目标函数，或内置任务名。`classification_supernet` 和 `h2_vqe_supernet` 会自动选择内置目标。 |
| `config`      | `SupernetConfig \| None`                            | 否   | 搜索空间、训练步数、学习率和随机种子等配置；传 `None` 使用默认值。                            |
| `dataset`     | `Mapping \| None`                                 | 否   | 分类任务数据集。为 `None` 时，内置分类任务会生成 3 维合成二分类数据。                         |
| `hamiltonian` | `Hamiltonian \| np.ndarray \| torch.Tensor \| None` | 否   | VQE 任务哈密顿量。为 `None` 时，`h2_vqe_supernet` 使用内置 4 量子比特 H2 哈密顿量。              |

返回值：

- `SupernetResult`
  - `best_architecture`：排序和微调后选中的 `Architecture`
  - `best_circuit`：选中架构对应的 aicir `Circuit`
  - `best_score`：分类任务为验证损失，H2 VQE 任务为微调后能量
  - `best_supernet_id`：排序阶段选中 ansatz 所属的 supernet 编号
  - `ranking_records`：候选架构排序记录
  - `supernet_log`：超网络训练日志
  - `finetune_log`：固定架构微调日志
  - `final_metrics`：任务相关最终指标

### 3.2 超参数（`SupernetConfig`）

| 字段                          |                       默认值 | 说明                                                                                 |
| ----------------------------- | ---------------------------: | ------------------------------------------------------------------------------------ |
| `n_qubits`                  |                        `3` | 量子比特数。                                                                         |
| `layers`                    |                        `3` | ansatz 层数。                                                                        |
| `single_qubit_gates`        |                  `("ry",)` | 单量子比特候选旋转门，目前支持 `rx`、`ry`、`rz`。                              |
| `two_qubit_pairs`           | `((0, 1), (0, 2), (1, 2))` | 允许搜索 CNOT/无 CNOT 的连接对，格式为 `(control, target)`。                       |
| `search_single_qubit_gates` |                     `True` | 是否搜索单量子比特门布局；关闭时每层使用第一个候选门。                               |
| `search_two_qubit_gates`    |                     `True` | 是否搜索双量子比特门 mask；关闭时默认启用所有给定连接对。                            |
| `supernet_num`              |                        `1` | 超网络数量 `W`。每个采样架构会在所有超网络上评估，并只更新损失最小的超网络。       |
| `supernet_steps`            |                      `100` | 一阶段超网络优化步数。                                                               |
| `ranking_num`               |                       `50` | 排序阶段采样的候选架构数量。                                                         |
| `finetune_steps`            |                       `20` | 对选中固定架构执行独立参数微调的步数。                                               |
| `learning_rate`             |                     `0.05` | 超网络共享参数学习率。                                                               |
| `finetune_learning_rate`    |                     `0.03` | 固定架构微调学习率。                                                                 |
| `seed`                      |                       `42` | 随机种子。                                                                           |
| `device`                    |                    `"cpu"` | `GPUBackend` 使用的设备。                                                        |
| `task`                      |         `"classification"` | 内置任务类型，如 `classification` 或 `h2_vqe`。                                  |
| `log_interval`              |                        `0` | 日志打印间隔；`0` 表示关闭。                                                       |
| `use_parameter_shift`       |                    `False` | 是否使用参数位移法更新梯度；默认使用 PyTorch autograd。                              |
| `track_best_validation`     |                     `True` | 分类任务中按补充材料记录验证准确率最优的 supernet 参数，并在 ranking 前恢复。        |
| `ranking_strategy`          |                 `"random"` | 排序阶段采样策略。默认随机采样；`"evolutionary"` 为补充材料扩展点，尚未实现。      |
| `use_evolutionary_ranking`  |                    `False` | 是否启用 evolutionary ranking 扩展点；启用时当前实现会抛出 `NotImplementedError`。 |
| `noise_mode`                |                   `"none"` | 噪声模式。当前仅支持无噪声；其他值会抛出 `NotImplementedError`。                   |

实践建议：

- 分类任务默认搜索空间为 3 量子比特、3 层、固定 `RY`，并在 `(0, 1)`、`(0, 2)`、`(1, 2)` 上搜索 CNOT/无 CNOT；当 `layers=3` 时逻辑搜索空间大小为 `8^3`。
- H2 VQE 默认搜索空间为 4 量子比特、3 层、`RY/RZ`，并在链式连接 `(0, 1)`、`(1, 2)`、`(2, 3)` 上搜索 CNOT/无 CNOT；当 `layers=3` 时逻辑搜索空间大小为 `128^3`。
- 权重共享规则与论文一致：两个 ansatz 只要在第 `l` 层具有相同单量子比特门布局，就共享该层可训练参数；共享关系不受双量子比特门选择和其他层布局影响。

### 3.3 分类任务示例

```python
from aicir.qas import config, run

cfg = config.supernet(
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

result = run("supernet_classification", config=cfg)
print(result.best_score)
print(result.best_circuit.show())
```

内置分类任务按补充材料构造 300 个 3 维样本，并划分为 100 个训练样本、100 个验证样本和 100 个测试样本。输入编码为 `RY(x1) ⊗ RY(x2) ⊗ RY(x3)`；默认标签由一个固定随机 `U*(theta*)` 量子核生成，测量量为最后一个量子比特处于 `|0>` 的概率：概率大于等于 `0.75` 标为 `1`，小于等于 `0.25` 标为 `0`，中间样本丢弃并重采样。训练 loss 使用 MSE，预测阈值为 `0.5`。默认搜索空间固定单量子比特门为 `RY`，并在 `(0, 1)`、`(0, 2)`、`(1, 2)` 上搜索是否添加 CNOT；当 `layers=3` 时，逻辑搜索空间大小为 `8^3`。

### 3.4 H2 VQE 示例

```python
from aicir.qas import config, run

cfg = config.supernet_h2(
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

result = run("supernet_h2", config=cfg)
print(result.final_metrics)
print(result.best_circuit.show())
```

内置 H2 任务使用补充材料 Eq. (19) 的 4 量子比特 Pauli 哈密顿量：

```text
H = -0.042 + 0.178(Z0 + Z1) - 0.243(Z2 + Z3)
    + 0.171 Z0Z1 + 0.123(Z0Z2 + Z1Z3)
    + 0.168(Z0Z3 + Z1Z2) + 0.176 Z2Z3
    + 0.045(Y0X1X2Y3 - Y0Y1X2X3 - X0X1Y2Y3 + X0Y1Y2X3)
```

默认搜索 `RY`/`RZ` 单量子比特门布局，以及链式连接对 `(0, 1)`、`(1, 2)`、`(2, 3)` 上的 CNOT/无 CNOT 选择。结果会报告 QAS 排序阶段最优能量、微调后能量、固定硬件高效 VQE baseline、选中线路和 CNOT 数量。当 `layers=3` 时，逻辑搜索空间大小为 `128^3`。

### 3.5 封装入口 `supernet_qas`（任意哈密顿量基态搜索）

对任意分子/自定义哈密顿量做 VQE ansatz 搜索时，无需手工拼装 `SupernetConfig`。封装函数 `supernet_qas` 把最常调节的几个旋钮直接做成参数，其余字段使用经过验证的 VQE 默认值；`n_qubits` 和 `two_qubit_pairs` 会自动从哈密顿量推断。

> 命名说明：函数名为 `supernet_qas`（区别于子模块 `aicir.qas.algorithms.supernet`，避免名称冲突），可直接从包层级导入：`from aicir.qas import supernet_qas`。

函数签名：

```python
from aicir.qas import supernet_qas

result = supernet_qas(
    hamiltonian,            # 必填；n_qubits 从 hamiltonian.n_qubits 推断
    layers=6,               # 主要旋钮 1：ansatz 深度 L
    supernet_num=5,         # 主要旋钮 2：超网络数量 W
    supernet_steps=250,     # 主要旋钮 3：超网络优化步数
    finetune_steps=250,     # 主要旋钮 4：选中架构微调步数
)
```

| 参数                       | 默认值                  | 说明                                                                     |
| -------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `hamiltonian`            | 必填                    | 目标哈密顿量；需暴露 `n_qubits`，否则用 `n_qubits=` 显式传入。         |
| `layers`                 | `6`                   | ansatz 深度 `L`。                                                       |
| `supernet_num`           | `5`                   | 权重共享超网络数量 `W`。                                                |
| `supernet_steps`         | `250`                 | 一阶段超网络优化步数。                                                    |
| `finetune_steps`         | `250`                 | 对选中架构独立微调的步数（最终能量的主要来源）。                          |
| `n_qubits`               | `None`                | 量子比特数；`None` 时从 `hamiltonian.n_qubits` 推断。                  |
| `ranking_num`            | `80`                  | 排序阶段采样的候选架构数量。                                              |
| `single_qubit_gates`     | `("i","h","rx","ry","rz")` | 单量子比特门池。                                                     |
| `two_qubit_gates`        | `("cx","rzz")`        | 双量子比特门池（`rzz` 为带参纠缠门）。                                  |
| `two_qubit_pairs`        | `None`                | 纠缠连接；`None` 时自动取最近邻 + 次近邻（线性链上 `1 <= j-i <= 2`）。 |
| `learning_rate`          | `0.1`                 | 超网络阶段 Adam 学习率。                                                  |
| `finetune_learning_rate` | `0.05`                | 微调阶段 Adam 学习率。                                                    |
| `seed`                   | `2`                   | 随机种子。                                                                |
| `device`                 | `"cpu"`               | Torch 设备字符串，如 `"cpu"`、`"cuda"`、`"npu:0"`。                  |
| `use_parameter_shift`    | `False`               | 是否改用参数移位梯度（默认使用 autograd）。                               |
| `**config_overrides`     | —                       | 其余任意 `SupernetConfig` 字段（如 `ranking_strategy`）。              |

返回值与 `train_supernet` 一致，为 `SupernetResult`：`final_metrics["fine_tuned_energy"]` 为微调后能量，`final_metrics["baseline_vqe_energy"]` 为固定 ansatz 的 VQE baseline，`best_circuit` 为选中并微调后的线路。

示例（自定义哈密顿量，CPU）：

```python
from aicir.operators import Hamiltonian
from aicir.qas import supernet_qas

ham = Hamiltonian(n_qubits=6, terms=[("IIIIII", -4.524), ("IIIIIZ", 0.515), ...])

result = supernet_qas(ham, layers=6, supernet_num=5,
                      supernet_steps=250, finetune_steps=250)
print(result.final_metrics["fine_tuned_energy"])
print(result.final_metrics["baseline_vqe_energy"])
print(result.best_circuit.show())
```

在 Ascend NPU 上只需改一个 `device` 参数（其余封装逻辑不变）：

```python
result = supernet_qas(ham, layers=6, supernet_num=5,
                      supernet_steps=250, finetune_steps=250,
                      device="npu:0")
```

### 3.6 VQE-QAS 一键闭环入口

VQE-QAS 的闭环代码放在 `aicir.qas.vqe_loop`，和 `algorithms/` 下的 MoG-VQE、PPO-RB 等算法实现并列但隔离。常规使用优先调用包级入口：

```bash
python -m aicir.qas.vqe_loop --hamiltonian lih_molecular_spec.json
```

也可以把 Hamiltonian JSON 作为位置参数：

```bash
python -m aicir.qas.vqe_loop lih_molecular_spec.json
```

默认运行参数为 `--rounds auto --batch-size auto --backend npu --dtype complex64`，输出目录会从文件名自动生成，例如 `lih_molecular_spec.json` 对应 `outputs/qas_lih_molecular_spec_loop`。这些参数仍可显式覆盖。

`--hamiltonian` 支持两类输入：

- 加权 Pauli 项：`[[coeff, pauli], ...]`，适合已有 qubit Hamiltonian 的场景。
- 分子规格：`{"kind": "molecular", "geometry": ..., "basis": ..., "active_space": ..., "mapping": ...}`，由 `aicir.chemistry.spec` 通过 PySCF/Qiskit Nature 一次性生成 Pauli Hamiltonian。`kind`、`basis`、`charge`、`spin`、`unit`、`driver` 和 `mapping` 都有默认值；LiH 这类双原子分子可只写 `{"molecule": "LiH", "distance": 0.1}`。

`hamiltonian_id` 默认由规范化 Hamiltonian 内容自动派生；只有需要兼容旧实验目录或人工指定 benchmark 标识时，才传 `--hamiltonian-id` 覆盖。

Python API 入口仍然保留：

```python
from pathlib import Path
from aicir.chemistry.spec import load_hamiltonian_input
from aicir.qas.vqe_loop import ClosedLoopConfig, run_vqe_qas_closed_loop

generated = load_hamiltonian_input("lih_molecular_spec.json")

result = run_vqe_qas_closed_loop(
    ClosedLoopConfig(
        output_dir=Path("outputs/qas_lih_loop"),
        n_qubits=generated.n_qubits,
        hamiltonian_terms=generated.terms,
        hamiltonian_id=generated.hamiltonian_id,
        hamiltonian_class=generated.hamiltonian_class,
        rounds=None,
        backend="npu",
        dtype="complex64",
    )
)
print(result.final_benchmark_table)
```

详细模块职责、调用顺序、协议边界、Hamiltonian 输入格式和辅助命令见 `aicir/qas/vqe_loop/README.md`。闭环逻辑不再放在 demos 或 cli 包里；服务模块可直接通过 `python -m aicir.qas.vqe_loop.<module>` 作为命令入口使用。

### 3.7 QAS fair-label 队列的多 NPU 分片

`aicir` 已经内置 `NPUBackend`，QAS 的 fair VQE runner 可通过 `--backend npu` 使用 Ascend NPU。对于 Stage 1.5 / Stage 2 产生的 label queue，推荐使用分片入口把候选行切成多份并行跑；每个分片是独立 VQE 任务，通过 `LOCAL_RANK` 绑定到一张 NPU，不会把单个态向量切成 HCCL 分布式任务。

```bash
python -m aicir.qas.vqe_loop.sharding \
  --queue outputs/qas_stage2/round1/round1_queue.csv \
  --output outputs/qas_stage2/round1/round1_labels.csv \
  --work-dir outputs/qas_stage2/round1/label_shards \
  --protocol aicir/qas/vqe_loop/fair_label_protocol.json \
  --backend npu \
  --dtype complex64 \
  --num-shards 4 \
  --device-offset 0 \
  --n-seeds 1 \
  --max-evals 300
```

关键参数：

- `--num-shards`：队列分片数。4 张卡可设为 `4`，也可以按机器资源改成 `1`、`2`、`8` 等。
- `--device-offset`：第 0 个 shard 使用的 `LOCAL_RANK` 偏移量；默认 `0` 会使用 `npu:0` 开始的连续设备。例如只想跳过 `npu:0`，可用 `--device-offset 1 --num-shards 3` 使用 `npu:1` 到 `npu:3`。
- `--backend npu` / `--dtype complex64`：使用 Ascend NPU 后端。若要做 CPU smoke，可改为 `--backend numpy --dtype complex128`。
- `--work-dir`：保存每个 shard 的临时 queue、临时 label CSV 和最终 summary。

分片器会把原始 queue 连续切块，并给底层 `python -m aicir.qas.vqe_loop.labeling` 传入 `--seed-index-offset`，因此同一全局队列行在单进程或多分片运行时使用一致的 seed 派生规则。最终输出仍是一个 benchmark table CSV，可继续用于 oracle reliability feedback、Stage-2/P1 planning 和标签回流。

## 4. MoG_VQE：基于 NSGA-II 的多目标遗传 VQE 拓扑搜索

`MoG_VQE.py` 实现 MoG-VQE 的线路拓扑搜索部分。输入是 block-based hardware-efficient ansatz，算法把线路表示为二量子比特 block 的有序列表，通过插入 block、删除 block 和大尺度变异修改拓扑，并使用 NSGA-II 同时最小化能量和 CNOT 数量。输出是修改后的 aicir `Circuit`、最终 Pareto 前沿和每一代搜索摘要。

论文依据：

- D. Chivilikhin, A. Samarin, V. Ulyantsev, I. Iorsh, A. R. Oganov, O. Kyriienko, *MoG-VQE: Multiobjective genetic variational quantum eigensolver*, arXiv:2007.04424, 2020.

当前实现保持 aicir 原生依赖，不强制引入 DEAP 或 CMA-ES。默认参数优化器是轻量的 `separable_es`；若需要严格复现实验中的 CMA-ES，可通过 `energy_evaluator(circuit)` 接入外部优化流程。`MoG_VQE` 当前作为底层专用接口使用，不经过统一 `run(method, ...)` 分发。

### 4.1 输入参数（`block_hardware_efficient_ansatz` / `run_mog_vqe`）

函数签名：

- `block_hardware_efficient_ansatz(n_qubits, layers=1, topology="linear", block_type="generalized_cnot")`
- `run_mog_vqe(initial_ansatz, hamiltonian=None, energy_evaluator=None, config=None, backend=None)`

| 参数                 | 类型                                                   | 必填 | 说明                                                                                           |
| -------------------- | ------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------- |
| `initial_ansatz`   | `MOGVQEIndividual \| Circuit \| Sequence[MOGVQEBlock]` | 是   | 初始 block-based HEA 拓扑；传入 `Circuit` 时会从其中的 CNOT 连接提取 block 拓扑。              |
| `hamiltonian`      | `Hamiltonian \| np.ndarray \| None`                    | 否   | 目标哈密顿量；提供后使用精确态向量能量评估。                                                   |
| `energy_evaluator` | `Callable[[Circuit], float] \| None`                   | 否   | 自定义能量评估函数；适合接入外部 VQE/CMA-ES/硬件评估流程。                                     |
| `config`           | `MOGVQEConfig \| None`                                 | 否   | NSGA-II、拓扑变异和参数优化超参数；传 `None` 使用默认值。                                      |
| `backend`          | `Any \| None`                                          | 否   | aicir 后端；默认为 `NumpyBackend`。                                                            |

`run_mog_vqe` 至少需要 `hamiltonian` 或 `energy_evaluator` 之一。返回值为 `MOGVQEResult`：

- `best_individual`：搜索到的最优 block 拓扑
- `best_circuit`：搜索到并绑定最优参数后的 aicir `Circuit`
- `best_energy`：最优线路能量
- `best_parameters`：最优连续旋转参数
- `pareto_front`：最终种群中的非支配 Pareto 前沿
- `population`：最终 NSGA-II 种群
- `history`：每一代的最优能量、CNOT 数量和 Pareto 前沿摘要

### 4.2 超参数（`MOGVQEConfig`）

| 字段                            | 默认值               | 说明                                                                 |
| ------------------------------- | -------------------- | -------------------------------------------------------------------- |
| `population_size`             | `16`               | NSGA-II 种群大小。                                                   |
| `generations`                 | `10`               | 拓扑搜索代数。                                                       |
| `mutation_insert_weight`      | `2.0`              | 插入 block 的变异权重。                                              |
| `mutation_delete_weight`      | `1.0`              | 删除 block 的变异权重。                                              |
| `mutation_big_weight`         | `0.25`             | 大尺度变异权重；一次执行多次插入/删除。                              |
| `big_mutation_steps`          | `10`               | 大尺度变异中的插入/删除次数。                                        |
| `min_blocks`                  | `0`                | 允许保留的最小 block 数。                                            |
| `max_blocks`                  | `None`             | 允许的最大 block 数；`None` 表示不设上限。                           |
| `block_type`                  | `"generalized_cnot"` | block 类型；支持 `"generalized_cnot"` 和 `"generalized_two_qubit"`。 |
| `allowed_edges`               | `None`             | 允许插入 block 的有向连接；`None` 时使用全连接有向边。               |
| `parameter_optimizer`         | `"separable_es"`   | 固定拓扑参数优化器；支持 `"separable_es"`、`"random"`、`"none"`。   |
| `parameter_generations`       | `8`                | 参数优化迭代代数。                                                   |
| `parameter_population_size`   | `8`                | 参数优化每代样本数。                                                 |
| `parameter_sigma`             | `0.5`              | `separable_es` 的初始扰动尺度。                                      |
| `parameter_bounds`            | `(-pi, pi)`        | 连续旋转参数范围。                                                   |
| `seed`                        | `None`             | 随机种子。                                                           |

### 4.3 最小示例（自定义能量函数）

```python
from aicir.qas import MOGVQEConfig, block_hardware_efficient_ansatz, run_mog_vqe

initial = block_hardware_efficient_ansatz(
    n_qubits=4,
    layers=2,
    topology="linear",
)

cfg = MOGVQEConfig(
    population_size=16,
    generations=20,
    parameter_generations=10,
    parameter_population_size=8,
    seed=42,
)

def energy_evaluator(circuit):
    # 替换为你的 VQE 能量、硬件测量能量或外部 CMA-ES 评估流程。
    return float(len(circuit.gates))

result = run_mog_vqe(initial, energy_evaluator=energy_evaluator, config=cfg)

print(result.best_energy)
print(result.best_individual.cnot_count)
print(result.best_circuit.show())
```

### 4.4 最小示例（哈密顿量能量）

```python
from aicir.operators import Hamiltonian
from aicir.qas import MOGVQEConfig, block_hardware_efficient_ansatz, run_mog_vqe

ham = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.2)])
initial = block_hardware_efficient_ansatz(n_qubits=2, layers=1, topology="linear")
cfg = MOGVQEConfig(population_size=8, generations=5, seed=7)

result = run_mog_vqe(initial, hamiltonian=ham, config=cfg)

print(result.best_energy)
print(result.best_circuit.show())
```

## 5. PPO_RB：基于真正近端策略优化的量子架构搜索

`PPO_RB` 的输入是目标密度矩阵，输出是策略参数 `theta` 与搜索得到的 `Circuit`。

论文依据：

- X. Zhu and X. Hou, *Quantum architecture search via truly proximal policy optimization*, Scientific Reports, 2023, doi: `10.1038/s41598-023-32349-2`.

### 5.1 输入参数（`ppo_rb_qas`）

函数签名：`ppo_rb_qas(target_density_matrix, epsilon, config=None)`

| 参数                      | 类型                         | 必填 | 说明                                                                                                          |
| ------------------------- | ---------------------------- | ---- | ------------------------------------------------------------------------------------------------------------- |
| `target_density_matrix` | `np.ndarray`               | 是   | 目标态密度矩阵，必须是方阵，维度为 `2^n × 2^n`。例如 3 比特时为 `(8, 8)`。内部会转为 `complex64`。     |
| `epsilon`               | `float`                    | 是   | 保真度阈值。环境在 `fidelity >= epsilon` 时结束当前 episode，并叠加 `terminal_bonus`。常用 `0.9~0.99`。 |
| `config`                | `PPORollbackConfig \| None` | 否   | 训练配置；传 `None` 使用默认超参数。                                                                        |

返回值：

- `theta: Dict[str, torch.Tensor]`：策略网络参数快照。
- `circuit: Circuit`：训练过程中发现的最优线路（若未记录到，则回退到当前策略贪婪推演得到的线路）。

### 5.2 超参数（`PPORollbackConfig`）

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

### 5.3 最小示例（GHZ）

```python
import numpy as np

from aicir.qas import config, run

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

theta, circuit = run("ppo_rb", target_density_matrix=rho_target, epsilon=0.95, config=cfg)

print(f"参数张量数量: {len(theta)}")
print(f"线路门数: {len(circuit.gates)}")
print(circuit.show())
```

## 6. PPR_DQL：基于持续强化学习和策略复用的量子架构搜索

`PPR_DQL` 的输入是目标 `State`，可直接返回 `Circuit`，也可以返回包含训练信息的结果对象。

论文依据：

- *Quantum Architecture Search via Continual Reinforcement Learning*, arXiv:`2112.05779v1`.

### 6.1 输入参数（`train_ppr_dql` / `ppr_dql_state_to_circuit`）

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

### 6.2 超参数（`PPRDQLConfig`）

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

### 6.3 自定义动作门集合

`config.ppr_dql(action_gates=...)` 支持自定义动作门集合。每个动作是一个门字典，格式与 `Circuit` 门定义一致。

注意：

- 不允许包含 `{"type": "unitary", ...}`
- 建议只使用 `gate_to_matrix` 可解析的门

示例：

```python
from aicir.qas import config, run

custom_actions = [
    {"type": "hadamard", "target_qubit": 0},
    {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
]

cfg = config.ppr_dql(action_gates=custom_actions)
result = run("ppr_dql", target_state=state, config=cfg)
circuit = result.circuit
```

### 6.4 最小示例（GHZ）

```python
import numpy as np

from aicir.backends.numpy_backend import NumpyBackend
from aicir.core.state import State
from aicir.qas import config, run

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

result = run("ppr_dql", target_state=state, config=cfg)
circuit = result.circuit

print(circuit)
print(circuit.show())
```

## 7. CRLQAS：面向硬件误差的课程强化学习量子架构搜索

`CRLQAS` 的目标是最小化给定哈密顿量的能量。结构搜索由 DDQN 决策，参数优化由 Adam-SPSA 执行。

论文依据：

- *Curriculum reinforcement learning for quantum architecture search under hardware errors*, arXiv:`2402.03500`.

### 7.1 输入参数（`train_crlqas` / `crlqas`）

函数签名：

- `train_crlqas(hamiltonian, config=None)`
- `crlqas(hamiltonian, config=None)`

| 参数            | 类型                         | 必填 | 说明                                                                              |
| --------------- | ---------------------------- | ---- | --------------------------------------------------------------------------------- |
| `hamiltonian` | `np.ndarray \| Hamiltonian` | 是   | 目标哈密顿量，支持直接传矩阵，或传 `aicir.operators.Hamiltonian` 对象。 |
| `config`      | `CRLQASConfig \| None`      | 否   | 训练超参数；传 `None` 使用默认值。                                              |

返回值：

- `train_crlqas(...) -> CRLQASResult`：包含最优 `circuit`、`minimum_energy`、课程阈值和训练轨迹。
- `crlqas(...) -> Tuple[Circuit, float]`：快捷接口，仅返回 `(circuit, minimum_energy)`。

### 7.2 超参数（`CRLQASConfig`）

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

### 7.3 最小示例（H2）

```python
from aicir.operators import Hamiltonian
from aicir.qas import config, run

h2 = Hamiltonian([
    ("II", -1.052373245772859),
    ("ZI", 0.39793742484318045),
    ("IZ", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156),
])

cfg = config.crlqas(
    max_episodes=300,
    n_act=10,
    adam_spsa={"iterations": 20},
    seed=42,
)

result = run("crlqas", hamiltonian=h2, config=cfg)
print(result.minimum_energy)
print(result.circuit.show())
```

## 8. 示例脚本

- `PPO_RB_demo_ghz4.py`：使用 PPO-RB 搜索 4 比特 GHZ 线路
- `PPR_DQL_demo_ghz3.py`：使用 PPR-DQL 搜索 3 比特 GHZ 线路，并导出 OpenQASM 3.0 到 `demos/ppr_dql_ghz3_circuit.qasm`
- `CRLQAS_demo_h2.py`：使用 CRLQAS 搜索小分子 H2 的低能量线路，并导出 OpenQASM 3.0 到 `demos/crlqas_h2_circuit.qasm`

运行示例：

```bash
python aicir/qas/demos/PPR_DQL_demo_ghz3.py
python aicir/qas/demos/CRLQAS_demo_h2.py
```


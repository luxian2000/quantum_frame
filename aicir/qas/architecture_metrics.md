# Noise-Adaptive QAS Architecture Metrics

本文档定义噪声自适应 QAS 中候选量子架构 / ansatz 的统一评分指标。当前实现遵循一个原则：每个指标组可以列出多种评估函数，但顶层加权时每组只使用一个 active 指标的 `score`，避免同类指标重复加权。

本文档保留在 `qas/` 下，因为它描述的是 QAS 如何组织四类目标、如何加权、哪些指标是 active、哪些指标是后续搜索策略的候选项。具体指标实现不再放在 QAS 内部：

```text
metrics/expressibility.py        无噪表达能力
metrics/noisy_expressibility.py  含噪表达能力
metrics/trainability.py          可训练性
metrics/hardware.py              硬件效率
noise/                              噪声模型、噪声配置、噪声分析、噪声指标
qas/evaluator.py                 QAS 指标编排和加权评分
```

## 1. 统一符号

```text
n = qubit 数
d = 2^n
G = gate 数
G1 = 单比特门数
G2 = 双比特门数
P = 参数数量
M = 参数采样次数
B = histogram bins 数
S = 噪声源数量，当前离子阱模型约 6 类
```

顶层总分：

```text
weighted_score =
    w_expr  * expressibility.score
  + w_train * trainability.score
  + w_noise * noise_robustness.score
  + w_hw    * hardware_efficiency.score
```

## 2. 当前 Active 指标

| 指标组 | active 指标 | 实现位置 | 说明 |
|---|---|---|---|
| 表达能力 | `kl_haar` | `metrics/expressibility.py::KL_Haar_divergence`, `evaluator.py::ArchitectureEvaluator` | 衡量参数化线路诱导态分布与 Haar 随机态分布的距离 |
| 可训练性 | `structure_proxy` | `metrics/trainability.py::structure_proxy` | 使用深度、双比特门比例、参数密度的低成本代理 |
| 噪声鲁棒性 | `ion_trap_error_budget_proxy` | `noise/metrics.py::ion_trap_error_budget_proxy`, `noise/ion_trap.py` | 使用默认离子阱噪声配置估计线路 error budget |
| 硬件效率 | `native_depth_twoq_efficiency` | `metrics/hardware.py::native_depth_twoq_efficiency` | 使用 native gate 比例、深度、双比特门密度评估硬件友好度 |

`structure_proxy` 和 `native_depth_twoq_efficiency` 是 `evaluator.py` 中登记的 active 指标名，真实计算分别在 `metrics/trainability.py` 和 `metrics/hardware.py` 中。`multi_objective_reward.py` 保留 legacy RL reward wrapper；新主线不再从 reward 层直接 import 具体指标函数。

## 3. 表达能力组

表达能力组衡量候选架构在随机参数下可生成态分布的丰富程度。`kl_haar`、`frame_potential`、`entangling_capability` 都属于表达能力组，不应在顶层重复加权。

### 3.1 `kl_haar`（active）

公式：

```text
F = |<psi(theta_i) | psi(theta_j)>|^2

P_PQC(F): 参数化线路采样得到的保真度分布
P_Haar(F) = (d - 1) * (1 - F)^(d - 2)

D_KL = sum_i P_PQC(i) * log(P_PQC(i) / P_Haar(i))
Exp_idle = (d - 1) * log(B)
raw = -log(D_KL / Exp_idle)
score = clip(raw, 0, 1)
```

含义：`D_KL` 越小，线路生成态分布越接近 Haar 随机态；`raw` 越大，表达能力越强。QAS evaluator 在聚合前将 active 指标分数裁剪到 `[0, 1]`，因此最终 `score` 越接近 1，表达能力越强。若线路没有参数化门，当前 evaluator 会回退到结构代理分数，避免候选库中的固定线路直接评估失败。

依据：Sim et al., *Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms*。

复杂度：

```text
O(M * cost(unitary/evolution) + B)
```

在当前 dense simulator 下，线路演化随 `d = 2^n` 指数增长，适合小比特精确评估，大规模搜索需要 surrogate。

### 3.2 `mmd_relative`（implemented，可选）

公式：

```text
MMD^2 = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
k(x, y) = exp(-||x-y||^2 / (4 sigma^2))
score = 1 - MMD
```

其中 `X` 是线路采样得到的概率向量，`Y` 是 Haar-like / Dirichlet 参考样本。

依据：MMD 是分布距离，避免 KL 对 binning 的敏感性。

复杂度：

```text
O(M^2 * d) + M 次线路执行
```

### 3.3 `frame_potential`（todo）

公式，state ensemble 版本：

```text
F_t = (1 / M^2) * sum_ij |<psi_i | psi_j>|^(2t)
F_t^Haar = 1 / C(d + t - 1, t)
```

常用 `t = 2`：

```text
F_2^Haar = 2 / (d * (d + 1))
score = exp(- |F_2 - F_2^Haar| / scale)
```

依据：frame potential 是判断 ensemble 是否接近 t-design 的标准指标，与 `kl_haar` 功能类似，属于表达能力组。

复杂度：

```text
O(M^2 * d) + M 次线路执行
```

### 3.4 `entangling_capability`（todo）

常用 Meyer-Wallach 指标：

```text
Q(|psi>) = 2 * (1 - (1/n) * sum_k Tr(rho_k^2))
score = E_theta[Q(psi(theta))]
```

其中 `rho_k` 是第 `k` 个 qubit 的单比特约化密度矩阵。

依据：Sim et al. 同时讨论 expressibility 和 entangling capability。该指标更偏“产生纠缠的能力”。

复杂度：

```text
O(M * n * d)
```

### 3.5 `transformer_predictor`（todo）

形式：

```text
score = f_phi(tokens(circuit))
```

输入线路 token 序列，输出预测表达能力。

依据：*Learning the expressibility of quantum circuit ansatz using transformer*。适合作为大规模架构搜索 surrogate。

复杂度：

```text
训练: 需要大量已评估线路样本
推理: O(L^2 * hidden_dim)
```

其中 `L` 是线路 token 长度。

## 4. 可训练性组

可训练性组衡量架构是否容易训练，是否容易进入 barren plateau。`gradient_variance`、`gradient_norm`、`barren_plateau_risk` 属于同一组，不应在顶层重复加权。

### 4.1 `structure_proxy`（active）

当前低成本代理：

```text
depth_proxy = 按门作用 qubit 估计的 layer-like 深度
twoq_ratio = G2 / G
params_per_qubit = P / n

depth_score = exp(-depth_proxy / alpha)
entanglement_score = exp(-beta * twoq_ratio)
param_score = exp(-params_per_qubit / gamma)

score = 0.4 * depth_score
      + 0.4 * entanglement_score
      + 0.2 * param_score
```

当前实现近似使用：

```text
alpha = 10
beta = 2
gamma = 5
```

依据：McClean et al. 指出随机深线路更易 barren plateau；Holmes et al. 指出过强表达能力、接近 2-design 时梯度可能变小；noise-induced barren plateau 工作指出噪声和深度会诱导训练退化。

复杂度：

```text
O(G)
```

### 4.2 `gradient_variance`（todo）

先定义 task-agnostic 或 task-aware cost：

```text
C(theta) = <0| U(theta)^dagger O U(theta) |0>
```

例如 `O` 可选局部 Pauli observable。梯度方差：

```text
VarGrad = mean_i Var_theta[dC / d theta_i]
```

score 建议使用目标区间，而不是越大越好：

```text
score = exp(- |log(VarGrad + eps) - log(target_var)| / scale)
```

依据：barren plateau 的核心表现是梯度方差随 qubit 数指数消失。

复杂度，参数移位规则：

```text
O(M * P * cost(circuit_eval))
```

每个参数通常需要正负两次评估，约 `2MP` 次线路执行。

### 4.3 `gradient_norm`（todo）

公式：

```text
GradNorm = E_theta[ ||grad C(theta)||_2 ]
score = exp(- |log(GradNorm + eps) - log(target_norm)| / scale)
```

依据：梯度太小训练不动，太大可能训练不稳定。

复杂度：

```text
O(M * P * cost(circuit_eval))
```

### 4.4 `barren_plateau_risk`（todo）

解释性派生项：

```text
risk = sigmoid(
    a * depth_proxy
  + b * twoq_ratio
  + c * expressibility_closeness
  + d * noise_strength * G
  - bias
)

score = 1 - risk
```

依据：汇总 McClean / Holmes / noise-induced barren plateau 的风险因素。不建议和 `trainability_score` 平级重复加权。

复杂度：

```text
O(G)
```

如果依赖表达能力，则加上表达能力指标成本。

## 5. 噪声鲁棒性组

噪声鲁棒性组衡量候选架构在默认离子阱噪声模型下预计有多稳健。当前 active 使用 `noise/ion_trap.py`：

```python
config = load_default_ion_trap_noise_config()
resolved = config.resolved_parameters()
noise_model = config.build_noise_model(qubits=list(range(n_qubits)))
```

### 5.1 `ion_trap_error_budget_proxy`（active）

使用配置中的：

```text
p1 = oneq_depol
p2 = twoq_depol
pct = cross_talk
pidle_1 = 1/2 * (1 - exp(-oneq_gate_time / T2))
pidle_2 = 1/2 * (1 - exp(-twoq_gate_time / T2))
```

串行离子阱 idle 规则：

```text
单比特门执行时，其余 n - 1 个 qubit idle
双比特门执行时，其余 n - 2 个 qubit idle
```

一阶小错误近似：

```text
C_gate = G1 * p1 + G2 * p2

C_idle = G1 * (n - 1) * pidle_1
       + G2 * max(n - 2, 0) * pidle_2

C_crosstalk = (G1 + G2) * n * pct

C_readout = N_measure * meas_bitflip
C_reset = N_reset * reset_bitflip

C_total = C_gate + C_idle + C_crosstalk + C_readout + C_reset

noise_robustness_score = exp(-C_total)
noise_degradation_proxy = 1 - exp(-C_total)
```

依据：使用当前离子阱配置文件的六类噪声参数；idle 使用报告公式 `1/2 * (1 - exp(-t/T2))`；小错误概率下的一阶 error budget 适合大量候选架构初筛。

复杂度：

```text
O(G)
```

### 5.2 `ideal_noisy_score_gap`（todo）

直接仿真：

```text
S_ideal = expressibility_score(circuit, no_noise)
S_noisy = expressibility_score(circuit, ion_trap_noise_model)

degradation = max(0, S_ideal - S_noisy)
score = exp(-degradation / scale)
```

依据：比 error budget 更真实，可捕捉不同线路结构在噪声演化下的实际退化。

复杂度，密度矩阵噪声仿真时：

```text
O(M * G * K * d^3)
```

其中 `K` 是 Kraus 数。小比特可用，大比特昂贵。

### 5.3 `noise_sensitivity`（todo）

公式：

```text
sensitivity = ideal_metric - noisy_metric
sensitivity_by_gate_type[g] = degradation caused after gates of type g
```

依据：解释哪些门类型或结构导致噪声敏感。

复杂度：

```text
O(number_of_gate_groups * noisy_eval_cost)
```

### 5.4 `per_source_ablation`（todo）

对每类噪声源做开关实验：

```text
full = score(noise_all_on)
score_without_idle
score_without_crosstalk
score_without_twoq
...

impact_source = score_without_source - full
```

依据：回答哪个噪声源影响最大。对离子阱尤其重要，因为 idle、twoq、crosstalk 的物理含义明确。

复杂度：

```text
O(S * noisy_eval_cost)
```

当前 `S` 约为 6。

## 6. 硬件效率组

硬件效率组衡量架构是否适合目标硬件执行，是否门少、浅、native、少路由。

### 6.1 `native_depth_twoq_efficiency`（active）

当前实现公式：

```text
native_ratio = native_gate_count / G

depth_proxy = 按门作用 qubit 估计的 layer-like 深度
depth_score = min(1, max_depth / max(1, depth_proxy * 10))

twoq_ratio = G2 / n
twoq_score = exp(-twoq_ratio / 3)

score = 0.4 * native_ratio
      + 0.3 * depth_score
      + 0.3 * twoq_score
```

依据：NISQ 硬件上双比特门通常更慢、更 noisy；native gate 比例越高，编译开销越低；深度越小，退相干与累计错误越低。

复杂度：

```text
O(G)
```

### 6.2 `connectivity_penalty`（todo）

若硬件 connectivity 给定为图 `E`：

```text
invalid_edges = count(twoq_gate edge not in E)
score = exp(-lambda * invalid_edges)
```

更精确时按最短路径估计 SWAP：

```text
routing_cost = sum_g max(0, shortest_path(control, target) - 1)
score = exp(-lambda * routing_cost)
```

依据：硬件连接约束会引入 SWAP 和额外错误。离子阱全连接时该项通常较高。

复杂度：

```text
邻接查表: O(G)
最短路: O(G * (|V| + |E|))，可预先 Floyd 后 O(G)
```

### 6.3 `calibrated_error_cost`（todo）

如果每类门或每条边有校准错误率 `p_g`：

```text
success_prob = product_g (1 - p_g)
error_cost = -sum_g log(1 - p_g)
score = exp(-error_cost)
```

小错误近似：

```text
error_cost ~= sum_g p_g
```

依据：QuantumNAS / hardware-aware compilation 常用真实设备 noise calibration 进行架构选择。和噪声鲁棒性有重叠，但这里更偏硬件映射成本。

复杂度：

```text
O(G)
```

### 6.4 `latency_cost`（todo）

如果每个门有时间 `t_g`：

```text
串行模型: T_total = sum_g t_g
并行调度模型: T_total = sum_layers max_{g in layer} t_g

score = exp(-T_total / T_ref)
```

也可结合相干时间：

```text
score = exp(-T_total / T2)
```

依据：离子阱门时间差异明显，twoq gate 通常更慢；latency 与 idle dephasing 直接相关。

复杂度：

```text
串行: O(G)
简单分层调度: O(G log G) 或 O(G * n)
```





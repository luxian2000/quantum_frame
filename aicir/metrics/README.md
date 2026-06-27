# aicir.metrics 使用手册

量子线路与 ansatz 模板的评估指标库——提供可复用、与具体任务无关的线路质量度量，可用于 QAS 架构搜索、VQE ansatz 筛选以及线路设计分析。

> **通用约定**：所有评分函数返回值裁剪到 `[0, 1]`，数值越大越好。所有指标不依赖 QAS demo 或实验运行器，可独立使用。

---

## 目录

| 文件 | 说明 |
| --- | --- |
| `trainability.py` | 可训练性指标（结构代理 + 梯度探针） |
| `expressibility.py` | 表达能力指标（KL-Haar + MMD，理想噪声自由） |
| `noisy_expressibility.py` | 含噪表达能力指标（KL / MMD 的噪声版本 + 对比分析） |
| `hardware.py` | 硬件效率指标（原生门兼容 + 拓扑映射） |
| `_utils.py` | 内部工具函数（门分类、深度代理等） |

---

## 1  可训练性指标

评估参数化线路在梯度优化下的可训练性，用于识别贫瘠高原（barren plateau）风险。

### 1.1  结构代理评分（零计算成本）

基于线路结构（无仿真）的极低开销评分，适合在大规模搜索中快速筛选。

```python
from aicir.metrics import structure_proxy, structure_proxy_details

score = structure_proxy(circuit)       # float ∈ [0, 1]
details = structure_proxy_details(circuit)
# {
#     "structure_proxy_score": 0.72,
#     "single_qubit_gate_count": 8,
#     "two_qubit_gate_count": 3,
#     "depth_proxy": 5.0,
# }
```

**评分因子**（加权求和）：

| 因子 | 权重 | 公式 | 含义 |
| --- | :---: | --- | --- |
| 深度代理 | 0.4 | exp(−depth / 10) | 浅线路得分更高 |
| 双量子比特门密度 | 0.4 | exp(−2 × two_qubit_ratio) | 纠缠门比例越低越好 |
| 参数门密度 | 0.2 | exp(−params_per_qubit / 5) | 每比特参数门越少越好 |

### 1.2  局部探针梯度指标（近零成本）

使用参数移位规则在局部探针目标（task-agnostic）上估计梯度统计，不需要在下游任务上训练 ansatz。

```python
from aicir.metrics import local_probe_gradient_statistics

stats = local_probe_gradient_statistics(
    circuit,
    samples=8,              # 随机参数采样次数
    seed=1234,              # 随机种子
    shift=np.pi / 2,        # 参数移位量
    parameter_scale=2*np.pi,# 参数采样范围 [-scale, scale]
    probe="mean_z",         # 探针目标："mean_z" 或 "mean_abs_z"
)
# {
#     "n_parameters": 6,
#     "n_gradient_samples": 8,
#     "probe": "mean_z",
#     "mean_gradient_norm": 0.42,
#     "gradient_variance": 0.015,
#     "mean_abs_gradient": 0.08,
#     "zero_gradient_fraction": 0.12,
# }
```

**返回字段**：

| 字段 | 说明 |
| --- | --- |
| `mean_gradient_norm` | 各采样点梯度向量的平均范数 |
| `gradient_variance` | 各参数维度梯度方差的平均值 |
| `mean_abs_gradient` | 梯度绝对值的全局均值 |
| `zero_gradient_fraction` | 梯度绝对值低于阈值的比例（贫瘠高原指示器） |

### 1.3  梯度评分（便捷函数）

把梯度统计量映射为 `[0, 1]` 评分：

```python
from aicir.metrics import gradient_norm_score, gradient_variance_score

norm_score = gradient_norm_score(circuit, samples=8)       # float ∈ [0, 1]
var_score = gradient_variance_score(circuit, samples=8)    # float ∈ [0, 1]
```

- `gradient_norm_score`：基于平均梯度范数，按 `1 − exp(−norm / scale)` 映射。
- `gradient_variance_score`：加权混合方差评分（0.65）与范数评分（0.35）。

### 1.4  局部探针目标函数

底层探针目标函数，可独立使用（如自定义梯度评估流程）：

```python
from aicir.metrics import local_probe_objective

value = local_probe_objective(
    circuit,
    parameters=[0.1, 0.2, 0.3],  # 可选：绑定具体参数值
    probe="mean_z",
)
```

---

## 2  表达能力指标（Expressibility）

衡量参数化量子线路能探索多大比例的 Hilbert 空间。

### 2.1  KL-Haar 散度

计算线路参数化保真度分布与 Haar 随机态保真度分布的 KL 散度，再取对数得到相对表达能力值：

```python
from aicir.metrics import KL_Haar_relative, KL_Haar_divergence

# KL_Haar_relative = -ln(KL_Haar / Exp_1_idle)，值越高 → 越 expressive
score = KL_Haar_relative(circuit, samples=1000, n_bins=100, backend=None)

# KL_Haar_divergence 是向后兼容别名
score = KL_Haar_divergence(circuit, samples=1000, n_bins=100, backend=None)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `samples` | 1000 | 参数对采样次数 |
| `n_bins` | 100 | 保真度直方图 bin 数 |
| `backend` | `None`（NumpyBackend） | 计算后端 |

> 电路必须包含至少一个参数化门，否则抛出 `ValueError`。

### 2.2  MMD 表达能力

基于最大均值差异（MMD）的表达能力估计：

```python
from aicir.metrics import MMD_relative

# Exp_2 = 1 − MMD，越接近 1 → 越 expressive
score = MMD_relative(circuit, samples=1000, sigma=0.01, backend=None)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `samples` | 1000 | 采样次数 |
| `sigma` | 0.01 | 高斯核带宽（必须为正数） |
| `backend` | `None`（NumpyBackend） | 计算后端 |

---

## 3  含噪表达能力指标

在噪声模型下评估线路的表达能力，需要 `aicir.noise` 模块。

### 3.1  KL_Haar_noisy

```python
from aicir.metrics import KL_Haar_noisy

kl = KL_Haar_noisy(
    circuit,
    backend=None,
    n_samples=500,
    n_bins=1000,
    noise_model=noise_model,         # aicir.noise.NoiseModel 实例
    initial_state=None,              # 可选初始态（默认 |+⟩^N）
    use_density_matrix=True,         # True 走密度矩阵路径
)
```

> 限制：`n_qubits ≤ 10`（超出时抛出 `ValueError`）。

### 3.2  MMD_noisy

```python
from aicir.metrics import MMD_noisy

mmd = MMD_noisy(
    circuit,
    backend=None,
    n_samples=500,
    sigma=0.1,
    noise_model=noise_model,
    initial_state=None,
    use_density_matrix=True,
)
```

### 3.3  对比分析

一次性计算理想与含噪指标，返回噪声退化比：

```python
from aicir.metrics import comparative_expressibility

result = comparative_expressibility(
    circuit,
    backend=None,
    n_samples=500,
    noise_model=noise_model,         # 可选，传 None 则只返回理想值
)
# {
#     "kl_ideal": 3.14,
#     "mmd_ideal": 0.02,
#     "kl_noisy": 2.81,              # 仅 noise_model 不为 None 时
#     "mmd_noisy": 0.05,
#     "noise_degradation_kl": -0.10,
#     "noise_degradation_mmd": 1.50,
# }
```

### 3.4  统一接口

自动选择 KL 或 MMD 方法的统一入口：

```python
from aicir.metrics import expressibility_score

score = expressibility_score(
    circuit,
    backend=None,
    n_samples=500,
    noise_model=None,                # 传入 NoiseModel 走含噪路径
    method="auto",                   # "auto" | "kl" | "mmd"
    initial_state=None,
)
```

`method="auto"` 时，`2^n ≤ 256` 使用 KL，否则使用 MMD。

---

## 4  硬件效率指标

评估线路在特定硬件上的执行效率，用于架构级筛选。

### 4.1  原生门 / 深度 / 双量子比特代理

不需要完整硬件描述的快速评分：

```python
from aicir.metrics import native_depth_twoq_efficiency

score = native_depth_twoq_efficiency(
    circuit,
    native_gates=("hadamard", "rx", "ry", "rz", "cx", "cnot"),  # 默认
    max_depth=100,
)
```

**评分因子**（加权求和）：

| 因子 | 权重 | 说明 |
| --- | :---: | --- |
| 原生门比例 | 0.4 | 线路中属于原生门集的门占比 |
| 深度代理 | 0.3 | `min(1, max_depth / (depth_proxy × 10))` |
| 双量子比特效率 | 0.3 | `exp(−two_qubit_ratio / 3)` |

查看详细分项：

```python
from aicir.metrics import native_depth_twoq_efficiency_details

details = native_depth_twoq_efficiency_details(circuit)
# {
#     "native_depth_twoq_efficiency_score": 0.85,
#     "native_gate_count": 12,
#     "native_gate_ratio": 0.92,
#     "two_qubit_gate_count": 3,
#     "depth_proxy": 5.0,
# }
```

### 4.2  拓扑映射效率

结合完整硬件描述（耦合图、保真度）的综合评分：

#### HardwareProfile

```python
from aicir.metrics import HardwareProfile

profile = HardwareProfile(
    native_gates=("hadamard", "rx", "ry", "rz", "cx", "cnot"),
    coupling_map=[(0, 1), (1, 2), (2, 3)],
    edge_fidelity=None,            # 可选：标量或 {(q_i, q_j): fidelity} 字典
    gate_durations={},             # 可选：门延迟（预留）
    max_depth=12,                  # 可选：深度上限
)
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `native_gates` | `Sequence[str]` | 目标设备支持的原生门名称 |
| `coupling_map` | `Sequence[tuple[int, int]]` | 量子比特耦合拓扑（无向边列表） |
| `edge_fidelity` | `float \| dict \| None` | 双量子比特门保真度（标量或按边字典） |
| `gate_durations` | `dict` | 门执行时长（预留字段） |
| `max_depth` | `int \| None` | 深度归一化上限 |

也可从 `aicir.devices.Target` 直接构造，`native_gates` 取自 `target.basis_gates`（空门集回退默认 `DEFAULT_NATIVE_GATES`），`coupling_map` 取自 `target.coupling_map`（全连接 Target → 空耦合）；其余字段可经关键字覆盖：

```python
from aicir.devices import Target
from aicir.metrics import HardwareProfile

target = Target(n_qubits=3, basis_gates=("rx", "ry", "rz", "cx"), coupling_map=[(0, 1), (1, 2)])
profile = HardwareProfile.from_target(target, max_depth=12)
```

#### 评分

```python
from aicir.metrics import topology_mapping_efficiency, topology_mapping_efficiency_details

score = topology_mapping_efficiency(circuit, profile)  # float ∈ [0, 1]

details = topology_mapping_efficiency_details(circuit, profile)
# {
#     "topology_mapping_efficiency_score": 0.78,
#     "native_gate_count": 12,
#     "native_gate_ratio": 0.92,
#     "non_native_ratio": 0.08,
#     "two_qubit_gate_count": 3,
#     "two_qubit_density": 0.75,
#     "depth_proxy": 5.0,
#     "depth_norm": 0.42,
#     "connectivity_violation_count": 0,
#     "routing_distance_cost": 1.0,
#     "routing_distance_per_twoq": 0.33,
#     "mapping_fidelity_score": 0.98,          # 仅 edge_fidelity 不为 None
#     "mapping_fidelity_note": "reported for mapping preference only; not included in the primary score",
# }
```

**评分公式**：

```
score = exp(−0.8 × routing_norm − 0.5 × depth_norm − 0.4 × non_native_ratio − 0.2 × twoq_density)
```

> `mapping_fidelity_score` 作为辅助信息报告，**不纳入**主硬件效率分数，以保持硬件效率与噪声鲁棒性的正交性。

---

## 5  公共 API 速查

### 从 `aicir.metrics` 直接导入

| 函数 | 模块 | 类别 |
| --- | --- | --- |
| `structure_proxy` | trainability | 可训练性 |
| `structure_proxy_details` | trainability | 可训练性 |
| `gradient_norm_score` | trainability | 可训练性 |
| `gradient_variance_score` | trainability | 可训练性 |
| `local_probe_gradient_statistics` | trainability | 可训练性 |
| `local_probe_objective` | trainability | 可训练性 |
| `KL_Haar_relative` | expressibility | 表达能力 |
| `KL_Haar_divergence` | expressibility | 表达能力 |
| `MMD_relative` | expressibility | 表达能力 |
| `KL_Haar_noisy` | noisy_expressibility | 含噪表达能力 |
| `MMD_noisy` | noisy_expressibility | 含噪表达能力 |
| `comparative_expressibility` | noisy_expressibility | 含噪表达能力 |
| `expressibility_score` | noisy_expressibility | 含噪表达能力 |
| `HardwareProfile` | hardware | 硬件效率 |
| `DEFAULT_NATIVE_GATES` | hardware | 硬件效率 |
| `native_depth_twoq_efficiency` | hardware | 硬件效率 |
| `native_depth_twoq_efficiency_details` | hardware | 硬件效率 |
| `topology_mapping_efficiency` | hardware | 硬件效率 |
| `topology_mapping_efficiency_details` | hardware | 硬件效率 |

---

## 6  使用建议

| 场景 | 推荐指标 | 说明 |
| --- | --- | --- |
| 大规模 QAS 快速筛选 | `structure_proxy` | 零仿真成本，纯结构分析 |
| ansatz 贫瘠高原检测 | `gradient_norm_score` / `gradient_variance_score` | 近零成本，参数移位探针 |
| ansatz 表达能力对比 | `KL_Haar_relative` / `MMD_relative` | 需要仿真，精度更高 |
| 含噪场景表达能力 | `expressibility_score(noise_model=...)` | 统一接口，自动选方法 |
| 硬件兼容性粗筛 | `native_depth_twoq_efficiency` | 无需耦合图 |
| 面向真机的硬件评估 | `topology_mapping_efficiency(circuit, profile)` | 需要 `HardwareProfile` |

# Metrics

本包提供可复用、与具体任务无关的量子线路与 ansatz 模板评估指标。它们被设计为轻量级构件，可用于 QAS、VQE ansatz 筛选，以及其他架构层面的分析流程。

## 可训练性指标

`trainability.py` 提供两层可训练性代理指标。

### 结构代理指标

```python
from aicir.metrics.trainability import structure_proxy, structure_proxy_details
```

`structure_proxy(circuit)` 是一个代价很低的结构评分，基于以下因素：

- 线路深度代理值
- 双量子比特门密度
- 参数门 / 单量子比特门密度

当基于仿真的探测成本太高时，它很适合用于快速筛选。

### 局部探针梯度指标

```python
from aicir.metrics.trainability import (
    gradient_norm_score,
    gradient_variance_score,
    local_probe_gradient_statistics,
)
```

`local_probe_gradient_statistics(circuit, ...)` 会在一个简单的局部探针目标上，估计与任务无关的参数平移梯度。它会返回：

- `mean_gradient_norm`
- `gradient_variance`
- `mean_abs_gradient`
- `zero_gradient_fraction`
- `n_parameters`
- `n_gradient_samples`

评分辅助函数会把这些统计量映射到 `[0, 1]`：

- `gradient_norm_score(circuit, ...)`
- `gradient_variance_score(circuit, ...)`

从 NAS/QAS 的角度看，这些指标几乎是零成本的：它们不需要在下游任务上训练 ansatz。

## 硬件指标

`hardware.py` 提供用于架构级筛选的硬件效率指标。

### 原生门 / 深度 / 双量子比特代理

```python
from aicir.metrics.hardware import native_depth_twoq_efficiency
```

`native_depth_twoq_efficiency(circuit, ...)` 会基于以下因素为线路打分：

- 原生门兼容性
- 深度代理值
- 双量子比特门密度

### 拓扑映射效率

```python
from aicir.metrics.hardware import HardwareProfile, topology_mapping_efficiency
```

`HardwareProfile` 用于描述目标设备：

```python
profile = HardwareProfile(
    native_gates=("hadamard", "rx", "ry", "rz", "cx", "cnot"),
    coupling_map=[(0, 1), (1, 2), (2, 3)],
    edge_fidelity=None,
    gate_durations={},
    max_depth=12,
)
```

`topology_mapping_efficiency(circuit, profile)` 会基于以下因素为线路打分：

- 原生门比例
- 耦合图兼容性
- 路由距离
- 深度
- 双量子比特门密度

可使用 `topology_mapping_efficiency_details(circuit, profile)` 查看底层组成项。

`edge_fidelity` 可以传入标量，也可以传入按边划分的字典。它会以 `mapping_fidelity_score` 的形式报告，但不会纳入主硬件效率分数。这样可以将硬件效率与噪声鲁棒性指标分开。

## 对外导出

`aicir.metrics` 提供主要导出：

```python
from aicir.metrics import (
    HardwareProfile,
    gradient_norm_score,
    gradient_variance_score,
    local_probe_gradient_statistics,
    native_depth_twoq_efficiency,
    topology_mapping_efficiency,
)
```

## 说明

- 所有分数都会裁剪到 `[0, 1]`，其中数值越大越好。
- 这些指标不依赖 QAS demo 或实验运行器。
- 可训练性梯度探针默认使用 NumPy 后端。

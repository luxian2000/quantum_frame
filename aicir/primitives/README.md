# aicir.primitives 使用手册

Primitives 是算法层的统一执行入口，为不同后端的采样与期望值估计提供高层封装。它剥离了底层的 `Measure` 和 `PauliEstimator` 细节，通过统一接口处理单/多电路和测量配对。

---

## 目录

| 文件 / 类 | 描述 |
| --- | --- |
| `ShotSampler` | 有限 shots 采样器（基类：`BaseSampler`） |
| `StatevectorEstimator` | 精确态向量期望值估计器（无散粒噪声） |
| `ShotEstimator` | 有限 shots 能量估计器（自带 Pauli 分组，基类：`BaseEstimator`） |
| `SampleResult` | `counts` 与 `probs` 采样结果载体 |
| `EstimateResult` | 期望值与方差结果载体 |

---

## 1 接口约定

Primitives 遵循以下入参归一化约定：

1. **已绑定参数**：接收的 `circuit` 必须是参数已绑定的具体电路（`parameter_values=` 延迟绑定接口尚未实现）。
2. **标量 / 列表行为**：
   - 传入单个电路，返回单个结果。
   - 传入序列（如 `list`），返回结果列表。
3. **可观测量广播 (Estimator)**：
   - 单个 `observable` 自动广播配对到所有 `circuits`。
   - 若两者皆为序列，必须等长，并按位置一对一配对计算。

---

## 2 采样 (Sampler)

`ShotSampler` 包装了后端的 `Measure` 组件，支持显式指定测量比特，或处理内嵌 `measure()` 门的线路。

```python
from aicir import Circuit, hadamard, cnot
from aicir.primitives import ShotSampler

# 贝尔态线路
bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

# 1. 初始化 Sampler
sampler = ShotSampler(shots=1024)

# 2. 执行采样
result = sampler.run(bell)

# 返回 SampleResult
print(result.counts)  # {'|00>': 511, '|11>': 513}
print(result.probs)   # {'|00>': 0.5, '|11>': 0.5} (基于解析概率的非零项)
print(result.measured_qubits) # (0, 1)
```

### ShotSampler 参数说明
- **`__init__(backend=None, shots=1024)`**: 默认采用 numpy 后端。
- **`run(circuits, shots=None, measure_qubits=())`**:
  - `shots` 覆盖实例级别的配置。
  - `measure_qubits` 显式指定要在末端测量的比特（当线路无内嵌测量操作时生效）。

---

## 3 期望值估计 (Estimator)

提供了精确无噪声的 `StatevectorEstimator`，和基于实际测量计数的 `ShotEstimator`。两者接口一致，区别仅在于是否接受 `shots` 参数。

### 3.1 StatevectorEstimator (精确模式)

直接计算 $\langle \psi | H | \psi \rangle$，拒绝 `shots` 参数，适用于理论模拟验证。可观测量支持 `Hamiltonian` 或稠密矩阵。

```python
import numpy as np
from aicir import Circuit, Hamiltonian, pauli_x
from aicir.primitives import StatevectorEstimator

cir = Circuit(pauli_x(0), n_qubits=1)
ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

estimator = StatevectorEstimator()

# 注意: StatevectorEstimator 拒绝传递 shots= 参数
result = estimator.run(cir, ham)

print(result.value)    # -1.0
print(result.variance) # None (精确模式无方差)
print(result.shots)    # None 
```

### 3.2 ShotEstimator (采样模式)

包装了 `aicir.measure.estimator.PauliEstimator`，内部执行 Qubit-wise Commuting (QWC) 泡利分组与基变换测量，并分配 shot。

```python
from aicir.primitives import ShotEstimator

noisy_estimator = ShotEstimator(shots=4096)
result = noisy_estimator.run(cir, ham)

print(result.value)    # ~ -1.0
print(result.variance) # float
print(result.shots)    # 4096

# 查看分组执行的 metadata
print(result.metadata["groups"]) 
```

#### VQE 兼容性

`ShotEstimator` 暴露了 `estimate(circuit, hamiltonian)` 直通方法，可直接作为 `BasicVQE(energy_estimator=...)` 的依赖注入，无需修改 VQE 现有代码。

---

## 4 结果对象字典

所有 Primitive 执行均返回规范的数据类（Dataclass）。

### `SampleResult`

| 属性 | 类型 | 描述 |
| --- | --- | --- |
| `counts` | `dict[str, int]` | 如 `{'\|00>': 512}` 的本征态命中次数。 |
| `probs` | `dict[str, float]` | 剔除 `0.0` 后的理论解析概率。 |
| `shots` | `int | None` | 实际分配的采样次数。 |
| `measured_qubits` | `tuple[int, ...]` | 此采样结果关联的被测比特索引。 |
| `metadata` | `Mapping[str, Any]` | 额外执行信息字典。 |

### `EstimateResult`

| 属性 | 类型 | 描述 |
| --- | --- | --- |
| `value` | `float` | $\langle H \rangle$ 估计值（实数）。 |
| `variance` | `float | None` | 估计方差。精确路径 (`StatevectorEstimator`) 为 `None`。 |
| `shots` | `int | None` | 分配的采样次数。精确路径为 `None`。 |
| `term_results` | `tuple[Any, ...] | None` | 逐 Pauli 项明细。精确路径为 `None`。 |
| `metadata` | `Mapping[str, Any]` | 包含 `method` (如 `"statevector"`, `"pauli_shots"`) 和 `"groups"` 的调试信息字典。 |

---

## 5 后续方向 (Roadmap)

当前 primitives 是在 `aicir.measure` 上的一层薄包装。计划的演进方向包括：

1. **噪声模拟支持**：未来的 `NoisySampler`/`NoisyEstimator`。目前可通过 `ShotEstimator(use_density_matrix=True, noise_model=...)` 透传到底层 `PauliEstimator`。
2. **硬件/远端代理**：`BackendSampler`/`BackendEstimator` 作为向 QPU 真实硬件发送任务的远端扩展点。
3. **参数延迟绑定**：实现 `run(circuit, parameter_values=...)`，避免频繁的电路副本生成。
4. **子模块迁移**：未来将 `vqc`, `qas`, `metrics` 底层替换为 Primitives 统一调用接口。

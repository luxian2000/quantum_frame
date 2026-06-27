# aicir.primitives 使用手册

Primitives 是算法层的统一执行入口，为不同后端的采样与期望值估计提供高层封装。它剥离了底层的 `Measure` 和 `PauliEstimator` 细节，通过统一接口处理单/多电路和测量配对。

---

## 目录

| 文件 / 类 | 描述 |
| --- | --- |
| `StatevectorSampler` | 精确解析概率采样器（无散粒噪声，拒绝 `shots=`） |
| `ShotSampler` | 有限 shots 采样器（基类：`BaseSampler`） |
| `NoisySampler` | 带噪声采样器（`noise_model` → 密度矩阵路径） |
| `StatevectorEstimator` | 精确态向量期望值估计器（无散粒噪声） |
| `ShotEstimator` | 有限 shots 能量估计器（自带 Pauli 分组，基类：`BaseEstimator`） |
| `NoisyEstimator` | 带噪声期望值估计器（密度矩阵；`shots=None` 确定性） |
| `BackendSampler` / `BackendEstimator` | 注入式 `runner` 扩展点（真实硬件/远端服务） |
| `estimator_for_target(target, ...)` | 按 `aicir.devices.Target` 能力选择并构造 Estimator |
| `SampleResult` | `counts` 与 `probs` 采样结果载体 |
| `EstimateResult` | 期望值与方差结果载体 |

---

## 1 接口约定

Primitives 遵循以下入参归一化约定：

1. **参数绑定**：默认接收已绑定参数的具体电路；也可传模板电路并经 `run(..., parameter_values=...)` 延迟绑定（单电路 → 一维数组；电路序列 → 数组序列）。
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
- **`run(circuits, shots=None, measure_qubits=(), parameter_values=None)`**:
  - `shots` 覆盖实例级别的配置。
  - `measure_qubits` 显式指定要在末端测量的比特（当线路无内嵌测量操作时生效）。
  - `parameter_values` 对模板电路延迟绑定（见 §5.3）。

### StatevectorSampler（精确概率）

返回演化态的解析概率分布，无散粒噪声（`counts` 为空、`shots` 为 `None`），拒绝 `shots=`。

```python
from aicir.primitives import StatevectorSampler

result = StatevectorSampler().run(bell)
print(result.probs)   # {'|00>': 0.5, '|11>': 0.5}
print(result.counts)  # {}（精确路径无采样计数）
```

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

`StatevectorEstimator` 也暴露 `estimate(circuit, hamiltonian)` 直通方法（忽略 shots/initial_state 等 kwargs），与 `ShotEstimator`/`NoisyEstimator` 一致满足 `BasicVQE(energy_estimator=...)` 注入契约——使 VQE 精确能量求值也能走 primitives。

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

## 5 噪声与扩展点

primitives 是在 `aicir.measure` 上的统一封装。除精确/采样路径外还提供：

### 5.1 噪声路径

`NoisySampler` / `NoisyEstimator` 把 `noise_model` 附加到线路、经密度矩阵模拟执行。`NoisyEstimator` 在 `shots=None` 时给出确定性密度矩阵期望（仅退相干、无采样噪声），`shots>=1` 叠加散粒统计。

```python
from aicir.primitives import NoisyEstimator

est = NoisyEstimator(noise_model=my_noise)          # shots=None → 确定性
result = est.run(circuit, ham)
# 也可作 VQE 的 energy_estimator 注入（暴露 estimate()）
vqe = BasicVQE(ham, ansatz=ansatz, energy_estimator=est)
```

### 5.2 注入式硬件/远端扩展点

仓内不内置 QPU 或远端后端，`BackendSampler`/`BackendEstimator` 包装用户传入的 `runner`，由 runner 负责真正执行：

```python
from aicir.primitives import BackendEstimator

def runner(circuit, observable, *, shots):      # 也可返回现成 EstimateResult
    return my_qpu.expectation(circuit, observable, shots=shots)

result = BackendEstimator(runner, shots=1024).run(circuit, ham)
```

### 5.3 延迟绑定

所有 `run(...)` 支持 `parameter_values=`，对模板电路延迟绑定，避免上层频繁手动生成电路副本。

### 5.4 按 Target 选择执行路径

`estimator_for_target(target, *, backend=None, noise_model=None, shots=None)` 据 `aicir.devices.Target` 的能力标志自动选择估计器，供下游按设备能力选执行路径而非各自硬编码：

```python
from aicir.devices import Target
from aicir.primitives import estimator_for_target

est = estimator_for_target(Target(n_qubits=2, supports_statevector=True))  # → StatevectorEstimator
est = estimator_for_target(target, shots=2048)                            # → ShotEstimator（要求 supports_shots）
est = estimator_for_target(target, noise_model=nm)                        # → NoisyEstimator（要求 supports_density_matrix）
```

选择优先级：给定 `noise_model` → `NoisyEstimator`；给定 `shots` → `ShotEstimator`；否则 `supports_statevector` → `StatevectorEstimator`，退而 `supports_shots` → `ShotEstimator`；无可用路径抛 `ValueError`。`BasicVQE(..., target=...)` 即经此工厂注入估计器。

### 5.5 子模块采用

`vqc`/`qas`/`metrics` 采用加性集成：可经现有注入点（如 `BasicVQE(energy_estimator=...)`、`BasicVQE(target=...)`）消费 primitives，未重写其内部 `Measure`/`PauliEstimator` 调用；全量内部迁移属可选后续。

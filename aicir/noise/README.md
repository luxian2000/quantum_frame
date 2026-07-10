# aicir.noise 噪声模块

`aicir.noise` 提供面向密度矩阵仿真的噪声通道、门后触发式噪声模型、
线路噪声敏感性分析，以及 QAS 使用的离子阱误差预算指标。它的目标是给
VQE、QAS、QML 和基础线路仿真提供一个轻量、可组合、可复用的噪声层。

本文档同时也是默认离子阱噪声配置文件。`aicir.noise.ion_trap` 会从本文档
中的参数块读取默认参数，因此编辑时请保留参数块结构。

## 模块结构

| 文件 | 作用 |
| --- | --- |
| `base.py` | 定义 `NoiseChannel` 抽象接口 |
| `channels.py` | 常用单量子比特、多量子比特和自定义 Kraus 噪声通道 |
| `model.py` | 组合多个通道规则的 `NoiseModel` |
| `analysis.py` | 含噪与无噪演化对比、线路噪声敏感性分析 |
| `metrics.py` | 噪声相关线路指标，目前包含离子阱误差预算代理指标 |
| `ion_trap.py` | 离子阱默认参数加载、公式推导和运行时噪声模型构建 |
| `README.md` | 模块说明与默认离子阱参数源 |

## 快速开始

手动创建一个简单噪声模型，并将它挂到线路上。

```python
from aicir import BitFlipChannel, Circuit, Measure, NoiseModel, NumpyBackend, ry

circuit = Circuit(ry(0.2, 0), n_qubits=1)
circuit.noise_model = NoiseModel().add_channel(
    BitFlipChannel(target_qubit=0, p=0.01),
    after_gates=["ry"],
)

result = Measure(NumpyBackend()).run(circuit, shots=None)
print(result.probabilities)
```

加载默认离子阱配置，并生成运行时 `NoiseModel`。

```python
from aicir.noise import load_default_ion_trap_noise_config

config = load_default_ion_trap_noise_config()
noise_model = config.build_noise_model()

print(config.summary())
```

分析一个线路对噪声的敏感性。

```python
from aicir import Circuit, cnot, hadamard, NumpyBackend
from aicir.noise import load_default_ion_trap_noise_config, noise_sensitivity

circuit = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
noise_model = load_default_ion_trap_noise_config().build_noise_model(qubits=[0, 1])

report = noise_sensitivity(circuit, backend=NumpyBackend(), noise_model=noise_model)
print(report.summary())
```

## 噪声通道

所有通道都实现 `NoiseChannel`，通过 Kraus 算子作用在完整 `n` 量子比特密度矩阵上。
当前内置通道如下。

| 通道 | 参数 | 物理含义 |
| --- | --- | --- |
| `DepolarizingChannel` | `p` | 单量子比特退极化 |
| `BitFlipChannel` | `p` | 单量子比特比特翻转 |
| `PhaseFlipChannel` | `p` | 单量子比特相位翻转 |
| `PauliChannel` | `px`, `py`, `pz` | 独立 X/Y/Z Pauli 错误 |
| `AmplitudeDampingChannel` | `gamma` | 从激发态向基态的振幅阻尼 |
| `PhaseDampingChannel` | `gamma` | 纯退相干，衰减相干项但保持布居 |
| `GeneralizedAmplitudeDampingChannel` | `gamma`, `p_excited` | 有限温度振幅阻尼 |
| `ResetChannel` | `p` | 以概率 `p` 将目标量子比特重置为 `|0>` |
| `ErasureChannel` | `p`, `erase_to` | 固定 Hilbert 空间内的 erasure 代理，替换为指定对角态 |
| `ReadoutErrorChannel` | `p01`, `p10` | 非对称读出混淆的测量前代理通道 |
| `TwoQubitDepolarizingChannel` | `qubit_1`, `qubit_2`, `p` | 双量子比特退极化，`p=1` 时目标子系统完全混合 |
| `CorrelatedTwoQubitPauliChannel` | `probabilities` | 显式相关的双量子比特 Pauli 错误 |
| `ThermalRelaxationChannel` | `t1`, `t2`, `gate_time`, `excited_population` | 由 T1/T2 和门时长推导的热弛豫 |
| `KrausChannel` | `kraus_ops`, `target_qubits` | 用户自定义 Kraus 通道 |

参数 `p`、`px`、`py`、`pz`、`gamma` 和 `p_excited` 都要求位于 `[0, 1]`，且
`PauliChannel` 要求 `px + py + pz <= 1`。通道会被嵌入到全系统维度，因此当前实现适合
中小规模密度矩阵仿真，不是大比特数稀疏噪声采样器。

`ErasureChannel` 和 `ReadoutErrorChannel` 是当前二能级密度矩阵框架内的代理模型。
真正带 flag 的 erasure 或 qutrit leakage 需要扩展 Hilbert 空间；本模块暂不实现 leakage
通道，以免和 `2^n` 态空间约定冲突。

## NoiseModel

`NoiseModel` 是门后触发规则的集合。每条规则包含一个噪声通道、一个可选的门名过滤器，
以及是否排除当前门作用量子比特的标志。

```python
from aicir.noise import BitFlipChannel, NoiseModel, PhaseFlipChannel

noise_model = (
    NoiseModel()
    .add_channel(BitFlipChannel(0, 0.01), after_gates=["measure"])
    .add_channel(PhaseFlipChannel(1, 0.001), after_gates=["cx"], exclude_gate_qubits=True)
)
```

`after_gates=None` 表示所有门后都应用该通道。`exclude_gate_qubits=True` 常用于建模
空闲退相干，也就是当前门操作未寻址的量子比特才受到该规则影响。

## 与仿真执行的关系

在常规线路执行中，可以把 `NoiseModel` 直接挂到 `Circuit.noise_model`。当 `Measure`
执行含噪线路时，会走密度矩阵路径，并在匹配的门后调用噪声模型。

手动分析时，`analysis.py` 提供更直接的工具。

| 函数 | 作用 |
| --- | --- |
| `evolve_density_gatewise` | 按门演化密度矩阵，可选应用噪声模型 |
| `noise_sensitivity` | 比较无噪与含噪概率分布并返回摘要 |
| `estimate_noise_strength` | 根据模型中的通道参数估计平均噪声强度 |
| `analyze_gate_type_sensitivity` | 给出按门类型聚合的结构性敏感性代理值 |

## 离子阱默认配置

`ion_trap.py` 将本文档中的参数块解析为 `IonTrapNoiseConfig`。该配置会推导单门退极化、
双门退极化、空闲退相干和串扰概率，并可构造可直接用于仿真的 `NoiseModel`。

```python
from aicir.noise import load_ion_trap_noise_config

config = load_ion_trap_noise_config()
model = config.build_noise_model(qubits=[0, 1, 2])
```

如果需要替换硬件 profile，可以复制本 README 的参数块到另一个 Markdown 文件，
再通过 `load_ion_trap_noise_config(path)` 加载。也可以用 `overrides` 临时覆盖参数。

```python
from aicir.noise import load_ion_trap_noise_config

config = load_ion_trap_noise_config(
    overrides={
        "parameters": {
            "enable_idle_dephasing_noise": False,
            "enable_crosstalk_noise": False,
        }
    }
)
```

## 离子阱误差预算指标

`ion_trap_error_budget_proxy(circuit)` 是 QAS 和架构筛选可用的快速代理指标。它统计线路中的
单量子比特门、双量子比特门、测量和 reset，并结合默认离子阱参数估计总误差预算。
返回值包括一个 `[0, 1]` 分数和详细预算字典。

```python
from aicir.noise import ion_trap_error_budget_proxy

score, details = ion_trap_error_budget_proxy(circuit)
print(score)
print(details["total_error_budget"])
```

该指标不是完整物理仿真。它适合作为架构搜索、候选线路排序或硬件效率粗筛的一项信号。

## 默认参数块

下列参数是默认离子阱运行时配置的唯一事实来源。加载器会忽略普通说明文字，只读取
这些扁平参数项。

```text
formula_profile: ion_trap_doc_2025

rounds: 25
basis: z
twoq_gate: zz_opt
data_qubits: [0, 1, 6, 3, 4, 2, 7]
ancillas: [5]
logical_label_mode: parity

enable_initialization_noise: True
enable_measurement_noise: True
enable_oneq_gate_noise: True
enable_twoq_gate_noise: True
enable_idle_dephasing_noise: True
enable_crosstalk_noise: True

T2: 0.2
meas_bitflip: 2.5e-4
reset_bitflip: 2.5e-4
oneq_gate_time: 0.0001
twoq_gate_time: 0.0006
reset_time: 1.0e-5
meas_time: 1.0e-3

idle_time_strategy: switch_by_gate_arity

oneq_avg_infidelity_deltaF1: 0.001
twoq_avg_infidelity_deltaF2: 0.0075

cross_talk: 2.8e-06
```

## 参数推导

当 `formula_profile` 为 `ion_trap_doc_2025` 时，加载器使用以下公式推导运行时概率。

```text
oneq_depol = 3/2 * oneq_avg_infidelity_deltaF1
twoq_depol = 5/4 * twoq_avg_infidelity_deltaF2
p_idle = 1/2 * (1 - exp(-t / T2))
```

空闲退相干的时间由 `idle_time_strategy` 决定。单量子比特门后使用 `oneq_gate_time`，
双量子比特门后使用 `twoq_gate_time`。当前策略假设离子阱门串行执行，因此空闲退相干
作用在非活动量子比特上。

串扰默认值已经是运行时概率，代表性公式如下。

```text
p_c = sin^2(epsilon_CT * theta / 2), with theta = pi / 2
```

代表性取值 `epsilon_CT ~= 0.00213` 会给出 `p_c ~= 2.8e-6`。

## 编辑建议

禁用某个噪声来源时，优先把对应开关改为 `False`，不要删除标定值。修改硬件标定时，
优先改原始字段，例如 `oneq_avg_infidelity_deltaF1` 和 `twoq_avg_infidelity_deltaF2`，
让加载器负责推导运行时概率。

所有时间参数都以秒为单位。若硬件标定数据使用微秒或毫秒，请先换算后再写入。

## 来源说明

公式配置来源于 2025 年文档 `离子阱量子计算系统噪声分析与模型设计`。

线路与实验设置保持与项目说明中使用的 `si1000_ion.yaml` 风格配置一致。

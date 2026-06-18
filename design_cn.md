# quantum_frame / aicir 混合模拟器设计

本文档将 `design_gemini.md`、`design_gpt.md` 和 `design_claude.md` 的架构结论合并为一个面向 `aicir` 实现的设计方案。

## 1. 设计决策

对于量子机器学习 (QML)、量子架构搜索 (QAS) 和量子纠错 (QEC) 而言，没有一种单一的模拟器方案是最佳的。这些工作负载面临不同的瓶颈：

| 工作负载 | 主要变量 | 主要瓶颈 | 最佳执行风格 |
| --- | --- | --- | --- |
| QML | 几乎固定电路上的连续参数 | 重复的可微分前向/反向传播 | 具有 GPU/NPU 自动求导的批处理张量/状态向量引擎 |
| QAS | 电路拓扑和拟设结构 (ansatz structure) | 评估、变异、排序和缓存大量候选电路 | 类型化 IR + 原语 + 结构预过滤器 + 批处理估计器 |
| QEC | 测量、重置、泡利噪声、综合征历史 | 大型重型 Clifford 电路和高 shot 吞吐量 | 稳定子/tableau + 泡利帧采样器，带密度/轨迹验证 |

因此核心设计如下：

```text
电路 / 类型化 IR 是唯一的单一真实来源。
原语 (Primitives) 是稳定的执行 API。
规划器 (planner) 为任务选择最佳的专用引擎。
QNode 风格的可调用对象仍然是可选的 QML 人体工程学设计，而不是核心抽象。
```

这符合 `aicir` 当前的发展方向：`Circuit` 和 `CircuitIR` 已经存在，`aicir.primitives` 提供了 `SVEstimator`、`ShotEstimator` 和 `ShotSampler`，`BatchSV` 提供了批处理状态向量路径，并且后端已经分离为 CPU/GPU/NPU 实现。

## 2. 主要替代方案

### 2.1 QNode 风格的可调用对象

QNode 将电路构建、设备/后端、测量、shots 和微分封装到一个可调用对象中。

优点：
- 对于简单的 QML 和变分模型来说，具有极佳的人机工程学。
- 自然地在 PyTorch 风格的训练循环中作为层 (layer) 暴露。
- 可以通过 `aicir.qml.diff.select_diff` 隐藏梯度选择。

缺点：
- 耦合了电路定义、执行和微分。
- 非常不适合 QAS，因为电路拓扑是搜索变量。
- 对于 QEC 动态电路、综合征历史和硬件控制流路径来说不够用。

在 `aicir` 中的使用：
- 仅在原语和规划器行为稳定之后，将其作为薄薄的可选前端添加。
- 不要让底层依赖它。

### 2.2 原语 (Primitives)：估计器和采样器

电路仍然是数据。执行通过 `Estimator` 和 `Sampler` 对象进行。

优点：
- 关注点清晰分离。
- 适用于许多电路、可观测量或 shots 的批处理优先 (Batch-first) 接口。
- 用于 QAS、VQE 风格基准测试和 QEC shot 收集的最佳公共界面。
- 已经与 `aicir.primitives` 匹配。

缺点：
- 对于最简单的 QML 示例，比 QNode 更加冗长。
- 需要一条良好的参数绑定/批处理路径才能达到峰值 QML 吞吐量。

在 `aicir` 中的使用：
- 保持 `SVEstimator`、`ShotEstimator` 和 `ShotSampler` 作为主要算法执行 API。
- 向 `EstimateResult` 和 `SampleResult` 添加规划器元数据，以便调用者知道使用了哪个引擎和计划。

### 2.3 类型化 IR 加上转换 (transforms)

电路表示为类型化数据，优化、微分、指标、转译和执行都是对该数据的转换。

优点：
- 用于 QAS 指标、QEC 分类器、门计数、Clifford 检测和转译的最佳自省模型。
- 允许 `CircuitIR` 在许多引擎上编译/运行，而无需重建用户代码。
- 保持与当前 `Circuit.gates` 的兼容性，同时允许内部消费者通过 `aicir.ir` 使用类型化结构。

缺点：
- 需要清晰的接口和谨慎的迁移纪律。
- 对于初次使用 QML 的用户来说，比单一的可调用对象稍微复杂一些。

在 `aicir` 中的使用：
- 将 `CircuitIR` 视为内部规划输入。
- 保留 `Circuit`、`Circuit.operations`、`Circuit.ir` 和旧版 `Circuit.gates` 的兼容性。
- 将结构分析放入小型模块中，而不是将其嵌入到原语中。

### 2.4 纯张量/自动微分图 (autodiff graph)

模拟表示为张量运算，以便 ML 框架记录整个计算图。

优点：
- 在 GPU/NPU 上进行可微分 QML 的最佳性能路径。
- 对于受支持的门，实现精确的反向传播。
- 与混合经典神经网络自然协同工作。
- `BatchSV` 已经实现了一条 NPU 安全的实数/虚数批处理路径。

缺点：
- 当 QAS 不断更改拓扑时，重建计算图的成本很高。
- 仅限模拟器；它不能直接映射到基于 shot 的硬件执行。
- 内存随存储的中间状态而增长。

在 `aicir` 中的使用：
- 将其作为固定拓扑可微分电路的首选 QML 引擎。
- 对于不可用自动求导或不合适的情况，保留参数平移 (parameter-shift)、有限差分 (finite difference)、SPSA 和伴随 (adjoint) 路径。

### 2.5 AOT/JIT 编译

电路或动态程序在重复执行之前被编译，例如编译为 OpenQASM 3.0、QIR、MLIR 或编译的内部内核。

优点：
- 减少重复的解释器开销。
- 对于包含中间测量和经典前馈的面向硬件的 QEC 而言非常重要。
- 最终可以支持门融合 (gate fusion)、调度和布局。

缺点：
- 工程成本最高。
- 当 QAS 拓扑每次迭代都改变时，编译开销是浪费的。
- 通过编译代码进行微分需要额外的设计。

在 `aicir` 中的使用：
- 把它作为规划器和 `aicir.transpile` 之后的稍后执行目标保留。
- 优先考虑模拟器引擎；编译路径应该消耗相同的 IR。

## 3. 模拟引擎

| 引擎 | 优点 | 缺点 | 最佳用处 |
| --- | --- | --- | --- |
| 状态向量 (state vector) | 精确，简单，非常适合无噪声 VQE/QAOA/QML/QAS | `O(2^n)` 内存，对大型 QEC 表现差 | 默认精确无噪声模拟器 |
| 批处理状态向量 (Batched state vector) | 针对数据/参数批次的高吞吐量，可映射到 GPU/NPU，已作为 `BatchSV` 开始 | 内存布局更复杂，最好在批处理成员共享结构时使用 | QML 训练和分组 QAS 评估 |
| 密度矩阵 (Density matrix) | 精确的混合状态和信道模拟 | `O(4^n)` 内存，仅限小系统 | 噪声正确性检查和小型噪声电路 |
| 量子轨迹 (Quantum trajectories) | 扩展噪声模拟的能力优于密度矩阵，跨轨迹并行 | 随机方差和更难的调试 | 中型噪声 QML/QAS 和近似 QEC 噪声研究 |
| 稳定子/Tableau (Stabilizer/tableau) | 对 Clifford 电路多项式缩放，非常适合综合征提取 | 无法直接表示任意非 Clifford 旋转 | QEC 核心模拟器 |
| 泡利帧 (Pauli frame) | 廉价跟踪 Pauli 修正和逻辑帧 | 需要明确的测量/重置语义 | QEC 解码和修正循环 |
| 张量网络/MPS (Tensor network/MPS) | 对于低纠缠局部电路可以超过状态向量大小 | 性能取决于纠缠和收缩顺序 | 可选的大量子比特 QML/QAS 引擎 |

## 4. 集成架构

推荐的架构是分层堆栈，而不是单一的单体模拟器类：

```text
前端 API (Frontend APIs)
  - 电路构建
  - 可选类似 QNode 的 QML 包装器
  - QAS 运行器
  - QEC 代码/电路构建器

类型化 IR (Typed IR)
  - aicir.core.Circuit
  - aicir.ir.CircuitIR
  - Operation / Measurement / Observable
  - 旧版 Circuit.gates 兼容性

分析与规划 (Analysis and planning)
  - 门和 Clifford 分类
  - 噪声分类
  - 拓扑、深度和硬件指标
  - 结构性 QAS 预过滤器
  - 引擎选择

原语 (Primitives)
  - SVEstimator
  - ShotEstimator
  - ShotSampler
  - 未来的 BatchedEstimator / QECSampler 门面（如果需要）

引擎 (Engines)
  - NumpyBackend / GPUBackend / NPUBackend 状态向量
  - BatchSV 批处理状态向量
  - 经过 State / Measure / noise 的密度矩阵路径
  - 未来的 TrajectoryEngine
  - 未来的 StabilizerEngine 和 PauliFrameEngine
  - 可选的 TensorNetworkEngine

结果层 (Result layer)
  - EstimateResult
  - SampleResult
  - 具有综合征和逻辑错误字段的未来 QECResult
```

规划器应该首先选择确定性规则，并允许显式覆盖：

```text
if task == "qec" and circuit is Clifford-compatible:
    use StabilizerEngine / PauliFrameEngine
elif noise_model is not None and n_qubits <= density_matrix_limit:
    use density matrix
elif noise_model is not None:
    use TrajectoryEngine
elif task in {"qml", "vqe", "qaoa"} and topology is fixed:
    use BatchSV or GPUBackend/NPUBackend state vector with autograd when possible
elif task == "qas":
    run structural prefilters, then SVEstimator or grouped BatchSV
elif circuit is local and expected entanglement is low:
    optionally use TensorNetworkEngine
else:
    use SVEstimator / state vector backend
```

覆盖示例：

```python
estimator = SVEstimator(backend=backend)
sampler = ShotSampler(backend=backend, shots=4096)

# 未来的规划器外观:
# estimator = Estimator(engine="auto", task="qas")
# sampler = Sampler(engine="stabilizer", task="qec")
```

## 5. 特定于工作负载的方案

### 5.1 QML

推荐方案：

```text
带有参数的 Circuit / CircuitIR
  -> 参数绑定或批处理参数输入
  -> BatchSV 或 GPUBackend/NPUBackend 状态向量
  -> 可微期望
  -> 自动微分 (autograd)、伴随 (adjoint)、参数平移 (parameter shift) 或 QNG 转换
```

实施方向：

- 在 GPU/NPU 上优先使用 `BatchSV` 进行批处理可微模型。
- 保留 `aicir.qml.deriv.auto` 用于由张量支持的 autograd。
- 保留 `psr`、`spsr`、`spsa`、`fd`、`ad` 和 QNG 方法作为显式转换，用于非 autograd 或需要几何感知（geometry-aware）的训练。
- 仅在原语和规划器元数据稳定后，添加类似 QNode 的可选包装器。

### 5.2 QAS

推荐方案：

```text
候选生成器
  -> Circuit / CircuitIR
  -> 结构预过滤器
  -> 拓扑/哈希缓存
  -> SVEstimator、ShotEstimator 或分组的 BatchSV
  -> 多目标排序 / 搜索策略
```

实施方向：

- 基于 `aicir.qas.ArchitectureSearch`、`ArchitectureEvaluator`、`aicir.metrics` 和 `aicir.primitives` 构建。
- 优先使用原语而不是每个候选一个 QNode。
- 为重复的候选添加电路哈希以及前缀/后缀缓存挂钩。
- 在批处理评估可行时，按拓扑对候选进行分组。
- 保留 MoG_VQE、CRLQAS、PPR_DQL、PPO_RB 和 VQA_QAS 作为消费相同评估器/原语层的算法前端。

### 5.3 QEC

推荐方案：

```text
QEC 电路 / 代码构建器
  -> 带有测量/重置/经典元数据的 CircuitIR
  -> Clifford 和泡利噪声分类器
  -> StabilizerEngine + PauliFrameEngine
  -> ShotSampler 风格的综合征采样
  -> 解码器和逻辑错误统计
```

实施方向：

- 为 Clifford 电路和 Pauli 测量添加一个稳定子/tableau 模拟器。
- 添加用于修正跟踪的泡利帧表示。
- 保留密度矩阵执行以进行精确的小型噪声验证。
- 增加量子轨迹，用于无法停留在稳定子范围内的较大型噪声研究。
- 将硬件/AOT 动态电路编译视为 `aicir.transpile` 和 OpenQASM 3.0/QIR 导出之后的后续目标。

## 6. 基于当前 aicir 的实施步骤

### 阶段 1：在不更改行为的情况下添加规划器契约

文件：
- `aicir/primitives/results.py`
- `aicir/primitives/estimator.py`
- `aicir/primitives/sampler.py`
- 新增 `aicir/primitives/planner.py`
- `tests/primitives/` 下的测试

步骤：
1. 添加一个轻量级不可变 `ExecutionPlan` 数据类，包含诸如 `task`、`engine`、`backend_name`、`shots`、`supports_grad`、`noisy`、`batched` 和 `warnings` 字段。
2. 将规划器元数据添加到 `EstimateResult.metadata` 和 `SampleResult.metadata` 中，而不破坏现有字段。
3. 实现初始的 `select_execution_plan(circuit, *, task="auto", backend=None, shots=None, noise_model=None, engine="auto")`。
4. 将现有的 `SVEstimator`、`ShotEstimator` 和 `ShotSampler` 路由通过此规划器（最初仅用于元数据）；执行行为应保持完全相同。
5. 添加测试，证明现有原语输出未发生变化（增加的元数据除外）。

验证：
```bash
env PYTHONPATH=. pytest tests/primitives
```

### 阶段 2：加强批处理 QML 和变分执行

文件：
- `aicir/core/batch.py`
- `aicir/qml/deriv.py`
- `aicir/qml/diff/registry.py`
- `aicir/primitives/estimator.py`
- `tests/core/`、`tests/qml/` 和 `tests/primitives/` 下的测试

步骤：
1. 添加辅助函数，当支持该门集时，通过 `BatchSV` 演化 `Circuit`。
2. 添加当电路无法使用 `BatchSV` 时的清晰回退元数据。
3. 为固定拓扑 VQE/QML 评估支持分组参数批次。
4. 保持非批处理调用下的 `NumpyBackend`、`GPUBackend` 和 `NPUBackend` 行为不变。
5. 添加测试，比较小电路下 `BatchSV` 输出与常规状态向量路径的结果。

验证：
```bash
env PYTHONPATH=. pytest tests/core tests/qml tests/primitives
```

### 阶段 3：添加 QAS 预过滤器、缓存挂钩和分组评估

文件：
- `aicir/qas/evaluator.py`
- `aicir/qas/architecture_search.py`
- `aicir/qas/_utils.py`
- `aicir/metrics/`
- `aicir/primitives/estimator.py`
- 新增 `tests/qas/test_qas_evaluation_planning.py`

步骤：
1. 添加基于 `CircuitIR` 运算、量子比特、控制和门名称的稳定电路结构哈希，当打算共享权重时，排除可训练参数值。
2. 添加可选的结构预过滤器，用于深度、双量子比特数、纠缠拓扑、可训练性代理和硬件效率。
3. 为候选分数和可重用的前缀/后缀状态添加缓存接口。
4. 为具有兼容拓扑的候选添加分组评估。
5. 确保 MoG_VQE 和现有的 QAS 算法可以使用相同的评估器路径。

验证：
```bash
env PYTHONPATH=. pytest tests/qas tests/metrics
```

### 阶段 4：添加 QEC 结果契约和 Clifford 分析

文件：
- 新增 `aicir/qec/`
- `aicir/ir/accessors.py`
- `aicir/gates/registry.py`
- `aicir/primitives/results.py`
- `aicir/measure/measure.py`
- `tests/qec/` 下的测试

步骤：
1. 添加 `QECResult`，包含 `syndrome_history`、`logical_error_rate`、`decoder_metadata`、`pauli_frame` 和 `metadata`。
2. 使用门注册表和类型化 IR 访问器添加 Clifford 门分类辅助工具。
3. 添加泡利测量和重置功能检查。
4. 添加针对 Clifford、非 Clifford、测量和重置电路分类的测试。

验证：
```bash
env PYTHONPATH=. pytest tests/qec tests/ir tests/gates
```

### 阶段 5：实现稳定子/tableau 和泡利帧引擎

文件：
- 新增 `aicir/qec/stabilizer.py`
- 新增 `aicir/qec/pauli_frame.py`
- 新增 `aicir/qec/sampler.py`
- 如果添加通用外观，涉及 `aicir/primitives/sampler.py`
- `tests/qec/` 下的测试

步骤：
1. 实现 tableau 状态初始化、Clifford 门更新、Pauli 测量、重置和 shot 采样。
2. 在物理门应用之外实现泡利帧修正跟踪。
3. 提供一个返回 `QECResult` 的 QEC 采样器外观。
4. 交叉检查小型 Clifford 电路对比现有的状态向量或密度矩阵路径。
5. 当电路兼容 Clifford 时，为 `task="qec"` 添加规划器调度。

验证：
```bash
env PYTHONPATH=. pytest tests/qec tests/measure
```

### 阶段 6：添加噪声扩展路径

文件：
- `aicir/noise/`
- `aicir/measure/trajectory.py`
- 新增 `aicir/qec/noise.py` 用于 QEC 特定的 Pauli 噪声辅助工具
- `tests/noise/`、`tests/measure/` 和 `tests/qec/` 下的测试

步骤：
1. 保持密度矩阵模拟作为精确的小系统参考。
2. 添加或扩展用于更大型噪声电路的轨迹执行。
3. 跨密度矩阵、轨迹和 QEC 路径共享 `NoiseModel` 定义。
4. 添加针对密度矩阵限制和轨迹回退的规划器规则。
5. 添加具有确定性种子和容差的统计测试。

验证：
```bash
env PYTHONPATH=. pytest tests/noise tests/measure tests/qec
```

### 阶段 7：添加可选的张量网络/MPS 引擎

文件：
- 新增 `aicir/tensor_network/`
- 阶段 1 中的规划器模块
- `tests/tensor_network/` 下的测试

步骤：
1. 从 1D 局部电路的选择性加入 (opt-in) MPS 支持开始。
2. 添加与小电路状态向量的精确比较。
3. 在基准测试证明其合理之前，不要启用自动规划器选择。
4. 保持可选依赖项可选。

验证：
```bash
env PYTHONPATH=. pytest tests/tensor_network
```

## 7. 公共 API 政策

- 保留 `Circuit`、`CircuitIR`、`Operation`、`Measurement`、`Observable`、`State`、`Measure`、`SVEstimator`、`ShotEstimator`、`ShotSampler`、`NumpyBackend`、`GPUBackend`、`NPUBackend` 和 `BatchSV`。
- 保留 `Circuit.gates` 兼容性，同时内部消费者继续转向类型化 IR 访问器。
- 将新引擎放在原语/规划器外观后面，而不是要求用户实例化底层模拟内部构件。
- 保持可选依赖项可选。张量网络、QIR、MLIR、Qiskit 或 PennyLane 集成绝不能成为核心模拟的强制要求。
- 当公共 API 更改时，更新 `README.md`、`CHANGELOG.md` 和相关子模块自述文件。仅仅设计文档不需要更改日志条目。

## 8. 最终建议

使用混合架构：

```text
QML -> GPU/NPU 上批处理可微状态向量，带有显式梯度转换
QAS -> 类型化 IR + 原语 + 结构预过滤器 + 缓存 + 分组评估
QEC -> 稳定子/tableau + 泡利帧，具有密度/轨迹验证
```

集成点在于：

```text
单一类型化 IR
单一原语/结果 API
单一自动但可覆盖的执行规划器
多个专用模拟引擎
```

这使得 `aicir` 保持实用性：QML 获得快速的张量路径，QAS 获得可作为可变搜索数据对待的电路，QEC 获得其所需的多项式时间表示，而不是将每个工作负载都强制通过密集的 state vector 模拟器。

## 9. 实施时间表 (Implementation Schedule)

本节将第 6 节中的每个阶段拆解为具体、可分配的工作项，并按**串行执行 (Serial
Execution)**（必须按顺序进行——依赖瓶颈）与**并行执行 (Parallel Execution)**
（供并发团队使用的独立轨道）进行排序。每个工作项均列出其依赖项；每个阶段的文件
和验证命令见第 6 节。

### 9.0 依赖关系图

```text
阶段 1 (规划器契约)  ── 串行根，阻塞一切
   │
   ├── 轨道 A ── 阶段 2 (批处理 QML) ──串行──> 阶段 3 (QAS 预过滤器)
   │
   ├── 轨道 B ── 阶段 4 (QEC 契约) ──串行──> 阶段 5 (稳定子引擎)
   │              阶段 6 (噪声扩展)  ──与 4/5 并行──
   │
   └── 轨道 C ── 阶段 7 (张量网络/MPS)  ──独立，最低优先级──
```

| 阶段 | 类型 | 依赖项 | 负责方 |
| :--- | :--- | :--- | :--- |
| 阶段 1 | 串行根 | 无 | 核心架构师 |
| 阶段 2 → 阶段 3 | 轨道 A 内串行 | 阶段 1 | 团队 A (QML/QAS) |
| 阶段 4 → 阶段 5 | 轨道 B 内串行 | 阶段 1 | 团队 B (QEC) |
| 阶段 6 | 轨道 B 内并行 | 阶段 1 | 团队 B (噪声) |
| 阶段 7 | 独立 | 阶段 1 | 团队 C (可选) |

阶段 1 落地后，轨道 A、B、C 完全并行运行。

### 9.1 串行执行——基础核心

必须在并行轨道集成之前完成。

**阶段 1：规划器契约与核心 API（全局瓶颈）**
*引入路由机制，但不改变行为。以下五个工作项本身是串行的（1.1 → 1.2/1.3 → 1.4 →
1.5）。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 1.1 | 定义不可变的 `ExecutionPlan` 数据类（`task`、`engine`、`backend_name`、`shots`、`supports_grad`、`noisy`、`batched`、`warnings`） | — |
| 1.2 | 将规划器元数据添加到 `EstimateResult.metadata` 和 `SampleResult.metadata`，不破坏现有字段 | 1.1 |
| 1.3 | 实现 `select_execution_plan(circuit, *, task="auto", backend=None, shots=None, noise_model=None, engine="auto")` | 1.1 |
| 1.4 | 将 `SVEstimator`、`ShotEstimator`、`ShotSampler` 路由通过规划器**仅用于元数据**——执行行为保持完全相同 | 1.2, 1.3 |
| 1.5 | 测试证明现有原语输出除新增元数据外保持不变 | 1.4 |

**关卡：** 阶段 1 必须通过（`pytest tests/primitives`）才能分叉下面的并行轨道。

### 9.2 并行执行轨道

阶段 1 完成后，人员分成三个独立轨道。

#### 轨道 A——QML 与 QAS（专注于张量与批处理）——团队 A

**阶段 2：加强批处理 QML 和变分执行**
*阶段 3 的串行前置条件。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 2.1 | 当支持该门集时，通过 `BatchSV` 演化 `Circuit` 的辅助函数 | 阶段 1 |
| 2.2 | 当电路无法使用 `BatchSV` 时的清晰回退元数据 | 2.1 |
| 2.3 | 为固定拓扑 VQE/QML 评估支持分组参数批次 | 2.1 |
| 2.4 | 保持非批处理调用下 `NumpyBackend` / `GPUBackend` / `NPUBackend` 行为不变（回归保护） | 2.1 |
| 2.5 | 比较小电路下 `BatchSV` 输出与常规状态向量路径的测试 | 2.1–2.4 |

**阶段 3：QAS 预过滤器、缓存挂钩、分组评估**
*依赖于阶段 2（复用批处理评估路径）。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 3.1 | 基于 `CircuitIR` 运算/量子比特/控制/门名称的稳定电路结构哈希，权重共享时排除可训练参数值 | 阶段 2 |
| 3.2 | 可选结构预过滤器：深度、双量子比特数、纠缠拓扑、可训练性代理、硬件效率 | 阶段 2 |
| 3.3 | 候选分数和可重用前缀/后缀状态的缓存接口 | 3.1 |
| 3.4 | 为拓扑兼容的候选添加分组评估 | 3.2, 3.3 |
| 3.5 | 确保 MoG_VQE 和现有 QAS 算法使用相同的评估器路径 | 3.4 |

#### 轨道 B——QEC 与噪声（专注于稳定子与轨迹）——团队 B

*轨道 B 有自己的串行链（阶段 4 → 阶段 5），外加与该链并行的阶段 6。人员充足时可
拆分为 B1（引擎：阶段 4→5）和 B2（噪声：阶段 6）。*

**阶段 4：QEC 结果契约与 Clifford 分析**
*阶段 5 的串行前置条件。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 4.1 | `QECResult`，包含 `syndrome_history`、`logical_error_rate`、`decoder_metadata`、`pauli_frame`、`metadata` | 阶段 1 |
| 4.2 | 使用门注册表 + 类型化 IR 访问器的 Clifford 门分类辅助工具 | 阶段 1 |
| 4.3 | 泡利测量和重置功能检查 | 4.2 |
| 4.4 | 针对 Clifford / 非 Clifford / 测量 / 重置电路分类的测试 | 4.1–4.3 |

**阶段 5：稳定子/tableau 与泡利帧引擎**
*依赖于阶段 4。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 5.1 | tableau 状态初始化、Clifford 门更新、Pauli 测量、重置、shot 采样 | 阶段 4 |
| 5.2 | 与物理门应用分离的泡利帧修正跟踪 | 阶段 4 |
| 5.3 | 返回 `QECResult` 的 QEC 采样器外观 | 5.1, 5.2, 4.1 |
| 5.4 | 交叉检查小型 Clifford 电路对比状态向量 / 密度矩阵路径 | 5.1 |
| 5.5 | 当电路兼容 Clifford 时为 `task="qec"` 添加规划器调度 | 5.3, 阶段 1 |

**阶段 6：噪声扩展路径**
*与阶段 4/5 并行（仅依赖于阶段 1）。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 6.1 | 保持密度矩阵模拟作为精确的小系统参考（保护） | 阶段 1 |
| 6.2 | 为更大型噪声电路添加/扩展轨迹执行 | 阶段 1 |
| 6.3 | 跨密度矩阵、轨迹和 QEC 路径共享 `NoiseModel` 定义 | 6.2 |
| 6.4 | 针对密度矩阵限制和轨迹回退的规划器规则 | 6.2, 阶段 1 |
| 6.5 | 具有确定性种子和容差的统计测试 | 6.1–6.4 |

> 工作项 6.3 涉及 `aicir/qec/noise.py`；在阶段 4 的 `aicir/qec/` 包就绪后，与 B1
> 协调共享的 `NoiseModel` 接口。

#### 轨道 C——可选引擎——团队 C

**阶段 7：张量网络/MPS 引擎**
*独立，最低优先级。仅依赖于阶段 1 的规划器模块。*

| 工作项 | 任务 | 依赖项 |
| :--- | :--- | :--- |
| 7.1 | 针对 1D 局部电路的选择性加入 (opt-in) MPS 支持 | 阶段 1 |
| 7.2 | 与小电路状态向量的精确比较 | 7.1 |
| 7.3 | 在基准测试证明合理之前，**不要**启用自动规划器选择 | 7.1 |
| 7.4 | 保持可选依赖项可选 | 7.1 |

### 9.3 调度摘要

| 时间 | 串行工作 | 并行工作 |
| :--- | :--- | :--- |
| Sprint 0 | **阶段 1**（1.1→1.5）由核心架构师完成 | — |
| Sprint 1+ | 阶段 2→3（团队 A）；阶段 4→5（团队 B1） | 阶段 6（团队 B2）；阶段 7（团队 C） |

**关键路径：** 阶段 1 →（阶段 2→3 *或* 阶段 4→5 中较长者）。阶段 6 和 7 绝不应
处于关键路径上；仅在依赖的串行链（A 和 B1）有人覆盖后才为其配备人员。

**横切关注点（依据第 7 节）：** 任何更改公共接口的工作项都必须同时更新
`README.md`、`CHANGELOG.md` 和相关子模块自述文件，并且必须保留所列公共符号和
`Circuit.gates` 兼容性。全程保持可选依赖项可选。

# Changelog

本文件记录 `aicir` 库的功能新增与重要接口变化。日期使用本地开发日期。

## 2026-06-15

### Added

- 新增 `DiffMethod` 策略注册表（NEXT.md 第 6 节第一片）：子包 `aicir.qml.diff` 提供冻结数据类 `DiffMethod`（字段含 `name`/`fn`/`aliases`/`exact`/`stochastic`/`requires_torch`/`supports_shots`/`supports_noise`）与注册表 API（`register_diff_method`/`unregister_diff_method`/`get_diff_method`/`registered_diff_methods`/`canonical_diff_name`/`resolve_diff_method`），并从 `aicir.qml` 顶层再导出。
- 新增纯函数选择器 `select_diff_method(*, backend=None, shots=None, noisy=False)`，按 auto → psr → fd 优先级自动推断梯度方法（`spsa`/`spsr` 不参与自动选择）；已有单元测试覆盖，暂未接入调用方（保留给后续 QNode）。
- 内置注册 fn-based 全梯度方法：`psr`/`fd`/`auto`/`spsa`/`spsr`；`mpsr`（返回标量混合偏导而非梯度向量）有意排除在注册表之外，仍作为 `qml.mpsr` 直接可用；基于线路的 `ad` 与预条件策略 `qng` 同样不纳入注册表。

### Changed

- `aicir/optimizer/params.py` 的 `_gradient_from_method` 改为经 `resolve_diff_method` 分发，`GD`/`Adam`/`ScipyMinimize` 现可统一访问所有内置梯度方法；对 `requires_torch=True` 的方法（即 `auto`）在经典黑盒目标路径上新增守卫，传入时抛出明确错误。
- `aicir/qml/deriv.py` 未改动；`vqc`/`qas` 保持原有 `from ..qml.deriv import psr` 路径，参数移位单一实现不变。

## 2026-06-14

### Added

- **批量态矢量路径 `BatchSV`**（`aicir.core.batch`，已在 `aicir` / `aicir.core` 顶层导出）：一次演化一批态矢量，供把变分线路当作神经网络一层的场景使用。特性：
  - 旋转门角度可逐样本（per-sample）不同（如数据编码角度依赖输入），亦兼容 0 维标量张量参数（保留 autograd 梯度）；
  - 全程以实部/虚部两个实张量表示，只用实数乘加，**NPU 安全**（规避 Ascend 缺失的 complex64 `aclnnAdd`/`aclnnMul`），反向不触发复数累加；
  - 端到端可微；门矩阵复用单态路径 `_single_qubit_base_for_gate` 的定义（常量门/标量旋转门），逐样本张量角度按相同公式构造批量基矩阵；
  - 端序与单态路径一致（qubit 0 为最高位）；提供 `apply_gate`、`probabilities`、`z_expectations`（逐比特 `<Z_q>`）。
  - 受控单比特门（`crx/cry/crz` 等）通过控制比特掩码通用实现。

### Changed（破坏性变更）

- **后端、噪声、算符上移到顶层（`aicir.channel` 包移除）**：`aicir.channel.backends` → `aicir.backends`、`aicir.channel.noise` → `aicir.noise`、`aicir.channel.operators` → `aicir.operators`。旧 `aicir.channel` 包及其 `__init__` 已删除，不保留兼容垫片。顶层再导出（`Backend`/`NumpyBackend`/`GPUBackend`/`NPUBackend`、`NoiseModel`/`NoiseChannel`/各噪声信道、`PauliOp`/`PauliString`/`Hamiltonian`）不变，`from aicir import ...` 无需改动；仅子模块路径导入（如 `from aicir.channel.backends.numpy_backend import NumpyBackend`）需改为新路径。
  - 顺带修复 `aicir/__init__.py` 中 `measure` 门构造器被 `aicir.measure` 子包属性覆盖的隐患（旧 `channel/__init__` 早期导入 `aicir.measure` 恰好掩盖了该问题）：现在导入 measure 子包后显式恢复门构造器绑定。
- **`measure`/`reset` 量子比特参数改为「单个 int 或列表」**：`measure([0,1], ...)` / `reset([0,1])`；不再支持 `measure(0, 1)` 多位置形式（请改用列表）。
- **统一测量模型**：删除"机制一/机制二互斥"框架；线路内嵌 `measure`/`reset` 与末端读出（`tm`/`measure_qubits`）现可共存，两者是同一模型的两个正交部分。
- **线路内 `measure(basis, id)` 改为投影测量**：由非坍缩边缘采样标记改为真正的两结果联合 Pauli 投影（`basis∈{Z,X,Y}`，同子空间保留相干）；新增 `id` 参数使 `result.output("id")` 可用；`basis`/`id` 字段在 IR 中作为一等字段序列化（JSON 往返保真）。
- **`reset` 改为无前置条件的信道**：删除"必须先有同一比特 `measure`"的限制；`reset` 可出现在任意位置；对纠缠目标施加 reset 使该轨迹升级为密度矩阵。
- **新增 `run()` 参数 `tm`/`sm`/`seed`**：`tm=True`（默认）控制末端测量；`sm="avg"` 为多轨迹聚合模式（`"shot"/"cond"` 待实现，传入报 `NotImplementedError`）；`seed` 单一 RNG 种子贯穿全部随机操作。
- **`output`/`counts`/`prob` 改为方法**：`result.output(i)`/`result.counts(i)`/`result.prob(i)` 按操作下标（或字符串 `id`）索引；末端结果用 `output(-1)`/`counts(-1)`。原字段访问方式（`result.output`/`result.counts`）已移除。
- **`state`/`final_state` 语义更新**：`state` = 末端测量前完整态（`ρ_pre`）；`final_state` = 末端测量后的态（`ρ_post`）；`shots>1` 时两者均为平均密度矩阵 `(2^n,2^n)`。
- **`shots=0` ≡ `None`（exact 模式）且覆盖 `tm`**：exact 模式下不执行末端测量（`final_state==state`，`output(-1)` 报错）；线路内 `measure()` 仍走一条随机轨迹并坍缩。这是对上传规则 23（`shots=0` 非法）与规则 36（`shots=None` 仍取末端结果）的显式偏差。
- **`Circuit.unitary()` 遇 measure/reset 改为报错**：默认对 `measure`/`reset` 一律 `raise`（原行为：`measure` 静默跳过）；新增 `ignore_nonunitary=True` 选项以丢弃非酉操作并返回酉部分。
- **QASM 导出限制**：联合（多比特）/ 非 Z 基 / 带 `id` 的 `measure` 导出时报 `NotImplementedError`；普通单比特 Z `measure` 仍按标准 QASM 2.0/3.0 导出。
- **末端输出顺序保留输入顺序**：`measure_qubits` 原始输入顺序保留到 `output(-1)` 列顺序，不做内部排序。
- **已移除**：`run_density_matrix`、`run_batch`、`scan_parameters` 旧入口；噪声通过 `circuit.noise_model` / `initial_density_matrix` 传入新 `run()`。
- **`sm="shot"/"cond"` 待实现**：传入时报 `NotImplementedError`，后续版本实现。

## 2026-06-12

### Added

- 新增 Qiskit 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_qiskit`/`circuit_from_qiskit` 以及短别名 `to_qiskit`/`from_qiskit`，位于 `aicir.core.io.qiskit_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`qiskit` 为可选依赖，仅在调用互操作函数时导入；当前支持基础门、参数旋转、受控门、`swap`、`rzz/rxx`、`u2/u3`、`ccx` 和线路内 `measure` 标记。
- 新增 PennyLane 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_pennylane`/`circuit_from_pennylane` 以及短别名 `to_pennylane`/`from_pennylane`，位于 `aicir.core.io.pennylane_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`pennylane` 为可选依赖，仅在调用互操作函数时导入；当前支持基础门、参数旋转、受控门、`swap`、`rzz/rxx`（PennyLane `IsingZZ`/`IsingXX`）、`u2/u3`、`ccx` 和 `identity`。
- 新增 WuYue 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_wuyue`/`circuit_from_wuyue` 以及短别名 `to_wuyue`/`from_wuyue`，位于 `aicir.core.io.wuyue_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`wuyue` 为可选依赖，仅在调用互操作函数时导入；当前支持 WuYue 原生基础门、参数旋转、`cx/cz`、`swap`、`rzz`（WuYue `IsingZZ`）、`u2/u3`、`ccx`、`identity` 和线路内 `measure` 标记。
- `Result` 新增 `state` 字段：始终返回测量前的完整末态（SV 路径为态向量，DM 路径为 flatten 密度矩阵），不受采样影响。
- `Result` 新增 `output` 字段：`shots=1` 单次测量结果——被测比特上 Z⊗…⊗Z 关联投影测量的本征值（±1，实现为对各被测比特分别做 Z 基投影后取乘积，与联合宇称测量的 ±1 分布一致且保证其余比特为纯态）；坍缩到的具体基态见 `counts` / `final_state`。
- `result.metadata` 新增 `final_state_kind`（`'state_vector'`/`'density_matrix'`/`None`）与 `final_state_qubits`（`final_state` 所描述的比特下标）。
- 新增线路内 `reset(*qubits)` 标记，与 `measure(*qubits)` 参数格式一致；每个 reset 目标必须先有同一比特的 `measure`，且二者之间不能有任何量子门作用于该比特。`Measure.run()`/`run_density_matrix()` 支持 reset 执行语义；matplotlib 线路图以与 `measure` 同色、标注 `Reset` 的虚线表示 reset，`Reset` 字号与 `Rz` 门主标签一致且虚线 dash 间距更大；虚线区间不叠加普通实线，没有后续量子门时延伸到线路末端；遇到 CNOT/SWAP 等无完整方框的后续门时，虚线停在虚拟方框左边界，虚拟方框内部恢复普通实线。
- 新增 `demos.reset_demo`，通过 `H(0) -> cnot(1,0) -> cnot(2,1) -> measure(1) -> reset(1) -> cnot(1,2)` 三比特线路的 `result.snap(...)` 记录 reset 前后中间态，验证 `reset(1)` 将已测量比特从 `|1>` 重置为 `|0>`，且后续门从重置后的态继续演化。

### Changed

- **破坏性**：`Measure.run`/`run_density_matrix` 的 `shots` 默认值由 `None` 改为 `1`；`shots=None`/`0` 表示不测量（无论是否传 `measure_qubits`）。
- **破坏性**：`result.final_state` 语义由"演化末态"改为"测量后的态"：`shots=None`/`0` 时与 `state` 相同；`shots=1` 时为坍缩后的态（子集读出时仅含未被测比特）；`shots>1` 时为对被测比特求偏迹后的约化密度矩阵（SV 路径为 `(2^m, 2^m)` 二维，DM 路径为 flatten；读出全部比特时无剩余比特，为 `None`）。需要演化末态请改用 `result.state`（内部消费方 `BasicVQE.ansatz_state`、`StatevectorEstimator` 已随之切换）。
- `shots` 为负数时显式抛出 `ValueError`。
- **破坏性**：统一量子态表示为单一 `State` 类，删除 `StateVector` 与 `DensityMatrix` 两个公开名称（顶层 `aicir` 与 `aicir.core` 不再导出它们）。
  - 新增 `State.from_matrix(...)` 从密度矩阵构造；`from_array`/`from_matrix`/`zero_state` 的 `backend` 改为可选（默认 `NumpyBackend`），`from_array`/`from_matrix` 可省略 `n_qubits`（按长度/形状推断）。
  - 新增属性 `.array`（纯态振幅向量，混合态为 `None`）、`.matrix`（密度矩阵）、`.ket`（Dirac 记号：纯态超叠加、混合态 Σρ_ij|i><j| 展开），均可直接打印。
  - 新增 `.is_density` 判定属性；密度矩阵方法（`purity`/`partial_trace`/`eigenvalues`/`von_neumann_entropy`/`is_pure`/`maximally_mixed`）并入 `State`。
  - 迁移指引：`isinstance(x, DensityMatrix)` 改用 `x.is_density`；`DensityMatrix(...)` 改用 `State.from_matrix(...)`；`StateVector(...)` 改用 `State(...)` / `State.from_array(...)`。

## 2026-06-11

### Added

- 新增 `aicir.gates` GateSpec 注册表（NEXT.md 第 7 节第一片）：`GateSpec`（门名/别名/目标比特数/参数个数/是否受控/QASM 名）、`register_gate`/`unregister_gate`/`get_gate_spec`/`registered_gate_names`；内置门集已全部注册。`num_qubits`/`num_params` 为 `None` 表示可变（`unitary`、`measure`、整寄存器 `identity`）。
- `Operation` 构造期接入 GateSpec 校验：已注册门检查目标比特数、参数个数与控制位要求，未注册门名保持宽松（自定义门不受限）。
- `aicir.transpile.ValidatePass` 升级为实质校验：qubit/控制位越界（相对 `n_qubits`）、目标与控制比特冲突、重复比特；原先仅做 round-trip 规范化。
- 新增 `aicir.gates.canonical_gate_name`：把别名门名（`X`/`cnot`/`ccnot`/`measurement` 等）解析为规范名，未注册名称原样返回。
- `aicir.transpile.CanonicalizePass` 升级为实质规范化：把门字典中的别名 `type` 重写为 GateSpec 规范名；原先仅做 round-trip 复制。
- QASM 导出门名改以 GateSpec 注册表为单一来源：`core/io/qasm.py` 的导出表由 `GateSpec.qasm_name` 派生，别名键（`X`/`cnot`/`ccnot` 等）从表中移除，导出时先经 `canonical_gate_name` 归一；导入表由导出表反推。导出结果与旧版完全一致（有回归测试钉住）。
- `GateSpec` 新增 `symbol` 字段（绘图显示符号，受控门为目标位符号），ASCII 与 matplotlib 绘图的符号/配色族查询改以注册表为单一来源：`_single_gate_symbol`/`_controlled_target_symbol` 由 `GateSpec.symbol` 派生（注册自定义门可携带 symbol 直接显示），`visual/plot.py` 的 `_FAMILY` 配色表只保留规范名键。
- 矩阵路径统一别名处理：`gate_to_matrix`/`apply_gate_to_state`/`_single_qubit_base_for_gate` 等在入口经 `canonical_gate_name` 归一后分发，全部 `["pauli_x", "X"]`/`["cnot", "cx"]`/`["toffoli", "ccnot"]` 式别名分支收敛为规范名（别名与规范名共享矩阵缓存）。行为与旧版完全一致（有别名等价回归测试钉住）。
- 新增 `aicir.primitives`（NEXT.md 第 4 节第一片）：`BaseSampler`/`BaseEstimator` 接口与最小统一结果对象 `SampleResult`/`EstimateResult`（第 9 节切片）；`ShotSampler` 包装 `Measure`，`StatevectorEstimator` 提供精确态向量期望，`ShotEstimator` 包装 `PauliEstimator` 并暴露 `estimate()` 直通方法（可直接作 `BasicVQE(energy_estimator=...)` 注入）。约定：接收已绑定参数的电路，单入参返回单结果、序列返回列表，单个可观测量可广播。

### Changed

- 门工厂函数（`pauli_x`/`hadamard`/`rx`/`cx`/`swap`/`rzz`/`u3`/`u2` 等）**签名与参数顺序完全不变**，但返回值由裸门字典升级为类型化 `Operation`；`measure(...)` 返回 `Measurement`。构造期即校验（量子比特下标、控制位/控制态长度等）。`Circuit` 内部存储的门字典与旧版完全一致，下游消费方无需改动。
- `Operation`/`Measurement` 新增旧门字典**只读**兼容层：支持 `gate["type"]`、`.get()`、`in`、`dict(gate)`、`len`/迭代等读取，以及与旧门字典的双向 `==` 比较；写入（`gate[...] = ...`）抛出 `TypeError`（对象不可变）。
- `aicir.visual` 的 `plot(...)`/`show(...)` 来源归一化现接受类型化指令（单个 `Operation`/`Measurement` 或其序列）。

## 2026-06-10

### Added

- `Operation` 新增显式 `label` 字段（默认 `None`），对齐 NEXT.md typed IR 规格；门字典中的 `label` 键现在提升为该字段而不再落入 `metadata`，`to_dict`/`from_dict` 保持 round-trip。
- 新增第一批架构演进目录占位：`aicir.ir`、`aicir.gates`、`aicir.transpile`、`aicir.transpile.passes`、`aicir.devices`、`aicir.primitives`，用于后续 typed IR、GateSpec、pass pipeline、Target 和 Sampler/Estimator primitives 迁移。
- `NEXT.md` 记录 `aicir` 目标目录结构和第一批已落地目录。
- 新增 typed IR `Operation`：支持从现有门字典构造、转换回门字典、通过 `normalize_gate` 兼容旧入口；`Circuit` 构造、`append`、`extend` 现在可接收 `Operation`，同时内部继续保存现有门字典格式。
- 补齐 typed IR 第一阶段剩余对象：`Measurement` 支持测量声明与现有 `measure` 门字典互转，`Observable` 支持包装 `PauliString`、`Hamiltonian` 和 dense matrix，`CircuitIR` 支持从现有 `Circuit` 构造、转回 `Circuit`，并保留 operation 序列、量子比特数、经典比特和 metadata。
- 新增 typed IR 访问 helper：`aicir.ir.circuit_instructions`、`circuit_gate_dicts`、`instruction_name`、`instruction_qubits`、`instruction_controls`、`instruction_parameter` 等，用于内部模块统一消费 `CircuitIR`、`Circuit.operations` 和旧 `Circuit.gates`。
- `Circuit` 新增 `.operations` 与 `.ir` typed IR 视图；`.gates` 继续保留为旧门字典公开 surface。
- 新增 `aicir.transpile` pass pipeline：提供 `TransformationPass`、`PassManager`、`default_optimization_pipeline`，以及 `ValidatePass`、`CanonicalizePass`、`CancelInversePass`、`MergeRotationsPass`、`CommuteSingleQubitPass`；`optimize_circuit` 和 `optimize_basic(Circuit)` 继续保留旧接口并委托给默认 pipeline。
- 新增 `aicir.optimizer.optimize_circuit` 公开入口，用于直接优化 `Circuit` 对象并保留 `n_qubits` 与 backend。
- 扩展 `aicir.optimizer.circuit` 的 dict/Circuit 路径：支持有限安全重排，可跨过不同量子比特的单比特门，以及已知可交换的 CNOT 模式来消去冗余门或合并 `rx/ry/rz`。

### Changed

- JSON/QASM/DAG 导出、绘图、测量、Pauli 估计、transpile/optimizer、QML 伴随梯度、metrics、noise 和 QAS 的主要内部读取路径迁移为优先消费 typed IR；需要旧门字典格式的矩阵、渲染和本地 rewrite 兼容层会显式从 typed instruction 生成门字典视图。

### Tests

- 新增 `tests/circuit/test_typed_ir_internal_migration.py`，覆盖 `CircuitIR` 直接进入 JSON/QASM、绘图、metrics、测量、QML 伴随梯度、transpile/optimizer 和核心门矩阵路径。

## 2026-06-09

### Fixed

- 修复 `aicir.qas.supernet` 计算 `Hamiltonian` 期望能量时的形状广播 bug：当态向量为 `(2^n, 1)` 列向量（训练/微调路径的默认形状）时，会与一维 phase/index 向量广播成 `(2^n, 2^n)` 并 `sum()`，导致能量被放大 `2^n` 倍。受影响时损失停在 `真实能量 × 2^n`、梯度趋近 0、架构排序失效，supernet 无法收敛到基态。现统一把态向量展平为一维后再计算，能量幅值正确，VQE/QAS 可正常收敛（MaxCut 等对角哈密顿量近似比从约 0.6 提升到 1.0）。
- 修正 MaxCut demo 的结果判读：区分能量期望对应的 `expected_cut` 与最佳显著读出比特串对应的 `achieved_cut`，避免把非基态叠加态误报为已完全收敛；同时将默认 ansatz 深度从 4 层提高到 6 层，并重新生成示例线路，使 5 节点示例的 `expected_cut` 接近精确最大割。
- MaxCut demo 新增 `--disable-rzz` 参数，可在 `supernet_qas` 搜索中禁用 `rzz`，只保留 `cx` 双比特门。
- 修正线路图中文字自适应：方块内文字渲染宽度限制在方块宽度的 0.8 倍以内，所有门内/门下文字高度限制在方块高度的 0.7 倍以内，方块下方参数文字宽度限制在方块宽度以内，避免长角度标签溢出。

### Added

- 新增 `demos/MaxCut/maxcut.py`：随机图 → MaxCut Ising 哈密顿量 → `supernet_qas` 搜索 VQE 基态线路 → 把哈密顿量与线路写入 `maxcut_hamiltonian.py`，并把随机图与线路一起绘制到 `maxcut_hamiltonian.png`。

## 2026-06-05

### Added

- 新增测量机制二：`measure(*qubits)` 门工厂，可在构造 `Circuit` 时内嵌测量标记，调用 `Measure.run()` 时自动仅读出标记比特，无需额外传入比特参数。
- `Measure.run()` 与 `Measure.run_density_matrix()` 自动检测线路中的 `measure` 门并跳过幺正演化；演化结束后计算指定比特子集的边际概率分布，输出对应多比特计数字典（MSB 顺序）。
- `Measure.run()` 结果的 `metadata["measured_qubits"]` 字段记录被读出的比特下标列表（机制二）；机制一下为 `None`。
- `Measure.run()` 与 `Measure.run_density_matrix()` 新增 `measure_qubits` 参数，可在机制一下显式指定要读出的比特子集（`metadata["measured_qubits"]` 会随之记录）；`run_batch` / `scan_parameters` 可通过 `per_circuit_options` 传递。
- 两种测量机制互斥：当电路已内嵌 `measure()` 门（机制二）时再传入 `measure_qubits`（机制一）会抛出 `ValueError`，避免两种读出方式相互冲突。
- `measure` 函数从 `aicir` 顶层导出，支持 `from aicir import measure`；`aicir.measure` 子包仍可通过 `from aicir.measure.measure import Measure` 访问。
- 线路图支持 `measure(q1, q2, ...)` 多比特测量门绘制：每个被测比特线上绘制独立测量框，测量框右侧不再延伸导线（符合量子线路惯例）。
- 新增 `Circuit.plot(...)` 语法，用于直接从电路对象输出线路图；默认文件位置为调用该方法的 `.py` 文件所在目录。
- 新增 `rxx(θ, q1, q2)` 双比特 XX 旋转门，并提供 `ms_gate` / `molmer_sorensen` 作为 Mølmer-Sørensen gate 别名。
- `rxx` 支持矩阵构造、逐门态演化、Torch autograd、QASM 导入导出、QML adjoint gradient、HEA entangler、metrics/QAS/noise 统计路径。
- 新增 `rxx` 与 Mølmer-Sørensen gate 的单元测试，覆盖矩阵定义、别名、QASM round-trip、Torch 梯度、绘图和 HEA/QML 路径。

### Changed

- `VQA_QAS` 的参数移位（parameter-shift）梯度改为复用 `aicir.qml.deriv.psr`，与 VQE/SSVQE/VQD 一致，将移位规则收敛到单一实现；梯度数值保持不变（标准 Pauli 旋转规则，shift=π/2、coefficient=0.5）。
- 改进线路图 layer packing，确保后出现且跨越相同 wire span 的门不会被绘制到前序多比特门之前。
- 线路图中 `Rzz`/`Rxx` 使用完整门名显示；`rzz`/`rxx` 参数值显示在两个对应门框内部。
- 线路图中 `rx`/`ry`/`rz` 及受控旋转门的参数值移入对应门框内部，显示在门名下方。
- 线路图中 `u2`/`u3` 参数值显示在门框下方的小字号文本中；`u2` 参数单行显示，`u3` 参数两行显示。
- README 和子模块文档补充 `Circuit.plot(...)`、`rxx`/Mølmer-Sørensen gate、QASM、Torch autograd、QML AD 和 HEA entangler 说明。

### Tests

- 全量测试通过：`env PYTHONPATH=. pytest`，共 299 项。
- 视觉绘图测试通过：`env PYTHONPATH=. pytest tests/visual/test_visual.py`，共 24 项。

## 2026-06-04

### Added

- 新增 `aicir.vqc.ansatz.hea`：标准 hardware efficient ansatz，支持 `Circuit`、`Parameter`、多种旋转门、entangler 和 topology。
- 新增 `aicir.vqc.ansatz.hea_ti`：trapped-ion HEA-TI ansatz，包含 TFIM/XY 全局演化、power-law 耦合和 HEA-TI 参数数量工具。
- 新增 `aicir.optimizer.params`：VQE/VQA 参数优化器，包括 `GD`、`Adam`、`SPSA`、`COBYLA`、`LBFGSB`、`ScipyMinimize`、`minimize`。
- 新增 `aicir.measure.estimator`：shot-based Pauli-term energy estimator，支持 Hamiltonian Pauli 项拆分、qubit-wise commuting 分组、基变换测量、shots 分配和能量方差统计。
- 新增通用 VQE orchestration：`BasicVQE` 支持 `Circuit`/callable ansatz、`Parameter` 绑定、`Hamiltonian`、backend、`Measure`、shots、density_matrix noise、初态配置、外部 optimizer 和可配置 `energy_estimator`。
- 新增 `aicir.chemistry.molecule`：预置已确认系数的 H2 qubit Hamiltonian，包括 parity 2-qubit、Jordan-Wigner 4-qubit 和 tapered 1-qubit 版本。
- 新增 `aicir.chemistry.README`：记录 chemistry 子包的当前接口、预置分子、示例和设计约束。
- 新增 `aicir.vqc.README`：记录 VQE orchestration、HEA、HEA-TI 和梯度工具配合方式。
- 新增 `demos.vqe_h2_demo`：从 H2 Hamiltonian preset、HEA ansatz、VQE 编排和 optimizer 出发，演示氢分子基态能量求解。

### Changed

- `aicir.optimizer.basic` 改名为 `aicir.optimizer.circuit`，用于线路结构优化相关工具。
- 删除 `optimizer/basic.py`，不再保留旧模块文件。
- 参数优化器只保留简短名称，不保留 `AdamOptimizer`、`SPSAOptimizer` 等长别名。
- chemistry molecule preset 使用 `h2`、`h2_jw`、`h2_tapered` 等简短 canonical 名称，不保留旧长名称或额外别名。
- 将 VQE 文档和注释中的 `dense-matrix` / `dense matrix` 统一为 `dense_matrix`。

### Tests

- 新增 HEA、HEA-TI、参数优化器、PauliEstimator、VQE orchestration、chemistry molecule preset 相关测试。

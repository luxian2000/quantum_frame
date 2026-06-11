# Changelog

本文件记录 `aicir` 库的功能新增与重要接口变化。日期使用本地开发日期。

## 2026-06-11

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

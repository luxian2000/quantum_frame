# Changelog

本文件记录 `aicir` 库的功能新增与重要接口变化。日期使用本地开发日期。

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

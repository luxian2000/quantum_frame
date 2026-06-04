# Changelog

本文件记录 `aicir` 库的功能新增与重要接口变化。日期使用本地开发日期。

## 2026-06-04

### Added

- 新增 `aicir.vqc.ansatz.hea`：标准 hardware efficient ansatz，支持 `Circuit`、`Parameter`、多种旋转门、entangler 和 topology。
- 新增 `aicir.vqc.ansatz.hea_ti`：trapped-ion HEA-TI ansatz，包含 TFIM/XY 全局演化、power-law 耦合和 HEA-TI 参数数量工具。
- 新增 `aicir.optimizer.params`：VQE/VQA 参数优化器，包括 `GD`、`Adam`、`SPSA`、`COBYLA`、`LBFGSB`、`ScipyMinimize`、`minimize`。
- 新增 `aicir.measure.estimator`：shot-based Pauli-term energy estimator，支持 Hamiltonian Pauli 项拆分、qubit-wise commuting 分组、基变换测量、shots 分配和能量方差统计。
- 新增通用 VQE orchestration：`BasicVQE` 支持 `Circuit`/callable ansatz、`Parameter` 绑定、`Hamiltonian`、backend、`Measure`、shots、density_matrix noise、初态配置和外部 optimizer。
- 新增 `aicir.chemistry.molecule`：预置已确认系数的 H2 qubit Hamiltonian，包括 parity 2-qubit、Jordan-Wigner 4-qubit 和 tapered 1-qubit 版本。
- 新增 `aicir.chemistry.README`：记录 chemistry 子包的当前接口、预置分子、示例和设计约束。
- 新增 `aicir.vqc.README`：记录 VQE orchestration、HEA、HEA-TI 和梯度工具配合方式。

### Changed

- `aicir.optimizer.basic` 改名为 `aicir.optimizer.circuit`，用于线路结构优化相关工具。
- 删除 `optimizer/basic.py`，不再保留旧模块文件。
- 参数优化器只保留简短名称，不保留 `AdamOptimizer`、`SPSAOptimizer` 等长别名。
- chemistry molecule preset 使用 `h2`、`h2_jw`、`h2_tapered` 等简短 canonical 名称，不保留旧长名称或额外别名。
- 将 VQE 文档和注释中的 `dense-matrix` / `dense matrix` 统一为 `dense_matrix`。

### Tests

- 新增 HEA、HEA-TI、参数优化器、PauliEstimator、VQE orchestration、chemistry molecule preset 相关测试。

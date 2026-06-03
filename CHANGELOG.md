# CHANGELOG

All notable changes to this project will be documented in this file.

## Unreleased (2026-05-06)

### Added
- `aicir/optimizer/basic.py`: 新增并改进优化器：
  - 固定点（fixed-point）优化：对 `dict`/`qasm`/`dag` 路径均采用迭代优化直至收敛（最多 64 轮）。
  - 新增相邻单量子比特旋转合并规则：合并 `rx/ry/rz` 在同一量子比特上的相邻旋转门（角度相加），合并后角度接近 0 时删除该门。
  - 增强 QASM 文本解析，支持解析并合并 `rx(...)`、`ry(...)`、`rz(...)`（角度支持 `pi` 表达式）。

- `aicir/encoder/demos/encode_1234_demo.py`: 在导出前对 `Circuit` 做一轮优化，并在导出后的 QASM 文本上再做一轮 `qasm` 优化，避免导出器插入的预/后 `x`（control_state=0 包装）导致冗余未被移除。

- 新增文档 `aicir/optimizer/README.md`，说明优化规则、固定点行为、示例与测试位置。

- 新增/更新测试：`tests/circuit/test_optimizer_basic.py`，覆盖 `rx/ry/rz` 合并及固定点收敛行为；并修正相关 IO 测试。

### Changed
- `aicir/core/io/qasm.py`: QASM 导出增强（已包含在此前提交）：
  - 对多控 `crx/cry/crz` 在 OpenQASM 3.0 下自动分解为 `ccx` 聚合 + 单控旋转 + 反计算序列，并在文件头声明 `anc` 辅助寄存器。
  - 对 `control_states=0` 的控制位自动插入前后 `x` 包裹以转换为 `|1>` 控制逻辑（原有行为，现与优化器配合以减少冗余）。

### Tests
- 全量测试运行结果：`63 passed`（包含新增测试）。

### Commit
- 提交: `7e6bfa7` — `optimization: fixed-point & rx/ry/rz merge; demo qasm post-opt`

---

（若需按照语义版本或发布版本记录，请告知要添加到哪个版本栏目下。）

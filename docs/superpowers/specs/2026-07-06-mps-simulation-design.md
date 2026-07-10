# MPS（矩阵乘积态）近似模拟引擎设计（Spec 2）

日期：2026-07-06
状态：已批准设计，待实现

## 目标

在 `aicir/simulator/` 中新增**带 bond dimension 截断的 MPS（Matrix Product State）近似
模拟引擎**，作为精确张量网络引擎（`aicir/simulator/network.py`/`contract.py`，Spec 1，
2026-07-02 已落地）之外的第二种模拟方法。目标场景：低纠缠、近 1D 拓扑的电路（如有限
纠缠深度的 HEA 类 ansatz），用截断换取比精确态矢量更大的可模拟比特数。

`aicir/simulator/README.md` 第 6 节已预留此设计位置：「MPS/张量网络截断近似（有损
压缩）不在本模块范围内，属于另一独立规划（Spec 2）」——本文档即该 Spec 2。

## 非目标（本 spec 不做）

- **含噪声 / 密度矩阵路径**：MPS 引擎只处理纯态；噪声仍走既有密度矩阵模拟。
- **NPU 支持**：`Backend.svd()` 在 `NPUBackend` 上显式 `NotImplementedError`（见「背景与
  关键事实」）。NPU 上的 MPS/SVD 留待后续独立评估。
- **`ControlFlow` 电路**：与精确张量网络引擎一致，直接拒绝，提示走 `Measure.run`
  的态矢量轨迹路径。
- **3+ 比特门的自动分解**：MPS 引擎只接受 1/2 比特门；3+ 比特门要求调用方先经
  `aicir.transpile.DecomposePass` 分解，引擎本身不内部调用 transpile。
- **重写既有精确引擎**：MPS 是并列的第三种模拟方法（态矢量 / 精确 TN / MPS），既有
  `State`/`Measure` 语义不变。

## 背景与关键事实

- `Backend`（`aicir/backends/base.py`）现有算子含 `tensordot/transpose/reshape/conj/
  take/add`（Spec 1 落地），但**没有 SVD**。全仓库 `grep -rn "svd"` 命中为零——MPS 截断
  是第一个需要奇异值分解的子系统。
- `NPUBackend` 对原始复数张量操作支持不全（`aclnnMatmul`/`aclnnInplaceAdd` 等对
  `DT_COMPLEX64` 报错，见 CLAUDE.md）；复数 SVD 反传在 Ascend 上目前没有已验证的安全
  路径，因此本 spec 将 NPU 上的 `svd()` 显式设为 `NotImplementedError`，不强行拼凑。
- `GPUBackend`（Torch）的 `tensordot`/`matmul` 均可微；`torch.linalg.svd` 本身支持
  autograd（奇异值不简并处的反传数值稳定），因此 `MPSEstimator` 在 GPU 后端上可提供
  梯度，与 `StatevectorEstimator`/精确 TN 引擎的可微性一致。
- 现有 primitives（`aicir/primitives/estimator.py`）里 `StatevectorEstimator`/
  `NoisyEstimator`/`ShotEstimator` 均遵循同一模式：`estimate(circuit, hamiltonian, **_)`
  给 `BasicVQE(energy_estimator=...)` 直通，`run(circuits, observables, *, shots=,
  parameter_values=)` 走统一 `Sampler`/`Estimator` 协议、返回 `EstimateResult`。
  `MPSEstimator` 复用同一形状。
- `aicir.transpile.RoutingPass`（2026-06-30 升级）已有「插入 SWAP 后不复位、把比特置换
  向前携带」的做法，`final_layout` 记录累计置换。MPS 引擎里非相邻双比特门的 SWAP 处理
  采用同样思路，但**独立实现**，不依赖 `aicir.transpile`（MPS 的“相邻”是 site 顺序，
  与 transpile 的物理耦合图是不同的坐标系）。

## 架构

```
aicir/backends/base.py          # 扩展：新增 svd(matrix) -> (U, S, Vh) 抽象方法
aicir/backends/numpy_backend.py # 实现：np.linalg.svd(matrix, full_matrices=False)
aicir/backends/gpu_backend.py   # 实现：torch.linalg.svd(matrix, full_matrices=False)
aicir/backends/npu_backend.py   # 覆写：raise NotImplementedError

aicir/simulator/mps.py          # 新增：MPSState、zero_state、门作用、截断、公共入口
aicir/simulator/mps_contract.py # 新增：MPS 张量网络构建/收缩辅助（镜像 network.py 拆分）
aicir/simulator/__init__.py     # 扩展：re-export mps_statevector/mps_expectation
aicir/simulator/README.md       # 扩展：新增 MPS 方法说明，第 6 节移除“另立 Spec 2”措辞

aicir/primitives/mps_estimator.py  # 新增：MPSEstimator(BaseEstimator)
aicir/primitives/__init__.py       # 扩展：re-export MPSEstimator

tests/simulator/test_mps_*.py      # 新增：正确性 / 截断误差 / 可微 / 拒绝路径测试
tests/primitives/test_mps_estimator.py  # 新增
```

### 1. `Backend.svd()`

新增抽象方法，签名 `svd(self, matrix) -> tuple`，返回 reduced SVD `(U, S, Vh)`（`S` 为
一维奇异值向量，降序）：

| 后端 | 实现 |
| --- | --- |
| `NumpyBackend` | `np.linalg.svd(matrix, full_matrices=False)` |
| `GPUBackend` | `torch.linalg.svd(matrix, full_matrices=False)`（保留计算图） |
| `NPUBackend` | `raise NotImplementedError("MPS SVD 暂不支持 NPU；见 CLAUDE.md NPU complex64 限制")` |

### 2. `MPSState` 数据结构（`mps.py`）

```python
class MPSState:
    tensors: list        # 每比特一个 rank-3 张量，形状 (bond_left, 2, bond_right)；
                          # 边界 site 的外侧 bond=1
    n_qubits: int
    backend: Backend
    max_bond_dim: int | None   # None = 不设硬上限，仅按 cutoff 截断
    cutoff: float              # 相对奇异值截断阈值，默认 1e-10
    truncation_error: float    # 累计丢弃权重（sum of squared discarded singular values）
```

- **规范形式**：每次门作用后维持左规范（left-canonical），保证截断 SVD 数值条件良好，
  与标准 TEBD 两比特更新一致。
- `MPSState.zero_state(n, backend, *, max_bond_dim=None, cutoff=1e-10)`：构造全 `|0>`
  乘积态，所有 bond dim 为 1。
- `MPSState.to_statevector() -> State`：完整收缩回稠密 `(2^n, 1)` 态矢量。仅用于测试/
  小比特数消费，文档明确标注「对大 n 会失去 MPS 的意义，仅为验证/调试工具」。

### 3. 门作用

- **单比特门**：把门矩阵 `tensordot` 到该 site 张量的物理指标上，无需截断。
- **相邻双比特门 `(i, i+1)`**：把两个 site 张量收缩成一个双 site 张量，作用门，
  reshape 成矩阵后做 `backend.svd`，按 `min(max_bond_dim, count(S > cutoff * max(S)))`
  截断保留奇异值，拆回两个 site 张量（奇异值吸收进右侧张量以维持左规范）。这是唯一
  引入截断误差的步骤；每次调用把 `sum(discarded S ** 2)` 累加进
  `MPSState.truncation_error`。
- **非相邻双比特门 `(i, j)`**：插入 SWAP 链把 `j` 移到 `i` 相邻位置，作用门后**不换回**，
  把累计的逻辑↔物理 site 置换记录在 `MPSState.site_permutation`（做法与
  `RoutingPass.final_layout` 的“插入后不复位、向前携带置换”一致，但为 MPS 独立实现，
  不导入 `aicir.transpile`）。调用方可通过 `site_permutation` 在读出振幅/期望值时还原
  逻辑比特顺序。
- **3+ 比特门**：`apply_gate` 直接 `raise ValueError`，提示「MPS 引擎仅接受 1/2 比特门，
  请先用 aicir.transpile.DecomposePass 分解」。
- **`ControlFlow`**：与 `network.py` 一致，`raise ValueError` 提示走 `Measure.run` 的
  态矢量轨迹路径。

### 4. 公共 API（`mps.py`）

```python
def mps_statevector(circuit, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> MPSState
def mps_expectation(circuit, observable, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> complex
```

- 命名镜像精确引擎的 `tn_statevector`/`tn_expectation`。
- `mps_expectation` 优先走「有界支撑观测量」路径：`observable` 是 `PauliString`/
  `Hamiltonian` 且每项作用比特数有限时，直接在 MPS 上做局部收缩求期望，不做全量
  densification。若 `observable` 是任意稠密矩阵，回退到 `to_statevector()` 后
  `backend.expectation_sv`，并在文档中注明「仅对小比特数可行」。

### 5. `Measure.run(circuit, method="mps", ...)` 集成

与精确引擎的 `method="tensor"` 同构：接受 `max_bond_dim=`/`cutoff=` kwargs；沿用
`method="tensor"` 已有的限制——

- `snap` 仅接受 `None`/`[]`（无逐门快照语义）。
- 忽略调用方传入的 `initial_state`/`initial_density_matrix`，始终从 `|0...0>` 出发。
- 仅支持纯态、无噪声：`noise_model` 或内嵌 `measure(...)` 标记门 → `ValueError`。

### 6. `MPSEstimator`（`aicir/primitives/mps_estimator.py`）

```python
class MPSEstimator(BaseEstimator):
    def __init__(self, *, max_bond_dim=None, cutoff=1e-10, backend=None): ...
    def estimate(self, circuit, hamiltonian, **_ignored): ...  # BasicVQE(energy_estimator=...) 直通
    def run(self, circuits, observables, *, shots=None, parameter_values=None): ...
```

- 形状对齐 `StatevectorEstimator`：`shots is not None` 时 `raise ValueError`（MPS 精确
  收缩期望，不接受 shots；如需 shots 语义应叠加到 `Measure.run(method="mps", shots=...)`
  未来扩展，本 spec 不含）。
- `EstimateResult.metadata` 携带 `{"method": "mps", "max_bond_dim":..., "truncation_error":...}`，
  让调用方可检查本次估计的截断误差量级。

## 测试策略

- **精确匹配层**：小电路（≤8 比特），`max_bond_dim >= 2**(n//2)`（截断不可能发生）时，
  `mps_statevector` 结果需与 `tn_statevector`/`Circuit.unitary()` 精确态矢量在数值精度
  内一致。覆盖：单比特门、相邻双比特门、非相邻双比特门（验证 SWAP 网络 + 置换还原
  正确）、`mps_expectation` vs `StatevectorEstimator`。
- **截断误差层**：构造高纠缠电路（如多层随机双比特门 brickwork），用刻意偏小的
  `max_bond_dim`（如 2）运行，断言：(a) 结果与精确解的差异超过数值精度下限（证明截断
  确实发生，而非静默精确）；(b) 误差随 `max_bond_dim` 增大单调不增（小范围扫描
  `max_bond_dim ∈ {1,2,4,8}` 验证）；(c) `MPSState.truncation_error` 与实际保真度损失
  同数量级（`1 - fidelity` 与累计丢弃权重的关系在容差内吻合）。
- **后端覆盖**：Numpy 路径必须通过；GPUBackend（torch）测试用
  `pytest.importorskip("torch")` 门控，含 `MPSEstimator` 在 torch 参数张量上的
  autograd/梯度冒烟测试；NPUBackend 测试断言 `svd()` 抛 `NotImplementedError`。
- **拒绝路径**：3+ 比特门、`ControlFlow`、`noise_model`、非空 `snap`、
  `method="mps"` 下传入 `initial_state` 均需在测试中确认按设计抛出对应错误。

## 与其他子系统的关系

- `Measure`（`aicir/measure/`）：`method="mps"` 复用本模块 `mps_statevector`，其余
  聚合/读出逻辑与 `method="statevector"`/`"tensor"` 一致。
- `Backend`（`aicir/backends/`）：仅依赖新增 `svd()` 与既有张量原语；不感知具体后端
  实现细节。
- `aicir.transpile`：MPS 引擎的 SWAP 网络处理逻辑独立实现，不依赖 transpile；3+ 比特门
  要求调用方显式预先经 `DecomposePass` 处理。
- `aicir.vqc`：`MPSEstimator` 可经 `BasicVQE(energy_estimator=MPSEstimator(...))` 注入，
  与现有 `ShotEstimator`/`NoisyEstimator` 的加性集成方式一致；本 spec 不改动
  `BasicVQE`/`BasicQAOA` 默认路径。

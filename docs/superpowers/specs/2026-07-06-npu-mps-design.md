# NPU 支持 MPS 引擎设计（Spec 3）

日期：2026-07-06
状态：已批准设计，待实现

## 目标

让已有的 MPS 近似模拟引擎（`aicir/simulator/mps.py`，Spec 2 已落地：`MPSState`、
`mps_statevector`、`mps_expectation`、`MPSEstimator`、`Measure.run(method="mps")`）在真实
Ascend NPU 上**端到端可微**运行。当前 `NPUBackend.svd` 抛 `NotImplementedError`，是唯一
硬阻断；此外 `mps.py` 里有几处裸 torch 复数算子（`*`//`）在 NPU 上缺 complex64 内核。

v1 目标：single-card，全 autograd（能量 + 经 `MPSEstimator.gradient` 的梯度都在 NPU 上
可微），CPU/GPU 行为不变。

## 非目标

- **多卡 state sharding**：现有多卡为 task/data 并行（每 rank 全量 MPS），不在本 spec；
  与 CLAUDE.md 既有约束一致。
- **改动 MPS 算法/引擎结构**：复用 `mps.py`，只把裸复数算子改走 backend 方法；不改
  正交中心/SWAP/截断逻辑。
- **原生 complex SVD/QR**：Ascend 无此内核（已探明），一律走 real-embedding / real/imag
  分解。

## 背景与关键事实（真实硬件探针，2026-07-06）

见记忆 `npu-linalg-svd-capability`、`npu-complex64-unsupported`。要点：

- 实数 float32 `torch.linalg.svd` 前向+反传在 NPU 上**能跑**（梯度有限）。
- 原生 complex64 `torch.linalg.svd` **不存在**（`svd_npu only supported Float`）；complex
  `qr`/`matmul` 同样缺 complex64 内核。
- **real-embedding 复数 SVD**（`[[Re,-Im],[Im,Re]]` 2m×2n 实块）前向+反传**能跑**，梯度
  有限（探针 #4 直接验证）。
- NPU 广泛缺 complex64 内核（`aclnnMatmul`/`aclnnInplaceAdd`/`aclnnInplaceNormal`/逐元素
  `mul` 等）；所有复数运算必须经 `NPUBackend` real/imag 分解（如既有 `_NpuMatmulFn`）。
- NPU 初始化陷阱：`torch.npu.is_available()`/`get_device_name()` 作为**首个** NPU 调用会触发
  `AclSetCompileopt(ACL_PRECISION_MODE) 500001` 毒化整个进程；须先用平凡张量 op
  （`torch.ones(1).npu()`）预热。`demo_npu.py` 因经张量 op 初始化而正常。

`mps.py` 裸复数算子审计（须改走 backend 的位置）：

- 逐元素复数乘 `*`：`_move_center_right`（吸收奇异值 `vh * s`，行 103）、
  `_move_center_left`（`u * s`，行 118）、`_apply_two_site`（`vh_k * s_k`，行 153）、
  `mps_expectation` Pauli 项 `coef * _transfer(...)`（行 264）。
- 复数除 `/`：`mps_expectation` 的 `total / norm2`（行 267）。

其余算子已走 `backend.tensordot/conj/add/reshape/transpose/cast/real/expectation_sv/
inner_product` 或纯实数，NPU 上已安全。

## 架构

```
aicir/backends/base.py          # 扩展：新增抽象 mul / div（默认 raise NotImplemented，同 tensordot 风格）
aicir/backends/numpy_backend.py # 实现：mul= a*b, div= a/b
aicir/backends/gpu_backend.py   # 实现：mul= a*b, div= a/b（torch，autograd 天然）
aicir/backends/npu_backend.py   # 覆写：svd（real-embedding + 复数重建）、mul/div（real/imag 分解，可微）

aicir/simulator/mps.py          # 把 5 处裸 * / / 改走 backend.mul / backend.div（CPU/GPU 行为不变）

tests/backends/test_npu_svd.py      # NPUBackend.svd 重建 + 梯度 vs CPU（is_npu_available 门控）
tests/backends/test_npu_muldiv.py   # mul/div vs CPU（门控）
tests/simulator/test_mps_npu.py     # mps_* / MPSEstimator NPU-vs-CPU 平价 + autograd 契约（门控）
demos/demo_npu_mps.py               # 严格/--allow-cpu-fallback 真机验收（对齐 demo_npu_tensor.py）
```

### 1. `NPUBackend.svd`（real-embedding + 复数重建）

给定复数 `A`（m×n）：

1. 取 `Ar=real(A)`、`Ai=imag(A)`（NPU real/imag，实数张量）。
2. 组装实块 `R = cat([cat([Ar,-Ai],1), cat([Ai,Ar],1)], 0)`（2m×2n 实 float32）。
3. `Ur, Sr, VhR = torch.linalg.svd(R, full_matrices=False)`（实数，NPU 原生，可反传）。
4. **重建复数 `(U, S, Vh)`**：`R` 的奇异值为 `A` 每个 σ 出现两次；取去重后的前
   `min(m,n)` 个作为 `S`（实数）。复数左向量 `U[:,j] = Ur[:m, p(j)] + i·Ur[m:, p(j)]`，
   右向量类似从 `VhR` 重建（`p(j)` 为配对索引）。返回 `(U, S, Vh)`，与 numpy/gpu 契约一致
   （约化 SVD，S 一维降序实数，`A ≈ U @ diag(S) @ Vh`）。
5. 全程 `cat`/`svd`/`real/imag`/`torch.complex` 均为实数 op 或可微复数构造，autograd 从
   `Ur/Sr/VhR` 反传回 `A`（探针 #4 验证 `cat`+实 `svd` 反传可行）。

**技术风险集中于第 4 步**（配对 + 符号/规范/简并）。简并奇异值下 SVD 反传本就不稳定——
这是所有后端（含 GPU/原生复数 SVD）共有的 SVD-autograd 固有性质，本设计**继承而非加重**。
实现计划第一个任务即为「重建正确性 + 梯度正确性 对齐 CPU `torch.linalg.svd`」的硬门；
若简并配对被证明不稳，回退契约为「非简并输入」（与既有 SVD-autograd 契约一致）。

### 2. `Backend.mul` / `Backend.div` 原语与 `mps.py` NPU 安全化

新增两个 `Backend` 方法（仿既有 `add` 先例）：

| 方法 | numpy | gpu | npu（real/imag 分解，可微） |
| --- | --- | --- | --- |
| `mul(a, b)` | `a * b` | `a * b` | `(ar·br − ai·bi) + i(ar·bi + ai·br)` |
| `div(a, b)` | `a / b` | `a / b` | `mul(a, conj(b)) / |b|²`（分母实数） |

`mps.py` 把审计出的 5 处裸算子改为 `bk.mul(...)`/`bk.div(...)`。numpy/gpu 下 `mul/div`
就是 `*`//`，行为逐位不变；仅 NPU 走 real/imag。`mps.py` 保持后端无关，不引入 NPU 分支。

### 3. Autograd 正确性契约

NPU MPS 路径每一步都可微，使 `MPSEstimator.gradient`（及任意 autograd VQE）在 NPU 上成立：
`svd`（real-embedding，任务 1 保证）、`mul`/`div`（real 分解）、其余
（`tensordot`/`conj`/`add`/`reshape`/`transpose`/切片）已 autograd 安全。

契约测试：`npu:0` 上构造带 torch 张量参数的小参数化电路，跑 `mps_expectation` →
`.backward()`，断言梯度与 CPU MPS（同电路，`NumpyBackend`）在容差内一致（同 GPU MPS 的
跨后端核对）。简并 SVD-autograd 注意事项文档化。

### 4. 测试与真机验收（strict-NPU 门控）

- `tests/backends/test_npu_svd.py`、`test_npu_muldiv.py`、`tests/simulator/test_mps_npu.py`：
  均 `is_npu_available()` 门控（CPU/CI 干净 skip）。覆盖 svd 重建+梯度 vs CPU、mul/div vs
  CPU、`mps_statevector`/`mps_expectation`/`MPSEstimator` NPU-vs-CPU 平价（full-bond 与截断）、
  autograd 契约。
- `demos/demo_npu_mps.py`：严格 / `--allow-cpu-fallback` 运行器，对齐 `demo_npu_tensor.py`；
  暴露 `run_checks()` 供 CPU 冒烟（导入期测试）。
- 真机签核走既有 strict 路径（`scripts/npu/…`）；**未经真实 Ascend 通过前不宣称正确**。
  所有 NPU 入口先用平凡张量 op 预热，规避 `ACL_PRECISION_MODE 500001` 初始化毒化。

## 与其他子系统的关系

- `aicir/simulator/mps.py`：唯一改动是 5 处裸复数算子改走 `backend.mul/div`；引擎逻辑不变，
  CPU/GPU 结果不变。
- `aicir/backends`：新增 `mul/div` 抽象 + 三后端实现；`NPUBackend.svd` 从
  `NotImplementedError` 变为 real-embedding 实现。CLAUDE.md 的「NPU 复数走 NPUBackend 方法/
  自定义 autograd」契约得到延续。
- `MPSEstimator`/`Measure.run(method="mps")`：无需改动，随底层 NPU 化自动获得 NPU 能力。
- 记忆 `npu-linalg-svd-capability` / `npu-complex64-unsupported`：本 spec 落地后，前者
  「real-embedding 未建」的表述应更新为已建。

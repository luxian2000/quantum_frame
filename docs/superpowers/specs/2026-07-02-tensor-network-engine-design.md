# 精确张量网络模拟引擎设计（Spec 1）

日期：2026-07-02
状态：已批准设计，待实现

## 目标

在 aicir 中新增一套**精确张量网络（Tensor Network, TN）模拟引擎**，独立实现（不移植
WuYueSDK），交付 `compare.md` 中标注 aicir 缺失的三种模拟方法：

- **张量网络模拟**：把电路收缩成张量网络求整段/部分振幅，突破全态矢量的内存墙。
- **单振幅模拟**：只算某个基态振幅 `⟨x|U|0⟩`（两端边界都固定 → 标量收缩）。
- **部分振幅模拟**：只算感兴趣的振幅子集（输入端固定 `|0⟩`，部分输出比特开放或枚举）。

引擎围绕 aicir 的 `Backend` 抽象构建，**真正在 NPUBackend 上执行**（收缩走后端算子，
含 Ascend real/imag 分解），并**保留 torch 自动微分图**（GPU/NPU 后端上期望值可微）。

## 非目标（本 spec 不做）

- **近似模拟（bond 截断 / MPS）**：属独立的第二个引擎，另立 Spec 2（需 on-device SVD +
  截断，数据结构与算法都不同）。
- **含噪声 / 密度矩阵路径**：TN 引擎只处理纯态；噪声仍走既有密度矩阵模拟。
- **重写既有态矢量引擎**：TN 是并列的可选模拟方法，既有 `State`/`Measure` 语义不变。

## 背景与关键事实

- 门是 dict，既有两条数值路径：`gate_to_matrix(gate, ...)`（全/局部 2^k 酉矩阵）与
  `_gate_local_matrix(gate, gate_type, backend) -> (local_matrix, axes, cache_key)`
  （`axes` 即门作用的量子比特）。这正是 TN 每个节点需要的「张量 + 连线比特」。
- `Backend`（`aicir/backends/base.py`）现有算子里**没有** `tensordot/transpose/reshape/conj`。
- NPUBackend 已有**自动微分安全的复数 `matmul`**（`_NpuMatmulFn`）与 `dagger`（real/imag 分解）。
- 约定：态矢量 shape `(2^n,1)`、比特序默认 msb、比特索引与既有引擎一致——必须对齐。

### 去风险核心洞见

任意两张量的成对收缩 = `transpose` + `reshape` + **`matmul`** + `reshape`。因此收缩器
可完全建立在 `backend.matmul` + 纯索引算子之上，**无需为任意 einsum 方程手写新的自动微分
Function**：NPU 复用既有 autograd-safe `matmul`，`transpose/reshape` 是纯索引操作（Ascend
上无复数运算隐患）。这让「on-device + 可微」从研究课题变成可控工程。

## 架构

引擎独立成包 `aicir/simulator/`，选择方式经 `Measure`/`State` 暴露。

```
aicir/backends/base.py         # 扩展：新增收缩原语（tensordot/transpose/reshape/conj）
aicir/backends/numpy_backend.py# 实现（np.*）
aicir/backends/gpu_backend.py  # 实现（torch.*）
aicir/backends/npu_backend.py  # 覆写 tensordot（走 autograd-safe matmul）、conj（real/imag）

aicir/simulator/__init__.py    # 新增：公共入口 + 顶层再导出
aicir/simulator/network.py     # 新增：Circuit -> (tensors, index_labels, output_labels)
aicir/simulator/contract.py    # 新增：路径选择 + 成对收缩执行

demos/demo_npu_tensor.py       # 新增：可远程执行的 NPU 测试脚本（见「NPU 测试脚本」）
tests/simulator/test_*.py      # 新增：正确性 / 可微 / 回退 测试
```

### 1. Backend 收缩原语

在 `Backend` 上新增四个方法（本仓库仅 3 个后端，采用「base 提供清晰 NotImplemented，
三后端各自实现」以免破坏外部子类）：

| 方法 | 语义 | numpy | gpu | npu |
| --- | --- | --- | --- | --- |
| `tensordot(a, b, axes)` | 沿 `axes` 收缩两张量 | `np.tensordot` | `torch.tensordot` | 覆写：permute→reshape→`self.matmul`→reshape（复用 autograd-safe 复数 matmul） |
| `transpose(a, axes)` | 轴置换 | `np.transpose` | `torch.permute` | 继承 gpu（纯索引，安全） |
| `reshape(a, shape)` | 变形 | `np.reshape` | `torch.reshape` | 继承 gpu（纯索引，安全） |
| `conj(a)` | 复共轭 | `np.conj` | `torch.conj` | 覆写：`complex(real(a), -imag(a))`（避开 complex64 conj 缺失） |

`tensordot` 是收缩器唯一的数值主算子；其余三个用于边界张量准备与 bra 侧共轭。

### 2. TN 构建器（`network.py`）

把 `Circuit` 转成一组带标签的张量：

- 为每个门取张量 + 作用比特：优先 `_gate_local_matrix`；返回 None 的门回退到
  `gate_to_matrix` 作用于该门比特集，reshape 成 rank-`2k` 张量（k 个 out 索引 + k 个 in 索引）。
  门比特集由 IR 访问器（`instruction_qubits`）+ 受控比特汇集。
- 逐门推进「每比特当前连线标签」，把门的 in 索引接到该比特的上一条线、out 索引开一条新线。
- **输入边界**：每比特接 `|0⟩` 向量（或固定输入比特）。
- **输出边界**：
  - 全/部分开放 → 对应比特留开放索引；
  - 单振幅 / 部分固定 → 对该比特接固定 bra 向量（`⟨0|`/`⟨1|`）。
- 产出 `(tensors, index_labels, output_labels)` 交给收缩器；参数门保留 torch 图（可微来源）。

### 3. 收缩器（`contract.py`）

- **路径**：装了 `opt_einsum` 用 `opt_einsum.contract_path`（近最优）；否则用内置**小型
  贪心**回退（反复选「结果最小 / 共享索引最多」的一对收缩）。路径仅在**主机**依据张量
  shape 计算一次，不搬运数据。
- **执行**：按路径逐对调用 `backend.tensordot`，中间张量始终是后端原生张量（保留 autograd 图）。
- 返回后端原生张量，shape 为 `2^(#开放输出比特)`（单振幅为标量）。

### 4. 公共入口与集成

- `Measure.run(circuit, method="tensor")`：经 TN 收缩得全态矢量，再喂既有概率/期望机制
  （既有默认 `method="statevector"` 不变，纯态、无噪声路径下二者结果一致）。
- 新模块函数（`aicir/simulator/`，顶层再导出）：
  - `single_amplitude(circuit, bitstring, *, backend=None) -> 复标量`（不建 2^n 态矢量）。
  - `partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None)`
    （二选一：`open_qubits` 留开放子集返回 `2^len` 向量；`bitstrings` 枚举给定基态返回对应振幅）。
  - `tn_statevector(circuit, *, backend=None)`：TN 全态矢量（供 `Measure` 复用与直接调用）。
- **可微期望**：`tn_expectation(circuit, observable, *, backend=None)` 把 `⟨ψ|O|ψ⟩` 作为
  网络收缩（bra 侧用 `conj`），在 torch/NPU 后端上对参数门可微。

> API 归属说明：模拟**方法选择**经 `Measure`/`State` 的 `method=`；而单/部分振幅是
> **Circuit 级查询**（其价值在于不构造全态矢量），故作为独立函数存在并顶层再导出。

## 数据流

```
Circuit → network.build(边界) → (tensors, labels, open) → contract.path(主机)
       → backend.tensordot 链 → 结果（标量 / 部分向量 / 全态矢量）
```

## 错误处理与边界

- opt_einsum 缺失 → 贪心回退（记录一次 info 级说明，不报错）。
- 收缩宽度过大（开放/中间索引维度爆炸）→ 在路径估算阶段按可配置阈值给出清晰错误
  （提示改用更少开放比特 / 单振幅 / 未来的 MPS）。
- `method="tensor"` 遇噪声电路 → 明确报错（TN 仅纯态）。
- `single_amplitude` 的 `bitstring` 长度须等于 `n_qubits`；`partial_amplitude` 的
  `open_qubits`/`bitstrings` 二者必须且只能提供其一。
- 未绑定参数的模板电路 → 复用既有校验报错。

## 测试策略

- **正确性对齐既有引擎**：随机小电路（含单/双比特、受控、`rzz/rxx`、`u3`）下
  - TN 全态矢量 ≈ `State`/`Measure` 态矢量；
  - `single_amplitude` ≈ 对应态矢量分量；
  - `partial_amplitude`（开放子集 / 枚举）≈ 对应分量；
  - numpy 与 gpu 后端都测；npu 无硬件时 `importorskip`/能力探测自动跳过。
- **可微**：小参数电路上 `tn_expectation` 的自动微分梯度 ≈ `aicir.qml.psr`。
- **回退**：opt_einsum 在场与强制走贪心两条路径都覆盖，结果一致。
- **约定**：比特序（msb）、比特索引与既有引擎一致的显式断言。

## NPU 测试脚本（交付物）

实现后交付一个**可在远程 Ascend 环境独立运行**的脚本 `demos/demo_npu_tensor.py`，
`python demos/demo_npu_tensor.py`（支持 `--allow-cpu-fallback`）覆盖：

1. NPU 上 TN 全态矢量 vs NPU 态矢量引擎的数值一致性；
2. NPU 上 `single_amplitude` / `partial_amplitude` 正确性；
3. NPU 上 `tn_expectation` 的可微性（对参数门反传得到 `.grad`）；
4. 打印后端 `name`、设备、是否走 real/imag 分解路径，便于远程核对。

脚本严格检查在 NPU 上运行（默认；`--allow-cpu-fallback` 供无卡开发），退出码非零表示失败。

## 影响与兼容性

- 纯新增：新包 `aicir/simulator/` + `Backend` 四个新原语 + `Measure.run` 的 `method=` 分支。
- 既有 `State`/`Measure`/`Circuit` 默认行为与签名不变。
- `opt_einsum` 为**可选**依赖（numpy 仍是唯一硬依赖）。
- 更新：`aicir/simulator/README.md`、`CHANGELOG.md`（2026-07-02 条目）、顶层再导出。

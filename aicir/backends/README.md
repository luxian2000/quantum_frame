# aicir 计算后端使用手册

aicir 提供三种可互换的计算后端，都实现统一的 `Backend` 抽象接口，可与 `Circuit` / `Measure` / `State` / `Hamiltonian` 无缝配合。切换后端只需替换一行构造代码，业务逻辑无需改动。

---

## 目录

1. [后端总览](#1--后端总览)
2. [快速上手](#2--快速上手)
3. [NumpyBackend — CPU 参考实现](#3--numpybackend)
4. [GPUBackend — PyTorch CPU / CUDA](#4--gpubackend)
5. [NPUBackend — Ascend NPU](#5--npubackend)
6. [后端绑定与优先级](#6--后端绑定与优先级)
7. [执行策略：逐门演化 vs 全矩阵](#7--执行策略)
8. [电路拼接时的后端继承](#8--电路拼接时的后端继承)
9. [BatchSV — 批量态矢量路径](#9--batchsv)
10. [Backend 抽象接口参考](#10--backend-抽象接口参考)
11. [NPU complex64 兼容性详解](#11--npu-complex64-兼容性详解)
12. [硬件能力探测（npu_probe）](#53--硬件能力探测npu_probe)

---

## 1  后端总览

| 后端 | 底层库 | 运行设备 | 自动微分 | 典型用途 |
| --- | --- | --- | :---: | --- |
| `NumpyBackend` | NumPy | CPU | ✗ | 小规模验证、算法原型、无 PyTorch 环境 |
| `GPUBackend` | PyTorch | CPU / CUDA GPU | ✔ | 变分算法（VQE / QAOA / QML）、GPU 加速 |
| `NPUBackend` | PyTorch + `torch_npu` | Ascend NPU（可回退 CPU） | ✔ | 昇腾 NPU 上的仿真与训练 |

> **命名说明**：`TorchBackend` 是 `GPUBackend` 的**过时别名**，仅为向后兼容保留，新代码请使用 `GPUBackend`。`NPUBackend` 继承自 `GPUBackend`，复用其全部数学内核。

---

## 2  快速上手

三种后端的调用范式完全一致：

```python
from aicir import Circuit, Measure, NumpyBackend, GPUBackend, NPUBackend
from aicir import hadamard, cnot

# 构造后端（三选一，API 完全等价）
backend = GPUBackend(device="cpu")
# backend = NumpyBackend()
# backend = NPUBackend()

# 构建电路并绑定后端
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)

# 测量
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name, result.counts)
```

---

## 3  NumpyBackend

纯 CPU 参考实现，依赖最少。

```python
from aicir import NumpyBackend

backend = NumpyBackend()                       # 默认 complex64
backend = NumpyBackend(dtype="complex128")     # 可选：指定精度
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `dtype` | `np.complex64` | 复数数据类型 |

**适用场景**：教学、单元测试、小比特数验证。

**限制**：不支持自动微分。如需梯度训练，请使用 `GPUBackend`，或配合参数移位规则（`psr`）等数值方法。

---

## 4  GPUBackend

基于 PyTorch，支持 CPU / CUDA GPU 加速与 autograd 自动微分。

```python
import torch
from aicir import GPUBackend

backend = GPUBackend()                          # 有 CUDA 用 cuda，否则 cpu
backend = GPUBackend(device="cpu")              # 强制 CPU
backend = GPUBackend(device="cuda:0")           # 指定某张 GPU
backend = GPUBackend(device=torch.device("cuda"), dtype=torch.complex128)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `device` | 有 CUDA 用 `cuda`，否则 `cpu` | 接收字符串或 `torch.device` |
| `dtype` | `torch.complex64` | 复数精度，可设 `torch.complex128` |

### 自动微分支持

把 Torch 标量张量作为门参数（`rx` / `ry` / `rz` / `u2` / `u3`、受控旋转、`rzz` / `rxx`、自定义 `unitary`）即可保留计算图，用于 VQE / QAOA / QML 训练。

---

## 5  NPUBackend

面向昇腾（Ascend）NPU 设备，继承自 `GPUBackend`，API 完全一致。

### 5.1  基本构造

```python
from aicir import NPUBackend

backend = NPUBackend()                                        # 自动选 npu:0，不可用则回退 CPU
backend = NPUBackend(device="npu:1")                          # 指定某张 NPU 卡
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)   # 严格模式：不可用直接报错
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `device` | 自动 `npu:0` | 目标 NPU 设备 |
| `dtype` | `torch.complex64` | 复数精度 |
| `fallback_to_cpu` | `True` | NPU 不可用时是否回退 CPU；`False` 则抛 `RuntimeError` |

### 5.2  分布式环境（多卡 / 多节点）

> **并行粒度：任务/数据并行，非态/门分片。** 多 NPU（`torchrun --nproc_per_node=N`）下，
> 各 rank 按 `should_run_batch_index(i)` / `aicir.qas.core.sharding.owned_indices` 领取
> **批次项（整条电路 / 候选）** 的子集：每个 NPU 独立运行其分到的**整条电路**——单条电路的
> 门矩阵 `2^n×2^n` 与态向量 `2^n` 始终完整驻留在一个 NPU 上，不跨设备拆分；结束后经
> `gather_indexed_results` / `all_gather` / `all_reduce_mean` 归并。原因：Ascend NPU 无复数
> dtype 算子，集合通信仅搬运实数张量与 Python 对象。把单条电路跨 NPU 拆分（态向量分片或
> 门块分布）需另设计实/虚部拆分 + 新集合通信，当前未实现。

使用环境变量 `WORLD_SIZE` / `RANK` / `LOCAL_RANK` 自动绑定对应卡：

```python
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
print(backend.runtime_context)
# NPURuntimeContext(world_size=4, rank=0, local_rank=0, distributed=True)
```

`from_distributed_env` 额外参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `init_process_group` | `True` | 自动调用 `torch.distributed.init_process_group` |
| `process_group_backend` | `None`（NPU 用 `hccl`，CPU 用 `gloo`） | 通信后端 |

#### runtime_context 字段

| 字段 | 说明 |
| --- | --- |
| `world_size` | 总进程数 |
| `rank` | 全局进程编号 |
| `local_rank` | 本节点本地编号（对应 `npu:local_rank`） |
| `distributed` | `world_size > 1` 时为 `True` |
| `process_group_initialized` | 分布式进程组是否已初始化 |
| `process_group_backend` | 通信后端名称 |

#### 分布式任务分发

```python
# 判断当前进程是否负责某个批次项
if backend.should_run_batch_index(i):
    result = run_circuit(i)
    local_results.append((i, result))

# 跨进程汇总
all_results = backend.gather_indexed_results(local_results)
```

#### 启动方式

```bash
# 单卡验证
python demos/demo_npu.py

# 允许 CPU 回退
python demos/demo_npu.py --allow-cpu-fallback

# 多卡分布式
torchrun --nproc_per_node=4 your_script.py
```

### 5.3  硬件能力探测（npu_probe）

> **功能：** 探测 Ascend NPU 运行时硬件能力（dtype、算子、张量维度上限、内存），缓存结果以供后续脚本复用，并可映射为 `Target` 执行标志。

#### 模块入口

```python
from aicir.backends.npu_probe import probe_npu, NpuCapabilities, target_from_npu

# 探测能力（默认无缓存刷新）
caps = probe_npu(allow_cpu_fallback=False)  # NPU 不可用时报错
# caps = probe_npu(allow_cpu_fallback=True)  # 允许回退到 CPU 探测
# caps = probe_npu(refresh=True)  # 忽略缓存，强制重探

# 能力查询
print(f"device:               {caps.device}")
print(f"available:            {caps.available}")
print(f"torch / torch_npu:    {caps.torch_version} / {caps.torch_npu_version}")
print(f"complex64 support:    matmul={caps.supports_complex_matmul}, conj={caps.supports_complex_conj}, add={caps.supports_complex_add}")
print(f"max_ndim:             {caps.max_ndim}")
print(f"max_qubits (单卡):    {caps.max_qubits}")
print(f"max_qubits (分片):    {caps.max_qubits_sharded}  (world_size={caps.world_size})")
print(f"total_memory:         {caps.total_memory} bytes")

# 探测结果序列化
caps.to_dict()          # → 可 JSON 化字典
NpuCapabilities.from_dict(...)  # ← 反序列化

# 映射为 Target（电路标志）
target = target_from_npu(caps)                       # n_qubits 缺省用 caps.max_qubits
target = target_from_npu(caps, n_qubits=10)          # 显式指定 n_qubits
print(f"Target: {target}")
```

#### 缓存管理

缓存位置可通过 `AICIR_CACHE_DIR` 环境变量覆盖（默认 `~/.cache/aicir/`）：

```bash
# 使用 ~/.cache/aicir/npu_caps.json（默认）
python demos/demo_npu_probe.py

# 使用 /tmp/my_cache/npu_caps.json
AICIR_CACHE_DIR=/tmp/my_cache python demos/demo_npu_probe.py

# 忽略缓存，强制重探
python demos/demo_npu_probe.py --refresh
```

缓存失效键为 `device | torch_version | torch_npu_version`；不同版本或设备自动独立缓存。

#### 探测脚本

```bash
# 严格 NPU（不可用则报错）
python demos/demo_npu_probe.py

# 允许 CPU 回退
python demos/demo_npu_probe.py --allow-cpu-fallback

# 忽略缓存重探
python demos/demo_npu_probe.py --refresh
```

脚本输出能力表与派生的 `Target` 对象。

---

### 5.4  QAS supernet 集成

在 QAS supernet（`aicir/qas`）中无需手动构造后端：把 `device="npu:0"` 传入配置，框架会自动选用 `NPUBackend`；CPU / CUDA 设备则用 `GPUBackend`。

完整端到端示例：

```python
import math
from aicir import NPUBackend, Circuit, Measure, hadamard, cnot, rz

# 1. 构建后端
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

# 2. 构建电路
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

# 3. 测量
measure = Measure(backend)
result = measure.run(cir, shots=2048)

print(f"backend : {backend.name}")
print(f"probs   : {result.probabilities}")
print(f"counts  : {result.counts}")
print(f"summary : {result.summary()}")
```

### 5.5  BeH2 16 比特 supernet QAS 的 OOM/SIGKILL 排查与修复

`torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py`（16 比特、1313 Pauli 项）曾在进入
`start sharded supernet search` 后约 12 分钟被 `SIGKILL`（`exitcode -9`）终止；Ascend TBE 后台线程
随后输出的 `EOFError`/`ConnectionResetError` 是子进程被杀后的连带现象，非首要异常。

**定位**：`dmesg` 显示 **cgroup（容器）内存超限**而非主机 OOM（主机 2 TB、空闲 1.9 TB）；
单个 rank 的主进程涨到 **~238 GB RSS**——对 16 比特（态向量仅 1 MB）是病态的内存爆炸，必为泄漏/枚举。
`tracemalloc` 指向 `supernet.py` 的布局枚举/参数预建行（数千万次分配且持续增长）。

**根因**：`Supernet` 构造时枚举整个单比特布局空间
`product(single_qubit_gates, repeat=n_qubits)` = `gates ** n_qubits`（BeH2 即 `5 ** 16 ≈ 1.5e11`），
并为每个布局预建共享参数。只在测试用的极小 `n_qubits` 下可行，16 比特撑爆内存。

**修复**（懒采样 + 懒共享参数，见 `aicir/qas/algorithms/supernet.py` 顶部内存说明）：

- `sample_architecture` 改为**采样下标 + 解码**，与旧 `choice(枚举列表)` **字节等价**（rng 序列不变，
  golden 测试不受影响），但不物化布局空间。
- 共享参数**首次访问懒建**；每 supernet 优化器懒建并经 `add_param_group` 增长。
- 每个被访问架构的参数在**所有** supernet 上创建（仅评估分片、参数创建不分片），保持 `safe`/
  `aggressive` 分片下各 rank 的共享参数键集一致（梯度 all-reduce / `broadcast_parameters` 依赖此不变量）。

共享参数内存降为 `O(supernet_steps × layers × n_qubits)`。配套 `tests/test_supernet_lazy_layouts.py`
（含 16 比特秒级构造回归）。

> 注：该修复只解决 supernet 的内存爆炸（任务/数据并行，整态仍驻单卡）。把单条电路的态向量跨 NPU
> 拆分以突破单卡比特上限是另一项工作，见 §5.6。

---

### 5.6  态向量分片（Roadmap，未实现）

当前多 NPU 为**任务/数据并行**（§5.2）：整条电路的态向量始终完整驻留单卡。把**单条电路的态向量跨 NPU 拆分**以突破单卡比特上限，是一个已讨论、暂缓的大项目；本节记录结论与设计选项，待日后推进。

#### 目标（已定）

突破**单卡比特上限**：多 NPU 池化内存，跑比单卡更大的 `n`。complex64 态向量 `2^n × 8` 字节，64 GB 卡约到 `n=32`（见 `probe_npu` 的 `max_qubits`）；`n=33` 单卡放不下。P 卡分片 → `+log2(P)` 比特（4 卡 +2、8 卡 +3），即 `max_qubits_sharded`。

**有意不追求「同 n 提速」**：态向量仿真受内存带宽限制，global-qubit 门每次需跨卡搬大块振幅，通信通常吃掉并行收益，往往净变慢。同 `n` 提速应继续用现有任务/数据并行，而非态向量分片。

#### Ascend 硬约束

Ascend **无复数集合通信**（现有 `qas/core/sharding.py` 只搬实数张量），且 complex64 连原生 `add` 都没有。故跨卡交换复数振幅须**拆实/虚部做 2 次实数集合通信**（hccl）。比 GPU 上的态向量分片明显更难。

#### 拆解（建议顺序 A→E）

- **A 分片态模型与分区**：`2^n` 振幅跨 P rank 切。顶 `k=log2(P)` 比特为 **global**（其 bit 选 owning rank），其余 `n−k` 为 **local**；每 rank 持 `2^(n-k)` 振幅的连续块。含下标映射、分片 `|0…0>` 初始化。基础。
- **B local-qubit 门作用**：目标为 local 比特 → 各 rank 块内用现有逐轴核 `_apply_local_matrix_to_state` 直接作用，**无通信**。多为复用。
- **C global-qubit 门作用（难点）**：目标为 global 比特 → 配对振幅在不同 rank。推荐**分布式转置**（一次 `all_to_all` 重排）把 global 比特换进 local 槽、再本地作用，把通信集中到单一原语而非逐门两两交换；Ascend 上该重排须拆实/虚部。
- **D 分片执行器**：驱动 `Circuit` 跑分片态——每门路由到 B/C，处理双比特门，必要时触发转置。
- **E 分片态测量/期望**：概率、采样、`<H>` 经跨 rank `all_reduce` 归约；容量经 `max_qubits_sharded` 预检。

A+B 可**单进程测试**（一个进程内排布 P 个块、对分片算法逐比特对照单卡稠密参考），无需真 NPU；C 是风险/核心。

#### 开发可测性（关键）

设计须让**分片算法单进程可测**（块下标数学 + 本地集合通信 shim），真实 **hccl** 集合调用留作薄层、仅在多 NPU 机上验证；否则每次迭代都要多卡硬件。

> 状态：暂缓，非当前优先。日后推进时本节作为起点，按 A→E 各自走 spec → plan → 实现。

---

## 6  后端绑定与优先级

后端可在两处指定：`Circuit` 构造时或 `Measure` 构造时。

```python
# 方式 A：构建时绑定（推荐）
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)

# 方式 B：先构建再绑定
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
cir.bind_backend(backend)

# 方式 C：只在 Measure 指定
result = Measure(backend).run(cir, shots=1024)
```

**优先级规则**：`Measure` 优先使用 `circuit.backend`（若存在），否则使用 `Measure` 自身的后端。

**推荐绑定到 `Circuit` 的原因**：

- 当电路具有 `gates` 序列时，`Measure` 会逐门调用 `gate_to_matrix(..., backend=resolved_backend)` 在目标设备上构造并作用门矩阵，减少主机 ↔ 设备数据搬运。
- 若 `Circuit` 未绑定后端，可能回退到 NumPy 在 CPU 上拼装整条电路的全局矩阵，再 `backend.cast` 到设备，引起大规模数据迁移。
- 对于 `GPUBackend`，参数化门的 Torch 标量张量会通过 Torch 运算构造矩阵，从而保留 autograd 计算图。

---

## 7  执行策略

理解 aicir 的执行路径有助于选择最优配置：

| 阶段 | 行为 |
| --- | --- |
| **构建阶段** | `hadamard(0)` 等生成门描述字典（如 `{"type": "hadamard", "target_qubit": 0}`），`Circuit.__init__` 只存储描述，不生成数值矩阵 |
| **执行阶段（逐门路径）** | `Measure.run` / `run_density_matrix` 检测到 `gates` 序列时，按门依次作用到态 / 密度矩阵（**推荐**，内存友好） |
| **执行阶段（全矩阵路径）** | 若电路不提供 `gates` 序列，回退到 `Circuit.unitary()` 拼装完整 2ⁿ×2ⁿ 矩阵后一次性作用（兼容外部实现） |

**性能建议**：对大比特数电路，优先绑定后端走逐门路径，避免组装完整矩阵造成的内存开销和设备迁移。

---

## 8  电路拼接时的后端继承

使用 `+` 拼接两个 `Circuit` 时，新电路的后端优先采用**左侧**电路的后端；若左侧没有，则采用右侧的。

```python
# 推荐做法：拼接后显式统一后端
full = part_a + part_b
full.bind_backend(common_backend)
result = Measure(common_backend).run(full)
```

如果不显式统一，可能导致运行时后端不一致或意外回退。

---

## 9  BatchSV

`BatchSV`（`aicir.core.batch`，也可从顶层 `aicir` 导入）是面向深度学习场景的批量态矢量路径，适用于将变分量子线路作为神经网络层使用。

### 9.1  设计目标

| 特性 | 说明 |
| --- | --- |
| **批量模拟** | 一次演化一批 `(batch, 2ⁿ)` 态矢量 |
| **逐样本参数** | 旋转门角度可为 `(batch,)` 张量，逐样本不同 |
| **NPU 安全** | 全程以实部 / 虚部两个实张量表示，只用实数乘加，反向传播不触发复数累加 |
| **端到端可微** | 支持 PyTorch autograd |

### 9.2  代码示例

```python
import torch
from aicir import BatchSV, GPUBackend, hadamard, ry, crz

backend = GPUBackend()                      # 或 NPUBackend
bsv = BatchSV(n_qubits=3, batch_size=8, backend=backend)

# 逐样本数据编码角度
enc = torch.randn(8, 3, requires_grad=True)
theta = torch.zeros(3, requires_grad=True)  # 标量参数按 batch 广播

for q in range(3):
    bsv.apply_gate(hadamard(q))
    bsv.apply_gate(ry(enc[:, q], q))        # 逐样本角度
bsv.apply_gate(crz(theta[0], 1, [0]))       # 受控旋转门
bsv.apply_gate(ry(theta[1], 2))

z = bsv.z_expectations()                     # (batch, n_qubits) 逐比特 ⟨Z_q⟩
probs = bsv.probabilities()                  # (batch, 2^n)

loss = z.sum()
loss.backward()                              # 全程实张量，NPU 安全
```

### 9.3  API 参考

**构造参数**

| 参数 | 说明 |
| --- | --- |
| `n_qubits` | 量子比特数（≥ 1） |
| `batch_size` | 批大小（≥ 1） |
| `backend` | aicir 后端（用于获取 device / dtype） |
| `device`（可选） | 覆盖后端 device |
| `real_dtype`（可选） | 覆盖实张量 dtype（默认由后端复数 dtype 推断） |

**方法**

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `apply_gate(gate)` | `self` | 就地作用一个门，支持链式调用 |
| `probabilities()` | `(batch, 2ⁿ)` 实张量 | 计算基测量概率 |
| `z_expectations()` | `(batch, n_qubits)` 实张量 | 逐比特泡利 Z 期望值 ⟨Z_q⟩ |

**逐样本参数门支持**：`rx` / `ry` / `rz` 及其受控形式 `crx` / `cry` / `crz`。常量门与标量参数门复用单态路径定义，覆盖范围与之一致。

**量子比特端序**：与 aicir 主路径一致，qubit 0 为最高位。

---

## 10  Backend 抽象接口参考

所有后端均实现以下抽象方法（定义在 `base.py` 的 `Backend` 类）：

### 元信息

| 属性 / 方法 | 说明 |
| --- | --- |
| `name` (property) | 后端唯一名称标识符 |

### 张量工厂

| 方法 | 说明 |
| --- | --- |
| `zeros(shape, dtype=None)` | 创建全零张量 |
| `eye(dim)` | 创建 dim × dim 复数单位矩阵 |
| `cast(array, dtype=None)` | 将 numpy array / list / 标量转换为后端张量 |
| `to_numpy(tensor)` | 将后端张量转换为 numpy array |

### 量子态初始化

| 方法 | 说明 |
| --- | --- |
| `zeros_state(n_qubits)` | 创建 \|0⊗n⟩ 基态列向量，shape `(2ⁿ, 1)` |

### 线性代数

| 方法 | 说明 |
| --- | --- |
| `matmul(a, b)` | 矩阵乘法 `a @ b` |
| `kron(a, b)` | Kronecker 积 `a ⊗ b` |
| `dagger(matrix)` | 共轭转置 `matrix†` |
| `trace(matrix)` | 矩阵的迹 |
| `real(tensor)` | 逐元素取实部 |
| `abs_sq(tensor)` | 逐元素取模平方 \|x\|² |

### 量子操作

| 方法 | 说明 |
| --- | --- |
| `apply_unitary(state, unitary)` | 酉矩阵作用于态向量：\|ψ'⟩ = U\|ψ⟩ |
| `inner_product(bra, ket)` | 内积 ⟨bra\|ket⟩ |
| `measure_probs(state)` | 由态向量计算计算基测量概率分布 |
| `partial_trace(rho, keep, n_qubits)` | 对密度矩阵执行偏迹 |
| `sample(probs, shots)` | 按概率分布采样 |
| `expectation_sv(state, operator)` | 纯态期望值 ⟨ψ\|O\|ψ⟩ |
| `expectation_dm(rho, operator)` | 混合态期望值 Tr(ρO) |

### 便利方法（非抽象）

| 方法 | 说明 |
| --- | --- |
| `tensor_product(*matrices)` | 多矩阵 Kronecker 积（从左到右） |
| `matrix_product(*matrices)` | 多矩阵乘积（从左到右） |

---

## 11  NPU complex64 兼容性详解

Ascend NPU 在不同版本的 `torch_npu` 下对 `complex64` 的内核支持不完整，某些复数算子会直接报错（如 `aclnnMatmul ... DT_COMPLEX64 not implemented`）。`NPUBackend` 在后端层提供兼容路径。

### 11.1  处理原则

- **不需要**把整个项目改成"处处手工拆实虚部"。
- 只需确保"在 NPU 上实际执行的复数运算"都经过后端封装或 NPU 专用回退。
- 若出现新报错，按栈定位到具体算子，再做最小修复。

### 11.2  已覆盖的兼容算子

后端层（实 / 虚部拆分回退）：

| 算子 | NPU 兼容方式 |
| --- | --- |
| `matmul` / `apply_unitary` | 自定义 `torch.autograd.Function`（`_NpuMatmulFn`），实 / 虚部拆分不进入 autograd 图 |
| `kron` | 实 / 虚部拆分后四次实 `kron` |
| `dagger` / `trace` | 分别对实 / 虚部操作 |
| `inner_product` | 实 / 虚部点积 |
| `partial_trace` | 对实 / 虚部分别执行逐比特求迹 |
| `expectation_sv` | 自定义 `_NpuExpectationFn`，避免 fan-out 梯度累加 |
| `expectation_dm` | 经 `matmul` + `trace` 组合 |
| `abs_sq` / `measure_probs` | `real² + imag²` 替代 `abs()` |
| `eye` / `zeros_state` | 从实张量构造后拼为复数 |

门矩阵构造层（`aicir/core/gates.py`）：

- `rz` / `rzz` / `u2` / `u3` 等参数化门改用 `cos + i·sin` 构造，避免 `torch.exp` 对复数张量的调用。
- 各含梯度单元构造为独立复数张量、不做复数乘法、不复用同一张量，使梯度累加落在实数角度上。

### 11.3  训练（autograd backward）

`complex64` 的**反向**内核同样缺失。要让 `loss.backward()` 在 NPU 上跑通，需要整条反向路径里没有复数加 / 乘：

1. **线性代数层**：`matmul` 与 `expectation_sv` 用自定义 `torch.autograd.Function` 封装，实 / 虚部拆分只发生在 Function 的 forward / backward 内部（不进入 autograd 计算图），反向时不会发生复数梯度累加。
2. **门矩阵构造层**：参数化门按实 / 虚部独立拼装、各单元不复用、不做复数乘法，使复数梯度不会触发复数加 / 乘。

两者配合后，**NPU 上可以直接用 `loss.backward()` 训练**，QAS supernet 也因此走标准 autograd 路径。

> 仍可选用参数移位规则（`SupernetConfig(use_parameter_shift=True)`）作为对照或后备；每个参数每步需 2 次前向求值，通常更慢。

### 11.4  快速排查清单

遇到 NPU 复数报错时：

1. 检查报错是否包含 `DT_COMPLEX64 not implemented`。
2. 检查报错栈是否位于后端层**之外**（如业务代码直接做了 Torch 复数加法）。
3. 优先改为调用 `backend` 方法；必要时在 `NPUBackend` 增加拆分回退。

### 11.5  验证

```bash
# NPU 全链路 smoke 测试
python tests/smoke_npu_new_path.py --shots 512
```

示例输出：

```text
=== Smoke NPU New Path ===
backend: NPUBackend(dtype=torch.complex64, device=npu:0, npu_available=True)
runtime_context: NPURuntimeContext(world_size=1, rank=0, local_rank=0, distributed=False)
[PASS] single_gate
[PASS] controlled_gate
[PASS] parametric_gate
[PASS] density_matrix

Summary: PASS
```

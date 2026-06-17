# 计算后端的选择与使用

aicir 提供三种计算后端，都实现统一的 `Backend` 接口，可与 `Circuit` / `Measure` / `State` / `Hamiltonian` 无缝配合。它们的区别只在底层张量库与运行设备，业务代码无需改动即可切换：

| 后端                                    | 底层库                 | 运行设备                 | 自动微分 | 典型用途                                      |
| --------------------------------------- | ---------------------- | ------------------------ | -------- | --------------------------------------------- |
| `NumpyBackend`                        | NumPy                  | CPU                      | 否       | 小规模验证、算法原型、无 PyTorch 依赖环境     |
| `GPUBackend`（别名 `TorchBackend`） | PyTorch                | CPU / CUDA GPU           | 是       | 变分算法（VQE/QAOA/QML）、需要梯度或 GPU 加速 |
| `NPUBackend`                          | PyTorch +`torch_npu` | Ascend NPU（可回退 CPU） | 是       | 昇腾 NPU 上的仿真与训练                       |

> 命名说明：`TorchBackend` 是 `GPUBackend` 的**过时别名**，仅为向后兼容保留，新代码请使用 `GPUBackend`。`NPUBackend` 继承自 `GPUBackend`，复用其全部数学内核，并对 NPU 缺失的 `complex64` 算子做实部/虚部拆分回退（详见 6.4 / 6.5）。

三种后端的调用范式完全一致——构造一个后端实例，绑定到 `Circuit`（或传给 `Measure`）即可：

```python
from aicir import Circuit, Measure, NumpyBackend, GPUBackend, NPUBackend, hadamard, cnot

backend = GPUBackend(device="cpu")     # 换成 NumpyBackend() 或 NPUBackend() 完全等价
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name, result.counts)
```

> 后端可绑定在 `Circuit` 上，也可只传给 `Measure`；当两者都给定时以 `circuit.backend` 优先。推荐绑定到 `Circuit`，原因与逐门演化、减少主机↔设备搬运有关（详见 6.6）。

## 6.1 NumpyBackend（CPU 参考实现）

```python
from aicir import NumpyBackend

backend = NumpyBackend()               # 纯 CPU，默认 numpy complex64
backend = NumpyBackend(dtype="complex128")   # 可选：指定复数精度
```

- 依赖最少，无需安装 PyTorch。
- **不支持自动微分**，因此不能用于基于 autograd 的参数训练（如需梯度，请改用 `GPUBackend`，或配合 `aicir/qml/README.md` 中参数移位 `psr` 等数值方法）。
- 适合教学、单元测试与小比特数验证。

## 6.2 GPUBackend（PyTorch，CPU / CUDA）

```python
import torch
from aicir import GPUBackend

backend = GPUBackend()                       # 默认设备：有 CUDA 用 cuda，否则 cpu
backend = GPUBackend(device="cpu")           # 强制 CPU
backend = GPUBackend(device="cuda:0")        # 指定某张 GPU
backend = GPUBackend(device=torch.device("cuda"), dtype=torch.complex128)  # 自定义设备与精度
```

构造参数：

| 参数       | 默认                              | 说明                                         |
| ---------- | --------------------------------- | -------------------------------------------- |
| `device` | 有 CUDA 用 `cuda`，否则 `cpu` | 接收字符串或 `torch.device`                |
| `dtype`  | `torch.complex64`               | 复数精度，可设 `torch.complex128` 提高精度 |

- **支持 PyTorch autograd**：把 Torch 标量张量作为门参数（`rx/ry/rz/u2/u3`、受控旋转、`rzz/rxx`、自定义 `unitary`）即可保留计算图，用于 VQE/QAOA/QML 训练（见顶层 `README.md` 4.4 节与 `aicir/qml/README.md`）。
- **支持 CUDA GPU 加速**：把 `device` 指向 GPU 即可，门矩阵构造与态演化都在 GPU 上完成。

## 6.3 NPUBackend（Ascend NPU）

```python
from aicir import NPUBackend

backend = NPUBackend()                                       # 自动选 npu:0（不可用则回退 CPU）
backend = NPUBackend(device="npu:1")                         # 指定某张 NPU 卡
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)  # 严格模式：NPU 不可用直接报错
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)  # 多卡：按 LOCAL_RANK 自动绑卡
```

构造参数：

| 参数                | 默认                | 说明                                                                      |
| ------------------- | ------------------- | ------------------------------------------------------------------------- |
| `device`          | 自动 `npu:0`      | 目标 NPU 设备                                                             |
| `dtype`           | `torch.complex64` | 复数精度                                                                  |
| `fallback_to_cpu` | `True`            | NPU 不可用时是否回退 CPU；`False` 则抛 `RuntimeError`（用于平台验证） |

- 继承自 `GPUBackend`，API 完全一致，可直接替换其它后端。
- 依赖 `torch_npu`；在真正的 NPU 设备上会自动对缺失的 `complex64` 内核做兼容回退（见 6.4 / 6.5）。
- 严格模式见 6.7，多卡/分布式见 6.9，端到端示例见 6.10。

> 在 QAS supernet（`aicir/qas`）中无需手动构造后端：只要把 `device="npu:0"` 传入配置，框架会自动选用 `NPUBackend`（见 `aicir/qas/supernet.py` 的 `_make_backend`）；CPU/CUDA 设备则用 `GPUBackend`。

## 6.4 NPU 兼容性概述（complex64）

Ascend NPU 在不同版本的 `torch_npu` 组合下，对 `complex64` 的内核支持并不完整，某些复数算子会直接报错，例如：

- `aclnnMatmul ... DT_COMPLEX64 not implemented`
- `aclnnEye ... DT_COMPLEX64 not implemented`
- `aclnnAdd ... DT_COMPLEX64 not implemented`

`NPUBackend` 在后端层提供 NPU 专用兼容路径（workaround），核心思路：优先走后端抽象接口（`matmul/kron/trace/...`）而非业务层直接做 torch 复数运算；在 NPU 且输入为复数时，将计算拆成实部/虚部后重组，绕过缺失内核；并对常见初始化路径（如 `eye`、`|0...0>`）提供 NPU 安全实现。当前已覆盖的高频兼容算子包括：

- `matmul`, `apply_unitary`
- `kron`
- `dagger`, `trace`
- `inner_product`, `partial_trace`
- `expectation_sv`, `expectation_dm`
- `abs_sq`, `measure_probs`
- `eye`, `zeros_state`

此外，门矩阵构造层（`aicir/core/gates.py`）也避免对复数张量直接调用 `torch.exp`（`rz/rzz/u2/u3` 改用 `cos + i·sin` 构造），同样是为了绕过 NPU 缺失的复数内核。为保证**反向**同样可用（见下），参数化门矩阵的每个含梯度单元都构造为**独立**的复数张量、且不做复数乘法：早期 `rzz`/`rxx` 把同一个复数相位张量放进矩阵的多个位置，autograd 在累加其梯度时会触发 `aclnnAdd ... DT_COMPLEX64`；`u2`/`u3` 用复数乘法 `exp(i·x)·sin` 会触发 `aclnnMul`。现改为按实/虚部直接拼装、各单元互不复用，梯度累加因此落在**实数角度**上（NPU 支持的实数加法）。注意：这不代表 NPU 对所有复数算子都原生可用——若新增路径中出现“直接 torch 复数加减乘”，仍可能触发新报错（排查方法见 6.5）。

> **训练（autograd backward）**：`complex64` 的**反向**内核同样缺失——朴素实现中，复数张量的梯度累加/相乘会触发 `aclnnAdd` / `aclnnMul ... DT_COMPLEX64`，导致 `loss.backward()` 直接报错。要让 `loss.backward()` 在 NPU 上跑通，需要保证**整条反向路径里没有任何复数加/乘**，为此做了两层处理：
>
> 1. **线性代数层**：`matmul` 与 `expectation_sv` 用自定义 `torch.autograd.Function`（`_NpuMatmulFn` / `_NpuExpectationFn`）封装，实/虚部拆分只发生在 Function 的 forward/backward **内部**（不进入 autograd 计算图），因此计算图里只剩“线性使用”的复数节点，反向时不会发生复数梯度累加。
> 2. **门矩阵构造层**：如上一段所述，`rzz/rxx/u2/u3` 等参数化门按实/虚部独立拼装、各单元不复用、不做复数乘法，使来自门矩阵的复数梯度也不会触发复数加/乘。
>
> 两者配合后，梯度与原生复数 autograd 数值一致（已在 CPU 上以“禁用复数加/乘”的 dispatch 模式模拟 NPU 限制对齐验证，见 `tests/backends/test_npu_backend.py`），所以**在 NPU 上可以直接用 `loss.backward()` 训练**，QAS supernet 也因此走标准 autograd 路径（比参数移位快）。
>
> 仍可选用参数移位规则（`SupernetConfig(use_parameter_shift=True)`）作为对照或后备；它仅前向计算，对 `rx/ry/rz/rzz` 这类 Pauli 旋转门是精确梯度，但每个参数每步需 2 次前向求值，通常更慢。若自行在 NPU 上写训练循环并直接对复数张量做运算，请确保经过 `NPUBackend` 的封装方法（`matmul`/`expectation_sv` 等），否则裸的复数 backward 仍会报错。

## 6.5 NPU complex64 问题详解（建议先读）

### 6.5.1 根因

- 问题不在量子算法本身，而在底层内核支持矩阵。
- 同样的 Python 代码在 CPU/CUDA 可运行，不代表在 NPU 复数路径可运行。

### 6.5.2 典型触发点

- 前端构造电路矩阵时，直接对复数张量做 `+`、`*`、某些初始化操作。
- 绕过 `Backend` 接口，直接调用 torch 复数运算。

### 6.5.3 处理原则

- 不需要把整个项目都改成“处处手工拆实虚部”。
- 只需要确保“在 NPU 上实际执行的复数运算”都经过后端封装或 NPU 专用回退。
- 若出现新报错，按栈定位到具体算子点，再做最小修复。

### 6.5.4 快速排查清单

- 检查报错是否包含 `DT_COMPLEX64 not implemented`。
- 检查报错栈是否位于后端层之外（例如业务文件里直接做了 torch 复数加法）。
- 优先改为调用 `backend` 方法，必要时在 `NPUBackend` 增加拆分回退。

## 6.6 推荐方式：在 `Circuit` 绑定后端（也可在 `Measure` 指定）

推荐在构建电路时把目标后端绑定到 `Circuit`（即在 `Circuit(..., backend=...)` 或随后调用 `bind_backend()`）。

示例：

```python
from aicir import Circuit, Measure, NPUBackend, hadamard, cnot

backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

# Measure 也可以接收 backend，但会被 circuit.backend 优先覆盖
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name)
```

要点说明：

- **可以在两处指定 backend**：`Circuit` 或 `Measure` 都支持传入后端。
- **优先级**：`Measure` 会优先使用 `circuit.backend`（若存在），否则使用 `Measure` 自身的后端（见 `Measure._resolve_backend` 的实现）。因此将后端绑定到 `Circuit` 能避免回退到主机端拼装或与 Measure 中传入后端的混淆。
- **为什么推荐绑定到 `Circuit`**：当电路具有 `gates` 时，`Measure` 会逐门调用 `gate_to_matrix(..., backend=resolved_backend)` 在目标设备上构造并作用门矩阵，从而减少构造完整 2^n×2^n 矩阵的内存与主机→设备搬运；若 `unitary(backend=...)` 不被支持则会回退到无 backend 的 `unitary()`（在 CPU 上用 numpy 拼装整矩阵），然后再 `backend.cast` 到设备，这会引起大规模数据搬运。对于 `GPUBackend`，参数化门的 Torch 标量张量会通过 torch 运算构造矩阵，从而保留 autograd 计算图。

关于将多个电路合并（拼接）时的 backend 确定：

- 使用 `+` 操作符拼接两个 `Circuit`（`a + b`）时，新电路会按实现选择后端：优先采用左侧电路的 backend（`a._backend`）；若左侧没有，则采用右侧的 backend（`b._backend`）。这与 `Circuit.__add__` 的实现一致。
- 因此，若要把多个原本绑定到不同后端的 `Circuit` 连接成一个整体并在统一设备上运行，应在拼接后或拼接前显式统一后端：

```python
# 推荐做法：拼接后显式设置统一后端
full = part_a + part_b
full.bind_backend(common_backend)
result = Measure(common_backend).run(full)
```

- 如果不显式统一后端，拼接结果会继承左侧电路的 backend（若左侧没有则用右侧），这可能不是预期且可能导致在运行时出现回退或不一致的行为。

小结：将后端绑定到 `Circuit` 并在合并后或合并前统一后端，是既安全又高效的做法。

- 构建阶段只保存门描述: 调用 `hadamard(0)` 等构造的是门的描述字典（例如 `{"type": "hadamard", "target_qubit": 0}`），`Circuit.__init__` 只是把这些描述存起来，并不会在构建时把门转换成数值矩阵。
- 当前执行策略: `Measure.run`/`run_density_matrix` 在电路对象具备 `gates` 序列时，会优先走“逐门演化”路径（按门依次作用到态/密度矩阵），而不是先组装整条电路的全局矩阵后再一次性作用。
- 矩阵在组装时生成: 真正把门变为 2^n×2^n 的数值矩阵发生在调用 `Circuit.unitary(backend=...)` 或 `Measure` 等需要数值矩阵的地方。此时会调用 `gate_to_matrix(gate, cir_qubits, backend)` 来生成每个门的矩阵。
- backend 参数的作用: 当 `backend=None` 时，`gate_to_matrix` 会走 numpy 路径（例如调用 `_hadamard()` 等函数，在 CPU 上生成矩阵）；当传入 `backend` 时，`gate_to_matrix` 会使用后端分支（先构造 base 矩阵再通过 `_single_qubit_from_base_backend`/`_controlled_from_base_backend` 等路径调用 `backend.cast`、`backend.kron`、`backend.matmul` 等接口），从而在目标后端（CPU/GPU/NPU）上构造和组合张量。`rx`/`ry`/`rz`/`u2`/`u3`、受控旋转、`rzz`/`rxx` 和自定义 `unitary` 可在 `GPUBackend` 下保留 Torch 参数的梯度链路。
- 兼容回退路径: 若电路对象不提供 `gates` 序列，`Measure` 仍会回退到 `unitary()` 路径以兼容外部实现。
- 可能的设备搬运: 在 `unitary()` 回退路径中，`Measure` 现在优先直接 `backend.cast(unitary_raw)`，避免无必要的 `to_numpy` 主机往返。
- 性能建议: 对大 qubit 数，显式组装全矩阵会占用大量内存并产生迁移成本。若要最小化搬运，优先在构建时绑定后端（本节方式 B），或改为按门逐步在态上直接作用（逐门 apply），避免生成完整 2^n×2^n 矩阵；若需要彻底避免中间拷贝，可考虑修改 `Measure` 中的 `to_numpy` 使用点或直接在后端上逐门演化。

也可先构建再绑定：

```python
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
cir.bind_backend(backend)
```

适用场景：

- 你希望前端矩阵组装与执行严格在同一设备上
- 希望减少 CPU 和 XPU 之间的数据迁移

## 6.7 严格 NPU 模式（不允许回退）

```python
from aicir import NPUBackend

# NPU 不可用时直接抛 RuntimeError，用于验证平台
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)
```

## 6.8 运行示例

仓库示例脚本：`demos/demo_npu.py`

```bash
python demos/demo_npu.py
python demos/demo_npu.py --shots 2048 --allow-cpu-fallback
```

## 6.9 分布式环境（多卡/多节点）

使用环境变量 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 自动绑定对应卡：

```python
from aicir import NPUBackend

# 自动读取 LOCAL_RANK 决定 npu:LOCAL_RANK
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
print(backend.runtime_context)
# NPURuntimeContext(world_size=4, rank=0, local_rank=0, distributed=True)
```

启动方式：

```bash
# 单卡验证
python demos/demo_npu.py

# 允许 CPU 回退（本地调试用）
python demos/demo_npu.py --allow-cpu-fallback

# 多卡分布式启动
torchrun --nproc_per_node=4 your_script.py
```

## 6.10 完整端到端示例

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

## 6.11 runtime_context 字段说明

| 字段            | 说明                                      |
| --------------- | ----------------------------------------- |
| `world_size`  | 总进程数                                  |
| `rank`        | 全局进程编号                              |
| `local_rank`  | 本节点本地编号（对应 `npu:local_rank`） |
| `distributed` | `world_size > 1` 时为 True              |

## 6.12 远程 NPU 验证输出示例（新路径）

使用新路径 smoke 脚本进行全链路验证（单门、受控门、参数门、density matrix）：

```bash
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

## 6.13 BatchSV（批量态矢量路径，NPU 安全且可微）

`aicir` 主路径（`State` / `apply_gate_to_state`）一次只演化单个 `(2^n, 1)` 态矢量，且门参数为标量。深度学习场景（例如把变分量子线路当作神经网络的一层）需要：一次模拟一批态矢量；旋转门角度可逐样本（per-sample）不同（如数据编码角度依赖输入）；端到端可微（autograd）；以及 **Ascend NPU 安全**——NPU 缺少 `complex64` 的 `aclnnAdd` / `aclnnMul` 内核，因此 `BatchSV` 全程以实部/虚部两个实张量表示，只用实数乘加，反向传播不会触发复数累加。

`BatchSV`（规范路径 `aicir.core`，也可从顶层 `aicir` 导入）即为该批量路径。门矩阵与单态路径采用同一套定义（复用 `_single_qubit_base_for_gate`），保持单一事实来源；量子比特端序也与主路径一致（**qubit 0 为最高位**）。需要 `torch`。

```python
import torch
from aicir import BatchSV, GPUBackend, hadamard, ry, crz

backend = GPUBackend()                      # 或 NPUBackend；用于获取目标 device / dtype
bsv = BatchSV(n_qubits=3, batch_size=8, backend=backend)

# 逐样本数据编码角度：ry 的参数可以是 (batch,) 张量，逐样本不同。
enc = torch.randn(8, 3, requires_grad=True)
theta = torch.zeros(3, requires_grad=True)  # 0 维/标量参数按 batch 广播

for q in range(3):
    bsv.apply_gate(hadamard(q))
    bsv.apply_gate(ry(enc[:, q], q))        # 逐样本角度
bsv.apply_gate(crz(theta[0], 1, [0]))       # 受控旋转门同样支持
bsv.apply_gate(ry(theta[1], 2))

z = bsv.z_expectations()                     # (batch, n_qubits) 逐比特 <Z_q>
probs = bsv.probabilities()                  # (batch, 2^n) 基测量概率

loss = z.sum()
loss.backward()                              # 端到端可微，全程实张量
```

逐样本张量角度目前支持 `rx` / `ry` / `rz` 及其受控形式 `crx` / `cry` / `crz`；常量门与标量参数门复用单态路径定义，覆盖范围与之一致。`apply_gate` 返回自身以便链式调用。

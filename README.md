# nexq 使用手册

---

## 1. 模块导入

所有常用类与函数均可从顶层 `nexq` 包一次性导入。

状态类 `StateVector`、`DensityMatrix` 的规范模块路径为 `nexq.circuit`，同时也可从顶层 `nexq` 导入。

```python
# 后端
from nexq import TorchBackend, NumpyBackend, NPUBackend

# 量子态（规范路径）
from nexq.circuit import StateVector, DensityMatrix

# 量子门（构造函数，返回门字典）
from nexq import (
    pauli_x, pauli_y, pauli_z,
    hadamard,
    rx, ry, rz,
    s_gate, t_gate,
    cx, cnot, cy, cz,
    crx, cry, crz,
    swap,
    toffoli, ccnot,
    u2, u3,
)

# 电路
from nexq import Circuit

# 测量
from nexq import Measure, Result

# 哈密顿量
from nexq import PauliOp, PauliString, Hamiltonian

# 噪声
from nexq import (
    NoiseChannel, NoiseModel,
    DepolarizingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
)

# OpenQASM 互转
from nexq import (
    circuit_to_qasm, circuit_to_qasm3,
    circuit_from_qasm,
    load_circuit_qasm, save_circuit_qasm,
    load_circuit_qasm, save_circuit_qasm3,
)
```

---

## 2. 量子线路的搭建

### 2.1 门字典速查

| 函数                    | 参数                             | 说明            |
| ----------------------- | -------------------------------- | --------------- |
| `pauli_x(q)`          | target_qubit                     | X 门            |
| `pauli_y(q)`          | target_qubit                     | Y 门            |
| `pauli_z(q)`          | target_qubit                     | Z 门            |
| `hadamard(q)`         | target_qubit                     | H 门            |
| `s_gate(q)`           | target_qubit                     | S 门            |
| `t_gate(q)`           | target_qubit                     | T 门            |
| `rx(θ, q)`           | 角度, target_qubit               | Rx 旋转门       |
| `ry(θ, q)`           | 角度, target_qubit               | Ry 旋转门       |
| `rz(θ, q)`           | 角度, target_qubit               | Rz 旋转门       |
| `u2(φ, λ, q)`       | phi, lambda, target_qubit        | U2 门           |
| `u3(θ, φ, λ, q)`   | theta, phi, lambda, target_qubit | U3 通用单比特门 |
| `cx(t, [c])`          | target, control_list             | CNOT（控制-X）  |
| `cnot(t, [c])`        | target, control_list             | 同 cx           |
| `cy(t, [c])`          | target, control_list             | 控制-Y          |
| `cz(t, [c])`          | target, control_list             | 控制-Z          |
| `crx(θ, t, [c])`     | 角度, target, control_list       | 受控 Rx         |
| `cry(θ, t, [c])`     | 角度, target, control_list       | 受控 Ry         |
| `crz(θ, t, [c])`     | 角度, target, control_list       | 受控 Rz         |
| `swap(q1, q2)`        | qubit_1, qubit_2                 | SWAP 门         |
| `toffoli(t, [c0,c1])` | target, control_list             | Toffoli (CCX)   |
| `ccnot(t, [c0,c1])`   | target, control_list             | 同 toffoli      |

### 2.2 构建电路

```python
from nexq import Circuit, hadamard, cnot, rz, rx, pauli_x, swap, toffoli

# 方式一：构造时直接传入门列表（自动推断 n_qubits）
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),      # 目标比特=1，控制比特=[0]
    rz(0.5, 1),
    n_qubits=2,        # 也可手动指定
)

# 方式二：先构造空电路，再逐步追加
cir = Circuit(hadamard(0), n_qubits=3)
cir.append(cx(1, [0]))
cir.extend(ry(1.2, 2), rz(0.3, 2))

# 方式三：两段电路拼接
part_a = Circuit(hadamard(0), n_qubits=2)
part_b = Circuit(cnot(1, [0]), n_qubits=2)
full = part_a + part_b

# 获取电路酉矩阵（numpy complex64，2^n × 2^n）
U = full.unitary()
print(U.shape)   # (4, 4)
```

### 2.3 更多门用法示例

```python
import math
from nexq import Circuit, rx, ry, rz, u3, crx, swap, toffoli

cir = Circuit(
    rx(math.pi / 2, 0),             # Rx(π/2) 作用在 qubit 0
    ry(math.pi / 4, 1),             # Ry(π/4) 作用在 qubit 1
    u3(math.pi, 0, math.pi, 2),     # U3(π, 0, π) ≡ X 门，作用在 qubit 2
    crx(math.pi / 2, 2, [1]),       # 受控 Rx，控制=qubit1，目标=qubit2
    swap(0, 1),                     # SWAP qubit0 和 qubit1
    toffoli(2, [0, 1]),             # Toffoli，控制=[0,1]，目标=qubit2
    n_qubits=3,
)
```

---

## 3. 量子测量

`Measure` 对象绑定一个后端，`run()` 返回统一的 `Result` 对象。

### 3.1 基本用法：概率 + 采样计数

```python
from nexq import Circuit, Measure, TorchBackend, hadamard, cnot

backend = TorchBackend()
measure = Measure(backend)

cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

# shots=None 时只返回概率，不采样
result = measure.run(cir, shots=1024)

print(result.probabilities)     # array([0.5, 0. , 0. , 0.5])
print(result.counts)            # {'|00>': 512, '|11>': 512}
print(result.most_probable())   # ('|00>', 0.5)
print(result.summary())
```

### 3.2 仅获取概率（不采样）

```python
result = measure.run(cir, shots=None)
print(result.probabilities)     # 仅概率，counts 为 None
```

### 3.3 期望值测量

```python
import numpy as np
from nexq import Circuit, Measure, TorchBackend, hadamard

backend = TorchBackend()
# Z 算符矩阵
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

result = measure.run(
    Circuit(hadamard(0), n_qubits=1),
    shots=1024,
    observables={"Z0": Z},
)
print(result.expectation_values)   # {'Z0': ~0.0}
print(result.expectation_variances)
```

### 3.4 从 StateVector 直接测量

```python
from nexq.circuit import StateVector
from nexq import TorchBackend
import numpy as np

backend = TorchBackend()
sv = StateVector.zero_state(2, backend)

# 直接获取概率分布
probs = sv.probabilities()

# 模拟 512 次测量，返回 {bitstring: count}
counts = sv.measure(shots=512)
print(counts)   # {'|00>': 512}
```

### 3.5 Result 对象字段速查

| 字段                      | 类型                       | 说明                          |
| ------------------------- | -------------------------- | ----------------------------- |
| `probabilities`         | `np.ndarray`             | 各基态概率，shape `(2^n,)`  |
| `counts`                | `dict` or `None`       | `{'\|00>': N, ...}` 采样计数 |
| `shots`                 | `int` or `None`        | 采样次数                      |
| `expectation_values`    | `dict`                   | `{name: float}` 期望值      |
| `expectation_variances` | `dict`                   | `{name: float}` 方差        |
| `final_state`           | `np.ndarray` or `None` | 末态向量                      |
| `most_probable()`       | `(str, float)`           | 最高概率基态及其概率          |
| `summary()`             | `str`                    | 单行摘要字符串                |

---

## 4. 构建哈密顿量

nexq 提供三个层次的算符构建工具：`PauliOp`、`PauliString`、`Hamiltonian`。

### 4.1 单算符 PauliOp

```python
from nexq import PauliOp, TorchBackend

backend = TorchBackend()

# Z 作用在 qubit 0，在 2 比特空间里展开为 4×4 矩阵
Z0 = PauliOp('Z', qubit=0)
mat = Z0.to_matrix(n_qubits=2, backend=backend)
print(mat.shape)   # torch.Size([4, 4])
```

### 4.2 多体泡利串 PauliString

```python
from nexq import PauliString, TorchBackend

backend = TorchBackend()

# 0.5 × Z₀ ⊗ X₁（2 比特空间），注意参数顺序：coefficient 在 n_qubits 之前
ps = PauliString({'Z': [0], 'X': [1]}, coefficient=0.5, n_qubits=2)
mat = ps.to_matrix(backend)
print(ps)   # PauliString(0.5 × ZX)

# 或者省略 n_qubits，库会自动从 paulistring 中推断（最大索引 + 1）
ps_auto = PauliString({'Z': [0], 'X': [1]}, coefficient=0.5)
print(ps_auto)
```

### 4.3 哈密顿量 Hamiltonian

```python
from nexq import Hamiltonian, TorchBackend, Circuit, Measure, hadamard

backend = TorchBackend()

# H = -1.0 × Z₀Z₁  +  0.5 × X₀X₁  +  0.3 × Z₀
H = (Hamiltonian(n_qubits=2)
    .term(-1.0, {'Z': [0, 1]})
    .term( 0.5, {'X': [0, 1]})
    .term( 0.3, {'Z': [0]}))

# 转为后端矩阵
H_mat = H.to_matrix(backend)
print(H_mat.shape)   # torch.Size([4, 4])

# 计算期望值（通过 StateVector）
from nexq.circuit import StateVector
sv = StateVector.zero_state(2, backend)
print(H.expectation(sv, backend))   # 实数期望值

# 通过 Measure.run() 传入 observables 参数
measure = Measure(backend)
cir = Circuit(hadamard(0), n_qubits=2)
result = measure.run(cir, shots=None, observables={"H": H_mat})
print(result.expectation_values["H"])
```

### 4.4 噪声通道（开放量子系统）

```python
from nexq import (
    NoiseModel,
    DepolarizingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
    TorchBackend,
)
from nexq.circuit import DensityMatrix

backend = TorchBackend()
model = (NoiseModel()
         .add_channel(DepolarizingChannel(target_qubit=0, p=0.01))
         .add_channel(BitFlipChannel(target_qubit=1, p=0.02), after_gates=["hadamard"])
         .add_channel(PhaseFlipChannel(target_qubit=0, p=0.005)))

# 手动应用到密度矩阵
rho = DensityMatrix.zero_state(2, backend)
rho_noisy = model.apply(rho.data, n_qubits=2, backend=backend)
```

---

## 5. NPU 后端的使用

nexq 通过 `NPUBackend` 支持 Ascend NPU（依赖 `torch_npu`）。

说明：Ascend NPU 在不同版本的 `torch_npu` 组合下，对 `complex64` 的内核支持并不完整。某些复数算子会直接报错，例如：

- `aclnnEye ... DT_COMPLEX64 not implemented`
- `aclnnAdd ... DT_COMPLEX64 not implemented`

因此，nexq 在后端层提供了 NPU 专用兼容路径（workaround），核心思路是：

- 优先走后端抽象接口（`matmul/kron/trace/...`），避免业务层直接做 torch 复数运算。
- 在 NPU 且输入为复数时，将计算拆成实部/虚部后重组，绕过缺失内核。
- 对常见初始化路径（如 `eye`、`|0...0>`）提供 NPU 安全实现。

当前已覆盖的高频兼容算子包括：

- `matmul`, `apply_unitary`
- `kron`
- `dagger`, `trace`
- `inner_product`, `partial_trace`
- `expectation_sv`, `expectation_dm`
- `abs_sq`, `measure_probs`
- `eye`, `zeros_state`

注意：这不代表 NPU 对所有复数算子都原生可用。若新增路径中出现“直接 torch 复数加减乘”，仍可能触发新报错。

### 5.1 NPU complex64 问题详解（建议先读）

#### 5.1.1 根因

- 问题不在量子算法本身，而在底层内核支持矩阵。
- 同样的 Python 代码在 CPU/CUDA 可运行，不代表在 NPU 复数路径可运行。

#### 5.1.2 典型触发点

- 前端构造电路矩阵时，直接对复数张量做 `+`、`*`、某些初始化操作。
- 绕过 `Backend` 接口，直接调用 torch 复数运算。

#### 5.1.3 处理原则

- 不需要把整个项目都改成“处处手工拆实虚部”。
- 只需要确保“在 NPU 上实际执行的复数运算”都经过后端封装或 NPU 专用回退。
- 若出现新报错，按栈定位到具体算子点，再做最小修复。

#### 5.1.4 快速排查清单

- 检查报错是否包含 `DT_COMPLEX64 not implemented`。
- 检查报错栈是否位于后端层之外（例如业务文件里直接做了 torch 复数加法）。
- 优先改为调用 `backend` 方法，必要时在 `NPUBackend` 增加拆分回退。

### 5.2 方式 A（在测量阶段指定后端）

该方式保持电路对象与后端解耦，在 `Measure` 中指定后端。

```python
from nexq import Circuit, Measure, NPUBackend, hadamard, cnot

backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
)

result = Measure(backend).run(cir, shots=1024)
print(result.backend_name)
```

适用场景：

- 你希望电路对象可复用在多种后端上（CPU/GPU/NPU）
- 希望保持 API 简洁，执行时再选择设备

### 5.3 方式 B（前端构建电路时绑定后端）

该方式在 `Circuit` 构建阶段指定 backend，前端矩阵组装与后端执行保持同一 XPU。

```python
from nexq import Circuit, Measure, NPUBackend, hadamard, cnot

backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

# Measure 会优先使用 circuit.backend（若存在）
result = Measure(backend).run(cir, shots=1024)
print(result.backend_name)

更多说明（矩阵组装时机与 backend 使用）:

- 构建阶段只保存门描述: 调用 `hadamard(0)` 等构造的是门的描述字典（例如 `{"type": "hadamard", "target_qubit": 0}`），`Circuit.__init__` 只是把这些描述存起来，并不会在构建时把门转换成数值矩阵。
- 当前执行策略: `Measure.run`/`run_density_matrix` 在电路对象具备 `gates` 序列时，会优先走“逐门演化”路径（按门依次作用到态/密度矩阵），而不是先组装整条电路的全局矩阵后再一次性作用。
- 矩阵在组装时生成: 真正把门变为 2^n×2^n 的数值矩阵发生在调用 `Circuit.unitary(backend=...)` 或 `Measure` 等需要数值矩阵的地方。此时会调用 `gate_to_matrix(gate, cir_qubits, backend)` 来生成每个门的矩阵。
- backend 参数的作用: 当 `backend is None` 时，`gate_to_matrix` 会走 numpy 路径（例如调用 `_hadamard()` 等函数，在 CPU 上生成矩阵）；当传入 `backend` 时，`gate_to_matrix` 会使用后端分支（先构造 2×2 的 base 矩阵再通过 `_single_qubit_from_base_backend`/`_controlled_from_base_backend` 调用 `backend.cast`、`backend.kron`、`backend.matmul` 等接口），从而在目标后端（CPU/GPU/NPU）上构造和组合张量。
- 兼容回退路径: 若电路对象不提供 `gates` 序列，`Measure` 仍会回退到 `unitary()` 路径以兼容外部实现。
- 可能的设备搬运: 在 `unitary()` 回退路径中，`Measure` 现在优先直接 `backend.cast(unitary_raw)`，避免无必要的 `to_numpy` 主机往返。
- 性能建议: 对大 qubit 数，显式组装全矩阵会占用大量内存并产生迁移成本。若要最小化搬运，优先在构建时绑定后端（本节方式 B），或改为按门逐步在态上直接作用（逐门 apply），避免生成完整 2^n×2^n 矩阵；若需要彻底避免中间拷贝，可考虑修改 `Measure` 中的 `to_numpy` 使用点或直接在后端上逐门演化。

```

也可先构建再绑定：

```python
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
cir.bind_backend(backend)
```

适用场景：

- 你希望前端矩阵组装与执行严格在同一设备上
- 希望减少 CPU 和 XPU 之间的数据迁移

### 5.4 严格 NPU 模式（不允许回退）

```python
from nexq import NPUBackend

# NPU 不可用时直接抛 RuntimeError，用于验证平台
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)
```

### 5.5 运行示例

仓库示例脚本：`demo_npu.py`

```bash
python demo_npu.py
python demo_npu.py --shots 2048 --allow-cpu-fallback
```

### 5.6 分布式环境（多卡/多节点）

使用环境变量 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 自动绑定对应卡：

```python
from nexq import NPUBackend

# 自动读取 LOCAL_RANK 决定 npu:LOCAL_RANK
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
print(backend.runtime_context)
# NPURuntimeContext(world_size=4, rank=0, local_rank=0, distributed=True)
```

启动方式：

```bash
# 单卡验证
python demo_npu.py

# 允许 CPU 回退（本地调试用）
python demo_npu.py --allow-cpu-fallback

# 多卡分布式启动
torchrun --nproc_per_node=4 your_script.py
```

### 5.7 完整端到端示例

```python
import math
from nexq import NPUBackend, Circuit, Measure, hadamard, cnot, rz

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

### 5.8 runtime_context 字段说明

| 字段            | 说明                                      |
| --------------- | ----------------------------------------- |
| `world_size`  | 总进程数                                  |
| `rank`        | 全局进程编号                              |
| `local_rank`  | 本节点本地编号（对应 `npu:local_rank`） |
| `distributed` | `world_size > 1` 时为 True              |

### 5.9 远程 NPU 验证输出示例（新路径）

使用新路径 smoke 脚本进行全链路验证（单门、受控门、参数门、density matrix）：

```bash
python smoke_npu_new_path.py --shots 512
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

---

## 6. 与 OpenQASM 2.0 / 3.0 互转

### 6.1 Circuit → QASM 字符串

```python
from nexq import Circuit, hadamard, cnot, rz, circuit_to_qasm, circuit_to_qasm3
import math

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rz(math.pi / 4, 1),
    n_qubits=2,
)

# 导出 OpenQASM 2.0
qasm2 = circuit_to_qasm(cir, version="2.0")
print(qasm2)
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# h q[0];
# cx q[0],q[1];
# rz(pi/4) q[1];

# 导出 OpenQASM 3.0
qasm3 = circuit_to_qasm3(cir)
print(qasm3)
```

### 6.2 QASM 字符串 → Circuit

```python
from nexq import circuit_from_qasm

qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
rz(pi/4) q[1];
"""

cir = circuit_from_qasm(qasm_str)
print(cir)          # Circuit(n_qubits=2, gates=[...])
print(cir.unitary().shape)  # (4, 4)
```

### 6.3 读写 QASM 文件

```python
from nexq import (
    save_circuit_qasm, load_circuit_qasm,
    save_circuit_qasm3,
)

# 保存为 .qasm 文件（OpenQASM 2.0）
save_circuit_qasm(cir, "my_circuit.qasm")

# 保存为 OpenQASM 3.0
save_circuit_qasm3(cir, "my_circuit_v3.qasm")

# 从文件读取（自动识别 2.0/3.0 版本头）
cir2 = load_circuit_qasm("my_circuit.qasm")
```

### 6.4 OpenQASM 3.0 格式差异

nexq 在 3.0 模式下的主要差异：

| 项目           | 2.0                       | 3.0                         |
| -------------- | ------------------------- | --------------------------- |
| 版本头         | `OPENQASM 2.0;`         | `OPENQASM 3.0;`           |
| 标准库         | `include "qelib1.inc";` | `include "stdgates.inc";` |
| 量子寄存器声明 | `qreg q[2];`            | `qubit[2] q;`             |
| U3 门名称      | `u3(θ,φ,λ)`          | `u(θ,φ,λ)`             |

### 6.5 支持的 QASM 门集

| QASM 门名                          | nexq 对应函数                         |
| ---------------------------------- | ------------------------------------- |
| `x`, `y`, `z`                | `pauli_x`, `pauli_y`, `pauli_z` |
| `h`                              | `hadamard`                          |
| `s`, `t`                       | `s_gate`, `t_gate`                |
| `rx`, `ry`, `rz`             | `rx`, `ry`, `rz`                |
| `u2(φ,λ)`                      | `u2`                                |
| `u3(θ,φ,λ)` / `u(θ,φ,λ)` | `u3`                                |
| `cx`                             | `cx` / `cnot`                     |
| `cy`, `cz`                     | `cy`, `cz`                        |
| `swap`                           | `swap`                              |
| `crx`, `cry`, `crz`          | `crx`, `cry`, `crz`             |
| `ccx`                            | `toffoli` / `ccnot`               |

> **当前不支持**：`if`、`reset`、`opaque`、自定义 `gate`、`cp`/`cu` 系列门。
> `measure` 和 `barrier` 语句在导入时会被跳过。

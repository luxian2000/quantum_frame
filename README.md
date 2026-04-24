# nexq 使用手册

---

## 1. 模块导入

所有常用类与函数均可从顶层 `nexq` 包一次性导入。

```python
# 后端
from nexq import TorchBackend, NumpyBackend, NPUBackend

# 量子态
from nexq import StateVector, DensityMatrix

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
from nexq import StateVector, TorchBackend
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

# 0.5 × Z₀ ⊗ X₁（2 比特空间）
ps = PauliString({'Z': [0], 'X': [1]}, n_qubits=2, coefficient=0.5)
mat = ps.to_matrix(backend)
print(ps)   # PauliString(0.5 × ZX)
```

### 4.3 哈密顿量 Hamiltonian

```python
from nexq import Hamiltonian, TorchBackend, Circuit, Measure, hadamard

backend = TorchBackend()

# H = -1.0 × Z₀Z₁  +  0.5 × X₀X₁  +  0.3 × Z₀
H = (Hamiltonian(n_qubits=2)
     .add_term(-1.0, {'Z': [0, 1]})
     .add_term( 0.5, {'X': [0, 1]})
     .add_term( 0.3, {'Z': [0]}))

# 转为后端矩阵
H_mat = H.to_matrix(backend)
print(H_mat.shape)   # torch.Size([4, 4])

# 计算期望值（通过 StateVector）
from nexq import StateVector
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
    DensityMatrix,
    TorchBackend,
)

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

说明：针对 NPU 上 `complex64` 的算子兼容性，库内已经在后端层实现了 workaround（例如 `matmul/kron/trace/inner_product/partial_trace/expectation/abs_sq/measure_probs` 的复数拆分路径）。

### 5.1 方式 A（在测量阶段指定后端）

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

### 5.2 方式 B（前端构建电路时绑定后端）

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
```

也可先构建再绑定：

```python
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
cir.bind_backend(backend)
```

适用场景：

- 你希望前端矩阵组装与执行严格在同一设备上
- 希望减少 CPU 和 XPU 之间的数据迁移

### 5.3 严格 NPU 模式（不允许回退）

```python
from nexq import NPUBackend

# NPU 不可用时直接抛 RuntimeError，用于验证平台
backend = NPUBackend(device="npu:0", fallback_to_cpu=False)
```

### 5.4 运行示例

仓库示例脚本：`demo_npu.py`

```bash
python demo_npu.py
python demo_npu.py --shots 2048 --allow-cpu-fallback
```

### 5.3 分布式环境（多卡/多节点）

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
python demo.py

# 严格验证（NPU 不可用时报错）
python demo.py

# 允许 CPU 回退（本地调试用）
python demo.py --allow-cpu-fallback

# 多卡分布式启动
torchrun --nproc_per_node=4 your_script.py
```

### 5.4 完整端到端示例

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

### 5.5 runtime_context 字段说明

| 字段            | 说明                                      |
| --------------- | ----------------------------------------- |
| `world_size`  | 总进程数                                  |
| `rank`        | 全局进程编号                              |
| `local_rank`  | 本节点本地编号（对应 `npu:local_rank`） |

### 5.6 远程 NPU 验证输出示例（新路径）

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
| `distributed` | `world_size > 1` 时为 True              |

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

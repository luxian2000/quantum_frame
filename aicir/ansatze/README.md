# aicir.ansatze — 参数化线路模板

本模块提供可复用的参数化线路模板（ansatz），返回原生 `Circuit`，参数默认使用 `Parameter` 符号占位，可通过 `Circuit.bind_parameters(...)` 绑定数值后接入 `Measure`、`qml.deriv`、`aicir.vqc`（VQE/QAOA/VQD/SSVQE）或后端态向量模拟。本模块与 `aicir.vqc`（变分算法编排）、`aicir.chemistry`（分子 Hamiltonian）均无依赖耦合——只吃纯数据（比特数、层数、HF 占据/激发列表等），由调用方桥接。

---

## 1. Ansatz 选择指南

```
需要通用硬件高效 VQE / VQA 模板？
  -> hardware_efficient_ansatz / hea
     局域旋转层 + entangler 层，支持 linear/ring/all_to_all/custom topology。

需要 trapped-ion 论文中的 HEA-TI 通用结构？
  -> hea_ti_ansatz(variant="general")
     每层 Rx Ry Rx 单比特旋转 + TFIM 全局演化。

需要电子结构 / 电荷守恒对称性的 HEA-TI？
  -> hea_ti_ansatz(variant="chemistry") 或 variant="xy"
     每层 Rz 单比特旋转 + XY 全局演化，默认 evolution_time=0.4。

只想确认参数数量或初始化数组长度？
  -> hea_parameter_count / hea_ti_parameter_count / uccsd_parameter_count

需要自定义 trapped-ion 耦合矩阵？
  -> power_law_couplings 生成默认 Jij，或向 hea_ti_ansatz(couplings=...) 传入对称矩阵。

需要量子化学 UCCSD ansatz？
  -> uccsd(n_qubits, hf_occupation, excitations)
     吃纯数据（HF 占据 + 激发列表），与 aicir.chemistry 解耦；
     常见桥接：uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)。
```

---

## 2. 公共接口一览

| 函数 | 文件 | 返回 | 用途 |
| --- | --- | --- | --- |
| `hardware_efficient_ansatz` / `hea` | `hea.py` | `Circuit` | 标准硬件高效 ansatz |
| `hea_parameter_count` | `hea.py` | `int` | 计算 HEA 旋转和可参数化 entangler 参数数量 |
| `entangling_edges` | `hea.py` | `list[tuple[int, int]]` | 生成 `linear` / `ring` / `all_to_all` entangler 边 |
| `hea_ti_ansatz` / `hea_ti` | `hea_ti.py` | `Circuit` | trapped-ion HEA-TI ansatz |
| `hea_ti_parameter_count` | `hea_ti.py` | `int` | 计算 HEA-TI 旋转参数数量，可选计入演化时间 |
| `power_law_couplings` | `hea_ti.py` | `np.ndarray` | 生成 `J_ij = J0 / |i-j|^alpha` trapped-ion 耦合 |
| `trapped_ion_hamiltonian` | `hea_ti.py` | `np.ndarray` | 构造 TFIM 或 XY 全局哈密顿量 |
| `global_evolution_unitary` | `hea_ti.py` | `np.ndarray` | 构造 `exp(-i H t)` 全局演化矩阵 |
| `uccsd` | `uccsd.py` | `Circuit` | UCCSD 化学 ansatz，吃纯数据（HF 占据 + 激发列表），与 `aicir.chemistry` 解耦 |
| `uccsd_parameter_count` | `uccsd.py` | `int` | 计算 UCCSD 参数个数（激发数 × reps） |

---

## 3. 标准 HEA

### 线路结构

`hardware_efficient_ansatz` 构造如下重复结构：

```
for each layer:
  local rotation gates on every qubit
  entangler gates on selected topology edges

optional final local rotation layer
```

默认配置为：

```python
hardware_efficient_ansatz(
    n_qubits,
    layers=1,
    rotation_gates=("ry", "rz"),
    entangler="cx",
    topology="linear",
    final_rotation_layer=True,
)
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `n_qubits` | 量子比特数量，必须为正整数 |
| `layers` | rotation-entangler block 数量，可为 0 |
| `rotation_gates` | 局域旋转块，支持 `"rx"`、`"ry"`、`"rz"`、`"u2"`、`"u3"` |
| `entangler` | 支持 `"cx"` / `"cnot"`、`"cy"`、`"cz"`、`"crx"`、`"cry"`、`"crz"`、`"rzz"`、`"rxx"`、`"swap"` |
| `topology` | `"linear"`、`"ring"`、`"all_to_all"` / `"full"`，或自定义 `(control, target)` 边列表 |
| `final_rotation_layer` | 是否追加末尾局域旋转层 |
| `final_rotation_gates` | 末尾旋转块；默认等于 `rotation_gates` |
| `parameter_prefix` | 自动生成符号参数名的前缀，默认 `"theta"` |
| `parameters` | 可选 flat 参数序列；传入后直接生成数值线路 |
| `backend` | 可选绑定到返回 `Circuit` 的后端 |

### 示例

```python
import numpy as np
from aicir import NumpyBackend
from aicir.ansatze import hea, hea_parameter_count

n_qubits = 4
layers = 2

circuit = hea(n_qubits, layers=layers, topology="ring")
print(circuit.parameters)

theta = np.random.default_rng(7).uniform(
    -0.1,
    0.1,
    size=hea_parameter_count(n_qubits, layers=layers, topology="ring"),
)
bound = circuit.bind_parameters(theta)
unitary = bound.unitary(backend=NumpyBackend())
```

---

## 4. HEA-TI

`hea_ti_ansatz` 实现论文 *Hardware-efficient variational quantum algorithm in trapped-ion quantum computer* 中提出的 trapped-ion hardware-efficient ansatz。它用单比特旋转提供局域自由度，并用 trapped-ion 全局哈密顿量演化作为 entangler 层。全局演化当前以 `unitary` 门承载，因此返回值仍是标准 `Circuit`。

### 通用问题结构

`variant="general"` / `"tfim"` 使用 TFIM 全局演化：

```text
H_TFIM = sum_{i<j} J_ij X_i X_j + B sum_i Z_i
```

每层线路为：

```text
for every qubit i:
  Rx(theta_d_i_1) Ry(theta_d_i_2) Rx(theta_d_i_3)
global exp(-i H_TFIM t_d)
```

默认 `B=1`，默认耦合为 `J_ij = J0 / |i-j|^alpha`，其中 `J0=1`、`alpha=1.5`。

### 对称 / 化学问题结构

`variant="symmetry"` / `"chemistry"` / `"xy"` 使用 charge-conserving XY 全局演化：

```text
H_XY = 1/2 sum_{i<j} J_ij (X_i X_j + Y_i Y_j)
```

每层线路为：

```text
for every qubit i:
  Rz(theta_d_i)
global exp(-i H_XY t_d)
```

该结构对应论文中电子结构问题的对称性约束版本。默认 `evolution_time=0.4`，与论文化学模拟段固定 `t_d = 0.4` 的设置一致。

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `n_qubits` | trapped-ion 量子比特数量 |
| `layers` | HEA-TI 层数 `D` |
| `variant` | `"general"` / `"tfim"` 或 `"symmetry"` / `"chemistry"` / `"xy"` |
| `evolution_time` | 所有层共用的全局演化时间，默认 `0.4` |
| `evolution_times` | 可选每层演化时间序列，长度必须等于 `layers` |
| `couplings` | 可选对称 `J_ij` 矩阵；不传则使用 power-law trapped-ion 耦合 |
| `j0` / `alpha` | 默认 power-law 耦合参数 |
| `transverse_field` | TFIM 磁场 `B`；通用版本默认 `1`，对称版本默认 `0` |
| `rotation_first` | `True` 时每层为旋转后接全局演化；`False` 时顺序相反 |
| `parameter_prefix` | 自动生成符号旋转参数名的前缀 |
| `parameters` | 可选 flat 旋转参数序列；不包含全局演化时间 |
| `backend` | 可选绑定到返回 `Circuit` 的后端 |
| `dtype` | dense 哈密顿量和全局 unitary 的复数 dtype |

### 示例：通用 TFIM HEA-TI

```python
import numpy as np
from aicir import NumpyBackend
from aicir.ansatze import hea_ti, hea_ti_parameter_count

n_qubits = 4
layers = 3

circuit = hea_ti(n_qubits, layers=layers, variant="general")
count = hea_ti_parameter_count(n_qubits, layers=layers, variant="general")

theta = np.random.default_rng(1).uniform(-0.1, 0.1, size=count)
bound = circuit.bind_parameters(theta)
unitary = bound.unitary(backend=NumpyBackend())
```

### 示例：化学 / XY HEA-TI

```python
import numpy as np
from aicir.ansatze import hea_ti, hea_ti_parameter_count

n_qubits = 6
layers = 5
count = hea_ti_parameter_count(n_qubits, layers=layers, variant="chemistry")

circuit = hea_ti(
    n_qubits,
    layers=layers,
    variant="chemistry",
    evolution_time=0.4,
    parameters=np.zeros(count),
)
```

### 示例：自定义 trapped-ion 耦合

```python
from aicir.ansatze import hea_ti, power_law_couplings

couplings = power_law_couplings(5, j0=1.0, alpha=1.2)
circuit = hea_ti(
    5,
    layers=2,
    variant="general",
    couplings=couplings,
    evolution_times=[0.2, 0.4],
)
```

---

## 5. UCCSD

`uccsd` 构造 Unitary Coupled Cluster Singles and Doubles ansatz：先按 HF 占据在占据比特施 `pauli_x`，再对每个激发施加精确的费米子激发电路（相邻比特直接用 `single_excitation`/`double_excitation` 门；非相邻比特用 fermionic-SWAP 网络，等价处理 Jordan-Wigner Z-string，无 Trotter 误差）。只吃纯数据，不依赖 `aicir.chemistry`。

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `n_qubits` | 量子比特数 |
| `hf_occupation` | 长度 `n_qubits` 的 0/1 序列，HF 参考态占据 |
| `excitations` | `("single", (i, a))` / `("double", (i, j, a, b))` 的序列 |
| `reps` | 激发层重复次数，默认 `1` |
| `parameter_prefix` | 自动生成符号参数名的前缀，默认 `"theta"` |
| `parameters` | 可选 flat 参数序列（顺序：先 reps 外层、后激发内层） |
| `backend` | 可选绑定到返回 `Circuit` 的后端 |

### 示例：与 `aicir.chemistry` 桥接

```python
from aicir.chemistry import build_molecule
from aicir.ansatze import uccsd

mol = build_molecule("H 0 0 0; H 0 0 0.735", basis="sto-3g", mapping="jordan_wigner")
ansatz = uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)
```

`build_molecule` 在 `jordan_wigner`/`parity`/`bravyi_kitaev` 三种映射下都会填充 `hf_occupation`/`excitations`（`uccsd` 收到 `None` 会抛 `ValueError`），但只有 Jordan-Wigner 下的 `excitations` 是 mapper-correct 的化学 UCCSD 激发；parity/bravyi_kitaev 下 `hf_occupation` 是映射变换后的 bitstring、`excitations` 只是结构索引，不建议直接喂给 `uccsd` 构造物理正确的 ansatz。

---

## 6. 与梯度工具配合

`hea`、`hea_ti`、`uccsd` 返回的都是 `Circuit`。如果参数仍为符号 `Parameter`，需先绑定再调用 `unitary()` 或 `qml.ad()`：

```python
import numpy as np
from aicir import NumpyBackend
from aicir.qml import ad
from aicir.ansatze import hea

backend = NumpyBackend()
circuit = hea(2, layers=1)
theta = np.zeros(len(circuit.parameters))
bound = circuit.bind_parameters(theta)

z0 = np.diag([1, 1, -1, -1]).astype(np.complex64)
grad, value = ad(bound, z0, backend=backend, return_value=True)
```

注意：`hea_ti` 的全局演化当前是 dense `unitary` 门。`qml.ad` 只对 `rx/ry/rz/crx/cry/crz/rzz/rxx` 等结构化旋转门返回梯度，不会对 `unitary` 门中的演化时间求导。若需要优化 `t_d`，可使用黑盒目标配合 `psr` / `fd` / `spsa`，或直接传入新的 `evolution_times` 重建线路。

---

## 7. 与其他子系统的关系

- **`aicir.vqc`**：`BasicVQE`/`run_vqe` 等消费本模块产出的 `Circuit`（或 callable ansatz builder），但不感知本模块内部结构。
- **`aicir.chemistry`**：`build_molecule` 产出的 `MoleculeHamiltonian`（`n_qubits`/`hf_occupation`/`excitations`）桥接 `uccsd`；本模块不 import `aicir.chemistry`。
- **`aicir.qas`** / **`aicir.optimization.qubo`**：直接复用 `hea`/`hea_ti` 作为架构搜索候选或 QAOA 适配的模板来源。

---

## 8. 验证命令

```bash
PYTHONPATH=. pytest tests/ansatze/test_hea_ansatz.py
PYTHONPATH=. pytest tests/ansatze/test_hea_ti_ansatz.py
PYTHONPATH=. pytest tests/ansatze/test_uccsd_ansatz.py
PYTHONPATH=. pytest tests/ansatze/test_excitation_circuits.py
PYTHONPATH=. pytest tests/ansatze/test_uccsd_vqe_integration.py
```

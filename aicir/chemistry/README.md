# aicir.chemistry — 量子化学预置工具

`aicir.chemistry` 当前提供小型、固定设置的分子 qubit Hamiltonian 预置，主要用于 VQE 示例、单元测试和算法原型验证。它不是电子结构计算流水线，不负责从分子构型、基组或 active space 自动生成哈密顿量。

---

## 1. 公共接口一览

| 接口 | 返回 | 用途 |
| --- | --- | --- |
| `MoleculeHamiltonian` | dataclass | 记录固定分子 Hamiltonian 的元数据和 Pauli 项 |
| `available_molecules()` | `tuple[str, ...]` | 返回可用 canonical preset 名称 |
| `get_molecule(name)` | `MoleculeHamiltonian` | 按 canonical 名称读取 preset |
| `molecule_hamiltonian(name)` | `Hamiltonian` | 构造新的 `aicir.operators.Hamiltonian` |
| `molecule_matrix(name, backend=None)` | `np.ndarray` | 返回 dense_matrix 哈密顿量 |
| `iter_molecules(names=None)` | `tuple[MoleculeHamiltonian, ...]` | 批量读取 preset |

---

## 2. 当前预置

每个分子一个模块，**文件名用分子式大小写**（`molecules/H2.py`、`molecules/BeH2.py`…），import 时自注册进 `MOLECULES`。canonical 名称统一小写（`get_molecule("BeH2")` 与 `get_molecule("beh2")` 等价）。系数均来自 PySCF / Qiskit Nature。

| 名称 | 分子 | qubits | basis | mapping | 模块 | 验证 |
| --- | --- | ---: | --- | --- | --- | --- |
| `h2` | H2 | 2 | STO-3G | ParityMapper(two_qubit_reduction) | `molecules/H2.py` | 基态能量 |
| `h2_jw` | H2 | 4 | STO-3G | JordanWignerMapper | `molecules/H2.py` | 基态能量 |
| `h2_tapered` | H2 | 1 | STO-3G | TaperedQubitMapper | `molecules/H2.py` | 基态能量 |
| `lih` | LiH | 4 | STO-3G | JordanWignerMapper（2e/2o） | `molecules/LiH.py` | 基态能量 |
| `h2o` | H2O | 6 | STO-3G | JordanWignerMapper（4e/3o） | `molecules/H2O.py` | 基态能量 |
| `nh3` | NH3 | 12 | STO-3G | JordanWignerMapper（6e/6o） | `molecules/NH3.py` | 结构守卫 |
| `n2` | N2 | 14 | STO-3G | JordanWignerMapper（10e/7o） | `molecules/N2.py` | 结构守卫 |
| `beh2` | BeH2 | 16 | 3-21G | JordanWignerMapper（6e/8o） | `molecules/BeH2.py` | 结构守卫 |

新增分子：在 `molecules/` 下加一个分子式命名的模块，`register_molecule(MoleculeHamiltonian(...))`；小分子（≤6 qubit，dense 构造快）在 `tests/chemistry/test_molecules.py` 的 `_GROUND_ENERGIES` 补一条基态能量守卫，大分子加入 `_STRUCTURAL_ONLY`（≥12 qubit dense 构造过慢/过大，只做结构一致性检查，系数由上游 PySCF/Qiskit Nature 保证）。

---

## 3. 使用示例

### 获取 Hamiltonian 对象

```python
from aicir.chemistry import molecule_hamiltonian

hamiltonian = molecule_hamiltonian("h2")
print(hamiltonian.n_qubits)
```

### 获取 dense_matrix

```python
from aicir import NumpyBackend
from aicir.chemistry import molecule_matrix

matrix = molecule_matrix("h2_jw", backend=NumpyBackend())
print(matrix.shape)
```

### 接入 VQE

```python
import numpy as np
from aicir import NumpyBackend
from aicir.chemistry import molecule_hamiltonian
from aicir.optimizer import GD
from aicir.vqc import BasicVQE
from aicir.vqc.ansatz import hea

hamiltonian = molecule_hamiltonian("h2")
ansatz = hea(hamiltonian.n_qubits, layers=1)

solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    optimizer=GD(max_iters=100, learning_rate=0.1, gradient_method="psr"),
)
result = solver.run(init_params=np.zeros(len(ansatz.parameters)))
print(result.energy)
```

---

## 4. 设计约束

- preset 是固定数据，不做分子积分、fermion-to-qubit mapping 或 active-space 选择。
- `molecule_hamiltonian()` 每次返回新的 `Hamiltonian` 实例，避免调用方修改共享对象。
- `molecule_matrix()` 默认使用 `NumpyBackend`，也可以传入其他兼容 backend。
- 新增分子前必须确认几何、基组、映射方式和 Pauli 系数；不确定的分子不应加入。

---

## 5. 验证命令

```bash
PYTHONPATH=. pytest tests/chemistry/test_molecules.py
```

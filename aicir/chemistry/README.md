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

当前只预置已确认系数来源的 H2 Hamiltonian。未确认的分子不会写入注册表。

| 名称 | 分子 | qubits | basis | mapping | 说明 |
| --- | --- | ---: | --- | --- | --- |
| `h2` | H2 | 2 | STO-3G | ParityMapper(two_qubit_reduction) | 适合最小 VQE 示例 |
| `h2_jw` | H2 | 4 | STO-3G | JordanWignerMapper | 未做对称性约化的 4-qubit 版本 |
| `h2_tapered` | H2 | 1 | STO-3G | TaperedQubitMapper | taper 后的 1-qubit 版本 |

chemistry preset 使用简短 canonical 名称，不保留旧长名称或额外别名。

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
PYTHONPATH=. pytest tests/chemistry/test_molecule.py
```

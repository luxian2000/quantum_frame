# aicir.chemistry — 量子化学工具

`aicir.chemistry` 提供两条互补的路径：

- **固定预置**（本 README 第 2 节）：小型、零额外依赖的分子 qubit Hamiltonian，主要用于 VQE 示例、单元测试和算法原型验证。系数预先算好、写死在 `molecules/` 模块里。
- **`build_molecule` 现算流水线**（第 3 节）：给定任意分子几何、基组、映射方式，现场调用 Qiskit Nature（内部驱动 PySCF）算出 qubit Hamiltonian，并为 Jordan-Wigner、Parity、Bravyi-Kitaev 映射附带 HF 占据/激发元数据。Parity/BK 的激发元数据用于结构桥接，不声明 mapper-correct 化学 UCCSD。该路径需要可选依赖，见第 3 节的 `chem` extra。

两条路径并列：预置覆盖“常用小分子、免安装、免计算”场景；流水线覆盖“任意分子、需要现算”场景。

---

## 1. 公共接口一览

| 接口 | 返回 | 用途 |
| --- | --- | --- |
| `MoleculeHamiltonian` | dataclass | 记录分子 Hamiltonian 的元数据和 Pauli 项（预置与 `build_molecule` 现算结果共用同一 dataclass） |
| `available_molecules()` | `tuple[str, ...]` | 返回可用 canonical preset 名称 |
| `get_molecule(name)` | `MoleculeHamiltonian` | 按 canonical 名称读取 preset |
| `molecule_hamiltonian(name)` | `Hamiltonian` | 构造新的 `aicir.operators.Hamiltonian` |
| `molecule_matrix(name, backend=None)` | `np.ndarray` | 返回 dense_matrix 哈密顿量 |
| `iter_molecules(names=None)` | `tuple[MoleculeHamiltonian, ...]` | 批量读取 preset |
| `build_molecule(geometry, ...)` | `MoleculeHamiltonian` | 现算任意分子的 qubit Hamiltonian（需 `chem` extra），三种 mapper 均附带结构元数据 |

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

`h2`/`h2_jw`/`lih`/`h2o` 四个已做基态能量验证的 preset 已静态补齐 `n_electrons`/`hf_occupation`/`excitations`（用 `build_molecule` 现算同几何/基组/mapping 交叉验证后抄录，见各自模块内的补齐注释），可直接喂给 `uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)`，不必先跑第 3 节的现算流水线。`h2_tapered`（`TaperedQubitMapper` 无对应现算路径）与 12–16 qubit 结构守卫 preset 保持 `None`。

---

## 3. `build_molecule` 电子结构流水线

### 安装

```bash
pip install -e ".[chem]"    # qiskit-nature + pyscf；也并入 [all] / [dev]
```

未安装时调用 `build_molecule` 会抛 `ImportError` 并给出安装提示；其余 `aicir.chemistry` 接口（预置）不受影响，`numpy` 仍是唯一硬依赖。

### 用法

```python
from aicir.chemistry import build_molecule

mol = build_molecule("H 0 0 0; H 0 0 0.735", basis="sto-3g", mapping="jordan_wigner")
hamiltonian = mol.to_hamiltonian()
```

`build_molecule` 内部用 `PySCFDriver` 算分子积分，再用 Qiskit Nature 的 mapper（`jordan_wigner`/`parity`/`bravyi_kitaev`）把费米子哈密顿量映到 qubit 空间，返回一个与预置同构的 `MoleculeHamiltonian`。可选 `active_electrons`/`active_orbitals` 做 active-space 裁剪，`charge`/`spin` 指定电荷与自旋多重度，`two_qubit_reduction` 仅对 `parity` 映射生效。

### 分子元数据

`mapping="jordan_wigner"`（默认）、`"parity"` 和 `"bravyi_kitaev"` 均会填充以下三个字段。JW 下 `hf_occupation` 的 1-bit 数等于电子数，`excitations` 是直接的费米子激发 qubit 索引；Parity/BK 下 `hf_occupation` 是 mapper 变换后的计算基 bitstring，1-bit 数不等于电子数，`excitations` 是经过目标 qubit 数过滤的结构索引，保证 `uccsd(...)` 输入校验与小线路构造可运行，但不宣称对应 mapper 的物理 UCCSD 激发算符。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `n_electrons` | `int` | 总电子数 |
| `hf_occupation` | `tuple[int, ...]`，长度 `n_qubits` | HF 参考态 bitstring（0/1），比特序与 `terms` 一致；Parity/BK 下是变换后 bitstring |
| `excitations` | `tuple[(str, tuple[int, ...]), ...]` | singles + doubles 结构索引；JW 下是物理 UCCSD 激发索引，Parity/BK 下仅保证结构有效 |

### 桥接 UCCSD

现有 `aicir.ansatze.uccsd` 使用 JW-string/fSWAP 激发电路。物理化学意义上的 UCCSD 桥接应使用 `mapping="jordan_wigner"`；Parity/BK 元数据只保证结构可校验，不代表 mapper-aware UCCSD。

```python
from aicir.chemistry import build_molecule
from aicir.ansatze import uccsd

mol = build_molecule("H 0 0 0; H 0 0 0.735", basis="sto-3g", mapping="jordan_wigner")
ansatz = uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)
```

`uccsd` 只吃这三个纯数据字段，与 `aicir.chemistry` 完全解耦（见 `aicir/vqc/README.md` 的 UCCSD 一节）。完整端到端例子（`build_molecule` → `uccsd` → `BasicVQE`，H2 收敛到基态能量）见 `tests/vqc/test_uccsd_vqe_integration.py`。

---

## 4. 使用示例（预置）

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
from aicir.ansatze import hea

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

## 5. 设计约束

- preset 是固定数据，不做分子积分、fermion-to-qubit mapping 或 active-space 选择；这部分交给 `build_molecule`。
- `molecule_hamiltonian()` 每次返回新的 `Hamiltonian` 实例，避免调用方修改共享对象。
- `molecule_matrix()` 默认使用 `NumpyBackend`，也可以传入其他兼容 backend。
- 新增分子前必须确认几何、基组、映射方式和 Pauli 系数；不确定的分子不应加入。
- `build_molecule` 是可选能力（`chem` extra），失败时给出清晰安装提示，不影响核心路径的 `numpy`-only 依赖。

---

## 6. 验证命令

```bash
PYTHONPATH=. pytest tests/chemistry/test_molecules.py
PYTHONPATH=. pytest tests/chemistry/  # 含 build_molecule 流水线测试（缺 qiskit-nature/pyscf 时自动 skip）
PYTHONPATH=. pytest tests/vqc/test_uccsd_vqe_integration.py  # UCCSD + build_molecule + VQE 端到端
```

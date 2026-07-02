# UCCSD 与电子结构流水线设计（Spec）

日期：2026-07-02
状态：已批准设计，待实现

## 目标

给 aicir 补齐 `compare.md` 中标注的两项化学能力，两者一并交付：

1. **电子结构流水线**：让 `aicir/chemistry` 从「只有固定预置」升级为「可现算」——给定分子几何/基组/映射，运行时算出 qubit Hamiltonian（现有 `MoleculeHamiltonian` 形态），并对 Jordan-Wigner 映射额外产出 HF 参考态占据与激发列表。底层调 **Qiskit Nature（内部包 PySCF）**。
2. **UCCSD ansatz**：`aicir/vqc/ansatz/uccsd.py` 新增 `uccsd(...)`，用精确 CNOT-ladder 电路实现单/双激发（文献标准闭式），供 VQE 等变分循环消费。

两者**解耦**：流水线产出数据，UCCSD 消费纯数据（int/tuple），不互相 import。用户按既有 README 里 VQE 示例的方式自行拼接（与 `chemistry` 提供 Hamiltonian、`vqc` 消费 Hamiltonian 的现有模式一致）。

## 非目标（本 spec 不做）

- **非 JW 映射的 UCCSD**：parity/BK 映射仍能算出可用 Hamiltonian（可配硬件高效 ansatz），但**不**产出 `hf_occupation`/`excitations` 元数据（保持 `None`）。UCCSD 的激发门数学是 JW 原生的，parity/BK 的 UCCSD 留待后续 spec。
- **超越 singles+doubles 的激发**（triples 等）。
- **Trotter 化激发电路**：本 spec 走精确闭式 CNOT-ladder（无 Trotter 误差），不做通用 Pauli 指数分解路径。
- **core 里的 `Parameter` 代数**：UCCSD 每个激发对应一个符号 `Parameter`，任何 `θ/2` 之类缩放只在门矩阵 builder 内部（与既有 `_rx`/`_single_excitation` 内部处理 `theta/2` 一致），不在符号 `Parameter` 对象上做算术。
- **`qiskit_io` 的符号参数导入**：UCCSD 直接用 aicir 原生 gate dict 构建，完全绕开 Qiskit→aicir 电路导入这条路（该路径当前只支持已绑定数值参数）。
- **frozen-core 手工实现**：活性空间只用 Qiskit Nature 的 `ActiveSpaceTransformer` 给到的能力，不另造。

## 背景与关键事实

- `MoleculeHamiltonian`（`aicir/chemistry/molecules/_base.py`）是 `@dataclass(frozen=True)`，字段：`name/formula/n_qubits/terms/basis/mapping/geometry/source/description`。128 个测试依赖其现有形态，公共 API 不变（CLAUDE.md）。
- 现有预置系数「来自 PySCF / Qiskit Nature」的 tutorial —— 即本流水线要现算的对象。因此**每个现有预置都是流水线的免费回归 oracle**：用 `build_molecule(...)` 重算应能复现已提交的 `terms`。
- 激发门已存在且 autograd/NPU 安全：`single_excitation(θ,q1,q2)`、`double_excitation(θ,q1,q2,q3,q4)`（`aicir/core/circuit.py`；矩阵在 `aicir/core/gates.py`）。`_single_excitation` 是 `{|01⟩,|10⟩}` 子空间上的 Givens 旋转（`c=cos(θ/2), s=sin(θ/2)`，`+s/-s`）——**仅对相邻 qubit 精确**。非相邻激发需 JW Z-string（CNOT-ladder）包裹。
- 既有 ansatz（`hea`/`hea_ti`）都只吃结构参数（`n_qubits`/`layers`），不认识分子。UCCSD 照此风格：吃 `n_qubits + hf_occupation + excitations`，产出 `Circuit`。
- 可选依赖惯例（`pyproject.toml`）：按能力分 extra，`numpy` 是唯一硬依赖，需 torch/scipy 的测试用 `pytest.importorskip` 跳过。

## 架构与组件

### 组件 1：`MoleculeHamiltonian` 扩展字段

在 frozen dataclass 末尾（`description` 之后）追加三个可选字段，全部默认 `None`：

```python
n_electrons: int | None = None
hf_occupation: tuple[int, ...] | None = None      # 每个 qubit 0/1，aicir 比特索引约定
excitations: tuple[tuple[str, tuple[int, ...]], ...] | None = None
                                                   # ("single", (i, a)) / ("double", (i, j, a, b))
```

- 默认 `None` → 现有预置与 128 个测试零影响（新字段不出现在任何预置构造里，保持 `None`）。
- 仅 `mapping="jordan_wigner"` 时由流水线填充；parity/BK 保持 `None`。
- 激发索引与 `hf_occupation` 均用 aicir 的 qubit 索引约定（与流水线产出的 Hamiltonian 比特序一致）。

### 组件 2：电子结构流水线 `aicir/chemistry/pipeline.py`

`chem` extra 守卫（顶部惰性 import qiskit-nature/pyscf；缺失时给出清晰安装提示）。

```python
def build_molecule(
    geometry,                    # Qiskit Nature/PySCF 接受的几何描述（原子+坐标）
    *,
    basis="sto-3g",
    charge=0,
    spin=0,
    mapping="jordan_wigner",     # "jordan_wigner" | "parity" | "bravyi_kitaev"
    active_electrons=None,       # 传入则用 ActiveSpaceTransformer
    active_orbitals=None,
    two_qubit_reduction=False,   # 仅 parity 有意义
    name="custom",
) -> MoleculeHamiltonian
```

流程：`PySCFDriver(...)` → `ElectronicStructureProblem` → 可选 `ActiveSpaceTransformer` → 选 mapper → `mapper.map(second_q_op)` 得 Pauli 项 → 转成 aicir 的 `PauliTerm` 元组（对齐比特序/相位约定）→ 组装 `MoleculeHamiltonian`。

JW 分支额外：
- `n_electrons` 从（活性空间后的）粒子数取。
- `hf_occupation`：用 Qiskit Nature 的 `HartreeFock` 初态逻辑得到占据 bitstring，映射到 aicir 比特索引。
- `excitations`：用 Qiskit Nature 的 `generate_fermionic_excitations`（singles+doubles）拿到费米子激发对，转成 qubit 索引元组。**不**自己推导轨道排序约定——复用 Qiskit Nature helper，避免 spin-orbital ordering 的隐性 bug。

设计约束：`build_molecule` 每次返回新对象；比特序/相位必须与现有预置的约定逐位对齐（由「重算复现预置」测试守住）。

### 组件 3：UCCSD ansatz `aicir/vqc/ansatz/uccsd.py`

```python
def uccsd(
    n_qubits,
    hf_occupation,               # 序列，长度 == n_qubits，元素 0/1
    excitations,                 # ("single",(i,a)) / ("double",(i,j,a,b)) 的序列
    *,
    reps=1,                      # 激发层重复次数（每次重复用独立参数集）
) -> Circuit
```

- **纯数据入参，零 chemistry import**（镜像 `hea(n_qubits, layers)`）。可脱离 qiskit-nature 独立测试（手写激发列表即可）。
- 校验：`len(hf_occupation)==n_qubits`、元素 ∈{0,1}、激发索引界内、single/double 元数正确；不合法抛清晰 `ValueError`。
- 结构：先按 `hf_occupation` 在占据位施 `pauli_x`（HF 参考态），再对每个激发（× `reps`）分配一个符号 `Parameter`，用**精确激发电路 builder** 施加。
- 参数总数 = `len(excitations) * reps`；参数顺序确定（先 reps 外层、激发内层——实现固定此顺序并在 docstring 写明），供调用方 `bind_parameters` 对齐。

### 组件 4：精确激发电路 builder（UCCSD 内部私有）

- **相邻 qubit**：直接复用既有 `single_excitation`/`double_excitation` gate（已 autograd/NPU 安全）。
- **非相邻 qubit**：文献标准闭式——CNOT-ladder（JW Z-string）+ 受控旋转，等价于 PennyLane `FermionicSingleExcitation`/`FermionicDoubleExcitation`、Qiskit `UCC` 的单激发/双激发电路。双激发电路易出隐性 sign/ordering bug，**必须**由「测试」节的酉矩阵等价测试守住后才可信任。
- builder 只用既有 aicir gate（`cnot`/`cry`/`ry`/`pauli_x` 等 dict），不引入新 gate 类型，保持后端无关与 autograd 安全。

## 数据流

```
build_molecule(geometry, ...) ──> MoleculeHamiltonian
   │  (JW 分支填 n_electrons/hf_occupation/excitations)
   ▼
调用方读取 mol.n_qubits / mol.hf_occupation / mol.excitations
   ▼
uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations, reps=r) ──> Circuit（含符号 Parameter）
   ▼
BasicVQE(mol.to_hamiltonian(), ansatz=circuit, ...).run(init_params=...)
```

## 错误处理

- 缺 `chem` extra：`pipeline` 顶层 import 失败时抛带安装指引的 `ImportError`（`pip install -e ".[chem]"`）。
- `build_molecule` 非 JW 映射：正常返回 Hamiltonian，元数据字段 `None`（不报错）。
- `uccsd` 收到 `None` 的 `hf_occupation`/`excitations`（如来自非 JW 预置）：抛清晰 `ValueError`，提示需 JW 映射的分子元数据。
- `uccsd` 入参不合法（长度/界/元数）：`ValueError`，指明具体问题。

## 测试

激发电路推导有风险，测试是强制正确性闸门，非可选：

1. **`tests/chemistry/test_pipeline.py`**（`importorskip("qiskit_nature")`）：对每个现有小预置（h2/h2_jw/lih/h2o…）用 `build_molecule(...)` 重算，与已提交 `terms` 逐项比对（免费回归 oracle）。JW 分支额外校验 `hf_occupation`/`excitations` 非空且结构合理。
2. **`tests/vqc/test_uccsd_ansatz.py`**：
   - **激发 builder 酉矩阵等价**：新激发电路（相邻+非相邻多种 qubit 配置、多组随机 θ）的 `unitary()` 对比 Qiskit Nature `UCC` 的 ground-truth 电路（`qiskit.quantum_info.Operator`）——CNOT-ladder 推导的正确性闸门。（`importorskip("qiskit_nature")`）
   - **纯结构测试**（不需 qiskit-nature）：手写小激发列表，验证 `uccsd(...)` 参数个数、HF `pauli_x` 位置、`ValueError` 路径。
   - **端到端集成**：H2（4 qubit JW）UCCSD + VQE 收敛到已知基态能量。
3. torch/qiskit-nature 缺失时相关测试干净跳过；纯结构与 `ValueError` 测试始终跑。

## 打包

`pyproject.toml` 新增 extra，`pyscf` 是重编译依赖，**不**并入 `sci`，单列：

```toml
chem = ["qiskit-nature", "pyscf"]
```

并入 `all` 与 `dev`（使 CI/开发环境能跑流水线测试；缺失时 `importorskip` 跳过）。

## 文档

- `aicir/chemistry/README.md`：改写「不是电子结构流水线」的定位段——现在**既有**固定预置**又有** `build_molecule` 现算路径；补 `build_molecule` 用法、`chem` extra、新增元数据字段、JW-only 限制。
- `aicir/vqc/ansatz`（README 或模块 docstring）：补 `uccsd` 用法与「吃纯数据、由 chemistry 元数据桥接」的说明。
- `CHANGELOG.md`：2026-07-02 条目补该特性。

## 影响的现有约定

CLAUDE.md 目前写 `aicir/chemistry` 是「fixed preset qubit Hamiltonians only … NOT an electronic-structure pipeline」。本 spec **有意**推翻该边界（用户明确要求 chemistry 成为流水线）。实现落地时同步更新 CLAUDE.md 对应描述与 `chemistry/README.md`，使文档与新能力一致。

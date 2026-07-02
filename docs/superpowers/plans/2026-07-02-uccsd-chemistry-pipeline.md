# UCCSD 与电子结构流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 aicir 加两项化学能力——`aicir/chemistry` 的现算电子结构流水线（`build_molecule`，底层 Qiskit Nature/PySCF）与 `aicir/vqc/ansatz/uccsd.py` 的 UCCSD ansatz（精确 fermionic-SWAP 网络实现非相邻激发）。

**Architecture:** 流水线与 ansatz **解耦**：流水线产出 `MoleculeHamiltonian`（新增 JW-only 的 `n_electrons`/`hf_occupation`/`excitations` 元数据）；`uccsd()` 只吃纯数据（int/tuple），零 chemistry import。UCCSD 用既有 `single_excitation`/`double_excitation` gate 处理相邻激发，非相邻用 **fSWAP = SWAP·CZ** 网络把目标 orbital 搬到相邻位再施加、再搬回——JW Z-string 由 fSWAP 的 −1 相位精确处理，参数不经任何算术直接流入既有 gate。

**Tech Stack:** Python，numpy（唯一硬依赖），可选 `torch`（autograd/NPU）、`scipy`、`qiskit-nature`+`pyscf`（新 `chem` extra）。测试 `pytest`，缺可选依赖时 `pytest.importorskip` 跳过。

## Global Constraints

- 从 repo 根目录运行，`PYTHONPATH=.`；测试 `PYTHONPATH=. pytest`。
- `numpy` 是唯一硬依赖；`torch`/`scipy`/`qiskit-nature`/`pyscf` 全部可选，相关测试用 `pytest.importorskip(...)` 跳过，**不得**在 core 模块硬 import。
- 注释/docstring/README 用中文，跟随周边风格。
- `MoleculeHamiltonian` 公共 API 不变：新字段一律 `@dataclass` 末尾追加、默认 `None`，现有预置构造不传这些字段。
- UCCSD 只处理 Jordan-Wigner；`excitations`/`hf_occupation` 仅 JW 映射填充，否则 `None`。
- **不**引入新 gate 类型；**不**对符号 `Parameter` 做算术（缩放只在既有 gate 矩阵内部）。
- 激发电路精确闭式（fSWAP 网络，无 Trotter 误差），正确性由「酉矩阵等价」测试守住。
- 每个任务结束提交一次；提交信息末尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

### Task 1: `MoleculeHamiltonian` 扩展元数据字段

**Files:**
- Modify: `aicir/chemistry/molecules/_base.py`（`MoleculeHamiltonian` dataclass）
- Test: `tests/chemistry/test_molecule_metadata.py`（新建）

**Interfaces:**
- Produces: `MoleculeHamiltonian` 新增只读字段 `n_electrons: int | None = None`、`hf_occupation: tuple[int, ...] | None = None`、`excitations: tuple[tuple[str, tuple[int, ...]], ...] | None = None`。

- [ ] **Step 1: 写失败测试**

`tests/chemistry/test_molecule_metadata.py`：

```python
from aicir.chemistry import MoleculeHamiltonian, get_molecule


def test_existing_preset_metadata_defaults_to_none():
    mol = get_molecule("h2")
    assert mol.n_electrons is None
    assert mol.hf_occupation is None
    assert mol.excitations is None


def test_metadata_fields_are_settable():
    mol = MoleculeHamiltonian(
        name="toy",
        formula="H2",
        n_qubits=4,
        terms=((-1.0, "IIII"),),
        basis="STO-3G",
        mapping="JordanWignerMapper",
        geometry="toy",
        source="toy",
        n_electrons=2,
        hf_occupation=(1, 1, 0, 0),
        excitations=(("single", (0, 2)), ("double", (0, 1, 2, 3))),
    )
    assert mol.n_electrons == 2
    assert mol.hf_occupation == (1, 1, 0, 0)
    assert mol.excitations == (("single", (0, 2)), ("double", (0, 1, 2, 3)))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/chemistry/test_molecule_metadata.py -v`
Expected: FAIL（`TypeError: __init__() got an unexpected keyword argument 'n_electrons'`）

- [ ] **Step 3: 加字段**

在 `aicir/chemistry/molecules/_base.py` 的 `MoleculeHamiltonian` 里，`description: str = ""` 之后追加：

```python
    n_electrons: int | None = None
    hf_occupation: tuple[int, ...] | None = None
    excitations: tuple[tuple[str, tuple[int, ...]], ...] | None = None
```

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/chemistry/test_molecule_metadata.py tests/chemistry/test_molecules.py -v`
Expected: PASS（新测试 + 既有分子测试全绿）

- [ ] **Step 5: 提交**

```bash
git add aicir/chemistry/molecules/_base.py tests/chemistry/test_molecule_metadata.py
git commit -m "feat(chemistry): MoleculeHamiltonian 增加 JW 元数据可选字段"
```

---

### Task 2: `chem` extra + 流水线 import 守卫骨架

**Files:**
- Modify: `pyproject.toml`（`[project.optional-dependencies]`）
- Create: `aicir/chemistry/pipeline.py`
- Test: `tests/chemistry/test_pipeline_guard.py`（新建）

**Interfaces:**
- Produces: `aicir.chemistry.pipeline._require_qiskit_nature()`（缺依赖时抛带安装指引的 `ImportError`）；`build_molecule(...)` 函数签名（Task 5 填实现）。

- [ ] **Step 1: 写失败测试**

`tests/chemistry/test_pipeline_guard.py`：

```python
import pytest

from aicir.chemistry import pipeline


def test_require_qiskit_nature_raises_helpful_error_when_missing():
    if pipeline._qiskit_nature_available():
        pytest.skip("qiskit-nature 已安装，跳过缺失分支")
    with pytest.raises(ImportError, match=r"\[chem\]"):
        pipeline._require_qiskit_nature()


def test_build_molecule_is_exported():
    from aicir.chemistry import build_molecule  # noqa: F401
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/chemistry/test_pipeline_guard.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'aicir.chemistry.pipeline'`）

- [ ] **Step 3: 加 extra**

`pyproject.toml`：`[project.optional-dependencies]` 里，`sci = ["scipy"]` 之后加：

```toml
# 电子结构流水线（build_molecule；pyscf 为重编译依赖，单列）
chem = ["qiskit-nature", "pyscf"]
```

并把 `all`、`dev` 改为包含 `chem` 的依赖：

```toml
all = ["torch", "matplotlib", "scipy", "qiskit-nature", "pyscf"]
dev = ["pytest", "torch", "matplotlib", "scipy", "qiskit-nature", "pyscf"]
```

- [ ] **Step 4: 建流水线骨架**

`aicir/chemistry/pipeline.py`：

```python
"""电子结构流水线：给定分子几何/基组/映射，现算 qubit Hamiltonian。

底层调 Qiskit Nature（内部包 PySCF）。属可选能力，需 ``pip install -e ".[chem]"``。
与固定预置并列——预置是快速、零依赖的常用分子；本流水线支持任意分子。
"""

from __future__ import annotations

from .molecules._base import MoleculeHamiltonian

_CHEM_INSTALL_HINT = (
    "电子结构流水线需要 qiskit-nature 与 pyscf；请安装可选依赖："
    'pip install -e ".[chem]"'
)


def _qiskit_nature_available() -> bool:
    try:
        import qiskit_nature  # noqa: F401
    except ImportError:
        return False
    return True


def _require_qiskit_nature():
    if not _qiskit_nature_available():
        raise ImportError(_CHEM_INSTALL_HINT)


def build_molecule(*args, **kwargs) -> MoleculeHamiltonian:
    """现算分子 qubit Hamiltonian（Task 5 实现）。"""

    _require_qiskit_nature()
    raise NotImplementedError("build_molecule 将在 Task 5 实现")
```

在 `aicir/chemistry/__init__.py` 的 import 块加 `from .pipeline import build_molecule`，并把 `"build_molecule"` 加进 `__all__`。

- [ ] **Step 5: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/chemistry/test_pipeline_guard.py -v`
Expected: PASS（无 qiskit-nature 时走 ImportError 分支；有则 skip 第一条，第二条恒通过）

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml aicir/chemistry/pipeline.py aicir/chemistry/__init__.py tests/chemistry/test_pipeline_guard.py
git commit -m "feat(chemistry): 加 chem extra 与 build_molecule 流水线骨架（import 守卫）"
```

---

### Task 3: fSWAP 网络激发电路 builder（正确性核心）

**Files:**
- Create: `aicir/vqc/ansatz/_excitation.py`
- Test: `tests/vqc/test_excitation_circuits.py`（新建）

**Interfaces:**
- Consumes: 既有 gate 工厂 `single_excitation`/`double_excitation`/`swap`/`cz`（`aicir.core.circuit`）。
- Produces:
  - `single_excitation_ops(param, p, q) -> list`：orbital p<q 间单激发的 gate op 列表（相邻直接用 `single_excitation`，非相邻用 fSWAP 网络）。
  - `double_excitation_ops(param, p, q, r, s) -> list`：orbital p<q<r<s 间双激发的 op 列表。
  - `fswap_ops(i, j) -> list`：`fSWAP(i,j) = CZ·SWAP` 的 op 列表。

> **实现要点（fSWAP 网络）：** `fSWAP = SWAP·CZ` 是 JW 一致的费米子交换（在 |11⟩ 上带 −1，正好等于 JW Z-string 效应）。非相邻激发 = 用 fSWAP 把远端 orbital 逐格搬到与近端相邻、施加既有相邻激发 gate、再逐格搬回。参数 `param` 原样传入既有 gate，**不做任何算术**。**双激发把 4 个 orbital 搬到相邻后喂给 `double_excitation` gate 的 qubit 顺序，是本任务唯一有推导风险处——由下面的酉矩阵等价测试守住；测试失败时用 systematic-debugging 迭代该顺序。**

- [ ] **Step 1: 写失败测试（相邻 = 既有 gate；非相邻单激发 vs expm 生成元）**

`tests/vqc/test_excitation_circuits.py`：

```python
import numpy as np
import pytest

from aicir import Circuit, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.vqc.ansatz._excitation import (
    double_excitation_ops,
    fswap_ops,
    single_excitation_ops,
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron(paulis):
    out = np.array([[1]], dtype=complex)
    for p in paulis:
        out = np.kron(out, _PAULI[p])
    return out


def _circuit_unitary(ops, n_qubits):
    circ = Circuit(*ops, n_qubits=n_qubits)
    return np.asarray(NumpyBackend().to_numpy(circ.unitary()))


def _single_excitation_generator(n, p, q):
    # JW: G = (1/2)(X_p Z...Z Y_q - Y_p Z...Z X_q)，interior 为 Z 串
    def string(a_gate, b_gate):
        labels = ["I"] * n
        labels[p], labels[q] = a_gate, b_gate
        for k in range(p + 1, q):
            labels[k] = "Z"
        return _kron(labels)

    return 0.5 * (string("X", "Y") - string("Y", "X"))


def test_fswap_is_swap_times_cz():
    theta = 0.0  # 无关，fswap 无参
    got = _circuit_unitary(fswap_ops(0, 1), 2)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    assert np.allclose(got, swap @ cz, atol=1e-9)


def test_adjacent_single_excitation_matches_existing_gate():
    theta = 0.7
    ops = single_excitation_ops(theta, 0, 1)
    got = _circuit_unitary(ops, 2)
    ref = np.asarray(
        NumpyBackend().to_numpy(
            gate_to_matrix(
                {"type": "single_excitation", "qubit_1": 0, "qubit_2": 1, "parameter": theta},
                2,
                NumpyBackend(),
            )
        )
    )
    assert np.allclose(got, ref, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.1, -0.8])
def test_nonadjacent_single_excitation_matches_expm_generator(theta):
    from scipy.linalg import expm

    n, p, q = 4, 0, 3
    got = _circuit_unitary(single_excitation_ops(theta, p, q), n)
    gen = _single_excitation_generator(n, p, q)
    ref = expm(theta * gen)
    # 允许全局相位差
    phase = np.vdot(ref.reshape(-1), got.reshape(-1))
    phase /= abs(phase)
    assert np.allclose(got, phase * ref, atol=1e-8)
```

> 注：`_single_excitation_generator` 的符号约定与既有 `single_excitation` gate 对齐（`c=cos(θ/2)`、`+s/-s` Givens）。若非相邻测试因整体符号不符而挂，优先核对 fSWAP 搬运方向而非改生成元。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/vqc/test_excitation_circuits.py -v`
Expected: FAIL（`ModuleNotFoundError: aicir.vqc.ansatz._excitation`）

- [ ] **Step 3: 实现 fSWAP 网络 builder**

`aicir/vqc/ansatz/_excitation.py`：

```python
"""UCCSD 激发的精确电路 builder（fermionic-SWAP 网络）。

非相邻 orbital 间的费米子激发需处理 JW Z-string。用 fSWAP = CZ·SWAP（JW 一致的
费米子交换，在 |11⟩ 上带 −1）把远端 orbital 逐格搬到与近端相邻，施加既有相邻激发
gate，再逐格搬回。参数原样流入既有 gate，不做任何算术；只用既有 gate，不引入新类型。
"""

from __future__ import annotations

from ...core.circuit import cz, double_excitation, single_excitation, swap


def fswap_ops(i: int, j: int) -> list:
    """fSWAP(i, j) = CZ·SWAP：JW 一致的费米子交换。"""

    return [cz(j, [i]), swap(i, j)]


def _bring_adjacent(op_list: list, high: int, target: int) -> int:
    """把 qubit ``high`` 逐格 fswap 到 ``target``（target < high），返回它现在的位置。"""

    pos = high
    while pos > target:
        op_list.extend(fswap_ops(pos - 1, pos))
        pos -= 1
    return pos


def _undo_adjacent(op_list: list, low: int, high: int) -> None:
    """还原 ``_bring_adjacent(op_list, high, low)`` 的搬运。"""

    for pos in range(low, high):
        op_list.extend(fswap_ops(pos, pos + 1))


def single_excitation_ops(param, p: int, q: int) -> list:
    """orbital p<q 间单激发的 op 列表。"""

    if p >= q:
        raise ValueError(f"single excitation 需 p<q，得到 p={p}, q={q}")
    if q == p + 1:
        return [single_excitation(param, p, q)]
    ops: list = []
    _bring_adjacent(ops, q, p + 1)          # q 搬到 p+1
    ops.append(single_excitation(param, p, p + 1))
    _undo_adjacent(ops, p + 1, q)
    return ops


def double_excitation_ops(param, p: int, q: int, r: int, s: int) -> list:
    """orbital p<q<r<s 间双激发的 op 列表。

    把四个 orbital 用 fSWAP 网络聚到相邻位 (p, p+1, p+2, p+3)，施加既有
    ``double_excitation`` gate，再还原。喂给 gate 的 qubit 顺序由酉矩阵等价测试守住。
    """

    if not (p < q < r < s):
        raise ValueError(f"double excitation 需 p<q<r<s，得到 {(p, q, r, s)}")
    ops: list = []
    # 依次把 q、r、s 聚到 p+1, p+2, p+3（从最靠近的开始，避免互相错位）
    _bring_adjacent(ops, q, p + 1)
    _bring_adjacent(ops, r, p + 2)
    _bring_adjacent(ops, s, p + 3)
    ops.append(double_excitation(param, p, p + 1, p + 2, p + 3))
    _undo_adjacent(ops, p + 3, s)
    _undo_adjacent(ops, p + 2, r)
    _undo_adjacent(ops, p + 1, q)
    return ops
```

- [ ] **Step 4: 跑测试确认通过（scipy 分支需 scipy）**

Run: `PYTHONPATH=. pytest tests/vqc/test_excitation_circuits.py -v`
Expected: PASS（fSWAP、相邻单激发恒过；非相邻单激发对 expm 生成元通过——若挂，用 systematic-debugging 核对 `_bring_adjacent` 方向/顺序）

- [ ] **Step 5: 加非相邻双激发的独立 oracle 测试（Qiskit UCC）**

在 `tests/vqc/test_excitation_circuits.py` 末尾追加（独立权威，缺依赖跳过）：

```python
def test_nonadjacent_double_excitation_matches_qiskit_ucc():
    pytest.importorskip("qiskit_nature")
    from qiskit.quantum_info import Operator
    from qiskit_nature.second_q.circuit.library import UCC
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    # 2 空间轨道、2 电子 → 4 qubit JW；取其唯一 double 激发做对照
    ucc = UCC(
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        excitations="d",
        qubit_mapper=JordanWignerMapper(),
    )
    theta = 0.37
    bound = ucc.assign_parameters([theta])
    ref = Operator(bound).data

    # aicir 对应的 double 激发 orbital 索引（与 UCC 的 JW 约定对齐）
    ops = double_excitation_ops(theta, 0, 1, 2, 3)
    got = _circuit_unitary(ops, 4)
    phase = np.vdot(ref.reshape(-1), got.reshape(-1))
    phase /= abs(phase)
    assert np.allclose(got, phase * ref, atol=1e-7)
```

> 若该测试挂，double 激发喂给 gate 的 qubit 顺序（`p, p+1, p+2, p+3` 的排列）是唯一待迭代量——用 systematic-debugging 针对该 oracle 调整 `double_excitation(...)` 的四个索引顺序，直到全局相位内相等。

- [ ] **Step 6: 跑全套激发测试**

Run: `PYTHONPATH=. pytest tests/vqc/test_excitation_circuits.py -v`
Expected: PASS（有 qiskit-nature 时含 UCC 对照；无则该条 skip）

- [ ] **Step 7: 提交**

```bash
git add aicir/vqc/ansatz/_excitation.py tests/vqc/test_excitation_circuits.py
git commit -m "feat(ansatz): fSWAP 网络实现非相邻单/双激发精确电路"
```

---

### Task 4: `uccsd()` 公共 ansatz

**Files:**
- Create: `aicir/vqc/ansatz/uccsd.py`
- Modify: `aicir/vqc/ansatz/__init__.py`（导出 `uccsd`、`uccsd_parameter_count`）
- Test: `tests/vqc/test_uccsd_ansatz.py`（新建）

**Interfaces:**
- Consumes: `single_excitation_ops`/`double_excitation_ops`（Task 3）；`pauli_x`、`Circuit`、`Parameter`（`aicir.core.circuit`）；`hea.py` 的 `_ParameterStream`/`_flatten_parameters` 模式（在本模块内自建等价的小工具，避免跨模块私有依赖）。
- Produces:
  - `uccsd(n_qubits, hf_occupation, excitations, *, reps=1, parameter_prefix="theta", parameters=None, backend=None) -> Circuit`
  - `uccsd_parameter_count(excitations, *, reps=1) -> int`

- [ ] **Step 1: 写失败测试（纯结构 + ValueError，无需 qiskit）**

`tests/vqc/test_uccsd_ansatz.py`：

```python
import numpy as np
import pytest

from aicir import Circuit
from aicir.vqc.ansatz import uccsd, uccsd_parameter_count

_EXC = (("single", (0, 2)), ("single", (1, 3)), ("double", (0, 1, 2, 3)))


def test_parameter_count():
    assert uccsd_parameter_count(_EXC) == 3
    assert uccsd_parameter_count(_EXC, reps=2) == 6


def test_returns_circuit_with_symbolic_parameters():
    circ = uccsd(4, (1, 1, 0, 0), _EXC)
    assert isinstance(circ, Circuit)
    assert circ.n_qubits == 4
    assert len(circ.parameters) == 3


def test_hf_reference_applies_x_on_occupied_qubits():
    # 无激发 → 电路只含 HF 的 pauli_x，落在占据位
    circ = uccsd(4, (1, 1, 0, 0), ())
    x_targets = sorted(
        g.get("target_qubit") for g in circ.gates if g.get("type") == "pauli_x"
    )
    assert x_targets == [0, 1]


def test_bound_parameter_count_matches():
    circ = uccsd(4, (1, 1, 0, 0), _EXC, parameters=np.zeros(3))
    assert len(circ.parameters) == 0  # 全绑定为数值


def test_rejects_none_metadata():
    with pytest.raises(ValueError, match="hf_occupation"):
        uccsd(4, None, _EXC)
    with pytest.raises(ValueError, match="excitations"):
        uccsd(4, (1, 1, 0, 0), None)


def test_rejects_bad_occupation_length():
    with pytest.raises(ValueError, match="hf_occupation"):
        uccsd(4, (1, 1, 0), _EXC)


def test_rejects_out_of_range_excitation():
    with pytest.raises(ValueError, match="越界|out of range|索引"):
        uccsd(4, (1, 1, 0, 0), (("single", (0, 9)),))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/vqc/test_uccsd_ansatz.py -v`
Expected: FAIL（`ImportError: cannot import name 'uccsd'`）

- [ ] **Step 3: 实现 `uccsd`**

`aicir/vqc/ansatz/uccsd.py`：

```python
"""UCCSD ansatz 模板。

吃纯数据（n_qubits + HF 占据 + 激发列表），产出参数化 ``Circuit``，与 chemistry
子包解耦（镜像 ``hea`` 只吃结构参数的风格）。非相邻激发的精确电路见 ``_excitation``。
调用方通常从 ``MoleculeHamiltonian`` 的 JW 元数据桥接：
``uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...core.circuit import Circuit, Parameter, pauli_x
from ._excitation import double_excitation_ops, single_excitation_ops


def _flatten(parameters: Sequence[Any] | None) -> list[Any] | None:
    if parameters is None:
        return None
    if isinstance(parameters, (str, bytes)):
        raise TypeError("parameters 必须是非字符串序列")
    if hasattr(parameters, "reshape"):
        flat = parameters.reshape(-1)
        return [flat[i] for i in range(len(flat))]
    return list(parameters)


def uccsd_parameter_count(excitations, *, reps: int = 1) -> int:
    """UCCSD 参数个数 = 激发数 × reps。"""

    if reps < 1:
        raise ValueError(f"reps 必须 ≥1，得到 {reps}")
    return len(tuple(excitations)) * int(reps)


def _validate(n_qubits: int, hf_occupation, excitations) -> None:
    if hf_occupation is None:
        raise ValueError("hf_occupation 为 None；UCCSD 需 Jordan-Wigner 映射的分子元数据")
    if excitations is None:
        raise ValueError("excitations 为 None；UCCSD 需 Jordan-Wigner 映射的分子元数据")
    if len(hf_occupation) != n_qubits:
        raise ValueError(
            f"hf_occupation 长度 {len(hf_occupation)} 与 n_qubits={n_qubits} 不符"
        )
    if any(bit not in (0, 1) for bit in hf_occupation):
        raise ValueError("hf_occupation 元素必须为 0/1")
    for kind, idx in excitations:
        arity = {"single": 2, "double": 4}.get(kind)
        if arity is None:
            raise ValueError(f"未知激发类型 {kind!r}")
        if len(idx) != arity:
            raise ValueError(f"{kind} 激发需 {arity} 个索引，得到 {idx!r}")
        if any(not (0 <= i < n_qubits) for i in idx):
            raise ValueError(f"激发索引越界（out of range）：{idx!r}, n_qubits={n_qubits}")


def uccsd(
    n_qubits: int,
    hf_occupation,
    excitations,
    *,
    reps: int = 1,
    parameter_prefix: str = "theta",
    parameters: Sequence[Any] | None = None,
    backend: Any = None,
) -> Circuit:
    """构建 UCCSD ansatz 电路。

    Args:
        n_qubits: 量子比特数。
        hf_occupation: 长度 == n_qubits 的 0/1 序列，HF 参考态占据。
        excitations: ``("single",(i,a))`` / ``("double",(i,j,a,b))`` 的序列。
        reps: 激发层重复次数（每次重复用独立参数集）。
        parameter_prefix: 生成符号参数的前缀。
        parameters: 可选的扁平参数值序列（缺省则生成符号 ``Parameter``）；
            顺序为 **先 reps 外层、后激发内层**：``[rep0_exc0, rep0_exc1, ..., rep1_exc0, ...]``。
        backend: 可选，绑定到返回 ``Circuit`` 的后端。

    Returns:
        参数化 ``Circuit``（未提供 parameters 时含符号参数）。
    """

    n_qubits = int(n_qubits)
    if reps < 1:
        raise ValueError(f"reps 必须 ≥1，得到 {reps}")
    _validate(n_qubits, hf_occupation, excitations)
    excitations = tuple(excitations)

    values = _flatten(parameters)
    total = uccsd_parameter_count(excitations, reps=reps)
    if values is not None and len(values) != total:
        raise ValueError(f"需要 {total} 个参数值，得到 {len(values)}")

    gates: list[Any] = []
    # HF 参考态：占据位施 pauli_x
    for qubit, bit in enumerate(hf_occupation):
        if bit == 1:
            gates.append(pauli_x(qubit))

    index = 0
    for rep in range(reps):
        for kind, idx in excitations:
            param = values[index] if values is not None else Parameter(f"{parameter_prefix}_{index}")
            index += 1
            if kind == "single":
                p, q = sorted(idx)
                gates.extend(single_excitation_ops(param, p, q))
            else:
                p, q, r, s = sorted(idx)
                gates.extend(double_excitation_ops(param, p, q, r, s))

    return Circuit(*gates, n_qubits=n_qubits, backend=backend)
```

在 `aicir/vqc/ansatz/__init__.py` 导出 `uccsd`、`uccsd_parameter_count`（跟随该文件既有导出风格）。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/vqc/test_uccsd_ansatz.py -v`
Expected: PASS（全部纯结构 + ValueError 测试通过）

> 注：`pauli_x(qubit)` 返回 `Operation`，`circuit.gates` 存 dict（`{"type": "pauli_x", "target_qubit": qubit}`）——`test_hf_reference_applies_x_on_occupied_qubits` 读 dict 字段，与既有约定一致。若字段名不符，先 `python -c "from aicir import Circuit, pauli_x; print(Circuit(pauli_x(0), n_qubits=1).gates)"` 核对键名再调整断言。

- [ ] **Step 5: 提交**

```bash
git add aicir/vqc/ansatz/uccsd.py aicir/vqc/ansatz/__init__.py tests/vqc/test_uccsd_ansatz.py
git commit -m "feat(ansatz): 新增 uccsd() UCCSD 模板（吃纯数据，与 chemistry 解耦）"
```

---

### Task 5: `build_molecule` 流水线实现（JW 元数据）

**Files:**
- Modify: `aicir/chemistry/pipeline.py`（填实 `build_molecule`）
- Test: `tests/chemistry/test_pipeline.py`（新建）

**Interfaces:**
- Consumes: `MoleculeHamiltonian`（Task 1 的新字段）；qiskit-nature（`PySCFDriver`/`ElectronicStructureProblem`/`ActiveSpaceTransformer`/mappers/`generate_fermionic_excitations`/`HartreeFock`）。
- Produces: `build_molecule(geometry, *, basis="sto-3g", charge=0, spin=0, mapping="jordan_wigner", active_electrons=None, active_orbitals=None, two_qubit_reduction=False, name="custom") -> MoleculeHamiltonian`。

> 本任务重度依赖 qiskit-nature；测试全部 `pytest.importorskip("qiskit_nature")`，缺依赖时干净跳过（与 NPU 测试同策略）。**免费 oracle：** 用 `build_molecule` 重算现有预置，与已提交 `terms` 逐项比对。

- [ ] **Step 1: 写失败测试（importorskip）**

`tests/chemistry/test_pipeline.py`：

```python
import pytest

pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from aicir.chemistry import build_molecule, get_molecule

_H2_GEOMETRY = "H 0 0 0; H 0 0 0.735"  # PySCF 默认 H2 几何，与预置一致


def _term_map(mol):
    return {pauli: complex(coeff) for coeff, pauli in mol.terms}


def test_h2_jw_reproduces_preset_terms():
    preset = get_molecule("h2_jw")
    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner", name="h2_jw")
    assert built.n_qubits == preset.n_qubits
    pm, bm = _term_map(preset), _term_map(built)
    assert set(pm) == set(bm)
    for pauli in pm:
        assert abs(pm[pauli] - bm[pauli]) < 1e-4


def test_jw_populates_metadata():
    built = build_molecule(_H2_GEOMETRY, basis="sto-3g", mapping="jordan_wigner")
    assert built.n_electrons == 2
    assert built.hf_occupation is not None
    assert len(built.hf_occupation) == built.n_qubits
    assert sum(built.hf_occupation) == 2
    assert built.excitations is not None
    assert all(kind in ("single", "double") for kind, _ in built.excitations)


def test_parity_mapping_leaves_metadata_none():
    built = build_molecule(
        _H2_GEOMETRY, basis="sto-3g", mapping="parity", two_qubit_reduction=True
    )
    assert built.hf_occupation is None
    assert built.excitations is None
    assert built.terms  # Hamiltonian 仍可用
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/chemistry/test_pipeline.py -v`
Expected: FAIL（`NotImplementedError: build_molecule 将在 Task 5 实现`）或 skip（无 qiskit-nature）

- [ ] **Step 3: 实现 `build_molecule`**

替换 `aicir/chemistry/pipeline.py` 里 `build_molecule` 的桩，实现：

```python
def build_molecule(
    geometry,
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    mapping: str = "jordan_wigner",
    active_electrons=None,
    active_orbitals=None,
    two_qubit_reduction: bool = False,
    name: str = "custom",
) -> MoleculeHamiltonian:
    """给定分子几何/基组/映射，现算 qubit Hamiltonian。

    仅 ``mapping="jordan_wigner"`` 填充 ``n_electrons``/``hf_occupation``/
    ``excitations``；``parity``/``bravyi_kitaev`` 仍返回可用 Hamiltonian，但元数据为 None。
    """

    _require_qiskit_nature()
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    driver = PySCFDriver(atom=geometry, basis=basis, charge=charge, spin=spin)
    problem = driver.run()
    if active_electrons is not None and active_orbitals is not None:
        problem = ActiveSpaceTransformer(active_electrons, active_orbitals).transform(problem)

    second_q_op = problem.hamiltonian.second_q_op()

    mapping_key = mapping.lower()
    if mapping_key in ("jordan_wigner", "jw"):
        mapper = JordanWignerMapper()
    elif mapping_key == "parity":
        mapper = ParityMapper(num_particles=problem.num_particles) if two_qubit_reduction else ParityMapper()
    elif mapping_key in ("bravyi_kitaev", "bk"):
        mapper = BravyiKitaevMapper()
    else:
        raise ValueError(f"未知 mapping: {mapping!r}")

    qubit_op = mapper.map(second_q_op)
    terms = _sparse_pauli_to_terms(qubit_op)   # 转 aicir PauliTerm，对齐比特序
    n_qubits = qubit_op.num_qubits

    n_electrons = hf_occupation = excitations = None
    if mapping_key in ("jordan_wigner", "jw"):
        n_electrons = sum(problem.num_particles)
        hf_occupation = _jw_hf_occupation(problem, n_qubits)
        excitations = _jw_excitations(problem)

    return MoleculeHamiltonian(
        name=name,
        formula=name.upper(),
        n_qubits=n_qubits,
        terms=terms,
        basis=basis.upper(),
        mapping=mapping,
        geometry=str(geometry),
        source="aicir.chemistry.build_molecule (Qiskit Nature/PySCF)",
        n_electrons=n_electrons,
        hf_occupation=hf_occupation,
        excitations=excitations,
    )
```

并实现三个私有 helper（同文件）：

```python
def _sparse_pauli_to_terms(qubit_op):
    """Qiskit SparsePauliOp → aicir PauliTerm 元组，比特序对齐预置约定。

    aicir 约定 qubit 0 为 Pauli 串最左字符；Qiskit 标签最右为 qubit 0，故翻转。
    """

    out = []
    for label, coeff in sorted(qubit_op.to_list()):
        out.append((complex(coeff), label[::-1]))
    return tuple(out)


def _jw_hf_occupation(problem, n_qubits):
    """JW 下 HF 占据 bitstring（aicir 比特序：qubit 0 最左）。"""

    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, JordanWignerMapper())
    # HartreeFock 电路对占据轨道施 X；从其 bitstring 读占据，翻转到 aicir 比特序
    occ = _statevector_occupation(hf, n_qubits)
    return tuple(occ)


def _jw_excitations(problem):
    """singles+doubles 费米子激发 → aicir qubit 索引元组。"""

    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    out = []
    for order, kind in ((1, "single"), (2, "double")):
        raw = generate_fermionic_excitations(order, problem.num_spatial_orbitals, problem.num_particles)
        for exc in raw:
            occ, vir = exc
            idx = tuple(int(i) for i in (*occ, *vir))
            out.append((kind, idx))
    return tuple(out)
```

> `_statevector_occupation(hf, n_qubits)`：把 `HartreeFock` 电路作用在 |0…0⟩ 上读出占据位（可用 aicir `Circuit`/`StateVector` 或直接读 HartreeFock 的 `_bitstr`）。实现时选最直接可靠的一种；`test_jw_populates_metadata` 只要求 `sum==n_electrons` 与长度正确，`test_h2_jw_reproduces_preset_terms` 守住比特序整体一致性。若 `generate_fermionic_excitations` 的返回结构随 qiskit-nature 版本不同，用 `python -c` 打印其结构核对后再定 `idx` 解包。

- [ ] **Step 4: 跑测试确认通过（需 qiskit-nature+pyscf）**

Run: `PYTHONPATH=. pytest tests/chemistry/test_pipeline.py -v`
Expected: PASS（装了 chem extra 时）；否则 skip。装了但比特序/激发解包不符时用 systematic-debugging 对着 `test_h2_jw_reproduces_preset_terms` 调 `_sparse_pauli_to_terms`/`_jw_*`。

- [ ] **Step 5: 提交**

```bash
git add aicir/chemistry/pipeline.py tests/chemistry/test_pipeline.py
git commit -m "feat(chemistry): 实现 build_molecule（Qiskit Nature/PySCF；JW 填 HF/激发元数据）"
```

---

### Task 6: 端到端集成 + 文档

**Files:**
- Test: `tests/vqc/test_uccsd_vqe_integration.py`（新建）
- Modify: `aicir/chemistry/README.md`、`aicir/vqc/ansatz`（README 或 `uccsd.py` docstring 已含用法，补 README 段落）、`CHANGELOG.md`、`CLAUDE.md`

**Interfaces:**
- Consumes: `uccsd`（Task 4）、`molecule_hamiltonian`/`get_molecule`（既有）、`BasicVQE`（既有）。

- [ ] **Step 1: 写端到端集成测试**

`tests/vqc/test_uccsd_vqe_integration.py`：

```python
import numpy as np
import pytest

from aicir import NumpyBackend
from aicir.chemistry import molecule_hamiltonian
from aicir.optimizer import GD
from aicir.vqc import BasicVQE
from aicir.vqc.ansatz import uccsd

# H2 STO-3G JW 已知基态能量（与 tests/chemistry 的守卫一致）
_H2_GROUND = -1.1372838
# H2 (4q JW) HF 占据与激发，手写（不依赖 pipeline，保证本测试始终可跑）
_H2_HF = (1, 1, 0, 0)
_H2_EXC = (("single", (0, 2)), ("single", (1, 3)), ("double", (0, 1, 2, 3)))


def test_uccsd_vqe_reaches_h2_ground_energy():
    hamiltonian = molecule_hamiltonian("h2_jw")
    n_params = len(uccsd(4, _H2_HF, _H2_EXC).parameters)

    def ansatz_factory(params):
        return uccsd(4, _H2_HF, _H2_EXC, parameters=params)

    solver = BasicVQE(
        hamiltonian,
        ansatz=ansatz_factory,
        backend=NumpyBackend(),
        optimizer=GD(max_iters=200, learning_rate=0.3, gradient_method="psr"),
    )
    result = solver.run(init_params=np.zeros(n_params))
    assert result.energy < _H2_GROUND + 5e-3
```

> 注：`BasicVQE` 的 ansatz 入参形态（可调用工厂 vs 模板 Circuit）以 `aicir/vqc` 现有用法为准——实现前先看 `aicir/vqc/README.md` 与既有 VQE 测试，按其约定传 ansatz（若 `BasicVQE` 收模板 `Circuit`，则传 `uccsd(4,_H2_HF,_H2_EXC)` 并让 solver 内部 bind）。以既有 API 为准调整本测试的 ansatz 传法。

- [ ] **Step 2: 跑测试确认失败/通过**

Run: `PYTHONPATH=. pytest tests/vqc/test_uccsd_vqe_integration.py -v`
Expected: 先按既有 `BasicVQE` API 修正 ansatz 传法，再 PASS（能量收敛到 H2 基态附近）

- [ ] **Step 3: 更新 `aicir/chemistry/README.md`**

改写开头「它不是电子结构计算流水线」定位段，改为「既提供固定预置，也提供 `build_molecule` 现算流水线（需 `[chem]` extra）」。补一节 `build_molecule` 用法、`chem` extra 安装、新增元数据字段（`n_electrons`/`hf_occupation`/`excitations`，JW-only）、以及桥接 UCCSD 的示例：

```python
from aicir.chemistry import build_molecule
from aicir.vqc.ansatz import uccsd

mol = build_molecule("H 0 0 0; H 0 0 0.735", basis="sto-3g", mapping="jordan_wigner")
ansatz = uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)
```

- [ ] **Step 4: 更新 ansatz 文档与 CHANGELOG/CLAUDE.md**

- `aicir/vqc/ansatz` README（或在其 README 补 `uccsd` 段）：说明 `uccsd` 吃纯数据、与 chemistry 解耦、fSWAP 网络处理非相邻激发、参数顺序（先 reps 后激发）。
- `CHANGELOG.md`：2026-07-02 `### Added` 下补两条——`aicir.chemistry.build_molecule` 电子结构流水线（`chem` extra）与 `aicir.vqc.ansatz.uccsd` UCCSD 模板。
- `CLAUDE.md`：把 `aicir/chemistry` 段「NOT an electronic-structure pipeline」改为「预置 + 可选 `build_molecule` 现算流水线（`chem` extra，Qiskit Nature/PySCF；JW 填 HF/激发元数据）」；在 `aicir/vqc/` 段的 ansatz 列表补 `uccsd`。

- [ ] **Step 5: 跑全量回归**

Run: `PYTHONPATH=. pytest -q`
Expected: 全绿（新测试通过；qiskit-nature 缺失的流水线测试 skip）

- [ ] **Step 6: 提交**

```bash
git add tests/vqc/test_uccsd_vqe_integration.py aicir/chemistry/README.md aicir/vqc/ansatz CHANGELOG.md CLAUDE.md
git commit -m "feat(chemistry,ansatz): UCCSD+VQE 端到端集成与文档；chemistry 升级为流水线"
```

---

## Self-Review

**Spec coverage：**
- 组件 1（MoleculeHamiltonian 扩展字段）→ Task 1 ✅
- 组件 2（build_molecule 流水线）→ Task 2（骨架/extra）+ Task 5（实现）✅
- 组件 3（uccsd ansatz，吃纯数据）→ Task 4 ✅
- 组件 4（精确激发 builder，非相邻 fSWAP）→ Task 3 ✅
- 测试节（预置复现 oracle / 酉矩阵等价 / 结构+ValueError / H2 端到端）→ Task 5 / Task 3 / Task 4 / Task 6 ✅
- 打包（chem extra，并入 all/dev）→ Task 2 ✅
- 文档（chemistry README / ansatz / CHANGELOG）+ CLAUDE.md 边界更新 → Task 6 ✅
- 非目标（parity/BK 的 UCCSD、triples、Trotter、Parameter 代数、qiskit_io 符号导入）→ 计划未触及，符合 ✅

**Placeholder scan：** 各 step 均含实际代码/命令/预期。Task 3 双激发 qubit 顺序、Task 5 版本相关解包、Task 6 BasicVQE ansatz 传法三处标注为「对着 oracle/既有 API 迭代」——均给了具体 oracle 与调试指引，非空泛占位。

**Type consistency：** `single_excitation_ops`/`double_excitation_ops`/`fswap_ops`（Task 3）↔ Task 4 消费一致；`uccsd`/`uccsd_parameter_count` 签名（Task 4）↔ Task 6 调用一致；`MoleculeHamiltonian` 新字段名（Task 1）↔ Task 5 构造一致；`build_molecule` 签名（Task 2 骨架）↔ Task 5 实现一致。

**风险提示：** 唯一有推导风险的是 Task 3 非相邻**双**激发喂给 `double_excitation` gate 的 4-qubit 顺序——已用独立 oracle（Qiskit UCC 酉矩阵等价，Step 5）守住，失败时按 systematic-debugging 迭代该顺序。单激发有 expm 生成元 oracle，fSWAP 与相邻情形有确定性 oracle，均无推导风险。

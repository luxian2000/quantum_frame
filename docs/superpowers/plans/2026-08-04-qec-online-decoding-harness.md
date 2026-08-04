# QEC 在线实时解码实验平台 M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `aicir.qec` 子系统 M1 骨架——一个面向**新型在线实时纠错/解码算法**的实验平台，码 / syndrome 提取调度 / 在线解码器三处均可插拔。

**Architecture:** `StabilizerCode` 用 GF(2) 辛表示做纯代数核心（无线路、无后端）；`Schedule` 把码编译成逐轮线路 + `DetectorLayout`；运行器**逐轮交错「模拟 ↔ 解码」**，复用现有 `run_trajectory`（它已支持 `init_state` 串联、`measure→creg`、`reset`），不新增任何模拟器机制。解码器只经 `DetectorLayout` + detection event 流获得信息，因果性由「未来轮次尚未被模拟」这一结构事实保证。

**Tech Stack:** Python，numpy（唯一硬依赖）。纯态 statevector + 逐 shot 随机 Pauli 采样，`NumpyBackend` 即可，不需要 torch。

**Spec:** `docs/superpowers/specs/2026-08-04-qec-online-decoding-harness-design.md`（commit `bb94ef6`）

## Global Constraints

- 仓库根目录运行；测试 `PYTHONPATH=. pytest`。
- `numpy` 是 `aicir.qec` 的**唯一**依赖。M1 不得 import `torch` / `scipy` / `matplotlib` / `stim` / `pymatching`。
- 注释 / docstring / README 用**中文**，跟随周边风格。
- **提交信息中绝对不得出现 `Co-Authored-By: Claude ...` 或任何 Anthropic/Claude 署名**（仓库 CLAUDE.md 明确要求；本目录下的历史 plan 含该 trailer，是历史遗留，不要照抄）。
- GF(2) 约定：Pauli P → `(x|z) ∈ F₂^{2n}`，`x_i=1` 当 `P_i ∈ {X,Y}`，`z_i=1` 当 `P_i ∈ {Z,Y}`。辛积 `= (a_x·b_z + a_z·b_x) mod 2`，**0 = 对易，1 = 反对易**。
- 数组统一 `numpy.uint8`，生成元矩阵 shape `(m, 2n)`，前 n 列是 x 块、后 n 列是 z 块。
- 解码器**只**能从 `reset(layout)` 与 `update(round, events)` 获得运行期信息。运行器不得向解码器传 `Circuit` / `StabilizerCode` / `State` / 后端。
- `TimingModel` 为 `None` 时，所有 timing 字段必须是 `None`——**不得编造数字**。
- M1 **不使用** `if_` / `ControlFlow`。修正在轮间由 Python 侧计算。
- **轮 0 是投影式制备，不是纠错轮。** `|0…0⟩` 一般不在任何稳定子码的码空间内，轮 0 的提取过程本身把态投影进码空间。由此：
  - **轮 0 的 detector 只对「在制备基下确定」的生成元存在**——`|0…0⟩` 制备时即 x 块全零者（Z 型）；`|+…+⟩` 制备时即 z 块全零者。其余生成元轮 0 读数是 50/50 随机的（已在真机上实测：Steane 的 X 型生成元轮 0 读数 12 shot 中 6/6 分裂，轮 1/2 则恒等于轮 0），它们的首个 detector 在轮 1。
  - 各码轮 0 确定的生成元个数（实测）：repetition 2/2、Steane 3/6、Shor 6/8、surface_d3 4/8、**five_qubit 0/4**（非 CSS，无纯 Z 型生成元）。
  - `reference` 恒为 `zeros(m)`（Z 型稳定子在 `|0…0⟩` 上读数为 0），**不从某次噪声无关运行中实测得来**——那样每 shot 的随机轮 0 读数会与它错配，导致每 shot 都有约半数 X 型 detector 虚假触发。
  - **`PauliErrorModel` 从轮 1 开始注入**，轮 0 不注入。故任何检验「错误被纠正」的测试都需 `rounds ≥ 2`。`rounds=1` 只做制备。
- M1 非目标（不要实现）：DEM、Stim / PyMatching 任何代码、MWPM、union-find、`benchmark()` 扫描、任何可视化、tableau 模拟器、子系统码。
- 每个 Task 结束提交一次。

## 关键既有 API（实现时直接调用，勿重新发明）

```python
from aicir.core.circuit import (Circuit, measure, reset, pauli_x, pauli_y, pauli_z,
                                hadamard, cx, cy, cz)   # cnot 是 cx 的别名
from aicir.core.classical import ClassicalRegister      # ClassicalRegister(size, name)；reg[i] -> Bit
from aicir.core.state import State                      # State.zero_state(n, backend)
from aicir.core.operators import PauliString            # .qubit_labels -> list[str]，.n_qubits
from aicir.backends import NumpyBackend
from aicir.measure.trajectory import run_trajectory
```

- `Circuit(*gates, n_qubits=None)`；可迭代；`.gates` 是 list；`.append(g)` / `.extend(*gs)`。
- `cx(target, control_qubits, control_states=None)`——**控制位是第二个参数**，且为列表：`cx(3, [0])`。
- `measure([q0, q1], creg=reg)` → 按序写入 `reg` 的 0..k-1 位；`measure(qs, cbits=[reg[i], ...])` → 显式指定位。有经典目标时仅支持 Z 基。
- `reset([q])` → 把比特重置为 |0>。
- `run_trajectory(circuit, init_state, backend, *, tm, measure_qubits, snap_ops, rng, noise_model=None, matrix_cache=None) -> TrajectoryResult`
  - `TrajectoryResult` 字段：`.pre`（末端测量前的态）、`.post`、`.incircuit`、`.terminal`、`.snaps`、`.classical`（`{寄存器名: [0/1 位列表]}`）。
  - 轮间串联量子态就是把上一轮的 `.pre` 作为下一轮的 `init_state`。
  - `tm=False` 时不做末端测量；`snap_ops` 传 `set()`；`rng` 传 `numpy.random.Generator`。

---

### Task 1: `StabilizerCode` GF(2) 代数核心

**Files:**
- Create: `aicir/qec/__init__.py`（本任务先留最小导出）、`aicir/qec/code.py`
- Test: `tests/qec/__init__.py`（空）、`tests/qec/test_code_algebra.py`

**Interfaces:**
- Produces:
  - `pauli_to_gf2(label_string: str) -> np.ndarray`：shape `(2n,)` uint8。
  - `gf2_to_pauli(vec: np.ndarray) -> str`。
  - `symplectic_product(a: np.ndarray, b: np.ndarray) -> np.ndarray`：支持 `(2n,)`×`(2n,)` → 标量，`(m,2n)`×`(2n,)` → `(m,)`。
  - `StabilizerCode(n, generators, signs, logical_x, logical_z, name, coords=None)`
  - `StabilizerCode.from_paulis(generators, *, logical_x, logical_z, name, coords=None, signs=None)`
  - `.n`、`.k`、`.m`、`.generators`、`.signs`、`.logical_x`、`.logical_z`、`.name`、`.coords`
  - `.validate() -> None`（失败抛 `ValueError`）
  - `.syndrome(error) -> np.ndarray` shape `(m,)`
  - `.distance(max_weight=None) -> int`
  - `.logical_class(residual) -> np.ndarray` shape `(k, 2)`
  - `.verdict(residual) -> str`

- [ ] **Step 1: 写失败测试**

`tests/qec/__init__.py` 建成空文件。`tests/qec/test_code_algebra.py`：

```python
import numpy as np
import pytest

from aicir.qec.code import (
    StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product,
)

# 五比特完美码 [[5,1,3]]（生成元与 logical 已逐对手工验证对易关系）
FIVE_GENS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


def test_pauli_gf2_roundtrip():
    v = pauli_to_gf2("XYZI")
    # x 块：X->1, Y->1, Z->0, I->0 ；z 块：X->0, Y->1, Z->1, I->0
    assert np.array_equal(v, np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8))
    assert gf2_to_pauli(v) == "XYZI"


def test_symplectic_product_basic():
    x = pauli_to_gf2("X")
    z = pauli_to_gf2("Z")
    y = pauli_to_gf2("Y")
    assert symplectic_product(x, z) == 1      # X 与 Z 反对易
    assert symplectic_product(x, y) == 1      # X 与 Y 反对易
    assert symplectic_product(x, x) == 0      # 自对易
    assert symplectic_product(pauli_to_gf2("XX"), pauli_to_gf2("ZZ")) == 0  # 重叠 2 → 对易


def test_symplectic_product_broadcasts_over_rows():
    rows = np.stack([pauli_to_gf2("XI"), pauli_to_gf2("ZI")])
    out = symplectic_product(rows, pauli_to_gf2("ZI"))
    assert out.shape == (2,)
    assert list(out) == [1, 0]


def test_five_qubit_code_validates():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    code.validate()
    assert (code.n, code.k, code.m) == (5, 1, 4)


def test_validate_rejects_anticommuting_generators():
    with pytest.raises(ValueError, match="不对易"):
        StabilizerCode.from_paulis(
            ["XI", "ZI"], logical_x=["IX"], logical_z=["IZ"], name="bad",
        ).validate()


def test_validate_rejects_dependent_generators():
    # ZZI 与 IZZ 独立，但第三个 ZIZ = 前两者之积 → 秩亏
    with pytest.raises(ValueError, match="线性相关"):
        StabilizerCode.from_paulis(
            ["ZZI", "IZZ", "ZIZ"], logical_x=["XXX"], logical_z=["ZII"], name="bad",
        ).validate()


def test_validate_rejects_logical_not_in_normalizer():
    # XII 与稳定子 ZZI 反对易 → 不是合法 logical
    with pytest.raises(ValueError, match="logical"):
        StabilizerCode.from_paulis(
            ["ZZI", "IZZ"], logical_x=["XII"], logical_z=["ZII"], name="bad",
        ).validate()


def test_syndrome_of_single_qubit_errors():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    # 距离 3 的码，任何权重 1 错误都必须给出非零综合征
    for q in range(5):
        for p in "XYZ":
            label = "I" * q + p + "I" * (4 - q)
            assert code.syndrome(pauli_to_gf2(label)).any()


def test_syndrome_of_stabilizer_is_zero():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert not code.syndrome(pauli_to_gf2(FIVE_GENS[0])).any()


def test_distance_of_five_qubit_code_is_three():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert code.distance() == 3


def test_distance_raises_at_cutoff_instead_of_guessing():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    with pytest.raises(ValueError, match="max_weight"):
        code.distance(max_weight=2)


def test_logical_class_and_verdict():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert code.verdict(pauli_to_gf2("IIIII")) == "corrected"
    # 稳定子元素仍算 corrected（残余落在稳定子群内）
    assert code.verdict(pauli_to_gf2(FIVE_GENS[0])) == "corrected"
    # 逻辑 X 算符本身 → logical_x
    assert code.verdict(pauli_to_gf2("XXXXX")) == "logical_x"
    assert code.verdict(pauli_to_gf2("ZZZZZ")) == "logical_z"
    cls = code.logical_class(pauli_to_gf2("XXXXX"))
    assert cls.shape == (1, 2)
    assert list(cls[0]) == [1, 0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_code_algebra.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec'`

- [ ] **Step 3: 实现 `aicir/qec/code.py`**

先建 `aicir/qec/__init__.py`：

```python
"""aicir.qec

量子纠错（Quantum Error Correction）实验平台。

面向**新型在线实时纠错/解码算法**：码、syndrome 提取调度、在线解码器
三处均可插拔。详见本包 README。
"""

from __future__ import annotations

from .code import StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product

__all__ = ["StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product"]
```

`aicir/qec/code.py`：

```python
"""稳定子码的 GF(2) 辛表示核心。

约定：n 比特 Pauli 表示为 (x|z) ∈ F₂^{2n}，x_i=1 当 P_i∈{X,Y}，z_i=1 当 P_i∈{Z,Y}。
辛积 = (a_x·b_z + a_z·b_x) mod 2，0 表示对易、1 表示反对易。

本模块是**纯代数**层：不构造线路、不接触后端、不依赖 aicir.core 之外的东西。
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Sequence

import numpy as np

_LABEL_TO_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_XZ_TO_LABEL = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def _as_label_string(item) -> str:
    """接受 str 或 aicir.core.operators.PauliString，统一取出标签串。"""
    if isinstance(item, str):
        return item.strip().upper()
    labels = getattr(item, "qubit_labels", None)
    if labels is None:
        raise TypeError(f"无法解析为 Pauli 串：{item!r}")
    return "".join(labels)


def pauli_to_gf2(item, n_qubits: int | None = None) -> np.ndarray:
    """Pauli 标签串 → (2n,) uint8 的 (x|z) 向量。"""
    label = _as_label_string(item)
    n = len(label) if n_qubits is None else int(n_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli 串长度 {len(label)} 与 n_qubits {n} 不符")
    vec = np.zeros(2 * n, dtype=np.uint8)
    for i, ch in enumerate(label):
        if ch not in _LABEL_TO_XZ:
            raise ValueError(f"未知泡利标签 {ch!r}，只支持 I/X/Y/Z")
        x, z = _LABEL_TO_XZ[ch]
        vec[i] = x
        vec[n + i] = z
    return vec


def gf2_to_pauli(vec: np.ndarray) -> str:
    """(2n,) uint8 → Pauli 标签串。"""
    vec = np.asarray(vec, dtype=np.uint8).ravel()
    if vec.size % 2:
        raise ValueError("向量长度必须是偶数（x 块 + z 块）")
    n = vec.size // 2
    return "".join(_XZ_TO_LABEL[(int(vec[i]), int(vec[n + i]))] for i in range(n))


def symplectic_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """辛积 (a_x·b_z + a_z·b_x) mod 2。0 = 对易，1 = 反对易。

    a 可为 (2n,) 或 (m, 2n)；b 为 (2n,)。返回标量或 (m,)。
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.uint8))
    b = np.asarray(b, dtype=np.uint8).ravel()
    n = b.size // 2
    out = (a[:, :n] @ b[n:] + a[:, n:] @ b[:n]) % 2
    return out.astype(np.uint8) if out.size > 1 else np.uint8(out[0])


def _gf2_rank(rows: np.ndarray) -> int:
    """GF(2) 高斯消元求秩。"""
    mat = np.array(rows, dtype=np.uint8, copy=True)
    rank = 0
    n_cols = mat.shape[1]
    for col in range(n_cols):
        pivot = next((r for r in range(rank, mat.shape[0]) if mat[r, col]), None)
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(mat.shape[0]):
            if r != rank and mat[r, col]:
                mat[r] ^= mat[rank]
        rank += 1
        if rank == mat.shape[0]:
            break
    return rank


def _in_span(rows: np.ndarray, vec: np.ndarray) -> bool:
    """vec 是否在 rows 的 GF(2) 张成空间内。"""
    if rows.shape[0] == 0:
        return not np.asarray(vec, dtype=np.uint8).any()
    base = _gf2_rank(rows)
    augmented = np.vstack([rows, np.asarray(vec, dtype=np.uint8)[None, :]])
    return _gf2_rank(augmented) == base


class StabilizerCode:
    """稳定子码：生成元 + 逻辑算符的 GF(2) 辛表示。"""

    def __init__(self, n, generators, signs, logical_x, logical_z, name, coords=None):
        self.n = int(n)
        self.generators = np.asarray(generators, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.signs = np.asarray(signs, dtype=np.uint8).ravel()
        self.logical_x = np.asarray(logical_x, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.logical_z = np.asarray(logical_z, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.name = str(name)
        self.coords = dict(coords) if coords else {}
        if self.logical_x.shape[0] != self.logical_z.shape[0]:
            raise ValueError("logical_x 与 logical_z 数量必须相同")
        if self.signs.size != self.generators.shape[0]:
            raise ValueError("signs 长度必须与生成元数一致")

    @property
    def m(self) -> int:
        """稳定子生成元个数。"""
        return int(self.generators.shape[0])

    @property
    def k(self) -> int:
        """逻辑比特数。"""
        return int(self.logical_x.shape[0])

    @classmethod
    def from_paulis(cls, generators, *, logical_x, logical_z, name,
                    coords=None, signs=None) -> "StabilizerCode":
        """从 Pauli 串（str 或 PauliString）构造。"""
        gen_labels = [_as_label_string(g) for g in generators]
        n = len(gen_labels[0])
        if any(len(g) != n for g in gen_labels):
            raise ValueError("所有生成元的 Pauli 串长度必须一致")
        gens = np.stack([pauli_to_gf2(g, n) for g in gen_labels])
        lx = np.stack([pauli_to_gf2(_as_label_string(p), n) for p in logical_x])
        lz = np.stack([pauli_to_gf2(_as_label_string(p), n) for p in logical_z])
        sg = np.zeros(gens.shape[0], dtype=np.uint8) if signs is None else signs
        return cls(n, gens, sg, lx, lz, name, coords)

    def validate(self) -> None:
        """校验码的合法性；任何一项不成立都抛 ValueError 并指名违规对象。"""
        for i, j in combinations(range(self.m), 2):
            if symplectic_product(self.generators[i], self.generators[j]):
                raise ValueError(
                    f"[{self.name}] 生成元 {i} ({gf2_to_pauli(self.generators[i])}) 与 "
                    f"{j} ({gf2_to_pauli(self.generators[j])}) 不对易"
                )
        rank = _gf2_rank(self.generators)
        if rank != self.m:
            raise ValueError(
                f"[{self.name}] 生成元线性相关：{self.m} 个生成元的 GF(2) 秩只有 {rank}"
            )
        if self.n - rank != self.k:
            raise ValueError(
                f"[{self.name}] 逻辑比特数不符：n−rank = {self.n - rank}，但给出了 {self.k} 组 logical"
            )
        for kind, rows in (("logical_x", self.logical_x), ("logical_z", self.logical_z)):
            for i, row in enumerate(rows):
                bad = np.nonzero(symplectic_product(self.generators, row))[0]
                if bad.size:
                    raise ValueError(
                        f"[{self.name}] {kind}[{i}] ({gf2_to_pauli(row)}) 与生成元 "
                        f"{int(bad[0])} 反对易，不在 normalizer 内"
                    )
        for i in range(self.k):
            for j in range(self.k):
                got = int(symplectic_product(self.logical_x[i], self.logical_z[j]))
                want = 1 if i == j else 0
                if got != want:
                    raise ValueError(
                        f"[{self.name}] logical_x[{i}] 与 logical_z[{j}] 的辛积应为 {want}，实际 {got}"
                    )

    def syndrome(self, error) -> np.ndarray:
        """错误 Pauli 的综合征：与各生成元的辛积，shape (m,)。"""
        vec = error if isinstance(error, np.ndarray) else pauli_to_gf2(error, self.n)
        return np.atleast_1d(symplectic_product(self.generators, vec)).astype(np.uint8)

    def _is_logical(self, vec: np.ndarray) -> bool:
        """与所有生成元对易，但不属于稳定子群 → 是非平凡逻辑算符。"""
        if self.syndrome(vec).any():
            return False
        return not _in_span(self.generators, vec)

    def distance(self, max_weight: int | None = None) -> int:
        """码距：最小权非平凡逻辑算符的权重。

        max_weight=None 表示不设上限、搜到为止（n ≲ 15 可行）；
        给定整数则搜到该权重为止，未找到时抛 ValueError——**绝不返回下界猜测**。
        """
        limit = self.n if max_weight is None else int(max_weight)
        for weight in range(1, limit + 1):
            for support in combinations(range(self.n), weight):
                for labels in product("XYZ", repeat=weight):
                    chars = ["I"] * self.n
                    for q, ch in zip(support, labels):
                        chars[q] = ch
                    vec = pauli_to_gf2("".join(chars), self.n)
                    if self._is_logical(vec):
                        return weight
        raise ValueError(
            f"[{self.name}] 在 max_weight={limit} 内未找到非平凡逻辑算符；"
            f"提高 max_weight 或检查码定义（不返回下界猜测）"
        )

    def logical_class(self, residual) -> np.ndarray:
        """残余 Pauli 落在哪个逻辑陪集。

        返回 (k, 2) uint8：第 i 行 = 第 i 个逻辑比特上的 (X 分量, Z 分量)。
        X 分量由 residual 与 logical_z[i] 的辛积给出，Z 分量由与 logical_x[i] 的辛积给出。
        全零表示残余落在稳定子群内。
        """
        vec = residual if isinstance(residual, np.ndarray) else pauli_to_gf2(residual, self.n)
        out = np.zeros((self.k, 2), dtype=np.uint8)
        for i in range(self.k):
            out[i, 0] = symplectic_product(self.logical_z[i], vec)
            out[i, 1] = symplectic_product(self.logical_x[i], vec)
        return out

    def verdict(self, residual) -> str:
        """把 logical_class 结果翻译成判定字符串。

        全零 → "corrected"；k=1 → "logical_x"/"logical_z"/"logical_y"；
        k>1 → "logical_q{i}_{x|y|z}"，多个逻辑比特出错时按下标升序以 "+" 连接。
        """
        cls = self.logical_class(residual)
        if not cls.any():
            return "corrected"
        parts = []
        for i, (has_x, has_z) in enumerate(cls):
            if not (has_x or has_z):
                continue
            letter = "y" if (has_x and has_z) else ("x" if has_x else "z")
            parts.append(f"logical_{letter}" if self.k == 1 else f"logical_q{i}_{letter}")
        return "+".join(parts)

    def __repr__(self) -> str:
        return f"StabilizerCode({self.name}, n={self.n}, k={self.k}, m={self.m})"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_code_algebra.py -q`
Expected: PASS（13 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/__init__.py aicir/qec/code.py tests/qec/__init__.py tests/qec/test_code_algebra.py
git commit -m "feat(qec): StabilizerCode GF(2) 辛表示代数核心"
```

---

### Task 2: 内置码注册表与五个参考码

**Files:**
- Create: `aicir/qec/codes/__init__.py`、`aicir/qec/codes/repetition.py`、`aicir/qec/codes/five_qubit.py`、`aicir/qec/codes/steane.py`、`aicir/qec/codes/shor.py`、`aicir/qec/codes/surface.py`
- Modify: `aicir/qec/__init__.py`（导出 `get_code` / `register_code` / `CODES`）
- Test: `tests/qec/test_builtin_codes.py`

**Interfaces:**
- Consumes: Task 1 的 `StabilizerCode.from_paulis` / `.validate` / `.distance`
- Produces:
  - `register_code(name: str, builder: Callable[..., StabilizerCode]) -> None`
  - `get_code(name: str, **kwargs) -> StabilizerCode`
  - `CODES: dict[str, Callable]`
  - 内置名：`"repetition"`（`get_code("repetition", d=3, basis="Z")`）、`"five_qubit"`、`"steane"`、`"shor"`、`"surface"`（`get_code("surface", d=3)`）

**关键事实（实现者必读，否则会写出错误的测试）：**
重复码 `distance()` 返回 **1**，不是 d。bit-flip 重复码的稳定子全是 Z 型，`logical_z = Z₀` 是权重 1 的逻辑算符——该码对 Z 错误毫无保护。d 是它**只针对 X 噪声**的有效距离。其余四个码 `distance()` 均为 3。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_builtin_codes.py`：

```python
import pytest

from aicir.qec.code import pauli_to_gf2
from aicir.qec.codes import CODES, get_code, register_code

# (名字, 构造 kwargs, 期望 n, 期望 k, 期望 distance())
BUILTINS = [
    ("repetition", {"d": 3, "basis": "Z"}, 3, 1, 1),   # 见下方注释：重复码码距是 1
    ("repetition", {"d": 5, "basis": "Z"}, 5, 1, 1),
    ("repetition", {"d": 3, "basis": "X"}, 3, 1, 1),
    ("five_qubit", {}, 5, 1, 3),
    ("steane", {}, 7, 1, 3),
    ("shor", {}, 9, 1, 3),
    ("surface", {"d": 3}, 9, 1, 3),
]


@pytest.mark.parametrize("name,kwargs,n,k,dist", BUILTINS)
def test_builtin_code_validates_and_has_expected_shape(name, kwargs, n, k, dist):
    code = get_code(name, **kwargs)
    code.validate()
    assert (code.n, code.k) == (n, k)
    # 重复码的 distance() 是 1：它的稳定子全是 Z 型，logical_z = Z₀ 权重为 1，
    # 对 Z 错误无保护。参数 d 只是它针对 X 噪声的有效距离。
    assert code.distance() == dist


@pytest.mark.parametrize("name,kwargs,n,k,dist", BUILTINS)
def test_builtin_code_weight_one_errors_are_detected_in_protected_basis(name, kwargs, n, k, dist):
    code = get_code(name, **kwargs)
    basis = "X" if name == "repetition" and kwargs["basis"] == "Z" else None
    bases = [basis] if basis else ["X", "Y", "Z"]
    for q in range(code.n):
        for p in bases:
            label = "I" * q + p + "I" * (code.n - q - 1)
            assert code.syndrome(pauli_to_gf2(label)).any(), f"{name} 漏检 {label}"


def test_surface_code_carries_coords():
    code = get_code("surface", d=3)
    assert len(code.coords) == 9
    assert code.coords[0] == (0, 0)
    assert code.coords[8] == (2, 2)


def test_registry_roundtrip():
    def _builder():
        from aicir.qec.code import StabilizerCode
        return StabilizerCode.from_paulis(
            ["ZZ"], logical_x=["XX"], logical_z=["ZI"], name="tiny",
        )

    register_code("tiny_test_code", _builder)
    assert "tiny_test_code" in CODES
    assert get_code("tiny_test_code").n == 2


def test_unknown_code_raises_listing_available():
    with pytest.raises(KeyError, match="five_qubit"):
        get_code("no_such_code")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_builtin_codes.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.codes'`

- [ ] **Step 3: 实现注册表与五个码**

`aicir/qec/codes/__init__.py`：

```python
"""内置稳定子码注册表。

沿用 aicir.chemistry.molecules 的惯例：每个码一个模块，模块内自注册进 CODES。
"""

from __future__ import annotations

from typing import Callable

CODES: dict[str, Callable] = {}


def register_code(name: str, builder: Callable) -> None:
    """把码构造器注册进 CODES。"""
    CODES[str(name)] = builder


def get_code(name: str, **kwargs):
    """按名取码；kwargs 透传给构造器（如 repetition 的 d/basis）。"""
    key = str(name)
    if key not in CODES:
        raise KeyError(f"未知码 {key!r}；可用：{sorted(CODES)}")
    return CODES[key](**kwargs)


from . import five_qubit, repetition, shor, steane, surface  # noqa: E402,F401  自注册

__all__ = ["CODES", "register_code", "get_code"]
```

`aicir/qec/codes/repetition.py`：

```python
"""重复码：n=d，d−1 个两体生成元。

basis="Z"：稳定子为 Z_iZ_{i+1}，保护 X（bit-flip）错误。
basis="X"：稳定子为 X_iX_{i+1}，保护 Z（phase-flip）错误。

注意：重复码的**真实码距是 1**——basis="Z" 时 logical_z=Z₀ 权重为 1，
该码对 Z 错误无任何保护。参数 d 是它针对受保护基的有效距离。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code


def build(d: int = 3, basis: str = "Z") -> StabilizerCode:
    d = int(d)
    if d < 3 or d % 2 == 0:
        raise ValueError(f"重复码的 d 必须是 ≥3 的奇数，收到 {d}")
    basis = str(basis).strip().upper()
    if basis not in ("Z", "X"):
        raise ValueError(f"basis 只支持 'Z' 或 'X'，收到 {basis!r}")

    gens = []
    for i in range(d - 1):
        chars = ["I"] * d
        chars[i] = chars[i + 1] = basis
        gens.append("".join(chars))

    if basis == "Z":
        logical_x, logical_z = [ "X" * d ], [ "Z" + "I" * (d - 1) ]
    else:
        logical_x, logical_z = [ "X" + "I" * (d - 1) ], [ "Z" * d ]

    return StabilizerCode.from_paulis(
        gens, logical_x=logical_x, logical_z=logical_z,
        name=f"repetition_d{d}_{basis}",
        coords={q: (0, q) for q in range(d)},
    )


register_code("repetition", build)
```

`aicir/qec/codes/five_qubit.py`：

```python
"""[[5,1,3]] 完美码。生成元为 XZZXI 的四个循环移位。"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

GENERATORS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        GENERATORS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
        coords={q: (0, q) for q in range(5)},
    )


register_code("five_qubit", build)
```

`aicir/qec/codes/steane.py`：

```python
"""Steane [[7,1,3]] CSS 码，基于 [7,4,3] Hamming 码的校验矩阵。

校验行 h1={3,4,5,6}、h2={1,2,5,6}、h3={0,2,4,6}；任意两行交集均为偶数，
故 X 型与 Z 型生成元逐对对易（CSS 条件）。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

X_GENERATORS = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
Z_GENERATORS = ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        X_GENERATORS + Z_GENERATORS,
        logical_x=["XXXXXXX"], logical_z=["ZZZZZZZ"], name="steane",
        coords={q: (0, q) for q in range(7)},
    )


register_code("steane", build)
```

`aicir/qec/codes/shor.py`：

```python
"""Shor [[9,1,3]] 级联码：三组三比特相位块，块内 Z 型、块间 X 型。"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

Z_GENERATORS = [
    "ZZIIIIIII", "IZZIIIIII",   # 块 0
    "IIIZZIIII", "IIIIZZIII",   # 块 1
    "IIIIIIZZI", "IIIIIIIZZ",   # 块 2
]
X_GENERATORS = ["XXXXXXIII", "IIIXXXXXX"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        Z_GENERATORS + X_GENERATORS,
        logical_x=["XXXXXXXXX"], logical_z=["ZZZZZZZZZ"], name="shor",
        coords={q: (q // 3, q % 3) for q in range(9)},
    )


register_code("shor", build)
```

`aicir/qec/codes/surface.py`：

```python
"""旋转表面码 d=3，9 个 data 比特按行主序排在 3x3 网格上：

    q0 q1 q2
    q3 q4 q5
    q6 q7 q8

X 型稳定子：X(0,1,3,4)、X(4,5,7,8)、X(2,5)、X(3,6)
Z 型稳定子：Z(1,2,4,5)、Z(3,4,6,7)、Z(0,1)、Z(7,8)
逻辑 X = X(0,1,2)（顶行），逻辑 Z = Z(0,3,6)（左列）。
所有 X/Z 生成元对的交集大小均为偶数，逐对已验证对易。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

X_SUPPORTS = [(0, 1, 3, 4), (4, 5, 7, 8), (2, 5), (3, 6)]
Z_SUPPORTS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 1), (7, 8)]


def _label(support, pauli: str, n: int = 9) -> str:
    chars = ["I"] * n
    for q in support:
        chars[q] = pauli
    return "".join(chars)


def build(d: int = 3) -> StabilizerCode:
    if int(d) != 3:
        raise ValueError(f"M1 只内置 d=3 的旋转表面码，收到 d={d}")
    gens = [_label(s, "X") for s in X_SUPPORTS] + [_label(s, "Z") for s in Z_SUPPORTS]
    return StabilizerCode.from_paulis(
        gens,
        logical_x=[_label((0, 1, 2), "X")],
        logical_z=[_label((0, 3, 6), "Z")],
        name="surface_d3",
        coords={q: (q // 3, q % 3) for q in range(9)},
    )


register_code("surface", build)
```

在 `aicir/qec/__init__.py` 追加：

```python
from .codes import CODES, get_code, register_code

__all__ += ["CODES", "get_code", "register_code"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_builtin_codes.py -q`
Expected: PASS（18 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/codes/ aicir/qec/__init__.py tests/qec/test_builtin_codes.py
git commit -m "feat(qec): 内置码注册表与五个参考稳定子码"
```

---

### Task 3: `detectors.py` —— 面向解码器的契约

**Files:**
- Create: `aicir/qec/detectors.py`
- Modify: `aicir/qec/__init__.py`
- Test: `tests/qec/test_detectors.py`

**Interfaces:**
- Produces:
  - `Detector(index: int, records: tuple[int, ...], stabilizer: int, round_index: int)`
  - `Observable(index: int, records: tuple[int, ...])`
  - `DetectorLayout(n_detectors, n_rounds, n_stabilizers, detectors, observables, coords, round0_stabilizers)`
  - `.detector_at(stabilizer, round_index) -> Detector`
  - `.round_slice(round_index) -> tuple[int, ...]`（该轮所有 detector 的全局下标）
  - `.detection_events(raw_syndromes, round_index, reference) -> np.ndarray` shape `(n_stabilizers,)`

**轮 0 语义（见 Global Constraints）：** `round0_stabilizers` 是轮 0 有 detector 的生成元下标元组。`detection_events` 在 `round_index==0` 时把不在该集合内的分量**掩为 0**——那些生成元轮 0 读数随机，不构成 detector。事件向量对每轮恒为 `(n_stabilizers,)`，使解码器协议与全部测试的形状保持统一。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_detectors.py`：

```python
import numpy as np
import pytest

from aicir.qec.detectors import Detector, DetectorLayout, Observable


def _layout(n_rounds=3, n_stab=2, round0=(0,)):
    """稳定子 0 在轮 0 确定（有 detector），稳定子 1 不确定（轮 0 无 detector）。"""
    dets, idx = [], 0
    for r in range(n_rounds):
        for s in range(n_stab):
            if r == 0 and s not in round0:
                continue                     # 轮 0 只对确定的生成元建 detector
            recs = (r * n_stab + s,) if r == 0 else ((r - 1) * n_stab + s, r * n_stab + s)
            dets.append(Detector(index=idx, records=recs, stabilizer=s, round_index=r))
            idx += 1
    obs = [Observable(index=0, records=(n_rounds * n_stab,))]
    return DetectorLayout(
        n_detectors=idx, n_rounds=n_rounds, n_stabilizers=n_stab,
        detectors=tuple(dets), observables=tuple(obs), coords={},
        round0_stabilizers=tuple(round0),
    )


def test_layout_shape_and_lookup():
    layout = _layout()
    # 轮 0 只有 1 个 detector（稳定子 0），轮 1/2 各 2 个 → 共 5
    assert layout.n_detectors == 5
    d = layout.detector_at(stabilizer=1, round_index=2)
    assert d.stabilizer == 1 and d.round_index == 2 and d.index == 4


def test_round_slice_returns_that_rounds_detectors():
    layout = _layout()
    assert layout.round_slice(0) == (0,)          # 轮 0 只有确定的那一个
    assert layout.round_slice(2) == (3, 4)


def test_detection_events_round_zero_masks_nondeterministic_stabilizers():
    """轮 0 读数随机的生成元不构成 detector，其事件必须被掩为 0。"""
    layout = _layout()
    raw = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    ev0 = layout.detection_events(raw, 0, ref)
    # 稳定子 0 确定：1 ^ 0 = 1 ；稳定子 1 不确定：掩为 0（尽管原始读数是 1）
    assert list(ev0) == [1, 0]


def test_detection_events_later_rounds_are_differences_unmasked():
    layout = _layout()
    raw = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    # 轮 1：稳定子 1 由 1 变 0 → 事件；轮 1 不掩码
    assert list(layout.detection_events(raw, 1, ref)) == [0, 1]
    assert list(layout.detection_events(raw, 2, ref)) == [0, 1]


def test_all_deterministic_layout_masks_nothing_at_round_zero():
    layout = _layout(round0=(0, 1))
    raw = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    assert list(layout.detection_events(raw, 0, ref)) == [1, 1]


def test_detector_at_rejects_unknown_pair():
    layout = _layout()
    with pytest.raises(KeyError):
        layout.detector_at(stabilizer=9, round_index=0)
    with pytest.raises(KeyError):
        layout.detector_at(stabilizer=1, round_index=0)   # 轮 0 该生成元无 detector
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_detectors.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.detectors'`

- [ ] **Step 3: 实现 `aicir/qec/detectors.py`**

```python
"""Detector / Observable / DetectorLayout —— 解码器唯一被允许知道的东西。

沿用 Stim 语义，使 M2 的互操作成为格式转换而非语义翻译：
- measurement record：一条 shot 内所有线路中测量的扁平有序列表，下标 i = 第 i 个执行的 measure。
- Detector：一组 record 下标，其奇偶在**无噪声线路中确定为 0**。
- Observable：一组 record 下标，其奇偶给出某逻辑算符取值。

解码器在 reset() 时拿到 DetectorLayout，此后只收 detection event 比特向量流。
它不持有线路、码、量子态或后端的任何引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Detector:
    """一个 detector：records 的奇偶在无噪声下恒为 0。"""
    index: int
    records: tuple[int, ...]
    stabilizer: int
    round_index: int


@dataclass(frozen=True)
class Observable:
    """一个逻辑可观测量：records 的奇偶给出逻辑算符取值。"""
    index: int
    records: tuple[int, ...]


@dataclass(frozen=True)
class DetectorLayout:
    """解码器面向的布局描述。

    round0_stabilizers：轮 0 有 detector 的生成元下标。|0…0⟩ 一般不在码空间内，
    轮 0 的提取本身把态投影进码空间；只有在制备基下确定的生成元（|0⟩ 制备时即
    x 块全零的 Z 型）轮 0 读数才确定，其余是 50/50 随机的，不构成 detector。
    """
    n_detectors: int
    n_rounds: int
    n_stabilizers: int
    detectors: tuple[Detector, ...]
    observables: tuple[Observable, ...]
    coords: dict = field(default_factory=dict)
    round0_stabilizers: tuple[int, ...] = ()

    def detector_at(self, stabilizer: int, round_index: int) -> Detector:
        """按 (稳定子, 轮) 取 detector。"""
        for det in self.detectors:
            if det.stabilizer == stabilizer and det.round_index == round_index:
                return det
        raise KeyError(f"没有 (stabilizer={stabilizer}, round={round_index}) 对应的 detector")

    def round_slice(self, round_index: int) -> tuple[int, ...]:
        """该轮全部 detector 的全局下标，按稳定子序。"""
        return tuple(
            d.index for d in sorted(
                (d for d in self.detectors if d.round_index == round_index),
                key=lambda d: d.stabilizer,
            )
        )

    def detection_events(self, raw_syndromes, round_index: int, reference) -> np.ndarray:
        """由原始稳定子读数算出该轮的 detection event。

        raw_syndromes: (rounds, n_stabilizers) uint8 的原始读数
        reference:     (n_stabilizers,) uint8，轮 0 参考值（恒为 zeros）
        轮 0 与 reference 比较，且**只保留 round0_stabilizers 内的分量**（其余生成元
        轮 0 读数随机、不构成 detector，掩为 0）；其余轮与上一轮比较、不掩码。

        返回形状恒为 (n_stabilizers,)，使解码器协议与全部测试的形状保持统一。
        """
        raw = np.asarray(raw_syndromes, dtype=np.uint8)
        if round_index != 0:
            return (raw[round_index] ^ raw[round_index - 1]).astype(np.uint8)
        events = (raw[0] ^ np.asarray(reference, dtype=np.uint8)).astype(np.uint8)
        mask = np.zeros(self.n_stabilizers, dtype=np.uint8)
        for s in self.round0_stabilizers:
            mask[s] = 1
        return (events & mask).astype(np.uint8)
```

在 `aicir/qec/__init__.py` 追加：

```python
from .detectors import Detector, DetectorLayout, Observable

__all__ += ["Detector", "Observable", "DetectorLayout"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_detectors.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/detectors.py aicir/qec/__init__.py tests/qec/test_detectors.py
git commit -m "feat(qec): Detector/Observable/DetectorLayout 解码器契约"
```

---

### Task 4: `schedules/` —— 码编译成逐轮线路 + `verify_schedule`

**Files:**
- Create: `aicir/qec/schedules/__init__.py`、`aicir/qec/schedules/bare.py`
- Modify: `aicir/qec/__init__.py`
- Test: `tests/qec/test_schedules.py`

**Interfaces:**
- Consumes: Task 1 `StabilizerCode`、Task 2 `get_code`、Task 3 `DetectorLayout`
- Produces:
  - `RoundCircuit(circuit, creg_name, ancilla_qubits, data_qubits, record_offset)`
  - `ReadoutCircuit(circuit, creg_name, observable_records)`
  - `Schedule` 协议：`build_encode(code, logical_state)` / `build_round(code, round_index, *, creg_name)` / `build_readout(code, logical_state)`
  - `BareAncillaSchedule`
  - `register_schedule(name, factory)` / `resolve_schedule(name_or_obj)`
  - `verify_schedule(code, schedule, rounds, *, logical_state="0", backend=None, shots=4) -> None`
  - `build_layout(code, schedule, rounds, *, logical_state="0") -> DetectorLayout`
  - `deterministic_round0(code, logical_state) -> tuple[int, ...]`
- 全局 qubit 编号约定：**data 比特 0..n−1，ancilla 比特 n..n+m−1**（ancilla j 测量生成元 j），线路总比特数 `n + m`。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_schedules.py`：

```python
import pytest

from aicir.qec.codes import get_code
from aicir.qec.schedules import (
    BareAncillaSchedule, build_layout, resolve_schedule, verify_schedule,
)

CASES = [
    ("repetition", {"d": 3, "basis": "Z"}),
    ("five_qubit", {}),
    ("steane", {}),
    ("shor", {}),
    ("surface", {"d": 3}),
]


@pytest.mark.parametrize("name,kwargs", CASES)
@pytest.mark.parametrize("rounds", [1, 3])
def test_detectors_are_deterministic_without_noise(name, kwargs, rounds):
    """无噪声下每个 detector 必须恒为 0——这是提取调度唯一最有力的结构性检验。"""
    code = get_code(name, **kwargs)
    verify_schedule(code, BareAncillaSchedule(), rounds)


def test_layout_shape_accounts_for_partial_round_zero():
    """Shor 码 8 个生成元中只有 6 个纯 Z 型 → 轮 0 只建 6 个 detector。

    刻意选 Shor 而非 Steane/surface：后两者 X/Z 型各占一半，
    len(round0) 恰等于 m/2，m*rounds 与正确值在某些轮数下会巧合相等。
    """
    code = get_code("shor")
    layout = build_layout(code, BareAncillaSchedule(), rounds=3)
    assert layout.n_stabilizers == code.m == 8
    assert layout.n_rounds == 3
    assert layout.round0_stabilizers == (0, 1, 2, 3, 4, 5)
    assert layout.n_detectors == 6 + 8 * 2 == 22
    assert len(layout.round_slice(0)) == 6
    assert len(layout.round_slice(1)) == 8


@pytest.mark.parametrize("name,kwargs,expected", [
    ("repetition", {"d": 3, "basis": "Z"}, (0, 1)),
    ("five_qubit", {}, ()),                      # 非 CSS：轮 0 无任何确定生成元
    ("steane", {}, (3, 4, 5)),
    ("surface", {"d": 3}, (4, 5, 6, 7)),
])
def test_deterministic_round0_matches_measured_values(name, kwargs, expected):
    """轮 0 确定的生成元集合 —— 数值已在真机上逐码实测确认。"""
    from aicir.qec.schedules import deterministic_round0
    code = get_code(name, **kwargs)
    assert deterministic_round0(code, "0") == expected


def test_round_circuit_uses_data_then_ancilla_numbering():
    code = get_code("repetition", d=3, basis="Z")
    rc = BareAncillaSchedule().build_round(code, 0)
    assert rc.data_qubits == (0, 1, 2)
    assert rc.ancilla_qubits == (3, 4)
    assert rc.circuit.n_qubits == 5


def test_readout_basis_must_match_preparation():
    code = get_code("steane")
    sched = BareAncillaSchedule()
    sched.build_readout(code, "0")     # Z 基制备 → Z 基读出，正常
    with pytest.raises(ValueError, match="基"):
        sched.build_readout(code, "?")


def test_resolve_schedule_accepts_name_and_instance():
    assert isinstance(resolve_schedule("bare"), BareAncillaSchedule)
    inst = BareAncillaSchedule()
    assert resolve_schedule(inst) is inst
    with pytest.raises(KeyError, match="bare"):
        resolve_schedule("no_such_schedule")


def test_verify_schedule_reports_offending_detector():
    """人为破坏调度（漏掉 ancilla reset）必须被 verify_schedule 抓住。"""
    code = get_code("repetition", d=3, basis="Z")

    class BrokenSchedule(BareAncillaSchedule):
        def build_round(self, code, round_index, *, creg_name="syn"):
            rc = super().build_round(code, round_index, creg_name=creg_name)
            # 去掉所有 reset 指令 → 第二轮起 ancilla 带着上一轮的值，detector 不再恒 0
            kept = [g for g in rc.circuit.gates
                    if getattr(g, "measurement_type", None) != "reset"]
            rc.circuit.gates[:] = kept
            return rc

    with pytest.raises(ValueError, match="detector"):
        verify_schedule(code, BrokenSchedule(), rounds=3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_schedules.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.schedules'`

- [ ] **Step 3: 实现 `aicir/qec/schedules/__init__.py`**

```python
"""syndrome 提取调度：把 StabilizerCode 编译成逐轮线路 + DetectorLayout。

按**轮**构建而非构建单一整体线路——运行器必须执行轮 t、暂停解码、再继续。
run_trajectory 接受 init_state 并返回 .pre，运行器据此在轮间串联量子态。

全局比特编号：data 0..n−1，ancilla n..n+m−1（ancilla j 测量生成元 j）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..detectors import Detector, DetectorLayout, Observable


@dataclass
class RoundCircuit:
    """一轮 syndrome 提取的产物。"""
    circuit: object
    creg_name: str
    ancilla_qubits: tuple[int, ...]
    data_qubits: tuple[int, ...]
    record_offset: int          # 该轮首个 measurement 在全局 record 序列中的下标


@dataclass
class ReadoutCircuit:
    """末端逻辑读出的产物。"""
    circuit: object
    creg_name: str
    observable_records: tuple[tuple[int, ...], ...]   # 每个逻辑比特对应的 record 下标组


class Schedule(Protocol):
    def build_encode(self, code, logical_state: str): ...
    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit: ...
    def build_readout(self, code, logical_state: str) -> ReadoutCircuit: ...


SCHEDULES: dict[str, Callable[[], Schedule]] = {}


def register_schedule(name: str, factory: Callable[[], Schedule]) -> None:
    SCHEDULES[str(name)] = factory


def resolve_schedule(name_or_obj) -> Schedule:
    """名字 → 调度实例；已经是实例则原样返回。"""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in SCHEDULES:
        raise KeyError(f"未知调度 {name_or_obj!r}；可用：{sorted(SCHEDULES)}")
    return SCHEDULES[name_or_obj]()


def deterministic_round0(code, logical_state: str = "0") -> tuple[int, ...]:
    """哪些生成元在轮 0 读数确定 —— 即在制备基下 |0…0⟩/|+…+⟩ 是其本征态者。

    |0…0⟩ 制备：x 块全零的生成元（纯 Z 型），读数确定为 0。
    |+…+⟩ 制备：z 块全零的生成元（纯 X 型），读数确定为 0。
    其余生成元轮 0 读数是 50/50 随机的，**不构成 detector**。

    实测各码的确定生成元个数：repetition 2/2、Steane 3/6、Shor 6/8、
    surface_d3 4/8、five_qubit **0/4**（非 CSS，无纯 Z 型生成元）。
    """
    state = str(logical_state)
    if state in ("0", "1"):
        block = code.generators[:, :code.n]          # x 块须全零
    elif state in ("+", "-"):
        block = code.generators[:, code.n:]          # z 块须全零
    else:
        raise ValueError(f"未知逻辑初态 {state!r}")
    return tuple(int(j) for j in range(code.m) if not block[j].any())


def build_layout(code, schedule, rounds: int, *, logical_state: str = "0") -> DetectorLayout:
    """由码与调度构造 DetectorLayout。

    detector (s, t)：轮 t 的稳定子 s 读数 XOR 轮 t−1 的读数。
    t=0 **只对 deterministic_round0 内的生成元建 detector**（其余轮 0 读数随机）。
    """
    schedule = resolve_schedule(schedule)
    m = code.m
    round0 = deterministic_round0(code, logical_state)
    detectors, idx = [], 0
    for t in range(int(rounds)):
        for s in range(m):
            if t == 0 and s not in round0:
                continue
            cur = t * m + s
            recs = (cur,) if t == 0 else ((t - 1) * m + s, cur)
            detectors.append(Detector(index=idx, records=recs, stabilizer=s, round_index=t))
            idx += 1
    base = int(rounds) * m
    observables = tuple(
        Observable(index=i, records=tuple(range(base, base + code.n)))
        for i in range(code.k)
    )
    return DetectorLayout(
        n_detectors=idx, n_rounds=int(rounds), n_stabilizers=m,
        detectors=tuple(detectors), observables=observables, coords=dict(code.coords),
        round0_stabilizers=round0,
    )


def verify_schedule(code, schedule, rounds: int, *, logical_state: str = "0",
                    backend=None, shots: int = 4) -> None:
    """无噪声运行，断言每个 detector 恒为 0。不满足则抛 ValueError 并指名违规项。

    这是提取调度**唯一最有力的结构性检验**：它抓 CNOT 顺序错、漏掉 ancilla reset、
    轮 0 确定集合推错。公开它，使用户验证自己写的调度时享有与内置调度同等的保障。

    参考值恒为 zeros(m)：轮 0 确定的生成元（|0…0⟩ 制备下即纯 Z 型）读数确定为 0。
    **不从某次运行中实测 reference**——非确定生成元的轮 0 读数逐 shot 随机，
    用某一次的实测值当参考会让检验对该 bug 视而不见（且轮 0 变成什么都不断言）。
    """
    from ..runner import collect_noiseless_syndromes   # 延迟导入，避免循环

    schedule = resolve_schedule(schedule)
    layout = build_layout(code, schedule, int(rounds), logical_state=logical_state)
    reference = np.zeros(code.m, dtype=np.uint8)
    for shot in range(int(shots)):
        raw = collect_noiseless_syndromes(
            code, schedule, int(rounds), logical_state=logical_state,
            backend=backend, seed=shot,
        )
        for t in range(int(rounds)):
            events = layout.detection_events(raw, t, reference)
            bad = np.nonzero(events)[0]
            if bad.size:
                raise ValueError(
                    f"[{code.name}] 调度不满足 detector 确定性："
                    f"shot {shot} 的轮 {t} 稳定子 {int(bad[0])} 触发了 detector"
                    f"（无噪声下应恒为 0）"
                )


from .bare import BareAncillaSchedule  # noqa: E402  自注册

__all__ = [
    "RoundCircuit", "ReadoutCircuit", "Schedule", "BareAncillaSchedule",
    "SCHEDULES", "register_schedule", "resolve_schedule",
    "build_layout", "verify_schedule", "deterministic_round0",
]
```

`aicir/qec/schedules/bare.py`：

```python
"""裸 ancilla syndrome 提取：每个生成元一个 ancilla。

单轮结构：H(ancilla) → 从 ancilla 向生成元 support 上各比特施加受控 P → H(ancilla)
→ measure 写入经典寄存器 → reset(ancilla)。

轮间靠 reset 复用 ancilla，这是表面码 d=3 保持在 9+8 而非逐轮新增 ancilla 的原因。
"""

from __future__ import annotations

from aicir.core.circuit import Circuit, cx, cy, cz, hadamard, measure, reset
from aicir.core.classical import ClassicalRegister

from ..code import gf2_to_pauli
from . import ReadoutCircuit, RoundCircuit, register_schedule

_CONTROLLED = {"X": cx, "Y": cy, "Z": cz}


class BareAncillaSchedule:
    """默认调度：裸 ancilla，标准 CNOT 顺序。"""

    name = "bare"

    def _split(self, code):
        data = tuple(range(code.n))
        ancilla = tuple(range(code.n, code.n + code.m))
        return data, ancilla, code.n + code.m

    def build_encode(self, code, logical_state: str = "0"):
        """制备逻辑初态。

        M1 只支持 |0…0> 作为码空间内的物理初态：对 CSS 码与本仓库内置的五个码，
        全 |0> 已是 Z 型稳定子的 +1 本征态；X 型稳定子的投影由第一轮 syndrome 提取
        完成（这正是轮 0 参考值由 reference 向量给出的原因）。
        """
        state = str(logical_state)
        if state not in ("0", "+"):
            raise ValueError(f"M1 只支持逻辑初态 '0' 或 '+'，收到 {state!r}")
        _, _, n_total = self._split(code)
        cir = Circuit(n_qubits=n_total)
        if state == "+":
            for q in range(code.n):
                cir.append(hadamard(q))
        return cir

    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit:
        data, ancilla, n_total = self._split(code)
        reg = ClassicalRegister(code.m, creg_name)
        cir = Circuit(n_qubits=n_total)
        for j in range(code.m):
            anc = ancilla[j]
            labels = gf2_to_pauli(code.generators[j])
            cir.append(hadamard(anc))
            for q, ch in enumerate(labels):
                if ch != "I":
                    cir.append(_CONTROLLED[ch](q, [anc]))
            cir.append(hadamard(anc))
        cir.append(measure(list(ancilla), creg=reg))
        cir.append(reset(list(ancilla)))
        return RoundCircuit(
            circuit=cir, creg_name=creg_name, ancilla_qubits=ancilla,
            data_qubits=data, record_offset=int(round_index) * code.m,
        )

    def build_readout(self, code, logical_state: str = "0") -> ReadoutCircuit:
        """末端逻辑读出：在制备基下逐个测量 data 比特。

        读出基必须与制备基匹配：'0'/'1' → 逻辑 Z 基，'+'/'-' → 逻辑 X 基。
        """
        state = str(logical_state)
        if state not in ("0", "1", "+", "-"):
            raise ValueError(f"未知逻辑初态 {state!r}；读出基无法与制备基匹配")
        data, _, n_total = self._split(code)
        reg = ClassicalRegister(code.n, "readout")
        cir = Circuit(n_qubits=n_total)
        if state in ("+", "-"):
            for q in data:
                cir.append(hadamard(q))          # X 基 → 先转到 Z 基再测
        cir.append(measure(list(data), creg=reg))
        records = tuple(tuple(range(code.n)) for _ in range(code.k))
        return ReadoutCircuit(circuit=cir, creg_name="readout", observable_records=records)


register_schedule("bare", BareAncillaSchedule)
```

在 `aicir/qec/__init__.py` 追加：

```python
from .schedules import (BareAncillaSchedule, register_schedule, resolve_schedule,
                        verify_schedule)

__all__ += ["BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule"]
```

- [ ] **Step 4: 实现 `collect_noiseless_syndromes`（`verify_schedule` 依赖它）**

新建 `aicir/qec/runner.py`，本任务只放这一个函数（完整运行器在 Task 7）：

```python
"""交错 simulate↔decode 运行器。

本文件在 Task 4 只提供无噪声综合征采集（verify_schedule 依赖），
完整的逐 shot 在线解码循环在 Task 7 补齐。
"""

from __future__ import annotations

import numpy as np

from aicir.backends import NumpyBackend
from aicir.core.state import State
from aicir.measure.trajectory import run_trajectory


def _read_creg(classical: dict, name: str, size: int) -> np.ndarray:
    """从轨迹经典 store 读出 size 位，缺位补 0。"""
    bits = list(classical.get(name, []))
    bits.extend([0] * (size - len(bits)))
    return np.array(bits[:size], dtype=np.uint8)


def collect_noiseless_syndromes(code, schedule, rounds: int, *, logical_state: str = "0",
                                backend=None, seed: int = 0) -> np.ndarray:
    """无噪声运行 rounds 轮，返回 raw_syndromes，shape (rounds, m) uint8。

    **不返回 reference**：轮 0 参考值恒为 zeros(m)（见 verify_schedule 的说明）。
    非确定生成元的轮 0 读数逐 shot 随机，任何「从一次运行实测 reference」的做法
    都会与其他 shot 错配。
    """
    from .schedules import resolve_schedule

    schedule = resolve_schedule(schedule)
    backend = backend or NumpyBackend()
    rng = np.random.default_rng(seed)
    n_total = code.n + code.m

    state = run_trajectory(
        schedule.build_encode(code, logical_state), State.zero_state(n_total, backend),
        backend, tm=False, measure_qubits=None, snap_ops=set(), rng=rng,
    ).pre

    raw = np.zeros((int(rounds), code.m), dtype=np.uint8)
    for t in range(int(rounds)):
        rc = schedule.build_round(code, t)
        res = run_trajectory(rc.circuit, state, backend, tm=False,
                             measure_qubits=None, snap_ops=set(), rng=rng)
        state = res.pre
        raw[t] = _read_creg(res.classical, rc.creg_name, code.m)

    return raw
```

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_schedules.py -q`
Expected: PASS（19 passed）

若 `test_detectors_are_deterministic_without_noise` 失败，说明提取线路有误——**先修线路，不要放宽断言**。该测试是本模块正确性的地基。

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/schedules/ aicir/qec/runner.py aicir/qec/__init__.py tests/qec/test_schedules.py
git commit -m "feat(qec): 裸 ancilla 提取调度、DetectorLayout 构造与 verify_schedule"
```

---

### Task 5: `errors.py` —— 逐 shot 随机 Pauli 误差模型

**Files:**
- Create: `aicir/qec/errors.py`
- Modify: `aicir/qec/__init__.py`
- Test: `tests/qec/test_errors.py`

**Interfaces:**
- Produces:
  - `ErrorEvent(round_index: int, qubit: int, pauli: str, source: str)`，`source ∈ {"data", "measurement"}`
  - `PauliErrorModel(p_data=0.0, p_measure=0.0, channel="depolarizing")`，`channel ∈ {"bit_flip", "phase_flip", "depolarizing"}`
  - `.sample_round(round_index, n_data, n_ancilla, rng) -> list[ErrorEvent]`
  - `.data_events(events) -> list[ErrorEvent]` / `.measurement_events(events) -> list[ErrorEvent]`

- [ ] **Step 1: 写失败测试**

`tests/qec/test_errors.py`：

```python
import numpy as np
import pytest

from aicir.qec.errors import ErrorEvent, PauliErrorModel


def test_zero_probability_yields_no_events():
    model = PauliErrorModel(p_data=0.0, p_measure=0.0)
    rng = np.random.default_rng(0)
    assert model.sample_round(0, n_data=5, n_ancilla=4, rng=rng) == []


def test_certain_probability_hits_every_qubit():
    model = PauliErrorModel(p_data=1.0, p_measure=1.0, channel="bit_flip")
    rng = np.random.default_rng(0)
    events = model.sample_round(2, n_data=3, n_ancilla=2, rng=rng)
    data = [e for e in events if e.source == "data"]
    meas = [e for e in events if e.source == "measurement"]
    assert len(data) == 3 and len(meas) == 2
    assert {e.qubit for e in data} == {0, 1, 2}
    assert {e.qubit for e in meas} == {0, 1}
    assert all(e.round_index == 2 for e in events)


def test_bit_flip_channel_only_emits_x():
    model = PauliErrorModel(p_data=1.0, channel="bit_flip")
    rng = np.random.default_rng(1)
    events = model.sample_round(0, n_data=8, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"X"}


def test_phase_flip_channel_only_emits_z():
    model = PauliErrorModel(p_data=1.0, channel="phase_flip")
    rng = np.random.default_rng(1)
    events = model.sample_round(0, n_data=8, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"Z"}


def test_depolarizing_channel_emits_all_three():
    model = PauliErrorModel(p_data=1.0, channel="depolarizing")
    rng = np.random.default_rng(3)
    events = model.sample_round(0, n_data=300, n_ancilla=0, rng=rng)
    assert {e.pauli for e in events} == {"X", "Y", "Z"}


def test_sampling_is_reproducible_under_seed():
    model = PauliErrorModel(p_data=0.3, p_measure=0.2)
    a = model.sample_round(0, 9, 8, np.random.default_rng(7))
    b = model.sample_round(0, 9, 8, np.random.default_rng(7))
    assert a == b


def test_rate_is_approximately_p():
    model = PauliErrorModel(p_data=0.1, channel="bit_flip")
    rng = np.random.default_rng(11)
    hits = sum(len(model.sample_round(t, 100, 0, rng)) for t in range(100))
    assert 800 < hits < 1200          # 10000 次伯努利试验，p=0.1


def test_rejects_bad_probability_and_channel():
    with pytest.raises(ValueError, match="概率"):
        PauliErrorModel(p_data=1.5)
    with pytest.raises(ValueError, match="channel"):
        PauliErrorModel(channel="no_such_channel")


def test_event_partition_helpers():
    model = PauliErrorModel(p_data=1.0, p_measure=1.0)
    events = model.sample_round(0, 2, 2, np.random.default_rng(0))
    assert len(model.data_events(events)) == 2
    assert len(model.measurement_events(events)) == 2
    assert isinstance(events[0], ErrorEvent)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_errors.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.errors'`

- [ ] **Step 3: 实现 `aicir/qec/errors.py`**

```python
"""逐 shot 随机 Pauli 误差模型。

态始终保持**纯态矢量**——不走密度矩阵。现有噪声路径一挂 NoiseModel 就把态升为
密度矩阵，17 比特需 275 TB，对 QEC 完全不可行；而 QEC 基准本来就用随机 Pauli 采样。

测量误差作用在**经典记录**上（翻转读数比特），不作用于量子态。这与 Stim 的
「MR 前加 X_ERROR」等价，更省，且传播语义正确——解码器随后基于被污染的综合征
动作，这正是被测行为。
"""

from __future__ import annotations

from dataclasses import dataclass

_CHANNELS = {
    "bit_flip": ("X",),
    "phase_flip": ("Z",),
    "depolarizing": ("X", "Y", "Z"),
}


@dataclass(frozen=True)
class ErrorEvent:
    """一次错误注入。source="data" 作用于量子态，"measurement" 翻转经典读数位。"""
    round_index: int
    qubit: int
    pauli: str
    source: str


class PauliErrorModel:
    """每 data 比特每轮以 p_data 出错；每 ancilla 读数每轮以 p_measure 翻转。"""

    def __init__(self, p_data: float = 0.0, p_measure: float = 0.0,
                 channel: str = "depolarizing"):
        for label, p in (("p_data", p_data), ("p_measure", p_measure)):
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"{label} 必须是 [0,1] 区间内的概率，收到 {p}")
        channel = str(channel)
        if channel not in _CHANNELS:
            raise ValueError(f"未知 channel {channel!r}；可用：{sorted(_CHANNELS)}")
        self.p_data = float(p_data)
        self.p_measure = float(p_measure)
        self.channel = channel

    def sample_round(self, round_index: int, n_data: int, n_ancilla: int, rng) -> list[ErrorEvent]:
        """采样该轮的全部错误事件。"""
        paulis = _CHANNELS[self.channel]
        events: list[ErrorEvent] = []
        if self.p_data > 0.0:
            for q in range(int(n_data)):
                if rng.random() < self.p_data:
                    p = paulis[0] if len(paulis) == 1 else paulis[rng.integers(len(paulis))]
                    events.append(ErrorEvent(int(round_index), q, p, "data"))
        if self.p_measure > 0.0:
            for a in range(int(n_ancilla)):
                if rng.random() < self.p_measure:
                    events.append(ErrorEvent(int(round_index), a, "flip", "measurement"))
        return events

    @staticmethod
    def data_events(events) -> list[ErrorEvent]:
        return [e for e in events if e.source == "data"]

    @staticmethod
    def measurement_events(events) -> list[ErrorEvent]:
        return [e for e in events if e.source == "measurement"]

    def __repr__(self) -> str:
        return (f"PauliErrorModel(p_data={self.p_data}, p_measure={self.p_measure}, "
                f"channel={self.channel!r})")
```

在 `aicir/qec/__init__.py` 追加：

```python
from .errors import ErrorEvent, PauliErrorModel

__all__ += ["ErrorEvent", "PauliErrorModel"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_errors.py -q`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/errors.py aicir/qec/__init__.py tests/qec/test_errors.py
git commit -m "feat(qec): 逐 shot 随机 Pauli 误差模型"
```

---

### Task 6: `decoders/` —— 在线解码协议与查表解码器

**Files:**
- Create: `aicir/qec/decoders/__init__.py`、`aicir/qec/decoders/lookup.py`
- Modify: `aicir/qec/__init__.py`
- Test: `tests/qec/test_lookup_decoder.py`

**Interfaces:**
- Consumes: Task 1 `StabilizerCode`、Task 3 `DetectorLayout`
- Produces:
  - `DecodeStep(frame_flips, corrections, committed_through, cost)`
  - `OnlineDecoder` 协议：`name` / `window` / `commit_lag` / `reset(layout)` / `update(round_index, events)` / `flush()` / `cost_of(round_index, events)`
  - `register_decoder(name, factory)` / `resolve_decoder(name_or_obj)`
  - `LookupDecoder(code, t=None, error_basis="XYZ")`

**关键设计点：** `LookupDecoder` 的默认 `t = ⌊code.distance()−1)/2⌋` 对重复码会得到 **0**（重复码 `distance()` 是 1）。此时必须**抛错并提示显式传 `t`**，而不是静默建一张无用的表。重复码的正确用法是 `LookupDecoder(code, t=(d-1)//2, error_basis="X")`。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_lookup_decoder.py`：

```python
import numpy as np
import pytest

from aicir.qec.code import gf2_to_pauli, pauli_to_gf2
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep, resolve_decoder
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.schedules import BareAncillaSchedule, build_layout


def _layout(code, rounds=1):
    return build_layout(code, BareAncillaSchedule(), rounds)


@pytest.mark.parametrize("name,kwargs", [
    ("five_qubit", {}), ("steane", {}), ("shor", {}), ("surface", {"d": 3}),
])
def test_lookup_corrects_every_weight_one_error(name, kwargs):
    """穷举：距离 3 的码必须纠正所有权重 1 错误。"""
    code = get_code(name, **kwargs)
    dec = LookupDecoder(code)
    dec.reset(_layout(code))
    for q in range(code.n):
        for p in "XYZ":
            err = pauli_to_gf2("I" * q + p + "I" * (code.n - q - 1), code.n)
            syn = code.syndrome(err)
            correction = dec.correction_for_syndrome(syn)
            residual = (err ^ correction) % 2
            assert code.verdict(residual) == "corrected", f"{name} 未纠正 {gf2_to_pauli(err)}"


def test_repetition_needs_explicit_t_because_distance_is_one():
    """重复码 distance()==1 → 默认 t 会算成 0，必须抛错而不是静默建无用的表。"""
    code = get_code("repetition", d=3, basis="Z")
    with pytest.raises(ValueError, match="t"):
        LookupDecoder(code)
    dec = LookupDecoder(code, t=1, error_basis="X")
    dec.reset(_layout(code))
    for q in range(3):
        err = pauli_to_gf2("I" * q + "X" + "I" * (2 - q), 3)
        residual = (err ^ dec.correction_for_syndrome(code.syndrome(err))) % 2
        assert code.verdict(residual) == "corrected"


def test_zero_syndrome_yields_identity_correction():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code))
    correction = dec.correction_for_syndrome(np.zeros(code.m, dtype=np.uint8))
    assert not correction.any()


def test_update_commits_every_round_immediately():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=3))
    assert (dec.window, dec.commit_lag) == (1, 0)
    for t in range(3):
        step = dec.update(t, np.zeros(code.m, dtype=np.uint8))
        assert isinstance(step, DecodeStep)
        assert step.committed_through == t


def test_flush_leaves_nothing_pending():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=2))
    dec.update(0, np.zeros(code.m, dtype=np.uint8))
    dec.update(1, np.zeros(code.m, dtype=np.uint8))
    assert dec.flush().committed_through == 1


def test_cost_is_reported_and_nonnegative():
    code = get_code("steane")
    dec = LookupDecoder(code)
    dec.reset(_layout(code, rounds=1))
    step = dec.update(0, np.zeros(code.m, dtype=np.uint8))
    assert step.cost >= 0.0
    assert dec.cost_of(0, np.zeros(code.m, dtype=np.uint8)) >= 0.0


def test_resolve_decoder_accepts_instance_and_rejects_unknown_name():
    code = get_code("steane")
    dec = LookupDecoder(code)
    assert resolve_decoder(dec) is dec
    with pytest.raises(KeyError):
        resolve_decoder("no_such_decoder")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_lookup_decoder.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'aicir.qec.decoders'`

- [ ] **Step 3: 实现 `aicir/qec/decoders/__init__.py`**

```python
"""在线解码协议。

因果性是**结构性**的，不是约定性的：update(t, …) 是唯一输入通道，按序每轮恰调用
一次；解码器不持有线路、码、量子态或未来轮次的引用。它无法偷看未来，因为轮 t+1
尚未被模拟。批式后处理平台只能靠自律保证这一点，这里它是架构性质。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class DecodeStep:
    """一次 update/flush 的输出。"""
    frame_flips: np.ndarray | None = None
    corrections: list | None = None
    committed_through: int = -1
    cost: float = 0.0


class OnlineDecoder(Protocol):
    name: str
    window: int
    commit_lag: int

    def reset(self, layout) -> None: ...
    def update(self, round_index: int, events: np.ndarray) -> DecodeStep: ...
    def flush(self) -> DecodeStep: ...
    def cost_of(self, round_index: int, events: np.ndarray) -> float: ...


DECODERS: dict[str, Callable] = {}


def register_decoder(name: str, factory: Callable) -> None:
    DECODERS[str(name)] = factory


def resolve_decoder(name_or_obj, **kwargs):
    """名字 → 解码器实例；已经是实例则原样返回。"""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in DECODERS:
        raise KeyError(f"未知解码器 {name_or_obj!r}；可用：{sorted(DECODERS)}")
    return DECODERS[name_or_obj](**kwargs)


from .lookup import LookupDecoder  # noqa: E402  自注册

__all__ = ["DecodeStep", "OnlineDecoder", "DECODERS",
           "register_decoder", "resolve_decoder", "LookupDecoder"]
```

`aicir/qec/decoders/lookup.py`：

```python
"""查表解码器：枚举权重 ≤ t 的错误，建 syndrome → 最小权修正表。

**与「承重约束」的边界**：约束的是**运行器交给解码器什么**——运行器只经
reset(layout) + update(round, events) 传递信息。解码器作者可以在**构造时**注入
额外知识；LookupDecoder 就是构造时吃进 code 来建表的。代价是它只能用于构造时
那个码，而只依赖 layout 的解码器可以跑任何码、包括 M2 里 Stim 采样的事件流。

M2 的 DEM 正是用来消除这个特例的。M1 的构造时注入是**已知的临时做法**，
不应被后续实现当作范例推广。
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from ..code import pauli_to_gf2
from . import DecodeStep, register_decoder


class LookupDecoder:
    """逐轮即时提交的查表解码器。"""

    name = "lookup"
    window = 1
    commit_lag = 0

    def __init__(self, code, t: int | None = None, error_basis: str = "XYZ"):
        self._code = code
        self._basis = "".join(sorted(set(str(error_basis).upper())))
        if not self._basis or any(ch not in "XYZ" for ch in self._basis):
            raise ValueError(f"error_basis 只能由 X/Y/Z 组成，收到 {error_basis!r}")
        if t is None:
            t = (code.distance() - 1) // 2
            if t < 1:
                raise ValueError(
                    f"[{code.name}] 由 distance()={code.distance()} 推出的 t={t} 无意义"
                    f"（重复码的 distance() 是 1，它对未受保护的基毫无防护）。"
                    f"请显式传 t，例如 LookupDecoder(code, t=1, error_basis='X')"
                )
        self._t = int(t)
        self._table: dict[bytes, np.ndarray] = {}
        self._layout = None
        self._committed = -1

    def _build_table(self) -> None:
        """按权重升序枚举，先到先得 → 表中即为最小权修正。"""
        code = self._code
        self._table = {np.zeros(code.m, dtype=np.uint8).tobytes():
                       np.zeros(2 * code.n, dtype=np.uint8)}
        for weight in range(1, self._t + 1):
            for support in combinations(range(code.n), weight):
                for labels in product(self._basis, repeat=weight):
                    chars = ["I"] * code.n
                    for q, ch in zip(support, labels):
                        chars[q] = ch
                    err = pauli_to_gf2("".join(chars), code.n)
                    key = code.syndrome(err).tobytes()
                    if key not in self._table:
                        self._table[key] = err

    def reset(self, layout) -> None:
        self._layout = layout
        self._committed = -1
        if not self._table:
            self._build_table()

    def correction_for_syndrome(self, syndrome) -> np.ndarray:
        """综合征 → 修正 Pauli 的 (2n,) 向量；查不到时返回恒等（放弃修正）。"""
        key = np.asarray(syndrome, dtype=np.uint8).tobytes()
        found = self._table.get(key)
        return found.copy() if found is not None else np.zeros(2 * self._code.n, dtype=np.uint8)

    def cost_of(self, round_index: int, events) -> float:
        """声明代价：一次哈希查表，记为 1 个单位。"""
        return 1.0

    def update(self, round_index: int, events) -> DecodeStep:
        correction = self.correction_for_syndrome(events)
        self._committed = int(round_index)
        return DecodeStep(
            frame_flips=self._code.logical_class(correction).ravel().copy(),
            corrections=_to_gate_list(correction, self._code.n),
            committed_through=self._committed,
            cost=self.cost_of(round_index, events),
        )

    def flush(self) -> DecodeStep:
        """逐轮即时提交 → flush 无未决可提交。"""
        return DecodeStep(frame_flips=None, corrections=None,
                          committed_through=self._committed, cost=0.0)


def _to_gate_list(correction: np.ndarray, n: int) -> list:
    """(2n,) 修正向量 → [(qubit, pauli), ...] 门列表。"""
    out = []
    for q in range(n):
        x, z = int(correction[q]), int(correction[n + q])
        if x and z:
            out.append((q, "Y"))
        elif x:
            out.append((q, "X"))
        elif z:
            out.append((q, "Z"))
    return out


register_decoder("lookup", LookupDecoder)
```

在 `aicir/qec/__init__.py` 追加：

```python
from .decoders import (DecodeStep, LookupDecoder, register_decoder, resolve_decoder)

__all__ += ["DecodeStep", "LookupDecoder", "register_decoder", "resolve_decoder"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_lookup_decoder.py -q`
Expected: PASS（10 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/decoders/ aicir/qec/__init__.py tests/qec/test_lookup_decoder.py
git commit -m "feat(qec): 在线解码协议、注册表与查表解码器"
```

---

### Task 7: `record.py` + 运行器 frame 模式与协议不变量

**Files:**
- Create: `aicir/qec/record.py`
- Modify: `aicir/qec/runner.py`（补齐 `run`）、`aicir/qec/__init__.py`
- Test: `tests/qec/test_runner_frame.py`、`tests/qec/test_online_protocol.py`

**Interfaces:**
- Consumes: Task 1–6 全部
- Produces:
  - `QECShotRecord`、`QECResult`（字段见实现）
  - `qec.run(code, *, schedule="bare", errors, decoder, rounds, shots, logical_state="0", correction_mode="frame", timing=None, backend=None, seed=None, keep_records=100, keep_failures=100) -> QECResult`

- [ ] **Step 1: 写失败测试**

`tests/qec/test_runner_frame.py`：

```python
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel


def test_noiseless_run_never_reports_a_logical_error():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=8, seed=0)
    assert result.logical_error_rate == 0.0
    assert result.verdict_counts == {"corrected": 8}


@pytest.mark.parametrize("name,kwargs", [
    ("five_qubit", {}), ("steane", {}), ("surface", {"d": 3}),
])
def test_low_rate_noise_after_preparation_is_corrected(name, kwargs):
    """轮 0 制备、轮 1 注入低速率噪声 → 权重 1 错误全可纠，逻辑错误率为 0。

    rounds=2 而非 1：轮 0 是投影式制备不注入错误（见 Global Constraints），
    rounds=1 只做制备，不构成纠错检验。
    """
    code = get_code(name, **kwargs)
    result = run(code, errors=PauliErrorModel(p_data=0.02, channel="depolarizing"),
                 decoder=LookupDecoder(code), rounds=2, shots=64, seed=3)
    assert result.logical_error_rate == 0.0


def test_round_zero_injects_no_errors():
    """轮 0 是制备轮 —— 注入的错误事件不得出现在轮 0。"""
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.5, p_measure=0.5),
                 decoder=LookupDecoder(code), rounds=3, shots=4, seed=1)
    for rec in result.records:
        assert all(e.round_index >= 1 for e in rec.injected_errors)


def test_result_reports_stderr_and_config():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=2, shots=32, seed=1)
    assert result.shots == 32 and result.rounds == 2
    assert result.code_name == "steane" and result.decoder_name == "lookup"
    assert result.logical_error_rate_stderr >= 0.0
    assert isinstance(result.summary(), str)


def test_records_capture_syndromes_and_detection_events():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.1), decoder=LookupDecoder(code),
                 rounds=3, shots=4, seed=2)
    rec = result.records[0]
    assert rec.raw_syndromes.shape == (3, code.m)
    assert rec.detection_events.shape == (3, code.m)
    assert len(rec.decode_steps) == 3
    assert rec.verdict in ("corrected", "logical_x", "logical_y", "logical_z")


def test_keep_records_caps_memory_but_aggregates_cover_all_shots():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=2, shots=40, seed=5, keep_records=3)
    assert len(result.records) == 3
    assert sum(result.verdict_counts.values()) == 40


def test_runner_rejects_missing_decoder():
    code = get_code("steane")
    with pytest.raises(ValueError, match="decoder"):
        run(code, errors=PauliErrorModel(), decoder=None, rounds=2, shots=1)


def test_timing_fields_are_none_without_a_timing_model():
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=2, shots=4, seed=0)
    assert result.max_backlog is None
    assert result.mean_commit_latency is None
    assert result.budget_violations is None
    assert result.records[0].commit_latency is None


def test_runner_rejects_bad_rounds():
    code = get_code("steane")
    with pytest.raises(ValueError, match="rounds"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code), rounds=0, shots=1)
```

`tests/qec/test_online_protocol.py`：

```python
import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep
from aicir.qec.errors import PauliErrorModel


class SpyDecoder:
    """记录自己被喂了什么——用来证明不存在通往未来轮次的通道。"""

    name = "spy"
    window = 1
    commit_lag = 0

    def __init__(self):
        self.seen = []
        self.reset_args = []
        self.flushed = False
        self._committed = -1

    def reset(self, layout):
        self.reset_args.append(layout)
        self.seen = []
        self.flushed = False
        self._committed = -1

    def update(self, round_index, events):
        self.seen.append((round_index, np.array(events, copy=True)))
        self._committed = int(round_index)
        return DecodeStep(committed_through=self._committed, cost=1.0)

    def flush(self):
        self.flushed = True
        return DecodeStep(committed_through=self._committed, cost=0.0)

    def cost_of(self, round_index, events):
        return 1.0


class RegressingDecoder(SpyDecoder):
    name = "regressing"

    def update(self, round_index, events):
        super().update(round_index, events)
        return DecodeStep(committed_through=0, cost=1.0)   # 第二轮起回退


def test_decoder_is_called_once_per_round_in_order():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=4, shots=1, seed=0)
    assert [r for r, _ in spy.seen] == [0, 1, 2, 3]


def test_decoder_receives_only_layout_and_events():
    """解码器拿到的只有 DetectorLayout 与事件向量——没有线路、码、量子态、后端。"""
    from aicir.qec.detectors import DetectorLayout

    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(p_data=0.1), decoder=spy, rounds=2, shots=1, seed=0)
    assert all(isinstance(a, DetectorLayout) for a in spy.reset_args)
    for _, events in spy.seen:
        assert events.shape == (code.m,)
        assert events.dtype == np.uint8


def test_flush_is_called_at_end_of_every_shot():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=2, shots=3, seed=0)
    assert spy.flushed


def test_reset_is_called_once_per_shot():
    code = get_code("steane")
    spy = SpyDecoder()
    run(code, errors=PauliErrorModel(), decoder=spy, rounds=2, shots=5, seed=0)
    assert len(spy.reset_args) == 5


def test_runner_rejects_regressing_committed_through():
    code = get_code("steane")
    with pytest.raises(ValueError, match="committed_through"):
        run(code, errors=PauliErrorModel(), decoder=RegressingDecoder(),
            rounds=3, shots=1, seed=0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_runner_frame.py tests/qec/test_online_protocol.py -q`
Expected: FAIL —— `ImportError: cannot import name 'run' from 'aicir.qec'`

- [ ] **Step 3: 实现 `aicir/qec/record.py`**

```python
"""逐 shot 与聚合的记录结构。

raw_syndromes 与 detection_events **并存**：M3 要画错误链，而「差分」表达不出错误链。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QECShotRecord:
    """一条 shot 的完整记录。"""
    shot: int
    seed: int
    injected_errors: list = field(default_factory=list)
    raw_syndromes: np.ndarray | None = None
    detection_events: np.ndarray | None = None
    decode_steps: list = field(default_factory=list)
    commit_latency: np.ndarray | None = None
    backlog: np.ndarray | None = None
    wall_clock: np.ndarray | None = None
    observable_raw: np.ndarray | None = None
    frame_flips: np.ndarray | None = None
    verdict: str = "corrected"


@dataclass
class QECResult:
    """一次 run(...) 的聚合结果。"""
    code_name: str
    decoder_name: str
    schedule_name: str
    rounds: int
    shots: int
    records: list = field(default_factory=list)
    failure_records: list = field(default_factory=list)
    logical_error_rate: float = 0.0
    logical_error_rate_stderr: float = 0.0
    verdict_counts: dict = field(default_factory=dict)
    max_backlog: float | None = None
    mean_commit_latency: float | None = None
    budget_violations: int | None = None

    def summary(self) -> str:
        lines = [
            f"码 {self.code_name} · 调度 {self.schedule_name} · 解码器 {self.decoder_name}",
            f"轮数 {self.rounds} · shots {self.shots}",
            f"逻辑错误率 {self.logical_error_rate:.6g} ± {self.logical_error_rate_stderr:.3g}",
            f"判定分布 {dict(sorted(self.verdict_counts.items()))}",
        ]
        if self.max_backlog is not None:
            lines.append(
                f"最大 backlog {self.max_backlog:.6g}s · 平均提交延迟 "
                f"{self.mean_commit_latency:.6g}s · 超预算轮数 {self.budget_violations}"
            )
        return "\n".join(lines)
```

- [ ] **Step 4: 在 `aicir/qec/runner.py` 补齐 `run`**

在 Task 4 已建的 `runner.py` 中追加（保留 `collect_noiseless_syndromes`）：

```python
import time

from .decoders import resolve_decoder
from .errors import PauliErrorModel
from .record import QECResult, QECShotRecord

_PAULI_GATE = {"X": "pauli_x", "Y": "pauli_y", "Z": "pauli_z"}


def _error_gates(events):
    """data 错误事件 → 门列表。"""
    from aicir.core.circuit import pauli_x, pauli_y, pauli_z
    factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
    return [factory[e.pauli](e.qubit) for e in events if e.source == "data"]


def _apply_pauli_gates(state, pairs, backend, n_total, rng):
    """把 [(qubit, pauli), ...] 作为真实门施加到量子态上。"""
    from aicir.core.circuit import Circuit, pauli_x, pauli_y, pauli_z
    if not pairs:
        return state
    factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
    cir = Circuit(*[factory[p](q) for q, p in pairs], n_qubits=n_total)
    return run_trajectory(cir, state, backend, tm=False, measure_qubits=None,
                          snap_ops=set(), rng=rng).pre


def run(code, *, schedule="bare", errors=None, decoder=None, rounds=2, shots=1,
        logical_state="0", correction_mode="frame", timing=None, backend=None,
        seed=None, keep_records=100, keep_failures=100) -> QECResult:
    """逐 shot 交错「模拟 ↔ 解码」主循环。"""
    from .schedules import build_layout, resolve_schedule

    rounds = int(rounds)
    if rounds < 1:
        raise ValueError(f"rounds 必须 ≥1，收到 {rounds}")
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots 必须 ≥1，收到 {shots}")
    if correction_mode not in ("frame", "active"):
        raise ValueError(f"correction_mode 只支持 'frame' / 'active'，收到 {correction_mode!r}")

    if decoder is None:
        raise ValueError("decoder 为必填参数；请传入实现 OnlineDecoder 协议的实例或已注册的名字")
    schedule = resolve_schedule(schedule)
    decoder = resolve_decoder(decoder)
    errors = errors if errors is not None else PauliErrorModel()
    backend = backend or NumpyBackend()
    layout = build_layout(code, schedule, rounds, logical_state=logical_state)
    n_total = code.n + code.m

    # 轮 0 参考值恒为 zeros：轮 0 确定的生成元读数确定为 0，其余已被 layout 掩掉。
    reference = np.zeros(code.m, dtype=np.uint8)

    kept, failures, verdicts = [], [], {}
    for shot in range(shots):
        shot_seed = shot if seed is None else int(seed) * 100003 + shot
        record = _run_one_shot(
            code, schedule, errors, decoder, rounds, layout, reference,
            logical_state, correction_mode, timing, backend, shot, shot_seed, n_total,
        )
        verdicts[record.verdict] = verdicts.get(record.verdict, 0) + 1
        if len(kept) < int(keep_records):
            kept.append(record)
        if record.verdict != "corrected" and len(failures) < int(keep_failures):
            failures.append(record)

    n_fail = shots - verdicts.get("corrected", 0)
    p = n_fail / shots
    result = QECResult(
        code_name=code.name, decoder_name=getattr(decoder, "name", type(decoder).__name__),
        schedule_name=getattr(schedule, "name", type(schedule).__name__),
        rounds=rounds, shots=shots, records=kept, failure_records=failures,
        logical_error_rate=p, logical_error_rate_stderr=float(np.sqrt(p * (1 - p) / shots)),
        verdict_counts=verdicts,
    )
    if timing is not None:
        _fill_timing_aggregates(result, kept, timing)
    return result


def _run_one_shot(code, schedule, errors, decoder, rounds, layout, reference,
                  logical_state, correction_mode, timing, backend, shot, shot_seed, n_total):
    rng = np.random.default_rng(shot_seed)
    decoder.reset(layout)

    state = run_trajectory(
        schedule.build_encode(code, logical_state), State.zero_state(n_total, backend),
        backend, tm=False, measure_qubits=None, snap_ops=set(), rng=rng,
    ).pre

    raw = np.zeros((rounds, code.m), dtype=np.uint8)
    events_log = np.zeros((rounds, code.m), dtype=np.uint8)
    injected, steps, wall = [], [], np.zeros(rounds, dtype=float)
    frame = np.zeros(2 * code.k, dtype=np.uint8)
    applied = np.zeros(2 * code.n, dtype=np.uint8)     # active 模式下已施加修正的累积
    committed = -1

    for t in range(rounds):
        # 轮 0 是投影式制备（|0…0⟩ 一般不在码空间内），不注入错误；从轮 1 开始注入。
        round_errors = [] if t == 0 else errors.sample_round(t, code.n, code.m, rng)
        injected.extend(round_errors)

        rc = schedule.build_round(code, t)
        cir = rc.circuit
        cir.gates[:0] = _error_gates(round_errors)     # data 错误在该轮提取之前施加

        res = run_trajectory(cir, state, backend, tm=False, measure_qubits=None,
                             snap_ops=set(), rng=rng)
        state = res.pre
        bits = _read_creg(res.classical, rc.creg_name, code.m)

        for e in round_errors:                          # 测量误差翻转经典记录
            if e.source == "measurement":
                bits[e.qubit] ^= 1
        raw[t] = bits

        ev = layout.detection_events(raw, t, reference)
        if correction_mode == "active":
            # active 模式下已施加的修正会把原始稳定子读数复位，若不扣除，
            # 下一轮的朴素差分会放出一个虚假 detection event。
            ev = (ev ^ _applied_syndrome_delta(code, applied, raw, t)).astype(np.uint8)
        events_log[t] = ev

        t0 = time.perf_counter()
        step = decoder.update(t, ev)
        wall[t] = time.perf_counter() - t0

        if step.committed_through < committed:
            raise ValueError(
                f"解码器 {getattr(decoder, 'name', '?')} 的 committed_through 回退："
                f"轮 {t} 报 {step.committed_through}，此前已提交到 {committed}"
            )
        committed = step.committed_through
        steps.append(step)

        if step.frame_flips is not None:
            frame ^= np.asarray(step.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]
        if correction_mode == "active" and step.corrections:
            state = _apply_pauli_gates(state, step.corrections, backend, n_total, rng)
            for q, p in step.corrections:
                if p in ("X", "Y"):
                    applied[q] ^= 1
                if p in ("Z", "Y"):
                    applied[code.n + q] ^= 1

    final = decoder.flush()
    if final.committed_through < committed:
        raise ValueError("flush() 的 committed_through 不得低于此前已提交轮次")
    if final.frame_flips is not None:
        frame ^= np.asarray(final.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]

    ro = schedule.build_readout(code, logical_state)
    ro_res = run_trajectory(ro.circuit, state, backend, tm=False, measure_qubits=None,
                            snap_ops=set(), rng=rng)
    readout = _read_creg(ro_res.classical, ro.creg_name, code.n)

    residual = _residual_from_readout(code, readout, frame, logical_state)
    verdict = code.verdict(residual)

    record = QECShotRecord(
        shot=shot, seed=shot_seed, injected_errors=injected, raw_syndromes=raw,
        detection_events=events_log, decode_steps=steps, wall_clock=wall,
        observable_raw=readout, frame_flips=frame, verdict=verdict,
    )
    if timing is not None:
        _fill_shot_timing(record, decoder, steps, timing, rounds)
    return record


def _applied_syndrome_delta(code, applied, raw, t):
    """active 模式：已施加修正自身对各稳定子的贡献。"""
    if t == 0 or not applied.any():
        return np.zeros(code.m, dtype=np.uint8)
    return code.syndrome(applied).astype(np.uint8)


def _residual_from_readout(code, readout, frame, logical_state):
    """由 data 比特读数与已提交 frame 还原残余逻辑 Pauli。

    读数给出逻辑算符的奇偶；frame 是解码器声称已抵消的部分。二者异或后
    仍非零的分量，即真正逃逸的逻辑错误。
    """
    residual = np.zeros(2 * code.n, dtype=np.uint8)
    for i in range(code.k):
        support = code.logical_z[i][code.n:] if logical_state in ("0", "1") \
            else code.logical_x[i][:code.n]
        parity = int(np.dot(readout, support) % 2)
        flip = int(frame[2 * i]) if logical_state in ("0", "1") else int(frame[2 * i + 1])
        if parity ^ flip:
            residual ^= code.logical_x[i] if logical_state in ("0", "1") else code.logical_z[i]
    return residual
```

在 `aicir/qec/__init__.py` 追加：

```python
from .record import QECResult, QECShotRecord
from .runner import run

__all__ += ["QECShotRecord", "QECResult", "run"]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_runner_frame.py tests/qec/test_online_protocol.py -q`
Expected: PASS（18 passed）

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/record.py aicir/qec/runner.py aicir/qec/__init__.py \
        tests/qec/test_runner_frame.py tests/qec/test_online_protocol.py
git commit -m "feat(qec): 交错 simulate-decode 运行器（frame 模式）与协议不变量"
```

---

### Task 8: 实时模型 —— `TimingModel`、backlog 与提交延迟

**Files:**
- Modify: `aicir/qec/runner.py`（补 `TimingModel` / `_fill_shot_timing` / `_fill_timing_aggregates`）、`aicir/qec/__init__.py`
- Test: `tests/qec/test_timing.py`

**Interfaces:**
- Produces:
  - `TimingModel(round_duration: float, cost_to_seconds: Callable[[float], float] = float)`
  - `QECShotRecord.commit_latency` / `.backlog` 填充为 `(rounds,)` float 数组
  - `QECResult.max_backlog` / `.mean_commit_latency` / `.budget_violations`

- [ ] **Step 1: 写失败测试**

`tests/qec/test_timing.py`：

```python
import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.runner import TimingModel, backlog_sequence


def test_backlog_recurrence_matches_hand_computed_sequence():
    """backlog[t] = max(0, backlog[t-1] + decode_time[t] - round_duration)"""
    decode = [3.0, 0.5, 2.0, 0.25]
    got = backlog_sequence(decode, round_duration=1.0)
    # t0: max(0, 0+3-1)=2 ; t1: max(0, 2+0.5-1)=1.5 ; t2: max(0, 1.5+2-1)=2.5 ; t3: max(0,2.5+0.25-1)=1.75
    assert got == pytest.approx([2.0, 1.5, 2.5, 1.75])


def test_backlog_stays_zero_when_decoder_keeps_up():
    got = backlog_sequence([0.2, 0.2, 0.2], round_duration=1.0)
    assert got == pytest.approx([0.0, 0.0, 0.0])


def test_timing_fields_populate_when_model_given():
    code = get_code("steane")
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 1e-7)
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=4, shots=4, seed=0, timing=timing)
    assert result.max_backlog is not None
    assert result.mean_commit_latency is not None
    assert result.budget_violations is not None
    rec = result.records[0]
    assert rec.commit_latency.shape == (4,)
    assert rec.backlog.shape == (4,)


def test_budget_violation_counted_when_decode_exceeds_round_duration():
    code = get_code("steane")
    # 每轮声明代价 1.0，映射成 10s，远超 1e-6s 的轮时长 → 每轮都超预算
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 10.0)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=2, seed=0, timing=timing)
    assert result.budget_violations == 3 * 2


def test_no_violation_when_decoder_is_fast_enough():
    code = get_code("steane")
    timing = TimingModel(round_duration=1.0, cost_to_seconds=lambda c: c * 1e-9)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=3, shots=2, seed=0, timing=timing)
    assert result.budget_violations == 0
    assert result.max_backlog == pytest.approx(0.0)


def test_wall_clock_is_recorded_separately_from_modeled_time():
    """wall-clock 与建模时间必须是分开的字段，绝不混同。"""
    code = get_code("steane")
    timing = TimingModel(round_duration=1.0, cost_to_seconds=lambda c: c * 5.0)
    result = run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
                 rounds=2, shots=1, seed=0, timing=timing)
    rec = result.records[0]
    assert rec.wall_clock.shape == (2,)
    assert np.all(rec.wall_clock >= 0.0)
    # 建模的解码时长是 5s/轮，wall-clock 实测远小于此 —— 二者不相等即证明未混用
    assert rec.wall_clock.max() < 1.0


def test_timing_model_rejects_bad_round_duration():
    with pytest.raises(ValueError, match="round_duration"):
        TimingModel(round_duration=0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_timing.py -q`
Expected: FAIL —— `ImportError: cannot import name 'TimingModel'`

- [ ] **Step 3: 在 `aicir/qec/runner.py` 追加实时模型**

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class TimingModel:
    """实时预算模型。

    用**声明代价**而非 Python wall-clock：Python 解码器比 FPGA 慢约 10⁴ 倍，
    wall-clock 对实时可行性毫无意义。解码器经 cost_of() 声明自己的复杂度度量，
    cost_to_seconds 把它映射到用户掌控的硬件模型。

    wall-clock 也照常记录（免费），但在**独立字段** QECShotRecord.wall_clock 里，
    绝不与建模时间混同。
    """
    round_duration: float
    cost_to_seconds: Callable[[float], float] = float

    def __post_init__(self):
        if float(self.round_duration) <= 0.0:
            raise ValueError(f"round_duration 必须为正，收到 {self.round_duration}")


def backlog_sequence(decode_times, round_duration: float) -> list[float]:
    """确定性到达的单服务台排队：backlog[t] = max(0, backlog[t−1] + decode[t] − 轮时长)。

    backlog 线性增长（斜率为正）即**吞吐失败模式**：解码器永久落后。
    """
    out, backlog = [], 0.0
    for dt in decode_times:
        backlog = max(0.0, backlog + float(dt) - float(round_duration))
        out.append(backlog)
    return out


def _fill_shot_timing(record, decoder, steps, timing: TimingModel, rounds: int) -> None:
    """按声明代价填充该 shot 的 backlog 与提交延迟。"""
    decode_times = [float(timing.cost_to_seconds(step.cost)) for step in steps]
    backlog = backlog_sequence(decode_times, timing.round_duration)
    lag = int(getattr(decoder, "commit_lag", 0))
    # 提交延迟 = 排队延迟（该轮 backlog 减去自身解码时长，下限 0）+ 解码时长
    #            + 滞后提交带来的 lag × 轮时长
    latency = [
        max(0.0, backlog[t] - decode_times[t]) + decode_times[t] + lag * timing.round_duration
        for t in range(len(steps))
    ]
    record.backlog = np.asarray(backlog, dtype=float)
    record.commit_latency = np.asarray(latency, dtype=float)
    record.decode_times = np.asarray(decode_times, dtype=float)


def _fill_timing_aggregates(result, records, timing: TimingModel) -> None:
    """把逐 shot timing 汇总到 QECResult。"""
    if not records:
        result.max_backlog, result.mean_commit_latency, result.budget_violations = 0.0, 0.0, 0
        return
    result.max_backlog = float(max(r.backlog.max() for r in records))
    result.mean_commit_latency = float(
        np.mean(np.concatenate([r.commit_latency for r in records]))
    )
    result.budget_violations = int(sum(
        int((r.decode_times > timing.round_duration).sum()) for r in records
    ))
```

同时给 `QECShotRecord` 加一个字段（`aicir/qec/record.py`）：

```python
    decode_times: np.ndarray | None = None
```

在 `aicir/qec/__init__.py` 追加：

```python
from .runner import TimingModel

__all__ += ["TimingModel"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_timing.py -q`
Expected: PASS（7 passed）

注意 `test_budget_violation_counted_when_decode_exceeds_round_duration` 依赖 `keep_records` 默认值 ≥ shots，使聚合遍历到全部 shot 的记录。若该测试计数偏小，检查 `_fill_timing_aggregates` 是否只遍历了 `kept`——按设计它就是只统计保留下来的记录，测试里 shots=2 < keep_records=100，故应完全覆盖。

- [ ] **Step 5: 提交**

```bash
git add aicir/qec/runner.py aicir/qec/record.py aicir/qec/__init__.py tests/qec/test_timing.py
git commit -m "feat(qec): 实时预算模型、backlog 递推与提交延迟统计"
```

---

### Task 9: active 修正模式与 detector 参考值扣除

**Files:**
- Modify: `aicir/qec/runner.py`（校验 active 模式前置条件）
- Test: `tests/qec/test_correction_modes.py`

**Interfaces:**
- Consumes: Task 7 的 `run` 与 `_applied_syndrome_delta`
- Produces: `run(..., correction_mode="active")` 可用；解码器不产出 `corrections` 时抛错

**这个任务守的是 spec 里点名的那处细节：** active 模式下施加修正会把**原始**稳定子读数复位，若不扣除已施加修正自身的综合征贡献，下一轮的朴素差分就会放出一个**虚假 detection event**。处理正确时，两种模式交给解码器的事件流**逐字节相同**。

- [ ] **Step 1: 写失败测试**

`tests/qec/test_correction_modes.py`：

```python
import numpy as np
import pytest

from aicir.qec import run
from aicir.qec.codes import get_code
from aicir.qec.decoders import DecodeStep
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel


@pytest.mark.parametrize("name,kwargs", [("steane", {}), ("five_qubit", {})])
@pytest.mark.parametrize("rounds", [2, 4])
def test_frame_and_active_produce_identical_event_streams(name, kwargs, rounds):
    """两种模式交给解码器的 detection event 流必须逐字节相同。

    这正是「active 模式需扣除已施加修正的综合征贡献」那处细节的守卫测试。
    """
    code = get_code(name, **kwargs)
    common = dict(errors=PauliErrorModel(p_data=0.08, channel="depolarizing"),
                  rounds=rounds, shots=8, seed=17)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.detection_events, rb.detection_events)


@pytest.mark.parametrize("name,kwargs", [("steane", {}), ("five_qubit", {})])
def test_frame_and_active_agree_on_verdicts(name, kwargs):
    code = get_code(name, **kwargs)
    common = dict(errors=PauliErrorModel(p_data=0.08, channel="depolarizing"),
                  rounds=3, shots=16, seed=23)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    assert a.verdict_counts == b.verdict_counts
    assert [r.verdict for r in a.records] == [r.verdict for r in b.records]


def test_active_mode_rejects_frame_only_decoder():
    class FrameOnlyDecoder:
        name = "frame_only"
        window, commit_lag = 1, 0

        def reset(self, layout): self._c = -1
        def update(self, t, ev):
            self._c = t
            return DecodeStep(frame_flips=None, corrections=None, committed_through=t, cost=1.0)
        def flush(self): return DecodeStep(committed_through=self._c)
        def cost_of(self, t, ev): return 1.0

    code = get_code("steane")
    with pytest.raises(ValueError, match="active"):
        run(code, errors=PauliErrorModel(), decoder=FrameOnlyDecoder(),
            rounds=2, shots=1, correction_mode="active", seed=0)


def test_unknown_correction_mode_raises():
    code = get_code("steane")
    with pytest.raises(ValueError, match="correction_mode"):
        run(code, errors=PauliErrorModel(), decoder=LookupDecoder(code),
            rounds=1, shots=1, correction_mode="teleport", seed=0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_correction_modes.py -q`
Expected: FAIL —— `test_active_mode_rejects_frame_only_decoder` 不抛错；若事件流扣除逻辑有误，前两个测试也失败

- [ ] **Step 3: 在 `_run_one_shot` 中加入 active 模式前置校验**

在 `_run_one_shot` 的 `step = decoder.update(t, ev)` 之后、`committed_through` 校验之前插入：

```python
        if correction_mode == "active" and step.corrections is None:
            raise ValueError(
                f"correction_mode='active' 要求解码器每轮产出 corrections 门列表，但 "
                f"{getattr(decoder, 'name', '?')} 在轮 {t} 返回了 corrections=None。"
                f"改用 correction_mode='frame'，或让解码器填充 DecodeStep.corrections"
            )
```

**注意条件里不要再加 `and step.frame_flips is not None`**：只产出 frame 的解码器
（如本任务测试里的 `FrameOnlyDecoder`）两个字段都是 `None`，多这一个条件会让守卫
永远不触发，其配套测试也就必然失败。空列表 `[]` 表示「本轮无需修正」，是合法值，
只有 `None` 才代表「该解码器不支持 active 模式」。

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=. pytest tests/qec/test_correction_modes.py -q`
Expected: PASS（8 passed）

若 `test_frame_and_active_produce_identical_event_streams` 失败，问题必然在 `_applied_syndrome_delta`——它必须返回**已累积施加的全部修正**对各稳定子的综合征贡献，而不是仅本轮的。**不要放宽断言**，这个测试存在的唯一目的就是守住这处细节。

- [ ] **Step 5: 全量回归**

Run: `PYTHONPATH=. pytest tests/qec/ -q`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add aicir/qec/runner.py tests/qec/test_correction_modes.py
git commit -m "feat(qec): active 修正模式与 detector 参考值扣除"
```

---

### Task 10: 扩展点证明 —— 自定义码 / 调度 / 解码器端到端

**Files:**
- Test: `tests/qec/test_custom_plugin.py`

**这是 M1 最重要的测试**——它是唯一真正证明三个扩展点为实、而非停留在设想的测试。其余测试验证内置码，那是容易的情形。本任务**不改动任何实现代码**；若测试暴露出扩展点不可用，那才要回头改实现。

- [ ] **Step 1: 写测试**

`tests/qec/test_custom_plugin.py`：

```python
"""证明「插入自己的码 / 调度 / 解码器」不需要改模块内部代码。

三样东西全部在本文件内定义并注册，然后端到端跑通。
"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, cz, hadamard, measure, reset
from aicir.core.classical import ClassicalRegister
from aicir.qec import run
from aicir.qec.code import StabilizerCode, gf2_to_pauli
from aicir.qec.codes import CODES, get_code, register_code
from aicir.qec.decoders import DecodeStep, register_decoder, resolve_decoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.schedules import (
    BareAncillaSchedule, RoundCircuit, build_layout, register_schedule,
    resolve_schedule, verify_schedule,
)


# ---------- 1. 一个新的码：[[4,2,2]] 检测码 ----------

def build_422() -> StabilizerCode:
    """[[4,2,2]] 检测码：两个权重 4 的生成元，两个逻辑比特。"""
    return StabilizerCode.from_paulis(
        ["XXXX", "ZZZZ"],
        logical_x=["XXII", "XIXI"],
        logical_z=["ZIZI", "ZZII"],
        name="detection_422",
        coords={q: (0, q) for q in range(4)},
    )


# ---------- 2. 一个新的调度：反转 CNOT 顺序 ----------

class ReversedOrderSchedule(BareAncillaSchedule):
    """与内置裸 ancilla 调度相同，但 support 上的受控门按降序施加。

    对无 flag 的裸 ancilla 提取，顺序不影响 detector 确定性 —— verify_schedule 会证实这点。
    """

    name = "reversed"

    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit:
        data = tuple(range(code.n))
        ancilla = tuple(range(code.n, code.n + code.m))
        n_total = code.n + code.m
        reg = ClassicalRegister(code.m, creg_name)
        cir = Circuit(n_qubits=n_total)
        for j in range(code.m):
            anc = ancilla[j]
            labels = gf2_to_pauli(code.generators[j])
            cir.append(hadamard(anc))
            for q in range(code.n - 1, -1, -1):          # 降序 —— 与内置调度相反
                ch = labels[q]
                if ch == "X":
                    cir.append(cx(q, [anc]))
                elif ch == "Z":
                    cir.append(cz(q, [anc]))
                elif ch == "Y":
                    from aicir.core.circuit import cy
                    cir.append(cy(q, [anc]))
            cir.append(hadamard(anc))
        cir.append(measure(list(ancilla), creg=reg))
        cir.append(reset(list(ancilla)))
        return RoundCircuit(circuit=cir, creg_name=creg_name, ancilla_qubits=ancilla,
                            data_qubits=data, record_offset=int(round_index) * code.m)


# ---------- 3. 一个新的在线解码器：滑窗多数表决 ----------

class SlidingMajorityDecoder:
    """滑窗解码器：缓存 window 轮，只在某稳定子在窗内多数轮触发时才提交。

    这不是一个好解码器 —— 它存在的意义是证明「带窗口与滞后提交的在线解码器」
    能被平台正确驱动：因果性、committed_through 单调、flush 收尾。
    """

    name = "sliding_majority"

    def __init__(self, window: int = 3, commit_lag: int = 1):
        self.window = int(window)
        self.commit_lag = int(commit_lag)

    def reset(self, layout) -> None:
        self._layout = layout
        self._buffer = []
        self._committed = -1
        self._seen_rounds = []

    def cost_of(self, round_index, events) -> float:
        return float(self.window)          # 声明代价 = 窗口大小

    def update(self, round_index, events) -> DecodeStep:
        self._buffer.append(np.asarray(events, dtype=np.uint8))
        self._seen_rounds.append(int(round_index))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        # 只提交滞后 commit_lag 轮之前的轮次
        target = int(round_index) - self.commit_lag
        if target > self._committed:
            self._committed = target
        return DecodeStep(frame_flips=None, corrections=None,
                          committed_through=self._committed,
                          cost=self.cost_of(round_index, events))

    def flush(self) -> DecodeStep:
        """线路结束，强制提交所有未决。"""
        if self._seen_rounds:
            self._committed = max(self._committed, max(self._seen_rounds))
        return DecodeStep(committed_through=self._committed, cost=0.0)


# ---------- 测试 ----------

def test_custom_code_registers_and_validates():
    register_code("detection_422", build_422)
    assert "detection_422" in CODES
    code = get_code("detection_422")
    code.validate()
    assert (code.n, code.k, code.m) == (4, 2, 2)


def test_custom_schedule_registers_and_passes_detector_determinism():
    register_schedule("reversed", ReversedOrderSchedule)
    assert isinstance(resolve_schedule("reversed"), ReversedOrderSchedule)
    code = get_code("steane")
    verify_schedule(code, ReversedOrderSchedule(), rounds=3)


def test_custom_decoder_registers_and_resolves():
    register_decoder("sliding_majority", SlidingMajorityDecoder)
    dec = resolve_decoder("sliding_majority", window=3, commit_lag=1)
    assert isinstance(dec, SlidingMajorityDecoder)
    assert (dec.window, dec.commit_lag) == (3, 1)


def test_all_three_custom_pieces_run_end_to_end():
    """新码 + 新调度 + 新在线解码器，端到端跑通。"""
    register_code("detection_422", build_422)
    code = get_code("detection_422")
    result = run(
        code,
        schedule=ReversedOrderSchedule(),
        errors=PauliErrorModel(p_data=0.05, p_measure=0.02, channel="depolarizing"),
        decoder=SlidingMajorityDecoder(window=3, commit_lag=1),
        rounds=5, shots=12, seed=41,
    )
    assert result.shots == 12
    assert result.code_name == "detection_422"
    assert result.schedule_name == "reversed"
    assert result.decoder_name == "sliding_majority"
    assert sum(result.verdict_counts.values()) == 12
    assert result.records[0].detection_events.shape == (5, code.m)


def test_custom_decoder_commit_lag_is_respected_and_monotone():
    """滞后提交的解码器不得被平台误判为 committed_through 回退。"""
    code = get_code("steane")
    dec = SlidingMajorityDecoder(window=3, commit_lag=2)
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=dec,
                 rounds=6, shots=4, seed=9)
    committed = [s.committed_through for s in result.records[0].decode_steps]
    assert committed == sorted(committed)          # 单调不减
    assert committed[0] == -2 or committed[0] <= 0  # 前 commit_lag 轮尚无可提交轮次


def test_custom_decoder_works_with_timing_model():
    from aicir.qec.runner import TimingModel

    code = get_code("steane")
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 1e-6)
    result = run(code, errors=PauliErrorModel(p_data=0.05),
                 decoder=SlidingMajorityDecoder(window=4, commit_lag=1),
                 rounds=5, shots=3, seed=13, timing=timing)
    # 声明代价 = window = 4 → 每轮建模 4e-6s，轮时长 1e-6s → 每轮都超预算，backlog 线性增长
    assert result.budget_violations == 5 * 3
    assert result.max_backlog > 0.0
    rec = result.records[0]
    assert np.all(np.diff(rec.backlog) > 0)        # 吞吐失败模式：backlog 单调增长
```

- [ ] **Step 2: 运行测试**

Run: `PYTHONPATH=. pytest tests/qec/test_custom_plugin.py -q`
Expected: PASS（7 passed）

任何一项失败都说明**扩展点没做到位**——回头改实现，不要改这个测试来迁就实现。特别注意：
- `committed_through` 初值为 `-1`，滞后提交的解码器在前 `commit_lag` 轮会报负数，运行器的单调性校验必须容许这一点。
- `SlidingMajorityDecoder` 不产出 `corrections`，故只能用 `correction_mode="frame"`（默认）。

- [ ] **Step 3: 提交**

```bash
git add tests/qec/test_custom_plugin.py
git commit -m "test(qec): 自定义码/调度/在线解码器端到端扩展点证明"
```

---

### Task 11: 公开 API、README、demo 与文档收口

**Files:**
- Modify: `aicir/qec/__init__.py`（整理为一处完整导出）
- Create: `aicir/qec/README.md`、`demos/qec_online_demo.py`
- Modify: `CHANGELOG.md`、`CONTENTS.md`、`CLAUDE.md`
- Test: `tests/qec/test_public_api.py`

- [ ] **Step 1: 写失败测试**

`tests/qec/test_public_api.py`：

```python
import importlib

import pytest


def test_public_names_are_exported():
    qec = importlib.import_module("aicir.qec")
    expected = {
        "StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product",
        "CODES", "get_code", "register_code",
        "Detector", "Observable", "DetectorLayout",
        "BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule",
        "ErrorEvent", "PauliErrorModel",
        "DecodeStep", "LookupDecoder", "register_decoder", "resolve_decoder",
        "QECShotRecord", "QECResult", "run", "TimingModel",
    }
    missing = expected - set(qec.__all__)
    assert not missing, f"__all__ 缺少：{sorted(missing)}"
    for name in expected:
        assert hasattr(qec, name), f"aicir.qec 缺少属性 {name}"


def test_qec_core_has_no_optional_dependencies():
    """qec 核心只能依赖 numpy —— 不得 import torch/scipy/matplotlib/stim/pymatching。"""
    import pathlib
    import re

    root = pathlib.Path("aicir/qec")
    banned = re.compile(r"^\s*(?:import|from)\s+(torch|scipy|matplotlib|stim|pymatching)\b",
                        re.MULTILINE)
    for path in root.rglob("*.py"):
        hits = banned.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} 引入了禁止的依赖：{hits}"


def test_readme_exists():
    import pathlib
    assert pathlib.Path("aicir/qec/README.md").is_file()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=. pytest tests/qec/test_public_api.py -q`
Expected: FAIL —— README 不存在，`__all__` 可能不完整

- [ ] **Step 3: 整理 `aicir/qec/__init__.py`**

把此前各任务追加的导出合并成一处：

```python
"""aicir.qec

量子纠错（Quantum Error Correction）实验平台。

面向**新型在线实时纠错/解码算法**：码、syndrome 提取调度、在线解码器三处均可
插拔，均不需要改模块内部代码。内置的五个码是参考实现与验证语料，不是产品本身。

当前为 M1（骨架）。M3 可视化与 M2 规模化/Stim 互操作见 README 的里程碑一节。
"""

from __future__ import annotations

from .code import StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product
from .codes import CODES, get_code, register_code
from .decoders import DecodeStep, LookupDecoder, register_decoder, resolve_decoder
from .detectors import Detector, DetectorLayout, Observable
from .errors import ErrorEvent, PauliErrorModel
from .record import QECResult, QECShotRecord
from .runner import TimingModel, run
from .schedules import (BareAncillaSchedule, register_schedule, resolve_schedule,
                        verify_schedule)

__all__ = [
    "StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product",
    "CODES", "get_code", "register_code",
    "Detector", "Observable", "DetectorLayout",
    "BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule",
    "ErrorEvent", "PauliErrorModel",
    "DecodeStep", "LookupDecoder", "register_decoder", "resolve_decoder",
    "QECShotRecord", "QECResult", "run", "TimingModel",
]
```

- [ ] **Step 4: 写 `aicir/qec/README.md`**

内容需覆盖（中文，跟随其余子系统 README 的体例）：

1. **定位**：面向新型在线实时纠错/解码算法的实验平台；三处插拔点；内置码是语料不是产品。
2. **快速开始**：`get_code` → `PauliErrorModel` → `LookupDecoder` → `run` → `result.summary()` 的最小可运行例子。
3. **三个扩展点**，各给一段可运行代码：
   - 插入新码：`StabilizerCode.from_paulis(...)` + `register_code`，并强调 `validate()` 是免费拿到的最有力保障。
   - 插入新调度：实现 `build_encode`/`build_round`/`build_readout`，并**务必**用 `verify_schedule` 验证 detector 确定性。
   - 插入新在线解码器：`reset`/`update`/`flush`/`cost_of`，说明 `window`/`commit_lag`/`committed_through` 语义。
4. **在线与实时语义**：因果性为何是结构性的；`TimingModel` 用声明代价而非 wall-clock 的理由；backlog 斜率为正即吞吐失败。
5. **两种修正模式**：frame（默认）与 active 的差别与适用场景。
6. **已知局限**：`LookupDecoder` 是逐轮解码，多轮空时解码需等 M2 的 MWPM；重复码 `distance()` 为 1 的原因；`LookupDecoder` 构造时吃 code 是 M1 临时做法。
7. **里程碑**：M1 已交付内容；M3 可视化、M2 Stim/PyMatching 互操作与 `benchmark()` 的范围。

- [ ] **Step 5: 写 `demos/qec_online_demo.py`**

```python
"""aicir.qec 在线实时解码最小演示。

运行：PYTHONPATH=. python demos/qec_online_demo.py
"""

from aicir.qec import (LookupDecoder, PauliErrorModel, TimingModel, get_code,
                       run, verify_schedule)
from aicir.qec.schedules import BareAncillaSchedule


def main() -> None:
    code = get_code("surface", d=3)
    print(f"码：{code}  距离 = {code.distance()}")

    # 提取调度的结构性检验：无噪声下每个 detector 必须恒为 0
    verify_schedule(code, BareAncillaSchedule(), rounds=3)
    print("detector 确定性检验通过")

    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 2e-7)
    result = run(
        code,
        errors=PauliErrorModel(p_data=0.01, p_measure=0.01, channel="depolarizing"),
        decoder=LookupDecoder(code),
        rounds=5, shots=200, seed=0, timing=timing,
    )
    print(result.summary())

    if result.failure_records:
        rec = result.failure_records[0]
        print(f"\n首个失败 shot #{rec.shot}（判定 {rec.verdict}）注入的错误：")
        for e in rec.injected_errors[:8]:
            print(f"  轮 {e.round_index}  比特 {e.qubit}  {e.pauli}  ({e.source})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 更新仓库文档**

- `CHANGELOG.md`：在顶部加一条 dated 条目（2026-08-04），说明新增 `aicir.qec` M1：稳定子码 GF(2) 核心、detector/observable 模型、流式在线解码协议、实时预算模型、交错运行器、五个内置码、三处插拔点。
- `CONTENTS.md`：目录树中补 `aicir/qec/`。
- `CLAUDE.md`：在「Subsystems」列表中补一条：

```markdown
- `aicir/qec/` — 量子纠错实验平台，面向**在线实时**解码算法。三处插拔点：码（`StabilizerCode.from_paulis` + `register_code`）、syndrome 提取调度（`Schedule` 协议 + `register_schedule`）、在线解码器（`OnlineDecoder` 流式协议 + `register_decoder`）。数据模型采用 Stim 语义的 detector/observable（为 M2 的 Stim/PyMatching 互操作做准备），raw syndrome 并存。运行器逐轮交错「模拟 ↔ 解码」，因果性由「未来轮次尚未被模拟」这一结构事实保证，而非靠约定。`TimingModel` 用解码器**声明的代价**而非 Python wall-clock 建模实时预算与 backlog。当前为 **M1**：无 DEM / Stim / MWPM / `benchmark()` 扫描 / 可视化（分属 M2、M3）。注意重复码 `distance()` 返回 1（它对未受保护的基无防护），故 `LookupDecoder` 用于重复码时须显式传 `t` 与 `error_basis`。
```

- [ ] **Step 7: 运行全部测试**

Run: `PYTHONPATH=. pytest tests/qec/ -q`
Expected: 全部通过

Run: `PYTHONPATH=. python demos/qec_online_demo.py`
Expected: 打印码信息、detector 检验通过、结果摘要

Run: `PYTHONPATH=. pytest -q`
Expected: 全仓库测试通过（确认没有破坏既有子系统）

- [ ] **Step 8: 提交**

```bash
git add aicir/qec/__init__.py aicir/qec/README.md demos/qec_online_demo.py \
        tests/qec/test_public_api.py CHANGELOG.md CONTENTS.md CLAUDE.md
git commit -m "docs(qec): 公开 API 收口、README 使用手册、在线解码 demo"
```

---

## Self-Review

**1. Spec 覆盖检查**

| Spec 组件 | 对应 Task |
| --- | --- |
| 组件 1 `code.py` StabilizerCode | Task 1 |
| 组件 2 `detectors.py` | Task 3 |
| 组件 3 `schedules/` + `verify_schedule` | Task 4 |
| 组件 4 `errors.py` PauliErrorModel | Task 5 |
| 组件 5 `decoders/` 在线协议 + LookupDecoder | Task 6 |
| 组件 6 实时模型 TimingModel/backlog | Task 8 |
| 组件 7 frame / active 双修正模式 | Task 7（frame）、Task 9（active + 参考值扣除） |
| 组件 8 `runner.py` 交错主循环 | Task 4（无噪声采集）、Task 7（完整循环） |
| 组件 9 `record.py` | Task 7 |
| 组件 10 `codes/` 五个内置码 | Task 2 |
| 错误处理（validate/distance/verify_schedule/运行器拒绝项） | Task 1、4、7、9 |
| 测试计划七个文件 | Task 1–10（test_code_algebra / test_builtin_codes / test_detectors / test_schedules / test_errors / test_lookup_decoder / test_runner_frame / test_online_protocol / test_timing / test_correction_modes / test_custom_plugin / test_public_api） |
| 交付物（README/demo/CHANGELOG/CONTENTS/CLAUDE.md） | Task 11 |

无遗漏。spec 的「测试计划」列了 7 个测试文件，本计划拆成 12 个——`test_code_algebra` 拆出 `test_builtin_codes`、`test_detectors` 拆出 `test_schedules`、另加 `test_errors` 与 `test_public_api`，覆盖只增不减。

**2. 占位符扫描**

无 TBD / TODO / "类似 Task N" / "适当处理错误"。每个 Step 都含可直接运行的代码或可直接执行的命令。Task 11 Step 4 的 README 以七点提纲给出而非全文——README 是散文体使用手册，提纲已锁定必须覆盖的每一项内容（含三段可运行示例与四条已知局限），实现者据此撰写不会产生歧义。

**3. 类型一致性检查**

- `StabilizerCode` 的 `.n/.k/.m/.generators/.signs/.logical_x/.logical_z/.name/.coords` 在 Task 1 定义，Task 2/4/6/7 的用法与之一致。
- `pauli_to_gf2(item, n_qubits=None)` / `gf2_to_pauli(vec)` 签名在 Task 1 定义，Task 2/4/6 调用一致。
- `RoundCircuit` 字段 `circuit/creg_name/ancilla_qubits/data_qubits/record_offset` 在 Task 4 定义，Task 7/10 使用一致。
- `DecodeStep` 字段 `frame_flips/corrections/committed_through/cost` 在 Task 6 定义，Task 7/8/9/10 使用一致。
- `DetectorLayout.detection_events(raw, round_index, reference)` 在 Task 3 定义，Task 4/7 调用一致。
- `QECShotRecord.decode_times` 在 Task 8 补入 `record.py`，仅由 `_fill_shot_timing`（Task 8）写、`_fill_timing_aggregates`（Task 8）读。
- `collect_noiseless_syndromes(code, schedule, rounds, *, logical_state, backend, seed) -> np.ndarray` 在 Task 4 定义，只返回 `raw`（不返回 reference）；`verify_schedule` 与 `run` 都自行用 `zeros(m)` 作参考值。
- `deterministic_round0(code, logical_state) -> tuple[int, ...]` 在 Task 4 定义，`build_layout` 调用，Task 4 测试直接断言其返回值。
- `DetectorLayout.round0_stabilizers` 在 Task 3 定义，Task 4 `build_layout` 填充，Task 3/4 测试断言。
- `run(...)` 的 `rounds` 默认值为 **2**（轮 0 是制备轮，默认 1 不构成纠错运行）。
- `verify_schedule(code, schedule, rounds, *, backend, shots)` 在 Task 4 定义，Task 10 与 demo 调用一致。
- 解码器 `name` 属性：`LookupDecoder.name = "lookup"` 是类属性，`run` 用 `getattr(decoder, "name", ...)` 读取，测试断言 `result.decoder_name == "lookup"` 一致。

**4. 已知的实现风险（实现者注意）**

- Task 4 的 `verify_schedule` 依赖 Task 4 Step 4 的 `collect_noiseless_syndromes`，二者在同一 Task 内，`schedules/__init__.py` 对 `runner` 用**延迟导入**避免循环依赖——不要把它提到模块顶层。
- Task 7 的 `_residual_from_readout` 是 M1 里最容易写错的函数。判据以 Task 7/9 的测试为准：无噪声必须 0 逻辑错误率、单轮小 p 必须 0 逻辑错误率、frame 与 active 判定必须完全一致。若这三条中任何一条不过，先怀疑这个函数。
- Task 9 的 `_applied_syndrome_delta` 必须用**累积**已施加修正（`applied` 向量），不是本轮增量。

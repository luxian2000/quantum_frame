# MPS 近似模拟引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `aicir/simulator/` 新增 bond-dimension 截断的 MPS（矩阵乘积态）近似模拟引擎，支持低纠缠电路突破精确态矢量的内存墙，并接入 `Measure.run(method="mps")` 与新的 `MPSEstimator` primitive。

**Architecture:** MPS 把 n 比特纯态存成 n 个 rank-3 张量（`(Dl,2,Dr)`），维护正交中心（orthogonality center）做标准 TEBD 式演化：单比特门就地作用、相邻双比特门经 SVD 截断更新、非相邻双比特门用 SWAP 网络移到相邻并跟踪逻辑↔物理置换。截断由 `max_bond_dim`（硬上限）+ `cutoff`（相对奇异值阈值）共同控制。所有数值走 `Backend` 张量原语（`tensordot/transpose/reshape/conj` 已有，本计划新增 `svd`），因此 NumPy/GPU 后端通用且在 GPU 上对参数门可微。

**Tech Stack:** Python，NumPy，PyTorch（可选，`pytest.importorskip`），既有 `aicir.backends` / `aicir.core` / `aicir.simulator` / `aicir.primitives` / `aicir.measure`。

## Global Constraints

- 后端范围：仅 `NumpyBackend` + `GPUBackend`；`NPUBackend.svd()` 显式 `raise NotImplementedError`（CLAUDE.md NPU complex64 限制）。
- 引擎只接受 1/2 比特门；≥3 比特门 `raise ValueError`，提示先经 `aicir.transpile.DecomposePass`。
- 只支持纯态、无噪声；`ControlFlow` 指令 `raise ValueError`。
- 比特序 msb 约定：逻辑 qubit 0 为 MSB，与既有 `State`/`tn_statevector` 一致。
- 复数 dtype 跟随 backend（NumPy `complex64`、GPU `torch.complex64`）。
- 注释/docstring 用中文，风格对齐既有子包。
- 从 repo 根运行、`PYTHONPATH=.`；每个 Task 末尾 commit。
- 截断参数默认：`max_bond_dim=None`（无硬上限）、`cutoff=1e-10`。

---

## File Structure

- `aicir/backends/base.py` — 新增抽象方法 `svd`
- `aicir/backends/numpy_backend.py` / `gpu_backend.py` / `npu_backend.py` — 实现 `svd`
- `aicir/simulator/mps.py` — `MPSState` + 门作用 + 截断 + `mps_statevector`/`mps_expectation`
- `aicir/simulator/__init__.py` — re-export MPS 公共入口
- `aicir/primitives/mps_estimator.py` — `MPSEstimator`
- `aicir/primitives/__init__.py` — re-export `MPSEstimator`
- `aicir/measure/measure.py` — `Measure.run(method="mps", ...)` 分派
- `aicir/simulator/README.md` / `CHANGELOG.md` — 文档
- `tests/simulator/test_mps_*.py`、`tests/primitives/test_mps_estimator.py`、`tests/measure/test_mps_measure.py` — 测试

---

### Task 1: Backend `svd` 原语

**Files:**
- Modify: `aicir/backends/base.py`（在 `tensordot` 一族附近新增抽象 `svd`）
- Modify: `aicir/backends/numpy_backend.py`
- Modify: `aicir/backends/gpu_backend.py`
- Modify: `aicir/backends/npu_backend.py`
- Test: `tests/backends/test_svd.py`

**Interfaces:**
- Produces: `Backend.svd(matrix) -> (U, S, Vh)`，reduced SVD；`S` 一维降序奇异值（实数），`matrix ≈ U @ diag(S) @ Vh`。

- [ ] **Step 1: 写失败测试**

```python
# tests/backends/test_svd.py
import numpy as np
import pytest

from aicir.backends import NumpyBackend


def test_numpy_svd_reconstructs():
    bk = NumpyBackend()
    m = bk.cast(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.complex64))
    u, s, vh = bk.svd(m)
    u, s, vh = bk.to_numpy(u), bk.to_numpy(s), bk.to_numpy(vh)
    recon = u @ np.diag(s) @ vh
    assert np.allclose(recon, bk.to_numpy(m), atol=1e-5)
    assert u.shape == (3, 2) and s.shape == (2,) and vh.shape == (2, 2)
    assert np.all(np.diff(s.real) <= 1e-6)  # 降序


def test_gpu_svd_reconstructs():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    m = bk.cast(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64))
    u, s, vh = bk.svd(m)
    recon = torch.matmul(torch.matmul(u, torch.diag(s).to(u.dtype)), vh)
    assert torch.allclose(recon, m, atol=1e-4)


def test_npu_svd_not_implemented():
    from aicir.backends.npu_backend import NPUBackend

    with pytest.raises(NotImplementedError):
        NPUBackend.svd(object.__new__(NPUBackend), np.eye(2))
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_svd.py -q`
Expected: FAIL（`Backend` 无 `svd` / `AttributeError`）

- [ ] **Step 3: base.py 新增抽象方法**

在 `aicir/backends/base.py` 的 `conj` 方法定义之后插入：

```python
    def svd(self, matrix):
        """约化 SVD：返回 (U, S, Vh)，S 为一维降序奇异值。

        matrix ≈ U @ diag(S) @ Vh。MPS 截断用；仅 NumPy/GPU 后端实现，
        NPU 因 complex64 限制不支持（见 CLAUDE.md）。
        """
        raise NotImplementedError("子类需实现 svd")
```

- [ ] **Step 4: numpy 实现**

在 `aicir/backends/numpy_backend.py` 的 `conj` 之后插入：

```python
    def svd(self, matrix):
        m = np.asarray(matrix, dtype=np.complex128)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        return u.astype(self._dtype), s.astype(np.float64), vh.astype(self._dtype)
```

- [ ] **Step 5: gpu 实现**

在 `aicir/backends/gpu_backend.py` 的 `conj` 之后插入：

```python
    def svd(self, matrix):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u, s, vh
```

- [ ] **Step 6: npu 覆写**

在 `aicir/backends/npu_backend.py` 的 `conj` 覆写附近插入：

```python
    def svd(self, matrix):
        raise NotImplementedError(
            "MPS SVD 暂不支持 NPU；见 CLAUDE.md NPU complex64 限制"
        )
```

- [ ] **Step 7: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_svd.py -q`
Expected: PASS（GPU 用例在无 torch 时 skip；NPU 用例无 torch_npu 环境仍能构造裸实例并抛 NotImplementedError）

- [ ] **Step 8: Commit**

```bash
git add aicir/backends/base.py aicir/backends/numpy_backend.py aicir/backends/gpu_backend.py aicir/backends/npu_backend.py tests/backends/test_svd.py
git commit -m "feat(backends): add svd primitive for MPS truncation (numpy/gpu; npu NotImplemented)"
```

---

### Task 2: `MPSState` 骨架与 `to_statevector`

**Files:**
- Create: `aicir/simulator/mps.py`
- Test: `tests/simulator/test_mps_state.py`

**Interfaces:**
- Produces:
  - `MPSState(tensors, n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10)`；属性 `tensors`(list)、`n_qubits`、`backend`、`max_bond_dim`、`cutoff`、`truncation_error`(float,初值0)、`oc`(int,正交中心,初值0)、`logical_at`(list,物理site→逻辑qubit)、`site_of`(list,逻辑qubit→物理site)。
  - `MPSState.zero_state(n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10) -> MPSState`。
  - `MPSState.to_statevector() -> State`（按逻辑比特序还原，msb）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_state.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState


def test_zero_state_statevector():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-6)


def test_zero_state_shapes_and_bookkeeping():
    bk = NumpyBackend()
    mps = MPSState.zero_state(4, bk)
    assert mps.n_qubits == 4
    assert len(mps.tensors) == 4
    for t in mps.tensors:
        assert np.asarray(bk.to_numpy(t)).shape == (1, 2, 1)
    assert mps.logical_at == [0, 1, 2, 3]
    assert mps.site_of == [0, 1, 2, 3]
    assert mps.truncation_error == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_state.py -q`
Expected: FAIL（`aicir.simulator.mps` 不存在）

- [ ] **Step 3: 写 `mps.py` 骨架**

```python
# aicir/simulator/mps.py
"""MPS（矩阵乘积态）近似模拟引擎：bond 截断的纯态演化（Spec 2）。

每比特一个 rank-3 张量 (Dl, 2, Dr)，维护正交中心做 TEBD 式演化。
数值走 Backend 张量原语（tensordot/transpose/reshape/conj/svd），
NumPy/GPU 后端通用；GPU 上对参数门可微。仅纯态、无噪声、1/2 比特门。
"""

from __future__ import annotations

import numpy as np

from ..core.state import State


class MPSState:
    """bond 截断的矩阵乘积态。"""

    def __init__(self, tensors, n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10):
        self.tensors = list(tensors)
        self.n_qubits = int(n_qubits)
        self.backend = backend
        self.max_bond_dim = None if max_bond_dim is None else int(max_bond_dim)
        self.cutoff = float(cutoff)
        self.truncation_error = 0.0
        self.oc = 0
        self.logical_at = list(range(self.n_qubits))
        self.site_of = list(range(self.n_qubits))

    @classmethod
    def zero_state(cls, n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10):
        n = int(n_qubits)
        if n <= 0:
            raise ValueError("n_qubits 必须为正整数")
        tensors = []
        for _ in range(n):
            arr = np.zeros((1, 2, 1), dtype=np.complex64)
            arr[0, 0, 0] = 1.0
            tensors.append(backend.reshape(backend.cast(arr), (1, 2, 1)))
        return cls(tensors, n, backend, max_bond_dim=max_bond_dim, cutoff=cutoff)

    def to_statevector(self):
        """收缩为稠密态矢量并按逻辑比特序还原，返回 State（仅供小 n 验证）。"""
        bk = self.backend
        cur = bk.reshape(self.tensors[0], (2, self.tensors[0].shape[2]))
        for s in range(1, self.n_qubits):
            t = self.tensors[s]
            dl = t.shape[0]
            cur = bk.tensordot(cur, t, ([cur.ndim - 1], [0]))  # (..., 2, Dr)
            new_rows = 1
            shape = np.asarray(bk.to_numpy(cur)).shape
            for d in shape[:-1]:
                new_rows *= int(d)
            cur = bk.reshape(cur, (new_rows, shape[-1]))
        phys = bk.reshape(cur, (2,) * self.n_qubits)  # 物理 site 序
        perm = [self.site_of[q] for q in range(self.n_qubits)]  # 逻辑序
        logical = bk.transpose(phys, perm)
        data = bk.reshape(logical, (1 << self.n_qubits, 1))
        return State(data, self.n_qubits, bk)
```

注：`t.shape` 对 numpy/torch 张量均可用；`Dr` 取 `t.shape[2]`。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_state.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aicir/simulator/mps.py tests/simulator/test_mps_state.py
git commit -m "feat(simulator): MPSState skeleton with zero_state and to_statevector"
```

---

### Task 3: 单比特门就地作用

**Files:**
- Modify: `aicir/simulator/mps.py`
- Test: `tests/simulator/test_mps_single_gate.py`

**Interfaces:**
- Produces: `MPSState._apply_one_site(self, m2, site)`（`m2` 为 `(2,2)` 后端张量，`site` 为物理 site 下标）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_single_gate.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState

_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)


def test_apply_x_flips_qubit0():
    bk = NumpyBackend()
    mps = MPSState.zero_state(2, bk)
    mps._apply_one_site(bk.cast(_X), 0)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1.0  # qubit0=MSB -> |10>
    assert np.allclose(sv, expected, atol=1e-6)


def test_apply_h_superposition():
    bk = NumpyBackend()
    mps = MPSState.zero_state(1, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    assert np.allclose(sv, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_single_gate.py -q`
Expected: FAIL（`_apply_one_site` 不存在）

- [ ] **Step 3: 实现 `_apply_one_site`**

在 `MPSState` 内新增（`to_statevector` 之后）：

```python
    def _apply_one_site(self, m2, site):
        """单比特门就地作用于物理 site 的物理指标（不改正交性、不截断）。"""
        bk = self.backend
        t = self.tensors[site]  # (Dl, 2, Dr)
        # tensordot(m2 (o,i), t (Dl,i,Dr)) over i -> (o, Dl, Dr) -> (Dl, o, Dr)
        out = bk.tensordot(m2, t, ([1], [1]))
        self.tensors[site] = bk.transpose(out, [1, 0, 2])
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_single_gate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aicir/simulator/mps.py tests/simulator/test_mps_single_gate.py
git commit -m "feat(simulator): MPS single-qubit gate application"
```

---

### Task 4: 相邻双比特门（正交中心 + SVD 截断）

**Files:**
- Modify: `aicir/simulator/mps.py`
- Test: `tests/simulator/test_mps_two_gate.py`

**Interfaces:**
- Produces:
  - `MPSState._move_center_right(self, i)` / `_move_center_left(self, i)` / `_ensure_center(self, p)`（SVD 搬运正交中心，不截断）。
  - `MPSState._apply_two_site(self, op4, s, *, truncate)`（`op4` 为 `(2,2,2,2)` 张量，作用物理 site `(s, s+1)`；`truncate=True` 时按 `max_bond_dim`/`cutoff` 截断并累加 `truncation_error`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_two_gate.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.simulator.mps import MPSState

_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
)


def test_bell_state_exact():
    bk = NumpyBackend()
    mps = MPSState.zero_state(2, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    op4 = bk.reshape(bk.cast(_CNOT), (2, 2, 2, 2))
    mps._apply_two_site(op4, 0, truncate=True)
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(4, dtype=complex)
    expected[0b00] = expected[0b11] = 1 / np.sqrt(2)
    assert np.allclose(sv, expected, atol=1e-6)
    assert mps.truncation_error < 1e-9  # 无截断


def test_center_move_preserves_state():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    op4 = bk.reshape(bk.cast(_CNOT), (2, 2, 2, 2))
    mps._apply_two_site(op4, 0, truncate=True)  # 纠缠 0-1
    before = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    mps._ensure_center(2)
    mps._ensure_center(0)
    after = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    assert np.allclose(before, after, atol=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_two_gate.py -q`
Expected: FAIL（`_apply_two_site` 不存在）

- [ ] **Step 3: 实现截断辅助与中心搬运**

在 `mps.py` 模块顶层（`class MPSState` 之前）新增：

```python
def _keep_count(s_np, max_bond_dim, cutoff):
    """按 cutoff（相对最大奇异值）+ max_bond_dim 决定保留的奇异值个数（≥1）。"""
    if s_np.size == 0:
        return 0
    smax = float(s_np[0]) if s_np[0] > 0 else 1.0
    k = int((s_np > cutoff * smax).sum())
    k = max(k, 1)
    if max_bond_dim is not None:
        k = min(k, int(max_bond_dim))
    return k
```

在 `MPSState` 内新增（`_apply_one_site` 之后）：

```python
    def _move_center_right(self, i):
        """把正交中心从 i 移到 i+1（SVD，不截断）。"""
        bk = self.backend
        t = self.tensors[i]  # (Dl, 2, Dr)
        dl, dr = t.shape[0], t.shape[2]
        mat = bk.reshape(t, (dl * 2, dr))
        u, s, vh = bk.svd(mat)  # u:(dl*2,K) s:(K,) vh:(K,dr)
        k = int(np.asarray(bk.to_numpy(s)).shape[0])
        self.tensors[i] = bk.reshape(u, (dl, 2, k))
        s_c = bk.cast(s)
        carry = vh * bk.reshape(s_c, (k, 1))  # (K, dr)
        nxt = self.tensors[i + 1]  # (dr, 2, Dr2)
        self.tensors[i + 1] = bk.tensordot(carry, nxt, ([1], [0]))  # (K, 2, Dr2)
        self.oc = i + 1

    def _move_center_left(self, i):
        """把正交中心从 i 移到 i-1（SVD，不截断）。"""
        bk = self.backend
        t = self.tensors[i]  # (Dl, 2, Dr)
        dl, dr = t.shape[0], t.shape[2]
        mat = bk.reshape(t, (dl, 2 * dr))
        u, s, vh = bk.svd(mat)  # u:(dl,K) s:(K,) vh:(K,2*dr)
        k = int(np.asarray(bk.to_numpy(s)).shape[0])
        self.tensors[i] = bk.reshape(vh, (k, 2, dr))
        s_c = bk.cast(s)
        carry = u * bk.reshape(s_c, (1, k))  # (dl, K)
        prev = self.tensors[i - 1]  # (Dm, 2, dl)
        self.tensors[i - 1] = bk.tensordot(prev, carry, ([2], [0]))  # (Dm, 2, K)
        self.oc = i - 1

    def _ensure_center(self, p):
        while self.oc < p:
            self._move_center_right(self.oc)
        while self.oc > p:
            self._move_center_left(self.oc)

    def _apply_two_site(self, op4, s, *, truncate):
        """作用双比特算子 op4=(out_s,out_{s+1},in_s,in_{s+1}) 于物理 site (s, s+1)。"""
        bk = self.backend
        self._ensure_center(s)  # 中心在 s -> 右侧 s+1 为右规范
        a, b = self.tensors[s], self.tensors[s + 1]
        dl, dr = a.shape[0], b.shape[2]
        theta = bk.tensordot(a, b, ([2], [0]))  # (Dl, 2, 2, Dr)
        # 作用 op4 于两个物理指标 (axis 1,2)
        applied = bk.tensordot(op4, theta, ([2, 3], [1, 2]))  # (out_s, out_{s+1}, Dl, Dr)
        applied = bk.transpose(applied, [2, 0, 1, 3])  # (Dl, out_s, out_{s+1}, Dr)
        mat = bk.reshape(applied, (dl * 2, 2 * dr))
        u, sv, vh = bk.svd(mat)
        s_np = np.asarray(bk.to_numpy(sv)).real
        if truncate:
            k = _keep_count(s_np, self.max_bond_dim, self.cutoff)
        else:
            k = int(s_np.shape[0])
        total = float((s_np ** 2).sum()) or 1.0
        discarded = float((s_np[k:] ** 2).sum()) / total
        self.truncation_error += discarded
        u_k = u[:, :k]
        vh_k = vh[:k, :]
        s_k = bk.cast(sv[:k])
        self.tensors[s] = bk.reshape(u_k, (dl, 2, k))
        vh_scaled = vh_k * bk.reshape(s_k, (k, 1))  # 把奇异值吸收进右张量
        self.tensors[s + 1] = bk.reshape(vh_scaled, (k, 2, dr))
        self.oc = s + 1
```

注：切片 `u[:, :k]` / `vh[:k, :]` / `sv[:k]` 对 numpy 与 torch 张量均有效且保留 autograd。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_two_gate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aicir/simulator/mps.py tests/simulator/test_mps_two_gate.py
git commit -m "feat(simulator): MPS adjacent two-qubit gate with SVD truncation and center moves"
```

---

### Task 5: 非相邻双比特门（SWAP 网络 + 置换跟踪）

**Files:**
- Modify: `aicir/simulator/mps.py`
- Test: `tests/simulator/test_mps_swap.py`

**Interfaces:**
- Produces:
  - `MPSState._swap_adjacent(self, p)`（交换物理 site p 与 p+1，含 `logical_at`/`site_of` 记账，不截断）。
  - `MPSState.apply_two_qubit(self, matrix, axes)`（`matrix` 为 `(4,4)` 后端张量，`axes=[u, v]` 为逻辑比特，`matrix` 按 `axes[0]`=MSB、`axes[1]`=LSB 编码；自动 SWAP 到相邻并按物理顺序摆正 op4）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_swap.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot
from aicir.simulator import tn_statevector
from aicir.simulator.mps import MPSState

_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
)


def test_cnot_non_adjacent_exact():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 0)
    mps.apply_two_qubit(bk.cast(_CNOT), [0, 2])  # 非相邻 control=0,target=2
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0b000] = expected[0b101] = 1 / np.sqrt(2)  # |000>+|101>
    assert np.allclose(sv, expected, atol=1e-6)


def test_reversed_axes_control_gt_target():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._apply_one_site(bk.cast(_H), 2)
    mps.apply_two_qubit(bk.cast(_CNOT), [2, 0])  # control=2 (MSB), target=0
    sv = np.asarray(mps.to_statevector().to_numpy()).reshape(-1)
    expected = np.zeros(8, dtype=complex)
    expected[0b000] = expected[0b101] = 1 / np.sqrt(2)
    assert np.allclose(sv, expected, atol=1e-6)


def test_swap_bookkeeping_restored_state():
    bk = NumpyBackend()
    mps = MPSState.zero_state(3, bk)
    mps._swap_adjacent(0)
    assert mps.logical_at[0] == 1 and mps.logical_at[1] == 0
    assert mps.site_of[0] == 1 and mps.site_of[1] == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_swap.py -q`
Expected: FAIL（`apply_two_qubit`/`_swap_adjacent` 不存在）

- [ ] **Step 3: 实现 SWAP 与双比特调度**

在 `mps.py` 模块顶层新增 SWAP 常量：

```python
_SWAP4 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64
).reshape(2, 2, 2, 2)
```

在 `MPSState` 内新增（`_apply_two_site` 之后）：

```python
    def _swap_adjacent(self, p):
        """交换物理 site p 与 p+1（用 SWAP 门，不截断），并更新逻辑↔物理记账。"""
        bk = self.backend
        self._apply_two_site(bk.cast(_SWAP4), p, truncate=False)
        lu, lv = self.logical_at[p], self.logical_at[p + 1]
        self.logical_at[p], self.logical_at[p + 1] = lv, lu
        self.site_of[lu], self.site_of[lv] = p + 1, p

    def apply_two_qubit(self, matrix, axes):
        """作用双比特门 matrix(4x4) 于逻辑比特 axes=[u,v]（u=MSB, v=LSB）。

        自动用 SWAP 把 u,v 移到相邻物理 site，再按物理顺序摆正 op4 后作用。
        """
        bk = self.backend
        u, v = int(axes[0]), int(axes[1])
        pu, pv = self.site_of[u], self.site_of[v]
        op4 = bk.reshape(matrix, (2, 2, 2, 2))  # (out_u, out_v, in_u, in_v)
        if pu < pv:
            for p in range(pv - 1, pu, -1):  # 把 v 冒泡到 pu+1
                self._swap_adjacent(p)
            s = self.site_of[u]  # == pu
            self._apply_two_site(op4, s, truncate=True)  # site s=u, s+1=v
        else:
            for p in range(pu - 1, pv, -1):  # 把 u 冒泡到 pv+1
                self._swap_adjacent(p)
            s = self.site_of[v]  # == pv
            op4_t = bk.transpose(op4, [1, 0, 3, 2])  # 物理序 (v, u)
            self._apply_two_site(op4_t, s, truncate=True)
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_swap.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aicir/simulator/mps.py tests/simulator/test_mps_swap.py
git commit -m "feat(simulator): MPS non-adjacent two-qubit gates via SWAP network with permutation tracking"
```

---

### Task 6: 电路驱动 `mps_statevector` + 门分派 + 拒绝路径

**Files:**
- Modify: `aicir/simulator/mps.py`
- Modify: `aicir/simulator/__init__.py`
- Test: `tests/simulator/test_mps_statevector.py`

**Interfaces:**
- Consumes: `gate_tensors(gate, backend) -> [(matrix, axes), ...]`（`aicir.core.gates`）；`ControlFlow`（`aicir.ir`）；`State`（`aicir.core.state`）。
- Produces:
  - `mps_statevector(circuit, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> MPSState`。
  - `aicir.simulator.mps_statevector` 顶层可导入。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_statevector.py
import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz, toffoli
from aicir.core.circuit import if_
from aicir.core.classical import ClassicalRegister
from aicir.measure import measure
from aicir.simulator import tn_statevector, mps_statevector


def _random_circuit(n, depth, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=n)
    for _ in range(depth):
        for q in range(n):
            c.append(rx(q, float(rng.uniform(0, np.pi))))
            c.append(rz(q, float(rng.uniform(0, np.pi))))
        for q in range(n - 1):
            c.append(cnot(q + 1, [q]))
    return c


def test_mps_matches_exact_full_bond():
    bk = NumpyBackend()
    c = _random_circuit(5, 3, seed=1)
    mps_sv = np.asarray(mps_statevector(c, backend=bk).to_statevector().to_numpy()).reshape(-1)
    exact = np.asarray(tn_statevector(c, backend=bk).to_numpy()).reshape(-1)
    assert np.allclose(mps_sv, exact, atol=1e-5)


def test_three_qubit_gate_rejected():
    bk = NumpyBackend()
    c = Circuit(n_qubits=3)
    c.append(toffoli(2, [0, 1]))
    with pytest.raises(ValueError, match="1/2"):
        mps_statevector(c, backend=bk)


def test_control_flow_rejected():
    bk = NumpyBackend()
    reg = ClassicalRegister(1)
    c = Circuit(n_qubits=1)
    c.append(measure(0, creg=reg))
    c.append(if_(reg[0] == 1, [("x", 0)]))
    with pytest.raises(ValueError):
        mps_statevector(c, backend=bk)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_statevector.py -q`
Expected: FAIL（`mps_statevector` 未从 `aicir.simulator` 导出）

- [ ] **Step 3: 实现驱动函数**

在 `mps.py` 顶部补充导入：

```python
from ..core.gates import gate_tensors
from ..ir import ControlFlow
```

在 `mps.py` 末尾新增：

```python
def _resolve_backend(circuit, backend):
    if backend is not None:
        return backend
    bk = getattr(circuit, "backend", None)
    if bk is not None:
        return bk
    from ..backends import NumpyBackend
    return NumpyBackend()


def _build_mps(circuit, backend, max_bond_dim, cutoff):
    n = int(circuit.n_qubits)
    mps = MPSState.zero_state(n, backend, max_bond_dim=max_bond_dim, cutoff=cutoff)
    for gate in circuit.gates:
        if isinstance(gate, ControlFlow):
            raise ValueError("控制流指令不支持 MPS 模拟；请用 Measure.run 执行")
        for matrix, axes in gate_tensors(gate, backend):
            k = len(axes)
            if k == 1:
                m2 = backend.reshape(matrix, (2, 2))
                mps._apply_one_site(m2, mps.site_of[int(axes[0])])
            elif k == 2:
                mps.apply_two_qubit(matrix, list(axes))
            else:
                raise ValueError(
                    f"MPS 引擎仅接受 1/2 比特门（收到作用 {k} 比特）；"
                    "请先用 aicir.transpile.DecomposePass 分解"
                )
    return mps


def mps_statevector(circuit, *, max_bond_dim=None, cutoff=1e-10, backend=None):
    """经 MPS 演化电路，返回 MPSState（bond 截断的近似末态）。"""
    backend = _resolve_backend(circuit, backend)
    return _build_mps(circuit, backend, max_bond_dim, cutoff)
```

- [ ] **Step 4: 导出到 `__init__`**

在 `aicir/simulator/__init__.py` 末尾追加：

```python
from .mps import MPSState, mps_statevector  # noqa: E402
```

并把 `MPSState`, `mps_statevector` 加入该文件的 `__all__`（若无 `__all__` 则跳过）。

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_statevector.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/mps.py aicir/simulator/__init__.py tests/simulator/test_mps_statevector.py
git commit -m "feat(simulator): mps_statevector circuit driver with gate dispatch and rejection guards"
```

---

### Task 7: `mps_expectation`（Pauli transfer 路径 + 稠密回退）

**Files:**
- Modify: `aicir/simulator/mps.py`
- Modify: `aicir/simulator/__init__.py`
- Test: `tests/simulator/test_mps_expectation.py`

**Interfaces:**
- Consumes: `Hamiltonian`（`.terms -> List[PauliString]`）、`PauliString`（`.coefficient`、`.qubit_labels`）（`aicir.core.operators`）。
- Produces: `mps_expectation(circuit, observable, *, max_bond_dim=None, cutoff=1e-10, backend=None)`（返回后端标量：NumPy 上为 float，GPU 上为保留计算图的 torch 标量，与 `tn_expectation` 一致）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_mps_expectation.py
import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz
from aicir.core.operators import Hamiltonian, PauliString
from aicir.primitives import StatevectorEstimator
from aicir.simulator import mps_expectation


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    for q in range(4):
        c.append(rz(q, float(rng.uniform(0, np.pi))))
    return c


def test_hamiltonian_matches_statevector_estimator():
    bk = NumpyBackend()
    c = _circ(7)
    H = Hamiltonian([("ZZII", -1.0), ("IXXI", 0.5), ("ZIIZ", 0.3)])
    got = float(np.real(complex(mps_expectation(c, H, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, H).value
    assert abs(got - ref) < 1e-5


def test_paulistring_matches():
    bk = NumpyBackend()
    c = _circ(3)
    ps = PauliString("ZIZI", coefficient=0.7)
    got = float(np.real(complex(mps_expectation(c, ps, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, ps).value
    assert abs(got - ref) < 1e-5


def test_dense_matrix_fallback():
    bk = NumpyBackend()
    c = _circ(5)
    op = np.diag([1.0] * 8 + [-1.0] * 8).astype(np.complex64)  # 稠密 16x16
    got = float(np.real(complex(mps_expectation(c, op, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, op).value
    assert abs(got - ref) < 1e-5


def test_gpu_expectation_differentiable():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    theta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    c = Circuit(n_qubits=2)
    c.append(rx(0, theta))
    c.append(cnot(1, [0]))
    H = Hamiltonian([("ZI", 1.0)])
    val = mps_expectation(c, H, backend=bk)
    val.backward()
    assert theta.grad is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_expectation.py -q`
Expected: FAIL（`mps_expectation` 未导出）

- [ ] **Step 3: 实现 Pauli transfer 与公共入口**

在 `mps.py` 模块顶层新增 Pauli 常量：

```python
_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex64),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
}
```

在 `mps.py` 末尾新增（`_transfer` 做 transfer 收缩、`_pauli_terms` 归一化观测量、`mps_expectation` 组装）：

```python
def _transfer(mps, phys_labels):
    """按物理 site 序做 <psi| (⊗ P_site) |psi> 的 transfer 收缩，返回后端 (1,1) 标量。"""
    bk = mps.backend
    left = bk.cast(np.array([[1.0]], dtype=np.complex64))  # (bra_left, ket_left)
    for s in range(mps.n_qubits):
        a = mps.tensors[s]  # (Dl, 2, Dr) ket
        p = bk.cast(_PAULI[phys_labels[s]])  # (o, i)
        pa = bk.tensordot(p, a, ([1], [1]))  # (o, Dl, Dr)
        pa = bk.transpose(pa, [1, 0, 2])  # (Dl, o, Dr) ket with P
        conj_a = bk.conj(a)  # bra (Dl, 2, Dr)
        t = bk.tensordot(left, conj_a, ([0], [0]))  # (ket_left, 2, bra_right)
        left = bk.tensordot(t, pa, ([0, 1], [0, 1]))  # (bra_right, ket_right)
    return left  # 末端 left 为 (1,1)


def _pauli_terms(observable):
    """把 observable 归一为 [(coefficient, qubit_labels)] 列表；非 Pauli 返回 None。"""
    if hasattr(observable, "terms"):  # Hamiltonian
        return [(ps.coefficient, ps.qubit_labels) for ps in observable.terms]
    if hasattr(observable, "qubit_labels"):  # PauliString
        return [(observable.coefficient, observable.qubit_labels)]
    return None


def mps_expectation(circuit, observable, *, max_bond_dim=None, cutoff=1e-10, backend=None):
    """经 MPS 求期望 <psi|O|psi>。Pauli/Hamiltonian 走 transfer 收缩（不稠密化），
    任意稠密矩阵回退到 to_statevector。返回后端标量（GPU 上可微）。"""
    backend = _resolve_backend(circuit, backend)
    mps = _build_mps(circuit, backend, max_bond_dim, cutoff)
    terms = _pauli_terms(observable)
    if terms is None:  # 稠密矩阵回退：稠密化后走 expectation_sv
        psi = mps.to_statevector()
        operator = observable.to_matrix(backend) if hasattr(observable, "to_matrix") else backend.cast(observable)
        return backend.expectation_sv(psi.to_numpy(), operator)
    n = mps.n_qubits
    norm2 = _transfer(mps, ["I"] * n)  # <psi|psi>，(1,1)
    total = None
    for coef, labels in terms:
        phys = ["I"] * n
        for q in range(n):  # 逻辑 Pauli 放到其物理 site
            phys[mps.site_of[q]] = labels[q]
        contrib = backend.cast(np.array([[complex(coef)]], dtype=np.complex64)) * _transfer(mps, phys)
        total = contrib if total is None else backend.add(total, contrib)
    ratio = total / norm2  # (1,1) 逐元素相除；numpy/torch 均保留计算图
    return backend.real(backend.reshape(ratio, ()))  # 0 维标量的实部
```

说明：`total`/`norm2` 都是 `(1,1)` 后端张量，`total / norm2` 对 numpy `ndarray` 与 `torch.Tensor` 均逐元素、保留 autograd（无需 backend 提供 div 原语）；`backend.reshape(ratio, ())` 压成 0 维标量，`backend.real` 取实部，得到与 `tn_expectation` 一致的「NumPy 上 float、GPU 上可微 torch 标量」返回值。

- [ ] **Step 4: 导出到 `__init__`**

修改 `aicir/simulator/__init__.py` 的 mps 导入行为：

```python
from .mps import MPSState, mps_statevector, mps_expectation  # noqa: E402
```

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_expectation.py -q`
Expected: PASS（GPU 用例在无 torch 时 skip）

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/mps.py aicir/simulator/__init__.py tests/simulator/test_mps_expectation.py
git commit -m "feat(simulator): mps_expectation with Pauli transfer contraction and dense fallback"
```

---

### Task 8: 截断误差层测试

**Files:**
- Test: `tests/simulator/test_mps_truncation.py`

**Interfaces:**
- Consumes: `mps_statevector`、`MPSState.truncation_error`、`tn_statevector`。

- [ ] **Step 1: 写测试**

```python
# tests/simulator/test_mps_truncation.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz
from aicir.simulator import tn_statevector, mps_statevector


def _brickwork(n, depth, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=n)
    for d in range(depth):
        for q in range(n):
            c.append(rx(q, float(rng.uniform(0, np.pi))))
            c.append(rz(q, float(rng.uniform(0, np.pi))))
        start = d % 2
        for q in range(start, n - 1, 2):
            c.append(cnot(q + 1, [q]))
    return c


def _l2_err(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b))


def test_full_bond_is_exact():
    bk = NumpyBackend()
    c = _brickwork(6, 4, seed=11)
    exact = tn_statevector(c, backend=bk).to_numpy()
    full = mps_statevector(c, max_bond_dim=2 ** 3, backend=bk)
    assert _l2_err(full.to_statevector().to_numpy(), exact) < 1e-5
    assert full.truncation_error < 1e-9


def test_small_chi_is_lossy_and_monotone():
    bk = NumpyBackend()
    c = _brickwork(8, 6, seed=22)
    exact = tn_statevector(c, backend=bk).to_numpy()
    errs = []
    for chi in (1, 2, 4, 16):
        mps = mps_statevector(c, max_bond_dim=chi, backend=bk)
        errs.append(_l2_err(mps.to_statevector().to_numpy(), exact))
    # chi=1 明显有损（证明截断真实发生，而非静默精确）
    assert errs[0] > 1e-3
    # 误差随 chi 增大单调不增（允许极小数值容差）
    for i in range(len(errs) - 1):
        assert errs[i + 1] <= errs[i] + 1e-6
    # chi=1 时累计 truncation_error 非零
    mps1 = mps_statevector(c, max_bond_dim=1, backend=bk)
    assert mps1.truncation_error > 0.0
```

- [ ] **Step 2: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_mps_truncation.py -q`
Expected: PASS

（若 `test_small_chi_is_lossy_and_monotone` 的单调性偶发失败：说明正交中心搬运有缺陷——Task 4 的 `_apply_two_site` 必须先 `_ensure_center(s)` 使 `s+1` 右规范，SVD 才给出最优截断。修正 Task 4 的实现而非放宽此断言。）

- [ ] **Step 3: Commit**

```bash
git add tests/simulator/test_mps_truncation.py
git commit -m "test(simulator): MPS truncation-error tier (exact at full bond, lossy+monotone at small chi)"
```

---

### Task 9: `Measure.run(method="mps")` 集成

**Files:**
- Modify: `aicir/measure/measure.py`（`run` 签名 + `method` 分派）
- Test: `tests/measure/test_mps_measure.py`

**Interfaces:**
- Consumes: `mps_statevector(circuit, max_bond_dim=, cutoff=, backend=).to_statevector()`。
- Produces: `Measure.run(circuit, ..., method="mps", max_bond_dim=None, cutoff=1e-10)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/measure/test_mps_measure.py
import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx
from aicir.measure import Measure, measure
from aicir.noise import NoiseModel


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    return c


def test_mps_matches_statevector_method():
    bk = NumpyBackend()
    c = _circ(9)
    sv = Measure(bk).run(c, shots=None, return_state=True, method="statevector")
    mp = Measure(bk).run(c, shots=None, return_state=True, method="mps")
    a = np.asarray(sv.state.to_numpy()).reshape(-1)
    b = np.asarray(mp.state.to_numpy()).reshape(-1)
    assert np.allclose(a, b, atol=1e-5)


def test_mps_rejects_noise():
    bk = NumpyBackend()
    c = _circ(1)
    c.noise_model = NoiseModel()
    with pytest.raises(ValueError):
        Measure(bk).run(c, method="mps")


def test_mps_rejects_embedded_measure():
    bk = NumpyBackend()
    c = Circuit(n_qubits=2)
    c.append(measure(0))
    with pytest.raises(ValueError):
        Measure(bk).run(c, method="mps")


def test_mps_rejects_nonempty_snap():
    bk = NumpyBackend()
    c = _circ(2)
    with pytest.raises(ValueError):
        Measure(bk).run(c, snap=[0], method="mps")


def test_mps_rejects_initial_state():
    bk = NumpyBackend()
    c = _circ(3)
    psi = bk.zeros_state(4)
    with pytest.raises(ValueError):
        Measure(bk).run(c, initial_state=psi, method="mps")
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/measure/test_mps_measure.py -q`
Expected: FAIL（`method="mps"` 不被接受）

- [ ] **Step 3: 修改 `Measure.run` 签名与分派**

在 `aicir/measure/measure.py` 的 `run` 方法签名中，把关键字段扩展（保持既有参数不动，新增两个 kw）：

```python
    def run(self, circuit, shots=1, measure_qubits=(), snap=None,
            sm="avg", seed=None, *,
            initial_state=None, initial_density_matrix=None,
            observables=None, return_state=True, method="statevector",
            max_bond_dim=None, cutoff=1e-10) -> Result:
```

把 `method` 校验行改为接受 `"mps"`：

```python
        if method not in ("statevector", "tensor", "mps"):
            raise ValueError(f"method 必须是 statevector/tensor/mps，收到 {method!r}")
```

在既有 `if method == "tensor":` 分支之后、`norm_shots = ...` 之前插入 `mps` 分支：

```python
        if method == "mps":
            if getattr(circuit, "noise_model", None) is not None:
                raise ValueError("method='mps' 仅支持纯态，无法用于含噪线路")
            if any(_is_measure(g) for g in circuit_instructions(circuit)):
                raise ValueError("method='mps' 不支持线路内嵌 measure 标记")
            if initial_state is not None or initial_density_matrix is not None:
                raise ValueError("method='mps' 始终从 |0...0> 出发，不接受 initial_state/initial_density_matrix")
            if snap not in (None, [], ()):  # 无逐门快照语义
                raise ValueError("method='mps' 不支持非空 snap")
            from ..simulator import mps_statevector
            psi = mps_statevector(circuit, max_bond_dim=max_bond_dim, cutoff=cutoff, backend=backend).to_statevector()
            from ..core.circuit import Circuit as _Circuit
            stripped = _Circuit(n_qubits=n)
            return self.run(
                stripped, shots=shots, measure_qubits=measure_qubits, snap=None,
                sm=sm, seed=seed, initial_state=psi, observables=observables,
                return_state=return_state, method="statevector",
            )
```

注：`initial_state` 拒绝校验放在 `method="mps"` 分支内（对递归调用无影响，因递归用 `method="statevector"`）。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_mps_measure.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aicir/measure/measure.py tests/measure/test_mps_measure.py
git commit -m "feat(measure): Measure.run(method='mps') with max_bond_dim/cutoff and purity guards"
```

---

### Task 10: `MPSEstimator` primitive

**Files:**
- Create: `aicir/primitives/mps_estimator.py`
- Modify: `aicir/primitives/__init__.py`
- Test: `tests/primitives/test_mps_estimator.py`

**Interfaces:**
- Consumes: `mps_expectation`（`aicir.simulator`）；`BaseEstimator`/`normalize_run_inputs`/`pair_observables`（`aicir.primitives.base`）；`EstimateResult`（`aicir.primitives.results`）。
- Produces:
  - `MPSEstimator(*, max_bond_dim=None, cutoff=1e-10, backend=None)`，`estimate(circuit, hamiltonian, **_) -> _EnergyResult`、`run(circuits, observables, *, shots=None, parameter_values=None) -> EstimateResult|list`。

- [ ] **Step 1: 写失败测试**

```python
# tests/primitives/test_mps_estimator.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import cnot, rx, rz
from aicir.core.operators import Hamiltonian
from aicir.primitives import StatevectorEstimator, MPSEstimator


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    for q in range(4):
        c.append(rz(q, float(rng.uniform(0, np.pi))))
    return c


_H = Hamiltonian([("ZZII", -1.0), ("IZZI", -1.0), ("IIZZ", -1.0), ("XIII", 0.5)])


def test_run_matches_statevector():
    bk = NumpyBackend()
    c = _circ(4)
    got = MPSEstimator(backend=bk).run(c, _H)
    ref = StatevectorEstimator(bk).run(c, _H).value
    assert abs(got.value - ref) < 1e-5
    assert got.metadata["method"] == "mps"
    assert "truncation_error" in got.metadata


def test_estimate_energy_contract():
    bk = NumpyBackend()
    c = _circ(6)
    res = MPSEstimator(backend=bk).estimate(c, _H)
    ref = StatevectorEstimator(bk).run(c, _H).value
    assert abs(res.energy - ref) < 1e-5


def test_shots_rejected():
    import pytest

    bk = NumpyBackend()
    c = _circ(1)
    with pytest.raises(ValueError):
        MPSEstimator(backend=bk).run(c, _H, shots=100)


def test_gradient_matches_statevector():
    bk = NumpyBackend()
    from aicir import Parameter

    theta = Parameter("t")
    c = Circuit(n_qubits=2)
    c.append(rx(0, theta))
    c.append(cnot(1, [0]))
    H = Hamiltonian([("ZI", 1.0)])
    x = np.array([0.7])
    g_mps = MPSEstimator(backend=bk).gradient(c, H, parameter_values=x, method="psr").gradient
    g_ref = StatevectorEstimator(bk).gradient(c, H, parameter_values=x, method="psr").gradient
    assert np.allclose(g_mps, g_ref, atol=1e-5)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/primitives/test_mps_estimator.py -q`
Expected: FAIL（`MPSEstimator` 不存在）

- [ ] **Step 3: 实现 `MPSEstimator`**

```python
# aicir/primitives/mps_estimator.py
"""MPS 近似期望估计 primitive：经 bond 截断的 MPS 求 <psi|H|psi>。"""

from __future__ import annotations

import numpy as np

from ..backends import NumpyBackend
from .base import BaseEstimator, normalize_run_inputs, pair_observables
from .estimator import _EnergyResult
from .results import EstimateResult


class MPSEstimator(BaseEstimator):
    """MPS 近似期望：bond 截断的纯态演化后求期望，无采样噪声。

    ``max_bond_dim`` 越大越接近精确（None 表示无硬上限）；``cutoff`` 为相对
    奇异值截断阈值。暴露 :meth:`estimate` 供 ``BasicVQE(energy_estimator=...)`` 注入。
    """

    def __init__(self, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> None:
        self.backend = backend if backend is not None else NumpyBackend()
        self.max_bond_dim = max_bond_dim
        self.cutoff = float(cutoff)

    def _expectation(self, circuit, observable):
        from ..simulator import mps_statevector, mps_expectation

        # 复用一次构建：先取 truncation_error，再求期望（两次构建成本相当，语义清晰）
        mps = mps_statevector(circuit, max_bond_dim=self.max_bond_dim, cutoff=self.cutoff, backend=self.backend)
        raw = mps_expectation(circuit, observable, max_bond_dim=self.max_bond_dim, cutoff=self.cutoff, backend=self.backend)
        return float(np.real(complex(self.backend.to_numpy(raw) if hasattr(raw, "shape") else raw))), float(mps.truncation_error)

    def estimate(self, circuit, hamiltonian, **_ignored):
        value, _err = self._expectation(circuit, hamiltonian)
        return _EnergyResult(value)

    def run(self, circuits, observables, *, shots=None, parameter_values=None):
        if shots is not None:
            raise ValueError("MPSEstimator 为（近似）精确路径，不接受 shots")
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = []
        for circuit, observable in zip(items, paired):
            value, err = self._expectation(circuit, observable)
            results.append(
                EstimateResult(
                    value=value,
                    metadata={"method": "mps", "max_bond_dim": self.max_bond_dim, "truncation_error": err},
                )
            )
        return results[0] if single else results
```

- [ ] **Step 4: 导出到 `__init__`**

在 `aicir/primitives/__init__.py` 中，仿照现有 estimator 导入新增 `MPSEstimator`，并加入 `__all__`：

```python
from .mps_estimator import MPSEstimator
```

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/primitives/test_mps_estimator.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add aicir/primitives/mps_estimator.py aicir/primitives/__init__.py tests/primitives/test_mps_estimator.py
git commit -m "feat(primitives): MPSEstimator for approximate MPS energy estimation"
```

---

### Task 11: 文档（README + CHANGELOG）

**Files:**
- Modify: `aicir/simulator/README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 更新 `aicir/simulator/README.md`**

在第 1 节方法选择指南末尾追加一条：

```
需要更大比特数的低纠缠电路近似末态/期望（bond 截断，有损）？
  -> mps_statevector(circuit, max_bond_dim=..., cutoff=...) / mps_expectation(...)
     或 Measure.run(circuit, method="mps", max_bond_dim=..., cutoff=...)
```

在第 2 节公共 API 之后新增一节「MPS 近似引擎」，说明：`mps_statevector` 返回 `MPSState`（`.to_statevector()` 还原稠密态、`.truncation_error` 累计丢弃权重）；`mps_expectation` 对 `Hamiltonian`/`PauliString` 走 transfer 收缩不稠密化、对稠密矩阵回退 `to_statevector`；`max_bond_dim`（硬上限，None=无）+ `cutoff`（相对奇异值阈值，默认 1e-10）共同控制截断；仅接受 1/2 比特门（≥3 比特门先经 `aicir.transpile.DecomposePass`）、非相邻双比特门自动 SWAP、仅纯态无噪声；后端仅 NumPy/GPU（GPU 上可微），NPU 因 complex64 SVD 限制不支持。

把第 6 节「MPS/张量网络截断近似（有损压缩）不在本模块范围内，属于另一独立规划（Spec 2）」改为：「MPS/张量网络截断近似（有损压缩）见本模块的 `mps_statevector`/`mps_expectation`（Spec 2，已落地）。」

- [ ] **Step 2: 更新 `CHANGELOG.md`**

在文件顶部新增 `## 2026-07-06` 段（若已存在则在其 `### Added` 下追加）：

```markdown
## 2026-07-06

### Added

- **`aicir.simulator` MPS（矩阵乘积态）近似模拟引擎（Spec 2）：`mps_statevector` /
  `mps_expectation`，并为 `Measure.run` 增加 `method="mps"`。** bond 截断由
  `max_bond_dim`（硬上限）+ `cutoff`（相对奇异值阈值，默认 1e-10）共同控制；正交
  中心 + SVD 的 TEBD 式演化，单比特门就地作用、相邻双比特门 SVD 截断、非相邻双比特
  门自动 SWAP 并跟踪逻辑↔物理置换。新增 `Backend.svd` 原语（NumPy/GPU 实现，NPU 因
  complex64 限制 `NotImplementedError`）。`mps_expectation` 对 `Hamiltonian`/
  `PauliString` 走 transfer 收缩不稠密化、GPU 上对参数门可微。新增
  `aicir.primitives.MPSEstimator`（可注入 `BasicVQE(energy_estimator=...)`）。仅纯态、
  无噪声、1/2 比特门（≥3 比特门先经 `DecomposePass`）。
```

- [ ] **Step 3: 全量回归**

Run: `PYTHONPATH=. pytest tests/simulator tests/backends tests/primitives tests/measure -q && PYTHONPATH=. pytest -q`
Expected: PASS（新用例全过，既有套件不回归）

- [ ] **Step 4: Commit**

```bash
git add aicir/simulator/README.md CHANGELOG.md
git commit -m "docs(simulator): document MPS engine (Spec 2) in README and CHANGELOG"
```

---

## Self-Review

**Spec 覆盖：**
- `Backend.svd`（numpy/gpu/npu）→ Task 1 ✓
- `MPSState` 数据结构 + `zero_state` + `to_statevector` → Task 2 ✓
- 单比特门 → Task 3 ✓
- 相邻双比特门 + SVD 截断 + `truncation_error` + 正交中心 → Task 4 ✓
- 非相邻双比特门 SWAP 网络 + `site_permutation`（`logical_at`/`site_of`）→ Task 5 ✓
- 3+ 比特门拒绝 + `ControlFlow` 拒绝 + `max_bond_dim`/`cutoff` API → Task 6 ✓
- `mps_statevector`/`mps_expectation`（Pauli transfer + 稠密回退）+ 顶层再导出 → Task 6/7 ✓
- `Measure.run(method="mps")` + 纯态/无噪声/无 initial_state/无 snap 守卫 → Task 9 ✓
- `MPSEstimator`（`estimate`/`run`/shots 拒绝/metadata truncation_error）→ Task 10 ✓
- 测试契约：精确匹配层（full bond）→ Task 6/8；截断误差层（小 chi 有损 + 单调）→ Task 8；后端覆盖（numpy 必过、gpu importorskip、npu NotImplementedError）→ Task 1/7；拒绝路径 → Task 6/9 ✓
- README + CHANGELOG + 移除「另立 Spec 2」措辞 → Task 11 ✓

**占位符扫描：** 无 TBD/TODO；每个代码步骤含完整可直接落地的代码与命令（Task 7 `mps_expectation` 为单一版本，无「见下」式占位）。

**类型一致性：** `MPSState.site_of`（逻辑→物理）在 Task 2 定义，Task 5/6/7 一致使用；`_apply_two_site(op4, s, *, truncate)` 签名 Task 4 定义、Task 5 `_swap_adjacent` 一致调用；`mps_statevector`/`mps_expectation`/`MPSState`/`mps_expectation` 返回后端标量约定与 `tn_expectation` 一致；`_EnergyResult` 复用自 `aicir.primitives.estimator`（Task 10）。

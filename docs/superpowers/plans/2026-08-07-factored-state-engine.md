# Factored State Simulation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a factored (product-state) simulation engine that keeps a pure state as a set of independent sub-system factors and joins them only when a gate entangles them, complementing — never replacing — the dense state vector.

**Architecture:** The engine lives in `aicir/simulator/factored.py`, above the `Backend` interface, exactly like the existing tensor-network and MPS engines. It performs **bookkeeping plus two backend primitives**: `backend.apply_statevector_local` for intra-factor gates and `backend.kron` for joins. Both are already implemented NPU-safe, so the engine gains Ascend support with no NPU-specific code — this is the portability claim of the paper's §4 exercised deliberately. Version 1 is **join-only**: factors merge but never re-split, which means no SVD and therefore no contact with Ascend's missing complex-SVD kernel. Re-splitting is explicitly deferred (Task 11 records why).

**Tech Stack:** Python, NumPy (required), PyTorch (optional, for GPU/NPU backends), pytest.

## Global Constraints

- Comments, docstrings, and READMEs are **Chinese**; match surrounding style (`CLAUDE.md`).
- Core must not gain hard dependencies on `torch`/`scipy`/`matplotlib`. Tests needing them use `pytest.importorskip`.
- **No bare `np.complex64` / `np.complex128` / `torch.complex64` literals in core paths.** dtype comes from the backend (`aicir/dtypes.py:resolve_dtype`).
- **Torch paths must not use bitwise ops** (`bitwise_and`, `bitwise_xor`, `>>`, `<<`): Ascend has no kernel and silently falls back to CPU (see `_apply_z_signs_` in `aicir/core/operators.py` for the approved alternative).
- **Torch paths must not index complex tensors** (`aclnnIndex` is unavailable on Ascend). Take `torch.real`/`torch.imag` views first — see `aicir/simulator/mps.py:_permute_basis`.
- Qubit ordering is **big-endian**: qubit `q` occupies bit `n_qubits-1-q`.
- Run tests from repo root with `PYTHONPATH=. pytest`.
- **Never** add `Co-Authored-By` or any AI-attribution trailer to commit messages.
- The dense path's behaviour must be **byte-identical** before and after this work. Every task runs the full suite before commit.

## File Structure

| File | Responsibility |
|---|---|
| `aicir/simulator/factored.py` (create) | `FactoredState` container, gate application, joining, materialisation, drivers |
| `aicir/simulator/__init__.py` (modify) | Export `FactoredState`, `factored_statevector`, `factored_expectation` |
| `aicir/measure/measure.py` (modify) | `method="factored"` dispatch and rejections |
| `tests/simulator/test_factored.py` (create) | Correctness vs dense oracle, invariants, rejections |
| `tests/simulator/test_factored_npu_safety.py` (create) | Dispatch-mode proof that no Ascend-missing op is issued |
| `scripts/npu/factored_probe.py` (create) | Real-hardware probe |
| `scripts/npu/factored.sh` (create) | Probe wrapper |
| `scripts/npu/run_npu_tests.py` (modify) | Register `factored` suite |
| `aicir/simulator/README.md` (modify) | User-facing documentation |
| `CHANGELOG.md` (modify) | Dated entry |

---

### Task 1: `FactoredState` container and invariants

**Files:**
- Create: `aicir/simulator/factored.py`
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: `aicir.dtypes.resolve_dtype`, `aicir.backends.base.Backend`
- Produces: `FactoredState(factors, n_qubits, backend)` where `factors` is a
  `list[tuple[tuple[int, ...], Any]]` of `(sorted_qubits, amplitude_tensor)`;
  properties `n_qubits`, `n_factors`, `max_factor_width`; classmethod
  `FactoredState.zero_state(n_qubits, backend)`; method `factor_index_of(qubit) -> int`.

- [ ] **Step 1: Write the failing test**

```python
"""因子化（乘积态）引擎：容器与不变量。"""

import numpy as np
import pytest

from aicir import NumpyBackend
from aicir.simulator.factored import FactoredState


class TestZeroState:
    def test_zero_state_is_fully_factored(self):
        st = FactoredState.zero_state(5, NumpyBackend())
        assert st.n_qubits == 5
        assert st.n_factors == 5           # |0>^5 完全可分
        assert st.max_factor_width == 1

    def test_zero_state_factors_cover_every_qubit_exactly_once(self):
        st = FactoredState.zero_state(4, NumpyBackend())
        seen = [q for qubits, _ in st.factors for q in qubits]
        assert sorted(seen) == [0, 1, 2, 3]

    def test_factor_index_of_locates_qubit(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        assert st.factor_index_of(2) == st.factor_index_of(2)
        idx = st.factor_index_of(1)
        assert 1 in st.factors[idx][0]

    def test_factor_qubits_are_sorted(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        for qubits, _ in st.factors:
            assert list(qubits) == sorted(qubits)

    def test_amplitudes_follow_backend_dtype(self):
        st = FactoredState.zero_state(2, NumpyBackend(dtype=np.complex64))
        assert np.asarray(st.factors[0][1]).dtype == np.complex64

    def test_rejects_non_positive_qubit_count(self):
        with pytest.raises(ValueError):
            FactoredState.zero_state(0, NumpyBackend())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aicir.simulator.factored'`

- [ ] **Step 3: Write minimal implementation**

```python
"""因子化（乘积态）模拟引擎：把纯态保存为若干互不纠缠的因子。

与稠密态矢量互补，**不替代**它：门只作用在自己所属的因子上，只有当一个门跨越
两个因子时才把它们合并（kron）。低纠缠线路因此只在小张量上演算。

v1 **只合并不拆分**——不做可分性检测。拆分需要 Schmidt/SVD，而昇腾没有复数 SVD
内核（real-embedding 变通在秩亏时对阈值极敏感），把它排除在 v1 之外可使本引擎
仅依赖 ``backend.apply_statevector_local`` 与 ``backend.kron`` 两个已 NPU-safe 的原语。
"""

from __future__ import annotations

from ..dtypes import resolve_dtype

__all__ = ["FactoredState"]


class FactoredState:
    """乘积态：``factors`` 为 ``[(sorted_qubits, amplitude_tensor), ...]``。

    不变量（每次修改后都应成立）：
      - 所有因子的 qubits 并集恰为 ``range(n_qubits)``，两两不交；
      - 每个因子的 qubits 升序排列；
      - 第 i 个因子的张量长度为 ``2 ** len(qubits_i)``，比特序为大端
        （该因子 qubits 列表中的第一个是最高位）。
    """

    def __init__(self, factors, n_qubits: int, backend):
        self._factors = [(tuple(int(q) for q in qs), amp) for qs, amp in factors]
        self._n_qubits = int(n_qubits)
        self._backend = backend

    @classmethod
    def zero_state(cls, n_qubits: int, backend) -> "FactoredState":
        """|0…0⟩：完全可分，每比特一个因子。"""
        n_qubits = int(n_qubits)
        if n_qubits <= 0:
            raise ValueError(f"n_qubits 必须为正整数，收到 {n_qubits}")
        dtype = resolve_dtype(backend)
        factors = []
        for qubit in range(n_qubits):
            amp = backend.zeros((2,), dtype=dtype)
            amp = backend.cast(_with_first_one(amp, backend))
            factors.append(((qubit,), amp))
        return cls(factors, n_qubits, backend)

    @property
    def factors(self):
        return list(self._factors)

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def backend(self):
        return self._backend

    @property
    def n_factors(self) -> int:
        return len(self._factors)

    @property
    def max_factor_width(self) -> int:
        return max((len(qs) for qs, _ in self._factors), default=0)

    def factor_index_of(self, qubit: int) -> int:
        """返回 ``qubit`` 所在因子的下标。"""
        for index, (qubits, _) in enumerate(self._factors):
            if qubit in qubits:
                return index
        raise ValueError(f"qubit {qubit} 不在任何因子中")


def _with_first_one(amp, backend):
    """把长度 2 的零张量的第 0 个分量置 1（避免依赖具体张量类型的原地赋值语义）。"""
    import numpy as np

    arr = np.asarray(backend.to_numpy(amp)).reshape(-1).copy()
    arr[0] = 1.0
    return arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: same pass count as before this task, 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py tests/simulator/test_factored.py
git commit -m "feat(simulator): FactoredState 容器与不变量"
```

---

### Task 2: Materialise a factored state to a dense state vector

**Files:**
- Modify: `aicir/simulator/factored.py`
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: `FactoredState` from Task 1, `aicir.simulator.mps._permute_basis`
- Produces: `FactoredState.to_statevector() -> aicir.core.state.State`

This is the subtlest task: `kron` of factors produces an amplitude vector whose
bit order is the **concatenation** of the factors' qubit lists, not canonical
`0..n-1`. A basis permutation is required.

- [ ] **Step 1: Write the failing test**

```python
from aicir import Circuit, Measure, NumpyBackend
from aicir.core.state import State
import aicir as A


class TestMaterialisation:
    def test_zero_state_materialises_to_computational_zero(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = 1.0
        np.testing.assert_allclose(vec, expected, atol=1e-12)

    def test_single_qubit_factors_kron_in_canonical_order(self):
        """因子内比特序必须还原成大端 0..n-1，而不是因子的拼接顺序。"""
        backend = NumpyBackend()
        # qubit0 = |1>, qubit1 = |0>, qubit2 = |1>  ->  |101> = index 5
        one = np.array([0.0, 1.0], dtype=np.complex128)
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0,), one), ((1,), zero), ((2,), one)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_out_of_order_factors_still_materialise_correctly(self):
        """因子列表顺序不应影响结果——排列由 qubits 决定，不由列表位置决定。"""
        backend = NumpyBackend()
        one = np.array([0.0, 1.0], dtype=np.complex128)
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((2,), one), ((0,), one), ((1,), zero)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_multi_qubit_factor_materialises_correctly(self):
        """一个跨 qubit 0 与 2 的两比特因子，中间夹着独立的 qubit 1。"""
        backend = NumpyBackend()
        # 因子 (0,2) 处于 |11>（局部 index 3），qubit1 = |0>  -> 全局 |101>
        pair = np.zeros(4, dtype=np.complex128); pair[3] = 1.0
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0, 2), pair), ((1,), zero)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_materialisation_preserves_norm(self):
        backend = NumpyBackend()
        rng = np.random.default_rng(0)
        a = rng.normal(size=2) + 1j * rng.normal(size=2); a /= np.linalg.norm(a)
        b = rng.normal(size=4) + 1j * rng.normal(size=4); b /= np.linalg.norm(b)
        st = FactoredState([((1,), a), ((0, 2), b)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k Materialisation`
Expected: FAIL — `AttributeError: 'FactoredState' object has no attribute 'to_statevector'`

- [ ] **Step 3: Write minimal implementation**

Add to `aicir/simulator/factored.py`:

```python
import numpy as np

from ..core.state import State
from .mps import _permute_basis


def _kron_all(tensors, backend):
    """按给定顺序做 Kronecker 积；``backend.kron`` 在 NPU 上已是 real/imag 分解。"""
    result = tensors[0]
    for tensor in tensors[1:]:
        result = backend.kron(result, tensor)
    return result


def _canonical_permutation(qubit_order, n_qubits):
    """构造把 ``qubit_order`` 位序还原成 0..n-1 的基态置换。

    ``kron`` 后的第 k 位对应 ``qubit_order[k]``（大端）。目标是让第 j 位对应
    qubit j。返回 ``src`` 数组，使 ``out[i] = flat[src[i]]``。
    """
    dim = 1 << n_qubits
    position = {q: k for k, q in enumerate(qubit_order)}
    src = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        source = 0
        for j in range(n_qubits):
            bit = (i >> (n_qubits - 1 - j)) & 1          # 目标态第 j 位（qubit j）
            if bit:
                source |= 1 << (n_qubits - 1 - position[j])
        src[i] = source
    return src
```

and as a method on `FactoredState`:

```python
    def to_statevector(self) -> State:
        """合并所有因子并还原成规范比特序的稠密 ``State``。

        注意 ``kron`` 得到的比特序是各因子 qubits 的**拼接**顺序，不是 0..n-1，
        故必须再做一次基态置换。置换走 ``mps._permute_basis``——torch 分支用
        实/虚部 ``index_select``，因为昇腾不支持复数张量索引。
        """
        ordered = sorted(self._factors, key=lambda item: item[0][0])
        qubit_order = [q for qubits, _ in ordered for q in qubits]
        amplitudes = _kron_all([amp for _, amp in ordered], self._backend)
        flat = self._backend.cast(amplitudes)
        src = _canonical_permutation(qubit_order, self._n_qubits)
        if not np.array_equal(src, np.arange(1 << self._n_qubits)):
            flat = _permute_basis(flat, src, self._backend)
        return State(
            self._backend.cast(flat).reshape(1 << self._n_qubits, 1),
            self._n_qubits,
            self._backend,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py tests/simulator/test_factored.py
git commit -m "feat(simulator): 因子化态还原为规范比特序的稠密态"
```

---

### Task 3: Apply a gate whose qubits lie inside one factor

**Files:**
- Modify: `aicir/simulator/factored.py`
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: `FactoredState`, `backend.apply_statevector_local(state, matrix, axes, n_qubits)`
- Produces: `FactoredState.apply_local(matrix, qubits) -> FactoredState` (returns a new
  state; does not mutate). Raises `ValueError` if `qubits` span more than one factor.

- [ ] **Step 1: Write the failing test**

```python
class TestIntraFactorGate:
    def test_single_qubit_gate_stays_within_its_factor(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        out = st.apply_local(h, (1,))
        assert out.n_factors == 3          # 未发生合并
        assert out.max_factor_width == 1
        vec = np.asarray(out.to_statevector().to_numpy()).reshape(-1)
        # |0>|+>|0> -> 幅度集中在 |000> 与 |010>
        nonzero = np.flatnonzero(np.abs(vec) > 1e-9)
        assert sorted(nonzero.tolist()) == [0b000, 0b010]

    def test_two_qubit_gate_inside_one_factor_does_not_merge(self):
        backend = NumpyBackend()
        pair = np.zeros(4, dtype=np.complex128); pair[0] = 1.0
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0, 1), pair), ((2,), zero)], 3, backend)
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
        out = st.apply_local(cnot, (0, 1))
        assert out.n_factors == 2

    def test_apply_local_does_not_mutate_input(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(2, backend)
        before = np.asarray(st.to_statevector().to_numpy()).copy()
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        st.apply_local(x, (0,))
        after = np.asarray(st.to_statevector().to_numpy())
        np.testing.assert_array_equal(before, after)

    def test_apply_local_rejects_qubits_spanning_factors(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
        with pytest.raises(ValueError, match="跨因子"):
            st.apply_local(cnot, (0, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k IntraFactor`
Expected: FAIL — `AttributeError: ... has no attribute 'apply_local'`

- [ ] **Step 3: Write minimal implementation**

```python
    def apply_local(self, matrix, qubits) -> "FactoredState":
        """把 ``matrix`` 作用在**同一个因子内**的 ``qubits`` 上，返回新状态。

        作用轴是该 qubit 在因子内的局部下标，因子宽度就是局部的 n_qubits——
        这正是 ``backend.apply_statevector_local`` 的契约，故无需任何新原语。
        """
        qubits = tuple(int(q) for q in qubits)
        indices = {self.factor_index_of(q) for q in qubits}
        if len(indices) != 1:
            raise ValueError(f"qubits {qubits} 跨因子，请先调用 join_for(qubits)")
        index = indices.pop()
        factor_qubits, amplitudes = self._factors[index]
        axes = [factor_qubits.index(q) for q in qubits]
        width = len(factor_qubits)

        updated = self._backend.apply_statevector_local(
            self._backend.cast(amplitudes).reshape(1 << width, 1),
            self._backend.cast(matrix),
            axes,
            width,
        )
        if updated is None:
            raise ValueError(f"后端不支持 {len(qubits)} 比特局部门")

        factors = list(self._factors)
        factors[index] = (factor_qubits, self._backend.cast(updated).reshape(-1))
        return FactoredState(factors, self._n_qubits, self._backend)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (15 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py tests/simulator/test_factored.py
git commit -m "feat(simulator): 因子内局部门应用（复用 backend.apply_statevector_local）"
```

---

### Task 4: Join factors when a gate spans them

**Files:**
- Modify: `aicir/simulator/factored.py`
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: `FactoredState`, `backend.kron`
- Produces: `FactoredState.join_for(qubits) -> FactoredState` — merges every factor
  containing any of `qubits` into one factor whose qubit tuple is sorted ascending.

- [ ] **Step 1: Write the failing test**

```python
class TestJoining:
    def test_join_merges_two_single_qubit_factors(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        joined = st.join_for((0, 2))
        assert joined.n_factors == 2
        widths = sorted(len(qs) for qs, _ in joined.factors)
        assert widths == [1, 2]

    def test_joined_factor_qubits_are_sorted(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        joined = st.join_for((2, 0))
        merged = [qs for qs, _ in joined.factors if len(qs) == 2][0]
        assert merged == (0, 2)

    def test_join_preserves_the_physical_state(self):
        backend = NumpyBackend()
        rng = np.random.default_rng(3)
        amps = []
        for _ in range(3):
            a = rng.normal(size=2) + 1j * rng.normal(size=2)
            amps.append(a / np.linalg.norm(a))
        st = FactoredState([((0,), amps[0]), ((1,), amps[1]), ((2,), amps[2])], 3, backend)
        before = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        after = np.asarray(st.join_for((0, 2)).to_statevector().to_numpy()).reshape(-1)
        np.testing.assert_allclose(before, after, atol=1e-12)

    def test_join_is_idempotent_when_already_together(self):
        backend = NumpyBackend()
        pair = np.zeros(4, dtype=np.complex128); pair[0] = 1.0
        st = FactoredState([((0, 1), pair)], 2, backend)
        assert st.join_for((0, 1)).n_factors == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k Joining`
Expected: FAIL — `AttributeError: ... has no attribute 'join_for'`

- [ ] **Step 3: Write minimal implementation**

```python
    def join_for(self, qubits) -> "FactoredState":
        """把包含 ``qubits`` 中任一比特的所有因子合并为一个因子。

        合并即 Kronecker 积。合并后 qubit 列表按升序排列，因此还需要把幅度按
        新的比特序重排——与 ``to_statevector`` 同一个置换问题，只是范围限于
        被合并的那些比特。
        """
        qubits = tuple(int(q) for q in qubits)
        target = sorted({self.factor_index_of(q) for q in qubits})
        if len(target) == 1:
            return FactoredState(self._factors, self._n_qubits, self._backend)

        merged_qubits = []
        tensors = []
        for index in target:
            factor_qubits, amplitudes = self._factors[index]
            merged_qubits.extend(factor_qubits)
            tensors.append(self._backend.cast(amplitudes).reshape(-1))

        combined = _kron_all(tensors, self._backend)
        width = len(merged_qubits)
        sorted_qubits = tuple(sorted(merged_qubits))
        local_order = [sorted_qubits.index(q) for q in merged_qubits]
        src = _canonical_permutation_local(local_order, width)
        if not np.array_equal(src, np.arange(1 << width)):
            combined = _permute_basis(self._backend.cast(combined), src, self._backend)

        remaining = [f for i, f in enumerate(self._factors) if i not in set(target)]
        remaining.append((sorted_qubits, self._backend.cast(combined).reshape(-1)))
        remaining.sort(key=lambda item: item[0][0])
        return FactoredState(remaining, self._n_qubits, self._backend)
```

and the local permutation helper:

```python
def _canonical_permutation_local(local_order, width):
    """与 ``_canonical_permutation`` 同构，但作用在因子局部的 ``width`` 个比特上。

    ``local_order[k]`` 表示 kron 结果的第 k 位在排序后应处的位置。
    """
    dim = 1 << width
    src = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        source = 0
        for k, position in enumerate(local_order):
            bit = (i >> (width - 1 - position)) & 1
            if bit:
                source |= 1 << (width - 1 - k)
        src[i] = source
    return src
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (19 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py tests/simulator/test_factored.py
git commit -m "feat(simulator): 跨因子门触发的因子合并（kron + 比特序重排）"
```

---

### Task 5: Circuit driver — `factored_statevector`

**Files:**
- Modify: `aicir/simulator/factored.py`
- Modify: `aicir/simulator/__init__.py`
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: `aicir.core.gates.gate_tensors(gate, backend) -> [(matrix, axes)]`,
  `aicir.ir.ControlFlow`, Tasks 1–4
- Produces: `factored_statevector(circuit, backend=None) -> FactoredState`

- [ ] **Step 1: Write the failing test**

```python
from aicir.simulator.factored import factored_statevector


def _dense(circuit, backend):
    return np.asarray(
        Measure(backend=backend).run(circuit, shots=None).final_state
    ).reshape(-1)


class TestDriverMatchesDense:
    """稠密路径是唯一 oracle：因子化只是表示方式不同，物理态必须一致。"""

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_ghz_matches_dense(self, n):
        backend = NumpyBackend()
        gates = [A.hadamard(0)] + [A.cnot(q + 1, [q]) for q in range(n - 1)]
        circuit = Circuit(*gates, n_qubits=n)
        got = np.asarray(factored_statevector(circuit, backend).to_statevector().to_numpy()).reshape(-1)
        np.testing.assert_allclose(got, _dense(circuit, backend), atol=1e-10)

    def test_product_circuit_never_joins(self):
        """全是单比特门的线路应保持完全因子化——这就是本引擎的收益来源。"""
        backend = NumpyBackend()
        gates = [A.rx(0.3, q) for q in range(6)] + [A.ry(0.7, q) for q in range(6)]
        circuit = Circuit(*gates, n_qubits=6)
        st = factored_statevector(circuit, backend)
        assert st.n_factors == 6
        assert st.max_factor_width == 1

    def test_disjoint_entanglers_produce_pairwise_factors(self):
        backend = NumpyBackend()
        gates = [A.hadamard(0), A.cnot(1, [0]), A.hadamard(2), A.cnot(3, [2])]
        circuit = Circuit(*gates, n_qubits=4)
        st = factored_statevector(circuit, backend)
        assert st.n_factors == 2
        assert st.max_factor_width == 2

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_circuit_matches_dense(self, seed):
        backend = NumpyBackend()
        rng = np.random.default_rng(seed)
        n = 5
        gates = []
        for _ in range(12):
            if rng.random() < 0.5:
                gates.append(A.ry(float(rng.uniform(0, 6.28)), int(rng.integers(n))))
            else:
                a = int(rng.integers(n - 1))
                gates.append(A.cnot(a + 1, [a]))
        circuit = Circuit(*gates, n_qubits=n)
        got = np.asarray(factored_statevector(circuit, backend).to_statevector().to_numpy()).reshape(-1)
        np.testing.assert_allclose(got, _dense(circuit, backend), atol=1e-10)

    def test_rejects_control_flow(self):
        from aicir.core.circuit import if_
        from aicir.core.classical import ClassicalRegister

        backend = NumpyBackend()
        creg = ClassicalRegister(1, "c")
        body = Circuit(A.pauli_x(0), n_qubits=1)
        circuit = Circuit(if_(creg[0] == 1, body), n_qubits=1)
        with pytest.raises(ValueError, match="控制流"):
            factored_statevector(circuit, backend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k Driver`
Expected: FAIL — `ImportError: cannot import name 'factored_statevector'`

- [ ] **Step 3: Write minimal implementation**

```python
from ..core.gates import gate_tensors
from ..ir import ControlFlow


def factored_statevector(circuit, backend=None) -> FactoredState:
    """按因子化表示演化 ``circuit``，返回 ``FactoredState``。

    门经 ``gate_tensors`` 降解为 ``[(matrix, axes)]``——与稠密路径、张量网络
    引擎同一个来源，故门语义不会分叉。
    """
    from ..backends.numpy_backend import NumpyBackend

    backend = backend if backend is not None else (circuit._backend or NumpyBackend())

    for gate in circuit.gates:
        if isinstance(gate, ControlFlow):
            raise ValueError("控制流指令无法用因子化引擎执行；请用 Measure.run 的轨迹路径")

    state = FactoredState.zero_state(circuit.n_qubits, backend)
    for gate in circuit.gates:
        for matrix, axes in gate_tensors(gate, backend):
            qubits = tuple(int(a) for a in axes)
            if len({state.factor_index_of(q) for q in qubits}) > 1:
                state = state.join_for(qubits)
            state = state.apply_local(matrix, qubits)
    return state
```

Export in `aicir/simulator/__init__.py`:

```python
from .factored import FactoredState, factored_statevector
```

adding both names to `__all__` if that module defines one.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (27 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py aicir/simulator/__init__.py tests/simulator/test_factored.py
git commit -m "feat(simulator): factored_statevector 线路驱动"
```

---

### Task 6: Expectation values without full materialisation

**Files:**
- Modify: `aicir/simulator/factored.py`
- Modify: `aicir/simulator/__init__.py`
- Test: `tests/simulator/test_factored.py`

A Pauli string factorises across independent factors: `⟨P⟩ = ∏ᵢ ⟨Pᵢ⟩` where
`Pᵢ` restricts `P` to factor `i`. Computing per factor avoids ever building the
`2^n` vector — this is where the engine's memory advantage becomes real.

**Interfaces:**
- Consumes: `FactoredState`, `aicir.core.operators.PauliString` (`.masks()`, `.n_qubits`,
  `.coefficient`, `.qubit_labels`), `Hamiltonian`
- Produces: `factored_expectation(state, observable) -> float`

- [ ] **Step 1: Write the failing test**

```python
from aicir import Hamiltonian
from aicir.simulator.factored import factored_expectation


class TestFactoredExpectation:
    def test_matches_dense_expectation_on_product_state(self):
        backend = NumpyBackend()
        gates = [A.ry(0.4, q) for q in range(5)]
        circuit = Circuit(*gates, n_qubits=5)
        st = factored_statevector(circuit, backend)
        ham = Hamiltonian(n_qubits=5, terms=[("Z", [q], 1.0) for q in range(5)])
        dense_state = st.to_statevector()
        assert factored_expectation(st, ham) == pytest.approx(
            ham.expectation(dense_state, backend), abs=1e-10
        )

    def test_matches_dense_expectation_after_entangling(self):
        backend = NumpyBackend()
        gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 2)]
        circuit = Circuit(*gates, n_qubits=3)
        st = factored_statevector(circuit, backend)
        ham = Hamiltonian(n_qubits=3, terms=[("ZZ", [0, 1], 1.0), ("X", [2], 0.5)])
        assert factored_expectation(st, ham) == pytest.approx(
            ham.expectation(st.to_statevector(), backend), abs=1e-10
        )

    def test_term_spanning_two_factors_multiplies_their_expectations(self):
        backend = NumpyBackend()
        gates = [A.ry(0.4, 0), A.ry(0.9, 1)]
        circuit = Circuit(*gates, n_qubits=2)
        st = factored_statevector(circuit, backend)
        assert st.n_factors == 2               # 前提：确实是两个因子
        ham = Hamiltonian(n_qubits=2, terms=[("ZZ", [0, 1], 1.0)])
        assert factored_expectation(st, ham) == pytest.approx(
            ham.expectation(st.to_statevector(), backend), abs=1e-10
        )

    def test_does_not_materialise_full_state(self):
        """20 比特全可分：稠密化需 16 MB，因子化只碰 20 个 2-维张量。"""
        backend = NumpyBackend()
        gates = [A.ry(0.2, q) for q in range(20)]
        circuit = Circuit(*gates, n_qubits=20)
        st = factored_statevector(circuit, backend)
        assert st.max_factor_width == 1
        ham = Hamiltonian(n_qubits=20, terms=[("Z", [0], 1.0), ("Z", [19], 1.0)])
        value = factored_expectation(st, ham)
        assert np.isfinite(value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k FactoredExpectation`
Expected: FAIL — `ImportError: cannot import name 'factored_expectation'`

- [ ] **Step 3: Write minimal implementation**

```python
from ..core.operators import Hamiltonian, PauliString


def _restrict_labels(labels, factor_qubits):
    """把全局 Pauli 标签限制到某因子的比特上，返回该因子的局部标签串。"""
    return "".join(labels[q] for q in factor_qubits)


def factored_expectation(state: FactoredState, observable) -> float:
    """在因子化表示上求 Pauli 可观测量期望，不构造完整态矢量。

    Pauli 串在互不纠缠的因子间是乘性的：``⟨P⟩ = ∏ᵢ ⟨Pᵢ⟩``。因此逐因子求值再
    连乘即可，代价只与最宽的因子有关，而非 ``2^n``。
    """
    terms = observable.terms if isinstance(observable, Hamiltonian) else [observable]

    total = 0.0
    for term in terms:
        labels = term.qubit_labels
        product = 1.0
        for factor_qubits, amplitudes in state.factors:
            local_labels = _restrict_labels(labels, factor_qubits)
            if set(local_labels) == {"I"}:
                continue                       # 恒等因子贡献 1
            local_state = State(
                state.backend.cast(amplitudes).reshape(1 << len(factor_qubits), 1),
                len(factor_qubits),
                state.backend,
            )
            product *= Hamiltonian(
                n_qubits=len(factor_qubits), terms=[(local_labels, 1.0)]
            ).expectation(local_state, state.backend)
        total += float(np.real(term.coefficient)) * product
    return float(total)
```

Export `factored_expectation` in `aicir/simulator/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (31 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/simulator/factored.py aicir/simulator/__init__.py tests/simulator/test_factored.py
git commit -m "feat(simulator): 因子化期望值（Pauli 串按因子相乘，不稠密化）"
```

---

### Task 7: Wire into `Measure.run(method="factored")`

**Files:**
- Modify: `aicir/measure/measure.py:176-205`
- Test: `tests/simulator/test_factored.py`

Follow the existing `method="tensor"` and `method="mps"` branches exactly; the
same rejections apply (pure states only, no embedded `measure()` markers, no
custom initial state, no `snap`).

**Interfaces:**
- Consumes: `factored_statevector` from Task 5
- Produces: `Measure.run(circuit, method="factored")` returning the standard `Result`

- [ ] **Step 1: Write the failing test**

```python
class TestMeasureIntegration:
    def test_factored_method_matches_statevector_method(self):
        backend = NumpyBackend()
        gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 2)]
        circuit = Circuit(*gates, n_qubits=3)
        a = np.asarray(Measure(backend=backend).run(circuit, shots=None, method="factored").final_state).reshape(-1)
        b = np.asarray(Measure(backend=backend).run(circuit, shots=None, method="statevector").final_state).reshape(-1)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_factored_rejects_noisy_circuits(self):
        """注意：拒绝检查读的是 **circuit.noise_model**，不是 Measure 的噪声模型
        （与 method='mps' 分支一致）。"""
        backend = NumpyBackend()
        circuit = Circuit(A.hadamard(0), n_qubits=1)
        circuit.noise_model = object()
        with pytest.raises(ValueError, match="仅支持纯态"):
            Measure(backend=backend).run(circuit, shots=None, method="factored")

    def test_factored_rejects_embedded_measure_markers(self):
        backend = NumpyBackend()
        from aicir.core.circuit import measure as measure_marker
        circuit = Circuit(A.hadamard(0), measure_marker(0), n_qubits=1)
        with pytest.raises(ValueError, match="measure"):
            Measure(backend=backend).run(circuit, shots=None, method="factored")

    def test_factored_rejects_initial_state(self):
        backend = NumpyBackend()
        circuit = Circuit(A.hadamard(0), n_qubits=1)
        init = np.array([[0.0], [1.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match="initial_state"):
            Measure(backend=backend).run(circuit, shots=None, method="factored", initial_state=init)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k MeasureIntegration`
Expected: FAIL — the first test errors because `method="factored"` is unrecognised

- [ ] **Step 3: Write minimal implementation**

In `aicir/measure/measure.py`, immediately after the `method == "mps"` branch
(which ends with its delegating `return self.run(...)`), insert:

```python
        if method == "factored":
            if getattr(circuit, "noise_model", None) is not None:
                raise ValueError("method='factored' 仅支持纯态，无法用于含噪线路")
            if any(_is_measure(g) for g in circuit_instructions(circuit)):
                raise ValueError("method='factored' 不支持线路内嵌 measure 标记")
            if initial_state is not None or initial_density_matrix is not None:
                raise ValueError("method='factored' 始终从 |0...0> 出发，不接受 initial_state/initial_density_matrix")
            if snap not in (None, [], ()):  # 无逐门快照语义
                raise ValueError("method='factored' 不支持非空 snap")
            from ..simulator import factored_statevector
            psi = factored_statevector(circuit, backend=backend).to_statevector()
            from ..core.circuit import Circuit as _Circuit
            stripped = _Circuit(n_qubits=n)
            return self.run(
                stripped, shots=shots, measure_qubits=measure_qubits, snap=None,
                sm=sm, seed=seed, initial_state=psi, observables=observables,
                return_state=return_state, return_probabilities=return_probabilities,
                method="statevector",
            )
```

This mirrors the `mps` branch exactly, including its delegation trick: evolve
with the alternate engine, then re-enter `self.run` on an **empty** circuit of
the same width with the resulting state as `initial_state`, so all the
downstream read-out, shots, and observable machinery is reused rather than
duplicated. `_is_measure`, `circuit_instructions`, `backend`, and `n` are
already in scope at that point in the function — do not re-derive them.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q`
Expected: PASS (35 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add aicir/measure/measure.py tests/simulator/test_factored.py
git commit -m "feat(measure): 接入 method='factored'"
```

---

### Task 8: Backend parity — GPU backend agreement

**Files:**
- Test: `tests/simulator/test_factored.py`

**Interfaces:**
- Consumes: everything above; `aicir.GPUBackend`

- [ ] **Step 1: Write the failing test**

```python
class TestBackendParity:
    def test_gpu_backend_matches_numpy(self):
        torch = pytest.importorskip("torch")
        from aicir import GPUBackend

        gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 2), A.cnot(2, [1])]
        circuit = Circuit(*gates, n_qubits=4)

        cpu = NumpyBackend(dtype=np.complex128)
        gpu = GPUBackend(device="cpu", dtype=torch.complex128)

        a = np.asarray(factored_statevector(circuit, cpu).to_statevector().to_numpy()).reshape(-1)
        b = np.asarray(factored_statevector(circuit, gpu).to_statevector().to_numpy()).reshape(-1)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_gpu_backend_factor_structure_matches(self):
        torch = pytest.importorskip("torch")
        from aicir import GPUBackend

        gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 2)]
        circuit = Circuit(*gates, n_qubits=4)
        cpu_state = factored_statevector(circuit, NumpyBackend())
        gpu_state = factored_statevector(circuit, GPUBackend(device="cpu", dtype=torch.complex64))
        assert [qs for qs, _ in cpu_state.factors] == [qs for qs, _ in gpu_state.factors]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored.py -q -k BackendParity`
Expected: If it fails, the cause is a dtype or reshape assumption in Tasks 1–5;
fix the engine, not the test. If it passes immediately, that is the expected
outcome of routing everything through `Backend` — record that in the commit.

- [ ] **Step 3: Fix any backend-specific assumption found**

Common cause: `np.asarray` applied to a torch tensor inside the engine. Every
tensor touch must go through `backend.cast` / `backend.to_numpy`.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 5: Commit**

```bash
git add tests/simulator/test_factored.py aicir/simulator/factored.py
git commit -m "test(simulator): 因子化引擎的 NumPy/GPU 后端一致性"
```

---

### Task 9: NPU-safety proof without hardware

**Files:**
- Create: `tests/simulator/test_factored_npu_safety.py`

This mirrors `tests/core/test_pauli_sparse_torch.py`. **The reverse-control test
is mandatory** — an interception that never fires proves nothing, and this
project has already shipped one such assertion (see the `device_residency`
history in `CHANGELOG.md`).

**Interfaces:**
- Consumes: `factored_statevector`, `torch.utils._python_dispatch.TorchDispatchMode`

- [ ] **Step 1: Write the failing test**

```python
"""因子化引擎的 NPU-safe 证明（无需真机）。

昇腾缺三类会被本引擎命中的内核：复数高级索引（aclnnIndex）、复数加/乘
（aclnnAdd/aclnnMul）、位运算（aten::bitwise_right_shift，且**不报错**而是
静默回落 CPU）。这里用 TorchDispatchMode 在 CPU 上复现这些缺口。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import aicir as A  # noqa: E402
from aicir import Circuit, GPUBackend  # noqa: E402
from aicir.simulator.factored import factored_statevector  # noqa: E402


class _BanAscendGaps(TorchDispatchMode):
    COMPLEX_BANNED = ("add", "mul", "sub", "index", "sum", "dot", "matmul", "mm", "gather")
    ALLOWED = ("view_as_real", "real", "imag", "conj_physical", "_conj", "resolve_conj")
    ALWAYS_BANNED = ("bitwise_", "__lshift__", "__rshift__")

    def __init__(self):
        super().__init__()
        self.violations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if any(tag in name for tag in self.ALWAYS_BANNED):
            self.violations.append(name)
        elif not any(tag in name for tag in self.ALLOWED):
            if any(tag in name for tag in self.COMPLEX_BANNED):
                for value in list(args) + list(kwargs.values()):
                    if isinstance(value, torch.Tensor) and torch.is_complex(value):
                        self.violations.append(name)
                        break
        return func(*args, **kwargs)


def _circuit(n=6):
    gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 3), A.cnot(4, [3]), A.rz(0.2, 5)]
    return Circuit(*gates, n_qubits=n)


class TestNPUSafety:
    def test_engine_issues_no_ascend_missing_op(self):
        backend = GPUBackend(device="cpu", dtype=torch.complex64)
        mode = _BanAscendGaps()
        with mode:
            factored_statevector(_circuit(), backend).to_statevector()
        assert not mode.violations, f"命中昇腾缺失内核: {sorted(set(mode.violations))}"

    def test_ban_mode_actually_fires_on_a_known_violation(self):
        """反证：拦截器必须能抓到真实违规，否则上一条断言毫无意义。"""
        mode = _BanAscendGaps()
        with mode:
            idx = torch.arange(8, dtype=torch.int64)
            torch.bitwise_xor(idx, torch.bitwise_right_shift(idx, 1))
        assert mode.violations, "拦截器没有检测能力"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored_npu_safety.py -q`
Expected: The reverse-control test PASSES; the engine test may FAIL, listing the
offending operations.

- [ ] **Step 3: Fix each reported violation in `aicir/simulator/factored.py`**

Likely causes and their fixes:
- `_canonical_permutation` builds the index array with Python ints and NumPy —
  keep it that way. Do **not** compute permutations with torch bitwise ops.
- Complex indexing in `_permute_basis` — already handled by `mps.py`; ensure
  the engine calls that helper rather than indexing directly.
- Complex `mul` in `_kron_all` — must go through `backend.kron`, never `*`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/simulator/test_factored_npu_safety.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add tests/simulator/test_factored_npu_safety.py aicir/simulator/factored.py
git commit -m "test(simulator): 因子化引擎 NPU-safe 拦截证明（含反证）"
```

---

### Task 10: Real-hardware probe

**Files:**
- Create: `scripts/npu/factored_probe.py`
- Create: `scripts/npu/factored.sh`
- Modify: `scripts/npu/run_npu_tests.py` (SUITES dict)
- Modify: `scripts/npu/README.md`

Model this file on `scripts/npu/pauli_sparse_probe.py`, including its `_sync`
and `_time` helpers — **NPU execution is asynchronous, and timing without
`torch.npu.synchronize()` measures kernel launch, not completion.**

**Interfaces:**
- Consumes: `factored_statevector`, `factored_expectation`, `NPUBackend`
- Produces: `scripts/npu/factored_probe.py` with cases
  `correctness_vs_cpu`, `factor_structure`, `no_ascend_gaps`, `product_state_scale`,
  `expectation_without_materialising`

- [ ] **Step 1: Write the probe**

Copy `scripts/npu/pauli_sparse_probe.py` and keep its `_backend`, `_sync`,
`_time`, and `main()` case-runner verbatim — only the case bodies differ. The
five cases:

```python
def _mixed_circuit(n):
    """单比特门 + 两对不相交 CNOT：保证既有因子内门也有合并。"""
    gates = [A.ry(0.3, q) for q in range(n)]
    gates += [A.cnot(1, [0]), A.cnot(3, [2])]
    return Circuit(*gates, n_qubits=n)


def case_correctness_vs_cpu(backend, n, metrics):
    circuit = _mixed_circuit(min(n, 10))
    got = np.asarray(
        factored_statevector(circuit, backend).to_statevector().to_numpy()
    ).reshape(-1)
    cpu = NumpyBackend(dtype=np.complex128)
    want = np.asarray(
        factored_statevector(circuit, cpu).to_statevector().to_numpy()
    ).reshape(-1)
    overlap = abs(complex(np.vdot(want, got)))
    metrics["correctness_overlap"] = overlap
    if abs(overlap - 1.0) > 1e-3:       # complex64 精度
        raise AssertionError(f"NPU 与 CPU 态重叠度 {overlap}，应为 1")


def case_factor_structure(backend, n, metrics):
    """因子划分是纯 Python 记账，必须与设备无关。不一致即说明混入了设备分支。"""
    circuit = _mixed_circuit(min(n, 10))
    npu_part = [qs for qs, _ in factored_statevector(circuit, backend).factors]
    cpu_part = [qs for qs, _ in factored_statevector(circuit, NumpyBackend()).factors]
    metrics["factor_partition"] = [list(qs) for qs in npu_part]
    if npu_part != cpu_part:
        raise AssertionError(f"因子划分不一致: NPU {npu_part} vs CPU {cpu_part}")


def case_no_ascend_gaps(backend, n, metrics):
    """真机复核：不发出复数索引/加乘、也不发出位运算。"""
    circuit = _mixed_circuit(min(n, 8))
    mode = _BanAscendGaps()             # 与 Task 9 的拦截器同一份实现
    with mode:
        factored_statevector(circuit, backend).to_statevector()
    metrics["violations"] = sorted(set(mode.violations))
    if mode.violations:
        raise AssertionError(f"命中昇腾缺失内核: {metrics['violations']}")


def case_product_state_scale(backend, n, metrics):
    """28 比特全可分：稠密需 2^28 x 8B = 2.1 GB，因子化只有 28 个 2 维张量。"""
    n_qubits = 28
    circuit = Circuit(*[A.ry(0.2, q) for q in range(n_qubits)], n_qubits=n_qubits)
    elapsed = _time(backend, lambda: factored_statevector(circuit, backend))
    state = factored_statevector(circuit, backend)
    metrics["scale_seconds"] = elapsed
    metrics["scale_factors"] = state.n_factors
    print(f"        n={n_qubits} 全可分 {elapsed*1000:.2f} ms, factors={state.n_factors}")
    if state.max_factor_width != 1:
        raise AssertionError(f"全可分线路却出现宽度 {state.max_factor_width} 的因子")


def case_expectation_without_materialising(backend, n, metrics):
    """28 比特求期望且**不得**稠密化——稠密化会立刻 OOM 或极慢。"""
    n_qubits = 28
    circuit = Circuit(*[A.ry(0.2, q) for q in range(n_qubits)], n_qubits=n_qubits)
    state = factored_statevector(circuit, backend)

    calls = {"n": 0}
    original = type(state).to_statevector

    def _guard(self):
        calls["n"] += 1
        return original(self)

    type(state).to_statevector = _guard
    try:
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("Z", [0], 1.0), ("Z", [27], 1.0)])
        value = factored_expectation(state, ham)
    finally:
        type(state).to_statevector = original

    metrics["expectation_value"] = float(value)
    metrics["materialised"] = calls["n"]
    if calls["n"] != 0:
        raise AssertionError("factored_expectation 稠密化了整个态")
    if not np.isfinite(value):
        raise AssertionError(f"期望值非有限: {value}")
```

Register them in `main()`'s `cases` list in this order, matching the
`pauli_sparse_probe.py` pattern:

```python
    cases = [
        ("correctness_vs_cpu", lambda: case_correctness_vs_cpu(backend, n, metrics)),
        ("factor_structure", lambda: case_factor_structure(backend, n, metrics)),
        ("no_ascend_gaps", lambda: case_no_ascend_gaps(backend, n, metrics)),
        ("product_state_scale", lambda: case_product_state_scale(backend, n, metrics)),
        ("expectation_without_materialising",
         lambda: case_expectation_without_materialising(backend, n, metrics)),
    ]
```

- [ ] **Step 2: Create the shell wrapper**

```sh
#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python}" "$SCRIPT_DIR/factored_probe.py" "$@"
```

Then `chmod +x scripts/npu/factored.sh`.

- [ ] **Step 3: Register the suite**

In `scripts/npu/run_npu_tests.py`, add to `SUITES`:

```python
    "factored": Suite(
        name="factored",
        description="Factored (product-state) engine: correctness, factor structure, and scale beyond the dense limit.",
        targets=(
            "tests/simulator/test_factored.py",
            "tests/simulator/test_factored_npu_safety.py",
        ),
        scripts=(("scripts/npu/factored_probe.py",),),
    ),
```

- [ ] **Step 4: Validate locally with CPU fallback**

Run: `./scripts/npu/factored.sh --allow-cpu-fallback`
Expected: 5/5 cases passed

Note in the output that `--allow-cpu-fallback` is **not** evidence of NPU
correctness — only that the script runs.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 6: Commit**

```bash
git add scripts/npu/factored_probe.py scripts/npu/factored.sh scripts/npu/run_npu_tests.py scripts/npu/README.md
git commit -m "test(npu): 因子化引擎真机探针与套件注册"
```

---

### Task 11: Documentation and honest limits

**Files:**
- Modify: `aicir/simulator/README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Document the engine in `aicir/simulator/README.md`**

Cover: what it is, when it helps, when it does not, and how to select it
(`Measure.run(method="factored")`). State plainly:

- **It helps** on circuits whose entanglement stays local — product-state
  preparation, disjoint entangling blocks, shallow encoders.
- **It does not help** on hardware-efficient ansätze, QAOA, or random circuits,
  where entanglement saturates within one or two layers and the state collapses
  to a single factor plus bookkeeping overhead. These are the framework's main
  variational workloads, which is why the dense path remains the default.
- **v1 never splits factors.** Once merged, factors stay merged, even if a later
  gate leaves them separable. Detecting separability requires a Schmidt
  decomposition, and Ascend has no complex SVD kernel — the real-embedding
  workaround exists but is highly sensitive to threshold choice on rank-deficient
  inputs. Splitting is therefore deferred, not forgotten.
- **Relation to MPS.** A product state is exactly MPS with bond dimension 1.
  This engine is the exact, automatic χ=1 case; `method="mps"` is the general,
  truncating case. If the future direction is an exact growing-χ MPS, this engine
  becomes its base case rather than a competitor.

- [ ] **Step 2: Add a dated CHANGELOG entry**

Record the new engine, `method="factored"`, the two new test files, the probe,
and the limits above. Include the measured factor counts from Task 5's tests as
concrete illustration (a 6-qubit single-qubit-gate-only circuit stays at 6
factors of width 1; two disjoint CNOT pairs give 2 factors of width 2).

- [ ] **Step 3: Run the full suite one final time**

Run: `PYTHONPATH=. pytest tests/ -p no:warnings -q`
Expected: 0 failures

- [ ] **Step 4: Commit**

```bash
git add aicir/simulator/README.md CHANGELOG.md
git commit -m "docs(simulator): 因子化引擎使用说明与边界"
```

---

## Deferred (explicitly out of scope)

Record these so a later reader knows they were considered, not overlooked:

1. **Factor splitting.** Requires Schmidt decomposition per gate; blocked on
   Ascend's missing complex SVD and the fragility of the real-embedding
   workaround at rank deficiency.
2. **Density matrices / noise.** The engine is pure-state only, matching the
   existing `tensor` and `mps` engine restrictions.
3. **Autograd through the factored representation.** Joining changes tensor
   shapes dynamically, which interacts poorly with a static autograd graph.
   Parameter-shift via the existing estimators remains available.
4. **Integration with `aicir.distributed`.** Sharding assumes a contiguous `2^n`
   buffer; a factored state has none. These target different regimes (low versus
   high entanglement) and should stay independent.
5. **Automatic engine selection.** Choosing `factored` versus `mps` versus dense
   from circuit structure is a separate piece of work and should not be bundled
   with introducing the engine.

# Result 状态字段返回 State 对象 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `aicir.measure.Result` 的 `state` / `final_state` / `snap()` / `snapshot_states` 返回统一 `State` 对象（而非裸 numpy 数组），与全局 State 迁移一致，并使 `result.state.ket` 等用法直接可用。

**Architecture:** 给 `State` 增加 `__array__`，使所有走 `np.asarray` / `np.allclose` / `backend.cast` 的隐式转换点继续工作；随后把 `Result` 的状态字段翻转为 `State`，仅迁移少数直接在 State 上调用 ndarray 方法（`.reshape`/`.shape`）的测试/演示点。

**Tech Stack:** Python、numpy、pytest。

参考规范：`docs/superpowers/specs/2026-06-15-result-state-objects-design.md`

运行测试统一用：`PYTHONPATH=. pytest`（仓库根目录）。

---

## Task 1: `State.__array__`

**Files:**
- Modify: `aicir/core/state.py`
- Test: `tests/core/test_state_array_protocol.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/core/test_state_array_protocol.py`（若 `tests/core/` 不存在则一并创建；本仓库测试用 PYTHONPATH 发现，无需 `__init__.py`，可对照 `tests/measure/`）：

```python
"""State 支持 numpy 数组协议（__array__），便于与裸数组互操作。"""

import numpy as np

from aicir import NumpyBackend, State


def test_asarray_vector_shape():
    s = State.from_array([1.0, 0.0, 0.0, 0.0], backend=NumpyBackend())  # 2 qubits
    arr = np.asarray(s)
    assert arr.shape == (4,)
    np.testing.assert_allclose(arr, [1, 0, 0, 0], atol=1e-6)


def test_asarray_density_shape():
    rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex64)
    s = State.from_matrix(rho)
    arr = np.asarray(s)
    assert arr.shape == (4, 4)
    np.testing.assert_allclose(arr, rho, atol=1e-6)


def test_allclose_between_states():
    a = State.from_array([1.0, 0.0], backend=NumpyBackend())
    b = State.from_array([1.0, 0.0], backend=NumpyBackend())
    assert np.allclose(a, b)


def test_asarray_dtype_arg():
    s = State.from_array([1.0, 0.0], backend=NumpyBackend())
    arr = np.asarray(s, dtype=np.complex128)
    assert arr.dtype == np.complex128
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. pytest tests/core/test_state_array_protocol.py -q`
Expected: FAIL（`np.asarray(State)` 产生 object/0 维数组，断言形状失败）。

- [ ] **Step 3: Implement**

在 `aicir/core/state.py` 的 `State` 类中，紧接 `to_numpy`（约 461 行）之后、`__len__` 之前，新增：

```python
    def __array__(self, dtype=None):
        """numpy 数组协议：向量态导出 (2^n,)，密度态导出 (2^n, 2^n)。

        使 ``np.asarray(state)`` / ``np.allclose(a, state)`` / ``backend.cast(state)``
        等隐式转换继续以与旧版裸数组一致的形态工作。
        """
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. pytest tests/core/test_state_array_protocol.py -q`
Expected: PASS（4 passed）。

- [ ] **Step 5: Run full suite (additive change, no regression)**

Run: `PYTHONPATH=. pytest -q`
Expected: 全绿（仅新增协议，不改变现有行为）。

- [ ] **Step 6: Commit**

```bash
git add aicir/core/state.py tests/core/test_state_array_protocol.py
git commit -m "feat(core): add State.__array__ numpy protocol"
```

---

## Task 2: 翻转 Result 状态字段为 State（原子变更 + 迁移断点）

本任务必须**一次性**改动 producer（`measure.py`）、`Result` 类型与 `snap()`、`primitives`，并同步迁移所有会因此报错的测试/演示，使全量测试保持绿色。

**Files:**
- Modify: `aicir/measure/result.py`
- Modify: `aicir/measure/measure.py`
- Modify: `aicir/primitives/estimator.py`
- Modify (tests/demos break sites): `tests/noise/test_noise_model.py`、`tests/measure/test_measure.py`、`tests/measure/test_measure_run_semantics.py`、`tests/measure/test_result_api.py`、`demos/snap_demo.py`

- [ ] **Step 1: 先确认断点（基线为红）**

不改代码，先把 producer 翻转所需的测试断言改成 State 语义，确认它们当前为红（证明迁移确有发生）。可跳过显式红步，直接进入实现后用 Step 6 全绿验证。建议至少手动核对一处：当前 `result.state` 是 ndarray，`result.state.array` 会 `AttributeError`。

- [ ] **Step 2: 改 `aicir/measure/result.py`**

文件顶部增加导入（与现有 `import numpy as np` 同区）：

```python
from ..core.state import State
```

字段类型改为 State（约 42–47 行）：

```python
    state: Optional[State] = None
    final_state: Optional[State] = None
    final_state_kind: Optional[str] = None
    expectation_values: Dict[str, float] = field(default_factory=dict)
    expectation_variances: Dict[str, float] = field(default_factory=dict)
    snapshot_states: Dict[int, State] = field(default_factory=dict)
```

`snap` 改为返回 State（约 85–87 行）：

```python
    def snap(self, op_index: int) -> Optional[State]:
        return self.snapshot_states.get(int(op_index))
```

`reduce` 不改：其内部 `arr = np.asarray(src)`（约 93 行）经 `State.__array__` 继续工作，返回 ndarray 不变。

- [ ] **Step 3: 改 `aicir/measure/measure.py` `_build_result`**

确保文件已 `from ..core.state import State`（第 20 行已有）。把 `_build_result`（约 225–286 行）按下述改写：

单轨迹/精确分支：让 state/final/snap 直接用轨迹里的 `State`（`tr.pre`/`tr.post`/`tr.snaps` 的值本就是 `State`）：

```python
        if exact or norm_shots == 1:
            tr = trajectories[0]
            state = tr.pre
            final = tr.post
            incircuit_outputs = ({op: tr.incircuit[op] for op in (s.op_index for s in specs)}
                                 if exact else
                                 {op: np.array([[tr.incircuit[op]]]) for op in (s.op_index for s in specs)})
            terminal_output = None
            if do_terminal and tr.terminal is not None:
                terminal_output = (np.array(tr.terminal) if exact
                                   else np.array(tr.terminal).reshape(1, -1))
            snap_states = dict(tr.snaps)
            probabilities = np.asarray(tr.pre.probabilities()).reshape(-1).astype(np.float64) \
                if hasattr(tr.pre, "probabilities") else np.abs(np.asarray(state).reshape(-1)) ** 2
            incircuit_counts = {}
            terminal_counts = None
            if not exact:
                incircuit_counts = {op: {int(tr.incircuit[op]): 1} for op in (s.op_index for s in specs)}
                if terminal_output is not None:
                    key = "".join("0" if b == 1 else "1" for b in tr.terminal)
                    terminal_counts = {key: 1}
```

聚合分支：把聚合产出的密度数组包成 State：

```python
        else:
            agg = aggregate_avg(trajectories, n, specs, terminal_qubits if do_terminal else None)
            state = State.from_matrix(np.asarray(agg["state"]), n)
            final = State.from_matrix(np.asarray(agg["final_state"]), n)
            incircuit_outputs = agg["incircuit_outputs"]; incircuit_counts = agg["incircuit_counts"]
            terminal_output = agg["terminal_output"]; terminal_counts = agg["terminal_counts"]
            snap_states = {t: State.from_matrix(np.asarray(s), n) for t, s in agg["snapshot_states"].items()}
            probabilities = agg["probabilities"]
```

> 实现期确认：`aggregate_avg` 的 `state`/`final_state`/`snapshot_states` 均为 `(2^n, 2^n)` 密度数组（见 `aicir/measure/aggregate.py`）。若某项实际是向量，请改用 `State.from_array(..., n)` 包装该项。

期望值与 `is_dm` 段（约 253–268 行）改用 `np.asarray(state)` 与 `State.is_density`：

```python
        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            state_arr = np.asarray(state)
            rho = state_arr if (state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]) else None
            vec = None if rho is not None else state_arr.reshape(-1, 1)
            for name, op in observables.items():
                op = np.asarray(op)
                if rho is not None:
                    exp_vals[name] = float(np.real(np.trace(rho @ op)))
                else:
                    exp_vals[name] = float(np.real((vec.conj().T @ op @ vec)[0, 0]))

        is_dm = bool(return_state and final.is_density) if return_state else False
        state_mode = "density_matrix" if (noise_model is not None or initial_density_matrix is not None or is_dm) else "state_vector"
```

`Result(...)` 构造段（约 274–286 行）保持字段名不变（现在 `state`/`final`/`snap_states` 已是 State）：

```python
        return Result(
            n_qubits=n, backend_name=type(backend).__name__,
            probabilities=probabilities, shots=norm_shots,
            measurement_specs=specs, incircuit_outputs=incircuit_outputs,
            incircuit_counts=incircuit_counts, terminal_output=terminal_output,
            terminal_counts=terminal_counts, terminal_qubits=terminal_qubits,
            state=(state if return_state else None),
            final_state=(final if return_state else None),
            final_state_kind=("density_matrix" if is_dm else "state_vector") if return_state else None,
            expectation_values=exp_vals, expectation_variances=exp_vars,
            snapshot_states=snap_states,
            metadata=meta,
        )
```

- [ ] **Step 4: 改 `aicir/primitives/estimator.py`**

第 26 行：

```python
        state = self.backend.cast(result.state)
```
改为：
```python
        state = self.backend.cast(result.state.to_numpy())
```

- [ ] **Step 5: 迁移测试/演示中的直接 ndarray 方法调用点**

- `tests/noise/test_noise_model.py:60`：`rho = result.final_state.reshape(2, 2)` → `rho = result.final_state.matrix.reshape(2, 2)`
- `tests/measure/test_measure.py:165`：`result.state.reshape(-1)` → `result.state.array`
- `tests/measure/test_measure_run_semantics.py`：
  - 49：`result.state.reshape(-1)` → `result.state.array`
  - 51：`result.final_state.reshape(-1), result.state.reshape(-1)` → `result.final_state.array, result.state.array`
  - 80、98：`result.state.reshape(-1)` → `result.state.array`
  - 176：`result.snap(0).reshape(-1)` → `result.snap(0).array`
  - 177：`result.snap(1).reshape(-1)` → `result.snap(1).array`
  - 148/159 的 `np.asarray(result.state).shape == (4, 4)` 与 158 的 `result.final_state_kind` **不改**（经 `__array__` / 保留字段仍工作）。
- `tests/measure/test_result_api.py:46`：构造 `Result(..., final_state=rho, final_state_kind="density_matrix")` 中 `rho` 为裸密度数组 → 改为 `final_state=State.from_matrix(rho)`；并在该文件顶部确保 `from aicir.core.state import State`（若无）。**先通读整文件**，把任何直接对 `r.state`/`r.final_state`/`r.snap(...)` 调用 ndarray 方法的断言一并迁移到 `.array`/`.matrix`/`np.asarray(...)`。
- `demos/snap_demo.py:48`：`result.snap(2).reshape(-1), result.final_state.reshape(-1)` → `result.snap(2).array, result.final_state.array`

- [ ] **Step 6: 运行受影响测试，再跑全量**

```bash
PYTHONPATH=. pytest tests/measure tests/noise tests/docs tests/vqc -q
PYTHONPATH=. pytest -q
```
Expected: 全绿。若 `tests/docs/test_readme_measure_examples.py`、`tests/vqc/test_vqe_orchestration.py` 出现红，按同样规则把直接 ndarray 方法调用迁到 `.array`/`.matrix`/`np.asarray`，`np.asarray(...)` 形式应已自动存活。把任何额外迁移点记入提交说明。

- [ ] **Step 7: Commit**

```bash
git add aicir/measure/result.py aicir/measure/measure.py aicir/primitives/estimator.py \
        tests/noise/test_noise_model.py tests/measure/test_measure.py \
        tests/measure/test_measure_run_semantics.py tests/measure/test_result_api.py \
        demos/snap_demo.py
git commit -m "feat(measure)!: Result.state/final_state/snap return State objects"
```

---

## Task 3: 文档与演示收尾

**Files:**
- Modify: `demos/multi_ctrl/multi_ctrl_demo.py`
- Modify: `aicir/measure/README.md`（若存在；否则跳过该项并在报告中说明）
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 简化 multi_ctrl 演示**

`demos/multi_ctrl/multi_ctrl_demo.py` 现在 `result.state` 已是 `State`，把：
```python
# result.state 是演化后末态的原始数组；用 State 包装即可打印狄拉克记号
print(State.from_array(result.state).ket)
```
改为：
```python
print(result.state.ket)
```
（`State` 导入可保留或移除；若移除请确保该行不再引用 `State`。）

运行验证（用非交互后端避免阻塞）：
```bash
MPLBACKEND=Agg PYTHONPATH=. python demos/multi_ctrl/multi_ctrl_demo.py
```
Expected: 打印形如 `…|000>+…|001>+…|100>+…|111>` 的狄拉克记号，无报错。

- [ ] **Step 2: measure README 迁移说明**

若 `aicir/measure/README.md` 存在，新增一小节说明：`Result.state`/`final_state`/`snap()` 现返回 `State`（`.is_density` 区分向量/密度）；`.ket`/`.array`/`.matrix`/`.probabilities()` 可直接用；裸数组迁移指引——`result.state.reshape(-1)` → `result.state.array`，`result.final_state.reshape(2,2)` → `result.final_state.matrix`；`np.asarray(result.state)` 经 `State.__array__` 仍有效。读取该 README 现有风格后追加。

- [ ] **Step 3: CHANGELOG 条目**

`CHANGELOG.md` 顶部按既有格式加 2026-06-15 条目（`### Changed`，标注 breaking）：`Result.state`/`final_state`/`snap()`/`snapshot_states` 返回 `State`；新增 `State.__array__` 保证 `np.asarray`/`np.allclose`/`backend.cast` 兼容；迁移指引 `.reshape(-1)` → `.array`、`.reshape(2,2)` → `.matrix`。

- [ ] **Step 4: 全量回归 + 提交**

```bash
PYTHONPATH=. pytest -q
git add demos/multi_ctrl/multi_ctrl_demo.py aicir/measure/README.md CHANGELOG.md
git commit -m "docs(measure): record Result State migration; simplify multi_ctrl demo"
```

---

## 完成校验

- [ ] `PYTHONPATH=. pytest -q` 全绿。
- [ ] `result.state` / `result.final_state` / `result.snap(i)` 为 `State`；`.ket`/`.array`/`.matrix`/`.is_density` 可用。
- [ ] `np.asarray(result.state)` 形态与旧版一致（向量 `(2^n,)`、密度 `(2^n,2^n)`）。
- [ ] `final_state_kind` 仍正确填充。
- [ ] `multi_ctrl_demo.py` 以 `result.state.ket` 直接打印成功。
- [ ] CHANGELOG 记录了 breaking change 与迁移指引。

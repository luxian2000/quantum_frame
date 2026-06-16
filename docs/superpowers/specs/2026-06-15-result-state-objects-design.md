# Result 状态字段返回 State 对象 — 设计

## 背景与目标

统一 `State` 迁移（`StateVector`/`DensityMatrix` → 单一 `State`）后，`aicir.measure.Result`
仍以原始 numpy 数组暴露 `state` / `final_state` / `snap()`，与全局方向不一致，并直接导致
`result.state.ket` 之类的自然用法报错（`.ket` 只在 `State` 上）。

**目标：** 让 `Result.state`、`Result.final_state`、`Result.snap()` 返回 `State` 对象，
`snapshot_states` 存 `State`。`State` 表达向量/密度两种形态（`.is_density`），并携带
`n_qubits`/backend，因此调用方无需手动 `State.from_array(..., n_qubits, backend)` 包装。

**非目标：** 不改 `reduce()` 的返回类型（它返回的是约化密度矩阵这一派生量，不是“态”本身；
其内部已用 `np.asarray(src)` 读取源态，迁移后经 `__array__` 仍可用）。不改 `state_mode`
metadata（VQE 测试依赖，语义不同）。不触碰 `measure/estimator.py` 的 `PauliEstimateResult`。

## 关键设计：`State.__array__` 作为兼容基石

给 `State` 增加 `__array__`，使 `np.asarray(state)` / `np.allclose(a, state)` /
`backend.cast(state)`（凡走 numpy 转换的路径）继续工作：

```python
def __array__(self, dtype=None):
    arr = self.to_numpy()          # 向量 (2^n,)；密度 (2^n, 2^n)
    return arr.astype(dtype) if dtype is not None else arr
```

形态与现状一致：当前向量态 `result.state` 即 `tr.pre.to_numpy()` = `(2^n,)`，密度为 `(2^n,2^n)`。
因此 `np.asarray(result.state).shape == (4,4)`、`np.asarray(result.state).reshape(-1)`、
`np.allclose(result.final_state, result.state)` 等调用点**无需修改**。

只有**直接在 State 上调用 ndarray 方法**（`.reshape(...)`、`.shape`）的少量调用点需要迁移到
`.array` / `.matrix` / `np.asarray(...)`。

## 受影响面（已勘定）

### 生产代码

1. **`aicir/core/state.py`** — 新增 `__array__`（上）。
2. **`aicir/measure/measure.py` `_build_result`** — 让 `state`/`final`/`snap_states` 为 `State`：
   - 单轨迹/精确分支：`state = tr.pre`、`final = tr.post`（二者本就是 `State`）；
     `snap_states = {t: s for t, s in tr.snaps.items()}`（`s` 已是 `State`）。
   - 聚合（多 shots）分支：`agg["state"]`/`agg["final_state"]`/`snapshot_states` 是密度数组，
     用 `State.from_matrix(arr, n)` 包装；`agg` 里的 snapshot 也逐个包装。
   - 内部期望值计算（约 253–263 行）改用 `np.asarray(state)` 取数组、`final.is_density` 判定密度，
     不再 `state.shape`/`np.asarray(final).ndim` 直读 State 属性。
   - `is_dm` 改由 `final.is_density` 得到；`final_state_kind` 仍按其填 `"density_matrix"`/`"state_vector"`。
3. **`aicir/measure/result.py`**：
   - 类型：`state: Optional[State]`、`final_state: Optional[State]`、`snapshot_states: Dict[int, State]`。
   - `snap(op_index)` 返回 `State`（从 `snapshot_states` 取；不存在返回 `None`）。
   - `final_state_kind` 字段**保留**（向后兼容；`test_measure_run_semantics` 仍读它）。
   - `reduce()` 内部 `arr = np.asarray(src)` 不变（经 `__array__` 工作），返回 ndarray 不变。
   - 导入 `State`（`from ..core.state import State`，文件已可用该路径）。
4. **`aicir/primitives/estimator.py:26`** — `self.backend.cast(result.state)` →
   `self.backend.cast(result.state.to_numpy())`（显式取数组，避免依赖 cast 的隐式转换）。
5. **`aicir/vqc/VQE.py`** — `ansatz_state` 已 `np.asarray(state)`，经 `__array__` 工作，**不改**
   （实现期确认无回归即可）。

### 测试 / 演示（直接 ndarray 方法调用点）

6. `tests/noise/test_noise_model.py:60` `result.final_state.reshape(2,2)` →
   `result.final_state.matrix.reshape(2, 2)`（密度态用 `.matrix`）。
7. `tests/measure/test_measure.py:165` `result.state.reshape(-1)` → `result.state.array`。
8. `tests/measure/test_measure_run_semantics.py`：49/51/80/98 `result.state.reshape(-1)` /
   `result.final_state.reshape(-1)` → `.array`；176/177 `result.snap(N).reshape(-1)` → `.array`；
   158 `result.final_state_kind` 保留不变；148/159 `np.asarray(result.state).shape == (4,4)` 经
   `__array__` 不变。
9. `tests/measure/test_result_api.py:46` 构造 `Result(..., final_state=rho, final_state_kind=...)`
   传原始 `rho` → 改为 `final_state=State.from_matrix(rho)`（字段现为 State）；同步检查该文件其余断言。
10. `demos/snap_demo.py:48` `result.snap(2).reshape(-1)` / `result.final_state.reshape(-1)` →
    `.array`（这些是向量态）。
11. `demos/multi_ctrl/multi_ctrl_demo.py` — 迁移后 `result.state` 已是 `State`，把
    `print(State.from_array(result.state).ket)` 简化回 `print(result.state.ket)`。

> 经 `__array__` 自动存活、无需改的点（实现期复核即可）：`tests/docs/test_readme_measure_examples.py`
> 的 `np.asarray(r.state)…`、`r.reduce(...)`、`np.asarray(r.snap(0))`；`test_measure_run_semantics`
> 的 `np.asarray(...).shape`；VQE `ansatz_state`。

## 验收标准

- `result.state` / `result.final_state` / `result.snap(i)` 均为 `State`（或 `None`），`.ket`/`.array`/
  `.matrix`/`.is_density` 可用。
- `snapshot_states` 值为 `State`。
- `__array__` 使 `np.asarray(result.state)` 形态与旧版一致（向量 `(2^n,)`、密度 `(2^n,2^n)`）。
- `final_state_kind` 仍正确填充。
- `multi_ctrl_demo.py` 以 `result.state.ket` 直接打印成功。
- 全量 `PYTHONPATH=. pytest` 通过（迁移上面 6–10 列出的测试后）。

## 风险与缓解

- **隐式 numpy 转换点**：`__array__` 覆盖；唯一显式改写的生产点是 `primitives/estimator.py`（改 `.to_numpy()`）。
- **聚合分支密度包装**：`State.from_matrix` 需要方阵；`agg` 产出本就是 `(2^n,2^n)`，符合。
- **CHANGELOG**：这是面向用户的 **breaking change**（返回类型从 ndarray 变 State，但保留 `__array__`
  使多数 numpy 用法兼容），需在 CHANGELOG/measure README 记录迁移指引（`.reshape(-1)` → `.array`）。

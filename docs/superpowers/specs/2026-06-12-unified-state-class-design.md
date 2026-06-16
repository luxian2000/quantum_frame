# 统一 `State` 类设计

日期：2026-06-12
状态：已批准（待实现）

## 目标

将当前分散的量子态表示——NumPy 振幅数组、`StateVector`（纯态向量）、`DensityMatrix`
（密度矩阵）——统一到**单一** `State` 类。对外**只保留 `State` API**，删除
`StateVector` 与 `DensityMatrix` 这两个名字，并迁移全部内部调用点。

期望用法：

```python
from aicir.core import State

# 从振幅数组构造（纯态）
psi = State.from_array([1, 0, 0, 1])        # 自动归一化 → 2 比特
# 从密度矩阵构造（可纯可混）
rho = State.from_matrix([[0.5, 0], [0, 0.5]])

psi.array      # numpy (4,) 振幅向量
psi.matrix     # numpy (4,4) 密度矩阵 |ψ><ψ|
psi.ket        # "1/\sqrt{2}|00>+1/\sqrt{2}|11>"

rho.array      # None（混合态无单一振幅向量）
rho.matrix     # numpy (2,2)
rho.ket        # "0.5|0><0|+0.5|1><1|"

print(psi.array); print(rho.matrix); print(rho.ket)   # 三种表示均可直接打印
```

## 核心模型与内部存储

`State` 内部保存**一种**表示，外加一个标记 `_kind ∈ {"vector", "matrix"}`：

- **vector 形态**：后端张量 `(2^n, 1)`，由纯态构造路径（`from_array`、`zero_state`）产生。
- **matrix 形态**：后端张量 `(2^n, 2^n)`，由 `from_matrix` 产生。

`.array` / `.ket` / `.matrix` 这三个用户属性按**量子态的实际性质**（是否为纯态）
按需派生并缓存，**与内部存储形态无关**：

- vector 形态恒为纯态。
- matrix 形态需按纯度（purity ≈ 1）判定纯/混。

派生缓存放在私有字段（如 `_array_cache` / `_matrix_cache`），首次访问时计算。

## 构造函数（`backend` 可选，默认 `NumpyBackend`）

`backend=None` 时实例化一个默认 `NumpyBackend()`
（`from aicir.channel.backends import NumpyBackend`）。

- `State.from_array(array, n_qubits=None, backend=None, bit_order="msb") -> State`
  - 振幅向量，自动归一化（沿用现有逻辑）。
  - `n_qubits=None` 时由数组长度推断（必须是 2 的幂，否则报错）。
  - 产生 vector 形态。
- `State.from_matrix(matrix, n_qubits=None, backend=None) -> State`
  - 密度矩阵；`n_qubits=None` 时由 `(2^n, 2^n)` 形状推断。
  - 产生 matrix 形态。**不**强制归一化迹，但形状须为方阵且边长为 2 的幂。
- `State.zero_state(n_qubits, backend=None, bit_order="msb") -> State`
  - 不变，vector 形态 `|0…0>`。

`__init__` 仍接受 `(data, n_qubits, backend, bit_order)`，但根据 `data` 维度
（1D/列向量 → vector；2D 方阵 → matrix）自动设定 `_kind`；`backend` 同样可选。

## 用户属性与打印

- `.array -> Optional[np.ndarray]`：纯态返回 numpy `(2^n,)` 振幅向量；混合态返回
  `None`。matrix 形态但为纯态时，经主特征向量（最大特征值对应）提取振幅。
- `.matrix -> np.ndarray`：**恒**返回 numpy `(2^n, 2^n)` 密度矩阵；vector 形态时计算
  `|ψ><ψ|`。
- `.ket -> str`：**恒**返回可打印的 Dirac 记号字符串：
  - 纯态：`a|0>+b|1>`（复用现有 `format()`）。
  - 混合态：遍历密度矩阵**所有非零** `ρ_ij`，输出 `ρ_ij|i><j|` 求和形式
    （用户示例 `0.5|0><0|+0.5|1><1|` 是其对角特例）。系数格式复用 `_format_amplitude`。
- 三者均可直接打印：`.array`/`.matrix` 为 numpy 数组（原生可打印），`.ket` 为 `str`。
- `__str__`/`print(state)` 仍输出 ket 形式（纯态超叠加，混合态算符展开）。
- 保留 `.data`（后端原生存储张量，按 `_kind` 返回向量或矩阵）与 `.to_numpy()`，
  供既有内部调用点使用。

## 方法与形态分派

合并两类方法到 `State`，按 `_kind` 分派：

- 纯态/通用：`evolve`、`probabilities`、`measure`、`expectation`、`inner_product`、
  `norm`、`format`、`reorder_endianness`/`msb`/`lsb`。
- 密度矩阵：`partial_trace`、`purity`、`eigenvalues`、`von_neumann_entropy`、
  `is_pure`，以及类方法 `maximally_mixed`。
- 分派示例：
  - `evolve`：vector → `U|ψ⟩`（`apply_unitary`）；matrix → `UρU†`。
  - `partial_trace`：返回 matrix 形态 `State`。
  - `purity`：vector 形态恒为 1.0；matrix 形态计算 `Tr(ρ²)`。
- 新增公开判定属性，替代基于类型的 `isinstance` 分派：
  - `.is_density -> bool`：当前是否为 matrix 形态（`_kind == "matrix"`）。
  - 供 `operators.py`、`measure.py` 等分派点使用。
- `to_density_matrix() -> State`：保留，返回 matrix 形态 `State`（行为等价于
  取 `.matrix` 并封装；用于既有调用点平滑迁移）。

## 迁移：删除 `StateVector` 与 `DensityMatrix`

对外只导出 `State`。改动点：

- `aicir/core/density.py`：**删除**整个 `DensityMatrix` 类；其能力并入 `State`。
- `aicir/core/state.py`：**删除** `StateVector = State` 别名；扩充 `State` 实现。
- 导出清理：
  - `aicir/core/__init__.py`：移除 `StateVector`、`DensityMatrix` 的 import 与
    `__all__` 条目，仅保留 `State`。
  - `aicir/__init__.py`：同上（第 77-78、139-140 行附近）。
- 内部调用点迁移：
  - `aicir/channel/operators.py`：`Union["StateVector","DensityMatrix"]` 类型注解改为
    `State`；`isinstance(state, DensityMatrix)` 改为 `state.is_density`。
  - `aicir/channel/noise/analysis.py`：`DensityMatrix(...)` → `State.from_matrix(...)`
    或 `State(...)`；返回类型注解改 `State`。
  - `aicir/measure/measure.py`：`DensityMatrix.zero_state` → `State.zero_state`
    （matrix 形态，见下）；`isinstance(..., DensityMatrix)` → `state.is_density`；
    `DensityMatrix(...)` → `State.from_matrix(...)`；类型注解改 `State`。
    - 注意：此处需要**矩阵形态的零态**。`State.zero_state` 保持 vector 形态、语义单一；
      迁移**统一用** `State.from_matrix` 构造 `|0…0><0…0|`（对角 `[0,0]=1` 矩阵）。
      不为 `zero_state` 增加 `as_density` 参数。
  - `aicir/channel/backends/base.py`、`aicir/metrics/expressibility.py`：文档/注解中的
    `StateVector` 字样改 `State`。
  - 其余出现 `StateVector` 的 demo（`demos/visual_*.py`）与测试同步改名。
- 文档与 `CHANGELOG.md`：记录这一破坏性接口变更（删除旧名、新增 `from_matrix`/
  `.array`/`.ket`/`.matrix`/`.is_density`、`backend` 变为可选）。

## 测试

- 既有 `tests/circuit/test_state.py`、`tests/backends/test_npu_backend.py` 等改名后
  仍需通过。
- 新增 `tests/core/test_state_unified.py`：
  - 构造：`from_array`（含 `n_qubits` 推断、自动归一化）、`from_matrix`（纯/混）、
    `zero_state`、`backend=None` 默认。
  - 属性派生：纯态 `from_array → .matrix → .ket`；纯态（matrix 形态）`.array` 经特征
    向量提取非空；混合态 `.array is None`、`.matrix` 正确、`.ket` 全非零项展开。
  - 打印：`print(.array)`/`print(.matrix)`/`print(.ket)`/`print(state)` 均产出合理输出。
  - 方法分派：`evolve`、`probabilities`、`measure`、`expectation`、`partial_trace`、
    `purity`、`von_neumann_entropy`、`is_density` 在两种形态下行为正确。
  - 回归：旧 `StateVector`/`DensityMatrix` 名称已不可导入（`ImportError`）。
- 全量 `PYTHONPATH=. pytest` 通过。

## 不做的事（YAGNI）

- 不保留 `StateVector` / `DensityMatrix` 的兼容别名或 shim。
- 不引入张量积 / 部分迹之外的新代数运算。
- `.ket` 不做特征分解形式 `Σ pₖ|ψₖ><ψₖ|`，统一用矩阵元 `ρ_ij|i><j|` 展开。

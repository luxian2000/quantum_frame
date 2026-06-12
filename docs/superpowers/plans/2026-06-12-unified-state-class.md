# 统一 `State` 类 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 NumPy 振幅数组、`StateVector`（纯态向量）、`DensityMatrix`（密度矩阵）统一到单一 `State` 类，对外只保留 `State` API。

**Architecture:** `State` 内部用 `_kind ∈ {"vector","matrix"}` 标记存储形态（向量 `(2^n,1)` 或密度矩阵 `(2^n,2^n)`），所有数值运算按 `_kind` 分派给注入后端。`.array`/`.matrix`/`.ket` 三个用户属性按量子态实际纯/混性质派生并缓存。先“加法”扩充 `State`（旧类仍在、套件全绿），再逐个迁移调用点，最后删除 `StateVector`/`DensityMatrix`。

**Tech Stack:** Python，NumPy，项目自带后端抽象（`aicir/channel/backends`），pytest（`PYTHONPATH=.`）。

参考规格：`docs/superpowers/specs/2026-06-12-unified-state-class-design.md`

---

## 文件结构

- 修改 `aicir/core/state.py` —— `State` 主体（新增 matrix 形态、构造函数、属性、密度方法、格式化 helper）。
- 删除 `aicir/core/density.py` —— `DensityMatrix` 能力并入 `State`（Task 9）。
- 修改 `aicir/core/__init__.py`、`aicir/__init__.py` —— 导出清理（Task 9）。
- 修改 `aicir/channel/operators.py`、`aicir/measure/measure.py`、`aicir/channel/noise/analysis.py`、`aicir/metrics/expressibility.py`、`aicir/channel/backends/base.py` —— 调用点迁移（Task 4-7）。
- 修改 `demos/visual_density_demo.py`、`demos/visual_state_demo.py`、`tests/backends/test_npu_backend.py`、`tests/circuit/test_state.py` —— 迁移与回归（Task 8-9）。
- 新增 `tests/circuit/test_state_unified.py` —— 统一行为测试（Task 10）。
- 修改 `CHANGELOG.md` —— 记录破坏性变更（Task 10）。

约定：所有 `pytest` 命令从仓库根运行，前缀 `PYTHONPATH=.`。

---

## Task 1: matrix 形态存储 + 构造函数 + `is_density`（backend 可选 / n_qubits 推断）

**Files:**
- Modify: `aicir/core/state.py`
- Test: `tests/circuit/test_state_unified.py`（新建）

- [ ] **Step 1: 写失败测试**

新建 `tests/circuit/test_state_unified.py`：

```python
import numpy as np
import pytest

from aicir.core import State


def test_from_array_infers_n_qubits_and_defaults_numpy_backend():
    s = State.from_array([1, 0, 0, 1])  # 无 n_qubits / 无 backend
    assert s.n_qubits == 2
    assert s.is_density is False
    assert s.backend is not None


def test_from_matrix_builds_density_form():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)  # 推断 1 比特
    assert s.n_qubits == 1
    assert s.is_density is True


def test_from_array_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        State.from_array([1, 0, 0])
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -v`
Expected: FAIL（`from_matrix`/`is_density` 不存在，或 `from_array` 缺省参数报错）

- [ ] **Step 3: 改 `aicir/core/state.py`**

在文件顶部 helper 区（`_format_amplitude` 之后）新增模块函数：

```python
def _infer_n_qubits(dim: int) -> int:
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"维数 {dim} 不是 2 的幂")
    return n


def _default_backend():
    from ..channel.backends import NumpyBackend
    return NumpyBackend()
```

把 `__init__` 整体替换为（支持 matrix 形态、backend 可选、初始化缓存）：

```python
    def __init__(self, data, n_qubits: int, backend: "Backend" = None, bit_order: str = "msb"):
        """
        参数:
            data:     后端张量。1D/(2^n,1) 视为纯态向量；(2^n,2^n) 视为密度矩阵
            n_qubits: 量子比特数
            backend:  计算后端实例；None 时使用默认 NumpyBackend
        """
        self._backend = backend if backend is not None else _default_backend()
        self._n_qubits = n_qubits
        self._bit_order = _normalize_bit_order(bit_order)
        self._array_cache = None
        self._matrix_cache = None

        casted = self._backend.cast(data)
        if casted is None:
            raise TypeError("backend.cast 返回了 None，无法构造 State")
        shape = tuple(int(axis) for axis in casted.shape)
        dim = 1 << n_qubits

        if len(shape) == 2 and shape[0] == dim and shape[1] == dim and dim > 1:
            self._kind = "matrix"
            self._data = casted
            return

        if len(shape) == 1:
            casted = self._backend.cast(casted).reshape(-1, 1)
            shape = tuple(int(axis) for axis in casted.shape)
        elif len(shape) == 2 and shape[1] != 1:
            raise ValueError(f"无法识别的数据形状 {shape}（n_qubits={n_qubits}）")
        self._kind = "vector"
        if shape[0] != dim:
            raise ValueError(
                f"数据长度 {shape[0]} 与 n_qubits={n_qubits} 不符（期望 {dim}）"
            )
        self._data = casted
```

替换 `zero_state`、`from_array`（backend 可选、n_qubits 推断），并新增 `from_matrix`、`is_density`：

```python
    @classmethod
    def zero_state(cls, n_qubits: int, backend: "Backend" = None, bit_order: str = "msb") -> "State":
        """创建 |0⊗n⟩ 计算基基态（向量形态）。"""
        backend = backend if backend is not None else _default_backend()
        data = backend.zeros_state(n_qubits)
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @classmethod
    def from_array(cls, array, n_qubits: int = None, backend: "Backend" = None, bit_order: str = "msb") -> "State":
        """从 numpy array / list 构造态向量（自动归一化）。n_qubits 省略时由长度推断。"""
        backend = backend if backend is not None else _default_backend()
        np_array = np.asarray(array, dtype=np.complex64).reshape(-1)
        if n_qubits is None:
            n_qubits = _infer_n_qubits(np_array.shape[0])
        norm = float(np.linalg.norm(np_array))
        if norm <= 0:
            raise ValueError("输入数组范数必须大于 0")
        data = backend.cast(np_array / norm)
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @classmethod
    def from_matrix(cls, matrix, n_qubits: int = None, backend: "Backend" = None) -> "State":
        """从密度矩阵 (2^n,2^n) 构造混合/纯态（matrix 形态）。n_qubits 省略时由形状推断。"""
        backend = backend if backend is not None else _default_backend()
        np_m = np.asarray(matrix, dtype=np.complex64)
        if np_m.ndim != 2 or np_m.shape[0] != np_m.shape[1]:
            raise ValueError("from_matrix 需要方阵 (2^n, 2^n)")
        if n_qubits is None:
            n_qubits = _infer_n_qubits(np_m.shape[0])
        return cls(backend.cast(np_m), n_qubits, backend)

    @property
    def is_density(self) -> bool:
        """当前是否以密度矩阵形态存储。"""
        return self._kind == "matrix"
```

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/core/state.py tests/circuit/test_state_unified.py
git commit -m "feat(state): 支持 matrix 形态存储、from_matrix、is_density、可选 backend"
```

---

## Task 2: 密度方法并入 + 形态分派（evolve / probabilities / measure / expectation / purity / partial_trace 等）

**Files:**
- Modify: `aicir/core/state.py`
- Test: `tests/circuit/test_state_unified.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/circuit/test_state_unified.py`：

```python
def test_matrix_form_methods_dispatch():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    assert s.purity() == pytest.approx(0.5)
    assert s.is_pure() is False
    assert s.von_neumann_entropy() == pytest.approx(np.log(2))
    probs = np.asarray(s.probabilities())
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-6)


def test_vector_form_purity_is_one_and_partial_trace_returns_density():
    s = State.from_array([1, 0, 0, 1], n_qubits=2)  # 贝尔态
    assert s.purity() == pytest.approx(1.0)
    red = s.partial_trace(keep=[0])
    assert red.is_density is True
    assert red.n_qubits == 1
    assert red.purity() == pytest.approx(0.5, abs=1e-6)


def test_to_density_matrix_returns_matrix_form_state():
    s = State.from_array([1, 0], n_qubits=1)
    rho = s.to_density_matrix()
    assert isinstance(rho, State)
    assert rho.is_density is True
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -k "dispatch or partial_trace or to_density_matrix" -v`
Expected: FAIL（`purity`/`partial_trace`/`von_neumann_entropy` 等不存在）

- [ ] **Step 3: 改 `aicir/core/state.py`**

新增私有 helper（放在 `data` property 附近）：

```python
    def _matrix_data(self):
        """返回密度矩阵形态的后端原生张量（向量形态时计算 |ψ><ψ|）。"""
        if self._kind == "matrix":
            return self._data
        bk = self._backend
        return bk.matmul(self._data, bk.dagger(self._data))
```

把 `evolve` 替换为按形态分派：

```python
    def evolve(self, unitary) -> "State":
        """酉演化：向量形态 U|ψ⟩；矩阵形态 UρU†。返回新 State。"""
        bk = self._backend
        if self._kind == "vector":
            new_data = bk.apply_unitary(self._data, unitary)
        else:
            new_data = bk.matmul(bk.matmul(unitary, self._data), bk.dagger(unitary))
        return State(new_data, self._n_qubits, bk, bit_order=self._bit_order)
```

把 `probabilities` 替换为：

```python
    def probabilities(self):
        """计算基测量概率。向量形态返回后端张量；矩阵形态返回 numpy。"""
        if self._kind == "vector":
            return self._backend.measure_probs(self._data)
        diag = self._backend.to_numpy(self._backend.real(self._data.diagonal()))
        diag = np.clip(np.asarray(diag), 0, None)
        total = diag.sum()
        return diag / total if total > 0 else diag
```

把 `measure` 替换为按形态分派：

```python
    def measure(self, shots: int = 1024, bit_order: Optional[str] = None) -> Dict[str, int]:
        """模拟 shots 次测量，返回各基态计数（仅非零项）。"""
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        if self._kind == "vector":
            counts_arr = self._backend.sample(self.probabilities(), shots)
            counts_np = self._backend.to_numpy(counts_arr).astype(int).reshape(-1)
        else:
            probs = self.probabilities()
            indices = np.random.choice(len(probs), size=shots, p=probs)
            counts_np = np.bincount(indices, minlength=len(probs))
        return {
            f"|{_basis_label(idx, self._n_qubits, order)}>": int(c)
            for idx, c in enumerate(counts_np)
            if c > 0
        }
```

把 `expectation` 替换为：

```python
    def expectation(self, operator) -> float:
        """期望值：向量形态 ⟨ψ|O|ψ⟩；矩阵形态 Tr(ρO)。"""
        if self._kind == "vector":
            value = self._backend.expectation_sv(self._data, operator)
        else:
            value = self._backend.expectation_dm(self._data, operator)
        if value is None:
            raise TypeError("backend expectation 返回了 None")
        return float(value)
```

把 `to_density_matrix` 替换为返回 matrix 形态 `State`（删除原 `DensityMatrix` 导入与构造）：

```python
    def to_density_matrix(self) -> "State":
        """转为密度矩阵形态 State：ρ = |ψ⟩⟨ψ|（向量形态）或原样（矩阵形态）。"""
        return State(self._matrix_data(), self._n_qubits, self._backend)
```

在 `to_numpy` 之前新增密度方法：

```python
    def partial_trace(self, keep) -> "State":
        """对子系统求偏迹，返回 matrix 形态 State（形状 2^k×2^k，k=len(keep)）。"""
        bk = self._backend
        rho_red = bk.partial_trace(self._matrix_data(), keep, self._n_qubits)
        return State(rho_red, len(keep), bk)

    def purity(self) -> float:
        """纯度 Tr(ρ²)。向量形态恒为 1.0。"""
        if self._kind == "vector":
            return 1.0
        bk = self._backend
        val = bk.trace(bk.matmul(self._data, self._data))
        return float(np.real(bk.to_numpy(val)))

    def eigenvalues(self) -> np.ndarray:
        """密度矩阵特征值（升序），numpy array。"""
        return np.linalg.eigvalsh(self.matrix)

    def von_neumann_entropy(self) -> float:
        """冯·诺依曼熵 S(ρ) = -Tr(ρ ln ρ)。"""
        eigs = self.eigenvalues()
        eigs = eigs[eigs > 1e-15]
        return float(-np.sum(eigs * np.log(eigs)))

    def is_pure(self, tol: float = 1e-5) -> bool:
        """是否纯态（purity ≈ 1）。"""
        return abs(self.purity() - 1.0) < tol

    @classmethod
    def maximally_mixed(cls, n_qubits: int, backend: "Backend" = None) -> "State":
        """最大混合态 ρ = I / 2^n（matrix 形态）。"""
        backend = backend if backend is not None else _default_backend()
        dim = 1 << n_qubits
        rho_np = np.eye(dim, dtype=np.complex64) / dim
        return cls(backend.cast(rho_np), n_qubits, backend)
```

把 `to_numpy` 替换为按形态返回：

```python
    def to_numpy(self) -> np.ndarray:
        """向量形态导出 (2^n,)；矩阵形态导出 (2^n, 2^n)。"""
        arr = self._backend.to_numpy(self._data)
        return arr.reshape(-1) if self._kind == "vector" else arr
```

删除文件顶部 `TYPE_CHECKING` 块中 `from .density import DensityMatrix` 一行（已不再引用）。

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -v`
Expected: PASS（全部通过）

- [ ] **Step 5: 回归既有 state 测试**

Run: `PYTHONPATH=. pytest tests/circuit/test_state.py -v`
Expected: PASS（旧测试不受影响）

- [ ] **Step 6: 提交**

```bash
git add aicir/core/state.py tests/circuit/test_state_unified.py
git commit -m "feat(state): 并入密度矩阵方法并按形态分派 evolve/measure/expectation 等"
```

---

## Task 3: `.array` / `.matrix` / `.ket` 属性与打印

**Files:**
- Modify: `aicir/core/state.py`
- Test: `tests/circuit/test_state_unified.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/circuit/test_state_unified.py`：

```python
def test_pure_state_three_representations():
    s = State.from_array([1, 0, 0, 1], n_qubits=2)
    np.testing.assert_allclose(
        s.array, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-6
    )
    assert s.matrix.shape == (4, 4)
    assert s.ket == "1/\\sqrt{2}|00>+1/\\sqrt{2}|11>"


def test_mixed_state_array_is_none_and_ket_is_operator_form():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    assert s.array is None
    assert s.matrix.shape == (2, 2)
    assert s.ket == "0.5|0><0|+0.5|1><1|"


def test_matrix_form_pure_state_array_extracted_via_eigenvector():
    s = State.from_array([0, 1], n_qubits=1).to_density_matrix()  # |1><1|
    assert s.is_density is True
    np.testing.assert_allclose(np.abs(s.array), [0, 1], atol=1e-6)
    assert s.ket == "1|1>"


def test_representations_are_printable():
    s = State.from_array([1, 0], n_qubits=1)
    # 不抛异常即可
    assert isinstance(str(s.ket), str)
    assert "1" in np.array2string(s.array)
    assert s.matrix.shape == (2, 2)
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -k "representation or array or ket" -v`
Expected: FAIL（`.array`/`.matrix`/`.ket` 不存在）

- [ ] **Step 3: 改 `aicir/core/state.py`**

把模块级的 `format` 逻辑抽到两个 helper（放在 `_format_amplitude` 之后）：

```python
def _format_ket(amplitudes, n_qubits: int, bit_order: str, atol: float = 1e-6) -> str:
    """Σ aᵢ|i> 形式。"""
    terms = []
    for idx, amplitude in enumerate(amplitudes):
        if abs(amplitude) < atol:
            continue
        coeff = _format_amplitude(amplitude, atol)
        label = _basis_label(idx, n_qubits, bit_order)
        terms.append(f"{coeff}|{label}>")
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr += term if term.startswith("-") else f"+{term}"
    return expr


def _format_density_ket(matrix, n_qubits: int, bit_order: str, atol: float = 1e-6) -> str:
    """Σ ρ_ij|i><j| 形式（遍历所有非零矩阵元）。"""
    terms = []
    dim = matrix.shape[0]
    for i in range(dim):
        for j in range(dim):
            val = matrix[i, j]
            if abs(val) < atol:
                continue
            coeff = _format_amplitude(val, atol)
            li = _basis_label(i, n_qubits, bit_order)
            lj = _basis_label(j, n_qubits, bit_order)
            terms.append(f"{coeff}|{li}><{lj}|")
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr += term if term.startswith("-") else f"+{term}"
    return expr
```

新增 `array` / `matrix` / `ket` 属性（放在 `data` property 之后）：

```python
    @property
    def array(self):
        """纯态返回 numpy (2^n,) 振幅向量；混合态返回 None。"""
        if self._array_cache is not None:
            return self._array_cache
        if self._kind == "vector":
            self._array_cache = self._backend.to_numpy(self._data).reshape(-1)
            return self._array_cache
        rho = self.matrix
        if not self.is_pure():
            return None
        evals, evecs = np.linalg.eigh(rho)
        idx = int(np.argmax(evals.real))
        vec = evecs[:, idx]
        nz = int(np.argmax(np.abs(vec) > 1e-9))
        phase = np.exp(-1j * np.angle(vec[nz])) if abs(vec[nz]) > 0 else 1.0
        self._array_cache = (vec * phase).astype(np.complex64)
        return self._array_cache

    @property
    def matrix(self) -> np.ndarray:
        """恒返回 numpy (2^n, 2^n) 密度矩阵。"""
        if self._matrix_cache is None:
            self._matrix_cache = self._backend.to_numpy(self._matrix_data())
        return self._matrix_cache

    @property
    def ket(self) -> str:
        """可打印 Dirac 记号：纯态 Σaᵢ|i>；混合态 Σρ_ij|i><j|。"""
        return self.format()
```

把 `format` 改为统一走 helper（支持混合态）：

```python
    def format(self, bit_order: Optional[str] = None, atol: float = 1e-6) -> str:
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        arr = self.array
        if arr is None:
            return _format_density_ket(self.matrix, self._n_qubits, order, atol)
        return _format_ket(arr, self._n_qubits, order, atol)
```

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -v`
Expected: PASS（全部通过）

- [ ] **Step 5: 回归 state 测试**

Run: `PYTHONPATH=. pytest tests/circuit/test_state.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add aicir/core/state.py tests/circuit/test_state_unified.py
git commit -m "feat(state): 新增 array/matrix/ket 属性与混合态 Dirac 展开打印"
```

---

## Task 4: 迁移 `aicir/channel/operators.py`

**Files:**
- Modify: `aicir/channel/operators.py:33`、`:391-415`

- [ ] **Step 1: 改类型注解与分派**

把 `expectation` 方法（约 391-415 行）替换为：

```python
    def expectation(
        self,
        state: "State",
        backend: "Backend",
    ) -> float:
        """
        计算量子态对哈密顿量的期望值。

        参数:
            state:   State 实例（向量或密度矩阵形态）
            backend: 计算后端
        返回:
            实数期望值
        """
        from ..core.state import State

        H_mat = self.to_matrix(backend)
        if isinstance(state, State):
            return state.expectation(H_mat)
        # 兼容原始后端张量（态向量）
        return backend.expectation_sv(state, H_mat)
```

把第 33 行 `TYPE_CHECKING` 块里的 `from ..core.density import DensityMatrix` 改为 `from ..core.state import State`（若已 import State 则删除 density 行）。

- [ ] **Step 2: 运行相关测试**

Run: `PYTHONPATH=. pytest tests/ -k "hamiltonian or pauli or expectation or operator" -q`
Expected: PASS（无 `DensityMatrix` 相关失败）

- [ ] **Step 3: 提交**

```bash
git add aicir/channel/operators.py
git commit -m "refactor(operators): Hamiltonian.expectation 改用统一 State"
```

---

## Task 5: 迁移 `aicir/measure/measure.py`

**Files:**
- Modify: `aicir/measure/measure.py:14-15`、`:261-336`

- [ ] **Step 1: 确认 numpy 已导入**

Run: `PYTHONPATH=. python -c "import ast,sys; src=open('aicir/measure/measure.py').read(); print('import numpy' in src)"`
Expected: `True`（若为 `False`，在导入区加 `import numpy as np`）

- [ ] **Step 2: 改导入**

把第 14-15 行的
```python
from ..core.density import DensityMatrix
from ..core.state import StateVector
```
替换为：
```python
from ..core.state import State
```

- [ ] **Step 3: 改 `_build_initial_state`（约 261-270 行）**

```python
    def _build_initial_state(self, n_qubits: int, backend, initial_state=None) -> State:
        if initial_state is None:
            return State.zero_state(n_qubits, backend)
        if isinstance(initial_state, State):
            if initial_state.n_qubits != n_qubits:
                raise ValueError("initial_state.n_qubits 与电路 n_qubits 不一致")
            return initial_state
        return State(initial_state, n_qubits, backend)
```

（保留该方法原有的 n_qubits 校验风格；若原方法体不同，按等价语义对齐。）

- [ ] **Step 4: 改 `_build_initial_density_matrix`（约 272-286 行）**

```python
    def _build_initial_density_matrix(
        self,
        n_qubits: int,
        backend,
        initial_density_matrix=None,
    ) -> State:
        if initial_density_matrix is None:
            dim = 1 << n_qubits
            rho = np.zeros((dim, dim), dtype=np.complex64)
            rho[0, 0] = 1.0 + 0j
            return State.from_matrix(rho, n_qubits, backend)

        if isinstance(initial_density_matrix, State) and initial_density_matrix.is_density:
            if initial_density_matrix.n_qubits != n_qubits:
                raise ValueError("initial_density_matrix.n_qubits 与电路 n_qubits 不一致")
            return initial_density_matrix

        return State.from_matrix(initial_density_matrix, n_qubits, backend)
```

- [ ] **Step 5: 改 gatewise 演化的类型注解与构造（约 302-336 行）**

把 `_evolve_state_vector_gatewise` 签名与体内 `StateVector(...)` 改为 `State`：

```python
    def _evolve_state_vector_gatewise(self, circuit, sv0: State, backend) -> State:
        sv = sv0
        for gate in circuit_instructions(circuit):
            if _is_measure_gate(gate):
                continue
            new_data = apply_gate_to_state(gate, sv.data, sv.n_qubits, backend)
            if new_data is None:
                gm = gate_to_matrix(gate, cir_qubits=sv.n_qubits, backend=backend)
                sv = sv.evolve(gm)
            else:
                sv = State(new_data, sv.n_qubits, backend, bit_order=sv.bit_order)
        return sv
```

把 `_evolve_density_matrix_gatewise` 的 `rho0: DensityMatrix` / `-> DensityMatrix` 改 `State`，体内 `DensityMatrix(rho_noisy, rho.n_qubits, backend)` 改为 `State(rho_noisy, rho.n_qubits, backend)`。

- [ ] **Step 6: 运行 measure 相关测试**

Run: `PYTHONPATH=. pytest tests/ -k "measure or density or noise" -q`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add aicir/measure/measure.py
git commit -m "refactor(measure): 初始态/演化全面改用统一 State"
```

---

## Task 6: 迁移 `aicir/channel/noise/analysis.py`

**Files:**
- Modify: `aicir/channel/noise/analysis.py:13`、`:55-75`（及其余 `DensityMatrix` 出现处）

- [ ] **Step 1: 删除 density 导入**

删除第 13 行 `from ...core.density import DensityMatrix`（`State` 已在第 15 行导入）。

- [ ] **Step 2: 改返回注解与构造**

把 `evolve_density_gatewise` 的返回注解 `-> DensityMatrix` 改 `-> State`；体内
```python
            rho = DensityMatrix(
                noise_model.apply(...),
                circuit.n_qubits,
                backend,
            )
```
改为：
```python
            rho = State(
                noise_model.apply(
                    rho.data,
                    n_qubits=circuit.n_qubits,
                    backend=backend,
                    gate_type=instruction_name(gate),
                    gate=gate,
                ),
                circuit.n_qubits,
                backend,
            )
```

Run: `PYTHONPATH=. grep -n "DensityMatrix" aicir/channel/noise/analysis.py`
Expected: 无输出（该文件已无 `DensityMatrix`）

- [ ] **Step 3: 运行 noise 测试**

Run: `PYTHONPATH=. pytest tests/ -k "noise or sensitivity or fidelity" -q`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add aicir/channel/noise/analysis.py
git commit -m "refactor(noise): analysis 改用统一 State"
```

---

## Task 7: 迁移 `aicir/metrics/expressibility.py` 与 `backends/base.py` 文档

**Files:**
- Modify: `aicir/metrics/expressibility.py:18,79,161,282`
- Modify: `aicir/channel/backends/base.py:14`

- [ ] **Step 1: 改 expressibility 导入与用法**

第 18 行 `from ..core.state import StateVector` → `from ..core.state import State`。
把第 79、161、282 行的 `StateVector` 全部改为 `State`（函数注解 `sv1: State, sv2: State`；`State.zero_state(...)`；`State.from_array(...)`）。

Run: `PYTHONPATH=. grep -n "StateVector" aicir/metrics/expressibility.py`
Expected: 无输出

- [ ] **Step 2: 改 base.py 文档字符串**

第 14 行注释中的 `StateVector` 改为 `State`（仅文档，无代码语义）。

- [ ] **Step 3: 运行 metrics 测试**

Run: `PYTHONPATH=. pytest tests/ -k "express or metric" -q`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add aicir/metrics/expressibility.py aicir/channel/backends/base.py
git commit -m "refactor(metrics): expressibility 改用统一 State；更新 backend 文档"
```

---

## Task 8: 迁移 demos 与 npu 测试

**Files:**
- Modify: `demos/visual_density_demo.py:12,28`
- Modify: `demos/visual_state_demo.py:12,18,27`
- Modify: `tests/backends/test_npu_backend.py:8,84`

- [ ] **Step 1: 改 demos**

两份 demo 里 `from aicir import ... StateVector ...` 的 `StateVector` 改为 `State`；
`StateVector.zero_state(...)` → `State.zero_state(...)`；
`visual_state_demo.py` 中 `-> StateVector` 注解改 `-> State`。

- [ ] **Step 2: 改 npu 测试**

`tests/backends/test_npu_backend.py` 第 8 行导入 `StateVector` 改 `State`；第 84 行 `StateVector.zero_state(2, backend)` 改 `State.zero_state(2, backend)`。

- [ ] **Step 3: 校验无残留（demos/tests 侧）**

Run: `PYTHONPATH=. grep -rn "StateVector" demos/ tests/backends/test_npu_backend.py`
Expected: 无输出

- [ ] **Step 4: 运行 npu 测试（无 NPU 时应 skip 而非 error）**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_backend.py -q`
Expected: PASS 或 SKIPPED（不得 import 错误）

- [ ] **Step 5: 提交**

```bash
git add demos/visual_density_demo.py demos/visual_state_demo.py tests/backends/test_npu_backend.py
git commit -m "refactor(demos,tests): visual demo 与 npu 测试改用统一 State"
```

---

## Task 9: 删除 `DensityMatrix` 与 `StateVector`，清理导出

**Files:**
- Delete: `aicir/core/density.py`
- Modify: `aicir/core/state.py`（删除 `StateVector = State`）
- Modify: `aicir/core/__init__.py:32-33,55-56`
- Modify: `aicir/__init__.py:77-78,139-140`
- Modify: `tests/circuit/test_state.py:6,43-44`

- [ ] **Step 1: 先更新 test_state.py（与删除同提交，保持绿）**

第 6 行导入改为：
```python
from aicir import NumpyBackend, State, TorchBackend
```
删除 `test_statevector_remains_alias` 方法（43-44 行），新增旧名已移除的回归测试：
```python
    def test_legacy_names_removed(self):
        with self.assertRaises(ImportError):
            from aicir import StateVector  # noqa: F401
        with self.assertRaises(ImportError):
            from aicir import DensityMatrix  # noqa: F401
```
把第 54 行 `StateVector.zero_state(2, backend)` 改为 `State.zero_state(2, backend)`。

- [ ] **Step 2: 删除 `StateVector` 别名**

删除 `aicir/core/state.py` 末尾的 `StateVector = State`。

- [ ] **Step 3: 删除 density.py**

```bash
git rm aicir/core/density.py
```

- [ ] **Step 4: 清理 `aicir/core/__init__.py`**

第 32 行 `from .state import State, StateVector` → `from .state import State`。
删除第 33 行 `from .density import DensityMatrix`。
从 `__all__` 删除 `"StateVector"`、`"DensityMatrix"` 两条。

- [ ] **Step 5: 清理 `aicir/__init__.py`**

删除第 77-78 行附近 `from .core.state import State, StateVector` 中的 `StateVector`（改为 `from .core.state import State`）及 `from .core.density import DensityMatrix` 整行。
从第 139-140 行附近的 `__all__` 删除 `"StateVector"`、`"DensityMatrix"`。

- [ ] **Step 6: 全局校验无残留**

Run: `PYTHONPATH=. grep -rn "StateVector\|DensityMatrix\|core.density\|import density" aicir/ tests/ demos/`
Expected: 无输出

- [ ] **Step 7: 运行受影响测试**

Run: `PYTHONPATH=. pytest tests/circuit/test_state.py tests/circuit/test_state_unified.py -v`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add -A
git commit -m "refactor!(state): 删除 StateVector/DensityMatrix，仅保留统一 State API"
```

---

## Task 10: 统一测试补全、CHANGELOG、全量回归

**Files:**
- Modify: `tests/circuit/test_state_unified.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 补充端序与默认后端测试**

追加到 `tests/circuit/test_state_unified.py`：

```python
def test_default_numpy_backend_used_when_omitted():
    from aicir.channel.backends import NumpyBackend

    s = State.zero_state(1)
    assert isinstance(s.backend, NumpyBackend)


def test_mixed_state_with_offdiagonal_ket_lists_all_terms():
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    ket = s.ket
    for sub in ("|0><0|", "|0><1|", "|1><0|", "|1><1|"):
        assert sub in ket
```

- [ ] **Step 2: 运行新测试**

Run: `PYTHONPATH=. pytest tests/circuit/test_state_unified.py -v`
Expected: PASS

- [ ] **Step 3: 写 CHANGELOG 条目**

在 `CHANGELOG.md` 顶部新增一段（沿用文件现有日期/格式风格）：

```markdown
## 2026-06-12

### Changed (Breaking)
- 统一量子态表示为单一 `State` 类：删除 `StateVector` 与 `DensityMatrix` 两个公开名称。
  - 新增 `State.from_matrix(...)` 从密度矩阵构造；`from_array`/`from_matrix`/`zero_state`
    的 `backend` 改为可选（默认 `NumpyBackend`），`from_array`/`from_matrix` 可省略
    `n_qubits`（按长度/形状推断）。
  - 新增属性 `.array`（纯态振幅向量，混合态为 `None`）、`.matrix`（密度矩阵）、
    `.ket`（Dirac 记号字符串：纯态超叠加、混合态 Σρ_ij|i><j| 展开），均可直接打印。
  - 新增 `.is_density` 判定属性；密度矩阵方法（`purity`/`partial_trace`/`eigenvalues`/
    `von_neumann_entropy`/`is_pure`/`maximally_mixed`）并入 `State`。
  - 迁移影响：`isinstance(x, DensityMatrix)` 改用 `x.is_density`；
    `DensityMatrix(...)` 改用 `State.from_matrix(...)`。
```

- [ ] **Step 4: 全量回归**

Run: `PYTHONPATH=. pytest -q`
Expected: PASS（无 import 错误；既有跳过项仍正常 skip）

- [ ] **Step 5: 提交**

```bash
git add tests/circuit/test_state_unified.py CHANGELOG.md
git commit -m "test,docs(state): 补全统一 State 测试与 CHANGELOG 破坏性变更记录"
```

---

## 自检对照

- 规格「核心模型与内部存储」→ Task 1（`_kind`/matrix 形态）+ Task 3（派生缓存）。
- 规格「构造函数」→ Task 1（`from_array`/`from_matrix`/`zero_state`、backend 可选、n_qubits 推断）。
- 规格「用户属性与打印」→ Task 3（`.array`/`.matrix`/`.ket` + helper）。
- 规格「方法与形态分派」+ `.is_density` → Task 1（`is_density`）+ Task 2（分派/密度方法/`to_density_matrix`）。
- 规格「迁移：删除旧名」→ Task 4-9。
- 规格「测试」→ Task 1-3、10（含旧名 ImportError 回归与全量 pytest）。
- 规格「不做的事」→ 计划未引入兼容别名、未加新代数运算、`.ket` 统一用 ρ_ij 展开。

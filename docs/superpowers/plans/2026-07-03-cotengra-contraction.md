# cotengra 路径优化 + 切片执行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `aicir.simulator` 接入 cotengra：HyperOptimizer 找收缩路径 + 内存超预算时切片执行，执行侧完全保留 `backend.tensordot` 通道（NPU 复数分解与 autograd 不破坏）。

**Architecture:** cotengra 只做规划（路径 + 切片指标），不做执行。切片执行 = 对每个切片赋值用新原语 `backend.take` 固定指标 → 同一 pairwise 收缩 → `backend.add` 累加（求和保梯度）。新原语在 NPUBackend 上按既有 `_NpuTransposeFn` 模式包 `torch.autograd.Function`（规避 `aclnnAdd DT_COMPLEX64`）。公共 API 四函数加 `optimize=`/`memory_limit=`。

**Tech Stack:** Python，numpy（唯一硬依赖）；可选 `cotengra`（新 `tn` extra）、`opt_einsum`、`torch`。

## Global Constraints

- 仓库根目录 `PYTHONPATH=.` 运行；测试 `PYTHONPATH=. pytest`。
- `numpy` 唯一硬依赖；cotengra/opt_einsum/torch 全部惰性 import，缺失时相关测试 `pytest.importorskip` 跳过。
- 注释/docstring/README 中文，跟随周边风格。
- cotengra 只规划不执行；执行必须走 `_pair_contract` → `backend.tensordot`。
- `memory_limit` = 最大中间张量元素数（int）；设定时必须 cotengra（缺失抛 `ImportError` 提示 `pip install -e ".[tn]"`）；与 `optimize in ("opt_einsum","greedy")` 组合抛 `ValueError`。
- 切片指标不得含 open index（`AssertionError` 防御）。
- `Measure.run(method="tensor")` 不加新参数。
- 每任务结束提交；提交信息末尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **对 spec 的一处显式偏差（执行前请人确认）**：spec 写 `optimize="auto"` 时 cotengra 优先。但 HyperOptimizer 每次搜索耗时可达秒级，小网络上纯浪费且会拖慢全量测试。本计划改为：`auto` 下仅当张量数 ≥ 24 才用 cotengra，否则走 opt_einsum/greedy；显式 `optimize="cotengra"` 或设 `memory_limit` 时始终用。

---

### Task 1: backend 原语 `take` / `add`（base + numpy + gpu）

**Files:**
- Modify: `aicir/backends/base.py`（`conj` 抽象方法之后追加两个抽象方法）
- Modify: `aicir/backends/numpy_backend.py`（`conj` 实现之后追加）
- Modify: `aicir/backends/gpu_backend.py`（`conj` 实现之后追加）
- Test: `tests/backends/test_take_add.py`（新建）

**Interfaces:**
- Produces: `backend.take(a, axis, index) -> tensor`（沿 `axis` 取下标 `index`，该轴消失）；`backend.add(a, b) -> tensor`（逐元素相加）。Task 2/3 依赖。

- [ ] **Step 1: 写失败测试**

`tests/backends/test_take_add.py`：

```python
import numpy as np
import pytest

from aicir.backends import NumpyBackend


def test_numpy_take_drops_axis():
    bk = NumpyBackend()
    a = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4)
    out = bk.to_numpy(bk.take(bk.cast(a), 1, 2))
    assert np.allclose(out, a[:, 2, :], atol=1e-5)
    assert out.shape == (2, 4)


def test_numpy_add():
    bk = NumpyBackend()
    a = np.array([1 + 2j, 3 + 4j])
    b = np.array([5 + 6j, 7 + 8j])
    assert np.allclose(bk.to_numpy(bk.add(bk.cast(a), bk.cast(b))), a + b, atol=1e-5)


def test_gpu_take_add():
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    a = (np.arange(8) + 1j * np.arange(8)).reshape(2, 2, 2)
    out = bk.to_numpy(bk.take(bk.cast(a), 0, 1))
    assert np.allclose(out, a[1], atol=1e-4)
    s = bk.to_numpy(bk.add(bk.cast(a), bk.cast(a)))
    assert np.allclose(s, 2 * a, atol=1e-4)


def test_base_take_add_not_implemented():
    from aicir.backends.base import Backend
    class Dummy(Backend):
        pass
    # Backend 抽象方法较多，直接实例化不可行时改为检查基类方法体抛 NotImplementedError
    import aicir.backends.base as base_mod
    import inspect
    src = inspect.getsource(base_mod.Backend.take)
    assert "NotImplementedError" in src
    src = inspect.getsource(base_mod.Backend.add)
    assert "NotImplementedError" in src
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_take_add.py -v`
Expected: FAIL（`AttributeError: ... has no attribute 'take'`）

- [ ] **Step 3: 实现**

`aicir/backends/base.py`（`conj` 之后，同风格）：

```python
    def take(self, a, axis, index):
        """沿 axis 取下标 index（该轴消失），用于张量网络切片固定指标。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 take")

    def add(self, a, b):
        """逐元素相加，用于切片收缩结果累加。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 add")
```

`aicir/backends/numpy_backend.py`：

```python
    def take(self, a, axis, index):
        return np.take(a, int(index), axis=int(axis))

    def add(self, a, b):
        return a + b
```

`aicir/backends/gpu_backend.py`：

```python
    def take(self, a, axis, index):
        return torch.select(a, int(axis), int(index))

    def add(self, a, b):
        return a + b
```

（两文件的 import 已有 np/torch，不需新增。）

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_take_add.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/base.py aicir/backends/numpy_backend.py aicir/backends/gpu_backend.py tests/backends/test_take_add.py
git commit -m "feat(backends): 新增 take/add 张量原语（切片收缩用）"
```

---

### Task 2: NPU autograd-safe `_NpuTakeFn` / `_NpuAddFn`

**Files:**
- Modify: `aicir/backends/npu_backend.py`（`_NpuConjFn` 类之后加两个 Function；`conj` override 之后加两个 override）
- Test: `tests/backends/test_take_add.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `take`/`add` 抽象与 GPU 实现（NPU 继承 GPUBackend，非复数/非 npu 走 `super()`）。
- Produces: NPU 复数路径下 autograd-safe 的 `take`/`add`。

> 背景：Ascend 上 `aclnnAdd` 不支持 `DT_COMPLEX64`（真机已爆过）；且在追踪图中对复数张量做 `torch.real(a)`/`torch.imag(a)` 双读会造成 fan-out，backward 需复数梯度累加同样触发该缺失算子。既有解法是把整个操作包成 `torch.autograd.Function`（见同文件 `_NpuTransposeFn`/`_NpuReshapeFn`/`_NpuConjFn`，约 251-311 行），复数张量在图中保持单条输入边。照抄该模式。

- [ ] **Step 1: 写失败测试（追加到 test_take_add.py）**

```python
def test_npu_take_fn_forward_backward_matches_native():
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import _NpuTakeFn

    a = torch.randn(2, 3, 4, dtype=torch.complex64, requires_grad=True)
    ref_in = a.detach().clone().requires_grad_(True)

    out = _NpuTakeFn.apply(a, 1, 2)
    ref = ref_in.select(1, 2)
    assert torch.allclose(out, ref, atol=1e-5)
    out.real.sum().backward()
    ref.real.sum().backward()
    assert torch.allclose(a.grad, ref_in.grad, atol=1e-5)


def test_npu_add_fn_forward_backward_matches_native():
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import _NpuAddFn

    a = torch.randn(5, dtype=torch.complex64, requires_grad=True)
    b = torch.randn(5, dtype=torch.complex64, requires_grad=True)
    ra = a.detach().clone().requires_grad_(True)
    rb = b.detach().clone().requires_grad_(True)

    out = _NpuAddFn.apply(a, b)
    ref = ra + rb
    assert torch.allclose(out, ref, atol=1e-5)
    out.real.sum().backward()
    ref.real.sum().backward()
    assert torch.allclose(a.grad, ra.grad, atol=1e-5)
    assert torch.allclose(b.grad, rb.grad, atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_take_add.py -v -k npu`
Expected: FAIL（`ImportError: cannot import name '_NpuTakeFn'`）

- [ ] **Step 3: 实现**

`aicir/backends/npu_backend.py`，`_NpuConjFn` 之后：

```python
class _NpuTakeFn(torch.autograd.Function):
    """切片取指标的 NPU 自动微分安全包装（原理同 ``_NpuTransposeFn``）。"""

    @staticmethod
    def forward(ctx, a, axis, index):
        ctx.axis = int(axis)
        ctx.index = int(index)
        ctx.in_shape = a.shape
        real = torch.real(a).select(ctx.axis, ctx.index)
        imag = torch.imag(a).select(ctx.axis, ctx.index)
        return torch.complex(real, imag)

    @staticmethod
    def backward(ctx, grad_output):
        gr = torch.zeros(ctx.in_shape, dtype=grad_output.real.dtype, device=grad_output.device)
        gi = torch.zeros(ctx.in_shape, dtype=grad_output.real.dtype, device=grad_output.device)
        gr.select(ctx.axis, ctx.index).copy_(torch.real(grad_output))
        gi.select(ctx.axis, ctx.index).copy_(torch.imag(grad_output))
        return torch.complex(gr, gi), None, None


class _NpuAddFn(torch.autograd.Function):
    """复数相加的 NPU 自动微分安全包装（规避 aclnnAdd DT_COMPLEX64）。"""

    @staticmethod
    def forward(ctx, a, b):
        real = torch.real(a) + torch.real(b)
        imag = torch.imag(a) + torch.imag(b)
        return torch.complex(real, imag)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
```

`NPUBackend` 类内，`conj` override 之后（照 `transpose` override 的门控风格）：

```python
    def take(self, a, axis, index):
        if self._is_npu_complex(a):
            return _NpuTakeFn.apply(a, int(axis), int(index))
        return super().take(a, axis, index)

    def add(self, a, b):
        if self._is_npu_complex(a) or self._is_npu_complex(b):
            return _NpuAddFn.apply(a, b)
        return super().add(a, b)
```

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/backends/ -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/npu_backend.py tests/backends/test_take_add.py
git commit -m "feat(npu): take/add 的 autograd-safe 复数包装（_NpuTakeFn/_NpuAddFn）"
```

---

### Task 3: contract.py — cotengra 规划 + 切片执行 + `optimize`/`memory_limit`

**Files:**
- Modify: `pyproject.toml`（`chem` extra 之后加 `tn = ["cotengra"]`；`all`/`dev` 追加 `"cotengra"`）
- Modify: `aicir/simulator/contract.py`
- Test: `tests/simulator/test_contract_cotengra.py`（新建）

**Interfaces:**
- Consumes: Task 1/2 的 `backend.take`/`backend.add`。
- Produces: `contract(tensors, indices, open_indices, backend, *, optimize="auto", memory_limit=None)`。Task 4 透传。

- [ ] **Step 0: 安装 cotengra（本机当前未装）**

Run: `pip install cotengra` 然后 `python -c "import cotengra; print(cotengra.__version__)"`
Expected: 打印版本号。

- [ ] **Step 1: pyproject.toml 加 extra**

`[project.optional-dependencies]` 中 `chem = [...]` 之后：

```toml
# 张量网络收缩路径优化 + 切片（cotengra planner）
tn = ["cotengra"]
```

并在 `all`、`dev` 列表各追加 `"cotengra"`。

- [ ] **Step 2: 写失败测试**

`tests/simulator/test_contract_cotengra.py`：

```python
import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, cnot, hadamard, ry
from aicir.core.state import State
from aicir.simulator import tn_statevector


def _ref_state(circ):
    bk = NumpyBackend()
    return np.asarray(bk.to_numpy(
        State.zero_state(circ.n_qubits, bk).evolve(circ.unitary(backend=bk)).data
    )).reshape(-1)


def _bell3():
    c = Circuit(n_qubits=3)
    c.append(hadamard(0))
    c.append(cnot(1, [0]))
    c.append(cnot(2, [1]))
    c.append(ry(0.7, 2))
    return c


def test_cotengra_path_parity():
    pytest.importorskip("cotengra")
    got = np.asarray(NumpyBackend().to_numpy(
        tn_statevector(_bell3(), optimize="cotengra").data
    )).reshape(-1)
    assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_sliced_contraction_parity_and_actually_slices():
    pytest.importorskip("cotengra")
    from aicir.simulator.contract import _cotengra_path
    from aicir.simulator.network import build_network

    bk = NumpyBackend()
    circ = _bell3()
    tensors, indices, open_idx = build_network(circ, bk, output_spec=[None] * 3)
    # memory_limit=4：全态 8 元素必超预算 → 必然切片
    _path, sliced = _cotengra_path([tuple(t) for t in indices], open_idx, 4)
    assert len(sliced) >= 1

    got = np.asarray(bk.to_numpy(
        tn_statevector(circ, optimize="cotengra", memory_limit=4).data
    )).reshape(-1)
    assert np.allclose(got, _ref_state(circ), atol=1e-5)


def test_optimize_greedy_and_opt_einsum_still_work():
    for opt in ("greedy", "auto"):
        got = np.asarray(NumpyBackend().to_numpy(
            tn_statevector(_bell3(), optimize=opt).data
        )).reshape(-1)
        assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_bad_optimize_raises():
    with pytest.raises(ValueError, match="optimize"):
        tn_statevector(_bell3(), optimize="nope")


def test_memory_limit_with_greedy_raises():
    with pytest.raises(ValueError, match="memory_limit"):
        tn_statevector(_bell3(), optimize="greedy", memory_limit=4)
```

（`test_bad_optimize_raises` 等两条依赖 Task 4 的 API 透传；本任务先让 contract 层测试通过——把这两条与 `test_cotengra_path_parity` 等先写成直接调 `contract(...)` 的版本亦可，但为免重复，直接按上面写，Task 3 结束时跑 `-k "not statevector"` 子集，Task 4 后全绿。更简单的处理：本任务先在测试文件顶部 `pytest.skip` 掉依赖公共 API 的用例？不要——**本任务把 Task 4 的 API 透传一并做掉太宽。折衷：本任务的测试直接用 `contract` 层接口**，改写如下，Task 4 再加公共 API 层测试。）

**用这份最终版**（替换上面草稿；全部走 contract 层，不依赖 Task 4）：

```python
import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, cnot, hadamard, ry
from aicir.core.state import State
from aicir.simulator.contract import contract
from aicir.simulator.network import build_network


def _ref_state(circ):
    bk = NumpyBackend()
    return np.asarray(bk.to_numpy(
        State.zero_state(circ.n_qubits, bk).evolve(circ.unitary(backend=bk)).data
    )).reshape(-1)


def _bell3():
    c = Circuit(n_qubits=3)
    c.append(hadamard(0))
    c.append(cnot(1, [0]))
    c.append(cnot(2, [1]))
    c.append(ry(0.7, 2))
    return c


def _net(circ, bk):
    tensors, indices, open_idx = build_network(circ, bk, output_spec=[None] * circ.n_qubits)
    return tensors, [tuple(t) for t in indices], open_idx


def test_cotengra_path_parity():
    pytest.importorskip("cotengra")
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    got = np.asarray(bk.to_numpy(
        contract(tensors, idx, open_idx, bk, optimize="cotengra")
    )).reshape(-1)
    assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_sliced_contraction_parity_and_actually_slices():
    pytest.importorskip("cotengra")
    from aicir.simulator.contract import _cotengra_path

    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    _path, sliced = _cotengra_path(idx, open_idx, 4)
    assert len(sliced) >= 1  # 防"没切到片的假通过"

    got = np.asarray(bk.to_numpy(
        contract(tensors, idx, open_idx, bk, optimize="cotengra", memory_limit=4)
    )).reshape(-1)
    assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_greedy_and_auto_parity():
    bk = NumpyBackend()
    for opt in ("greedy", "auto"):
        tensors, idx, open_idx = _net(_bell3(), bk)
        got = np.asarray(bk.to_numpy(
            contract(tensors, idx, open_idx, bk, optimize=opt)
        )).reshape(-1)
        assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_bad_optimize_raises():
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    with pytest.raises(ValueError, match="optimize"):
        contract(tensors, idx, open_idx, bk, optimize="nope")


def test_memory_limit_with_greedy_raises():
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    with pytest.raises(ValueError, match="memory_limit"):
        contract(tensors, idx, open_idx, bk, optimize="greedy", memory_limit=4)
```

- [ ] **Step 3: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_contract_cotengra.py -v`
Expected: FAIL（`TypeError: contract() got an unexpected keyword argument 'optimize'`）

- [ ] **Step 4: 实现 contract.py**

保留 `_pair_contract`/`_greedy_path`/`_opt_einsum_path` 不动。新增/改写：

```python
import itertools

_COTENGRA_HINT = '需要 cotengra；请安装可选依赖：pip install -e ".[tn]"'
_AUTO_COTENGRA_MIN_TENSORS = 24  # auto 下小网络不值得 HyperOptimizer 的搜索开销


def _cotengra_path(indices, open_indices, memory_limit):
    """cotengra 规划：返回 (path, sliced_labels)。仅规划，不执行。"""
    import cotengra as ctg

    ids = sorted({x for tup in indices for x in tup})
    sym = {x: f"i{k}" for k, x in enumerate(ids)}
    inputs = [tuple(sym[x] for x in tup) for tup in indices]
    output = tuple(sym[x] for x in open_indices)
    size_dict = {s: 2 for s in sym.values()}
    kwargs = {"progbar": False, "parallel": False}
    if memory_limit is not None:
        kwargs["slicing_opts"] = {"target_size": int(memory_limit)}
    opt = ctg.HyperOptimizer(**kwargs)
    tree = opt.search(inputs, output, size_dict)
    rev = {v: k for k, v in sym.items()}
    sliced = tuple(rev[s] for s in tree.sliced_inds)
    return list(tree.get_path()), sliced
```

（实现时先 `python -c` 探测所装版本的 `HyperOptimizer` 参数与 `tree.get_path()` 返回格式——须为 opt_einsum 兼容的线性 (i, j) 格式（cotengra 另有 `get_ssa_path()`，说明 `get_path()` 即线性格式）；若参数名有出入按实际 API 调整，意图不变。）

```python
def _plan(indices, open_indices, optimize, memory_limit):
    if optimize not in ("auto", "cotengra", "opt_einsum", "greedy"):
        raise ValueError(f"未知 optimize: {optimize!r}（可选 auto/cotengra/opt_einsum/greedy）")
    if memory_limit is not None and optimize in ("opt_einsum", "greedy"):
        raise ValueError("memory_limit（切片）只有 cotengra 规划器支持，不能与 optimize="
                         f"{optimize!r} 组合")
    if optimize == "cotengra" or memory_limit is not None:
        try:
            return _cotengra_path(indices, open_indices, memory_limit)
        except ImportError:
            raise ImportError(_COTENGRA_HINT)
    if optimize == "auto":
        if len(indices) >= _AUTO_COTENGRA_MIN_TENSORS:
            try:
                return _cotengra_path(indices, open_indices, None)
            except ImportError:
                pass
        try:
            return _opt_einsum_path(indices, open_indices), ()
        except ImportError:
            return _greedy_path(indices), ()
    if optimize == "opt_einsum":
        return _opt_einsum_path(indices, open_indices), ()  # 缺失时 ImportError 原样抛出
    return _greedy_path(indices), ()


def _execute(tens, idx, path, backend):
    """按 path 逐对收缩（原 contract 主循环提取）。"""
    for step in path:
        if len(step) < 2:
            continue  # 单张量网络的 (0,) 无操作步
        i, j = step[0], step[1]
        t, ids = _pair_contract(tens[i], idx[i], tens[j], idx[j], backend)
        for k in sorted((i, j), reverse=True):
            tens.pop(k)
            idx.pop(k)
        tens.append(t)
        idx.append(ids)
    return tens[0], idx[0]


def contract(tensors, indices, open_indices, backend, *, optimize="auto", memory_limit=None):
    idx = [tuple(t) for t in indices]
    path, sliced = _plan(idx, open_indices, optimize, memory_limit)

    if not sliced:
        result, ids = _execute(list(tensors), list(idx), path, backend)
    else:
        assert not (set(sliced) & set(open_indices)), "切片指标不得为 open index"
        result = ids = None
        for assign in itertools.product((0, 1), repeat=len(sliced)):
            ts, ix = [], []
            for t, tup in zip(tensors, idx):
                for lab, val in zip(sliced, assign):
                    if lab in tup:
                        t = backend.take(t, tup.index(lab), val)
                        tup = tuple(x for x in tup if x != lab)
                ts.append(t)
                ix.append(tup)
            r, ids = _execute(ts, ix, path, backend)
            result = r if result is None else backend.add(result, r)

    if open_indices:
        perm = [ids.index(x) for x in open_indices]
        if perm != list(range(len(perm))):
            result = backend.transpose(result, perm)
    return result
```

旧的 `_contraction_path` 删除（被 `_plan` 取代，无其他调用方——实现前 grep 确认）。

- [ ] **Step 5: 跑测试确认通过 + simulator 回归**

Run: `PYTHONPATH=. pytest tests/simulator tests/backends -q`
Expected: 全部 PASS（含既有 simulator 测试——`contract` 新参数全部有默认值，行为兼容）

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml aicir/simulator/contract.py tests/simulator/test_contract_cotengra.py
git commit -m "feat(simulator): cotengra 路径规划 + 切片执行（planner-only，执行走 backend.tensordot）"
```

---

### Task 4: 公共 API 透传 + 切片 autograd 测试

**Files:**
- Modify: `aicir/simulator/__init__.py`（四个公共函数 + `_statevector_tensor`）
- Test: `tests/simulator/test_api_cotengra.py`（新建）

**Interfaces:**
- Consumes: Task 3 的 `contract(..., optimize=, memory_limit=)`。
- Produces: `tn_statevector(circuit, *, backend=None, optimize="auto", memory_limit=None)`；`single_amplitude(circuit, bitstring, *, backend=None, optimize="auto", memory_limit=None)`；`partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None, optimize="auto", memory_limit=None)`；`tn_expectation(circuit, observable, *, backend=None, optimize="auto", memory_limit=None)`。

- [ ] **Step 1: 写失败测试**

`tests/simulator/test_api_cotengra.py`：

```python
import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, NumpyBackend, cnot, hadamard, ry
from aicir.simulator import partial_amplitude, single_amplitude, tn_expectation, tn_statevector


def _circ():
    c = Circuit(n_qubits=3)
    c.append(hadamard(0))
    c.append(cnot(1, [0]))
    c.append(ry(0.3, 2))
    return c


def test_api_kwargs_accepted_and_parity():
    pytest.importorskip("cotengra")
    base = np.asarray(NumpyBackend().to_numpy(tn_statevector(_circ()).data)).reshape(-1)
    sliced = np.asarray(NumpyBackend().to_numpy(
        tn_statevector(_circ(), optimize="cotengra", memory_limit=4).data
    )).reshape(-1)
    assert np.allclose(base, sliced, atol=1e-5)

    a0 = single_amplitude(_circ(), "000")
    a1 = single_amplitude(_circ(), "000", optimize="cotengra", memory_limit=4)
    assert abs(a0 - a1) < 1e-5

    p0 = partial_amplitude(_circ(), open_qubits=[0, 1])
    p1 = partial_amplitude(_circ(), open_qubits=[0, 1], optimize="cotengra", memory_limit=4)
    assert np.allclose(p0, p1, atol=1e-5)

    h = Hamiltonian([("ZII", 1.0)])
    e0 = complex(np.asarray(NumpyBackend().to_numpy(tn_expectation(_circ(), h))).reshape(()))
    e1 = complex(np.asarray(NumpyBackend().to_numpy(
        tn_expectation(_circ(), h, optimize="cotengra", memory_limit=4)
    )).reshape(()))
    assert abs(e0 - e1) < 1e-5


def test_sliced_expectation_autograd_matches_psr():
    pytest.importorskip("cotengra")
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    h = Hamiltonian([("ZII", 1.0)])
    theta = torch.tensor(0.4, requires_grad=True)

    c = Circuit(n_qubits=3)
    c.append(hadamard(1))
    c.append(cnot(2, [1]))
    c.append(ry(theta, 0))

    e = tn_expectation(c, h, backend=bk, optimize="cotengra", memory_limit=4)
    torch.real(e).backward()
    got = float(theta.grad)

    # <Z0> = cos(theta)（q0 与其余比特无纠缠），解析导数 -sin(theta)
    assert abs(got - (-np.sin(0.4))) < 1e-4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_api_cotengra.py -v`
Expected: FAIL（`TypeError: tn_statevector() got an unexpected keyword argument 'optimize'`）

- [ ] **Step 3: 实现透传**

`aicir/simulator/__init__.py`——`_statevector_tensor` 与四个公共函数签名各加 `optimize="auto"`, `memory_limit=None`，把两参传给每处 `contract(...)` 调用；`tn_statevector`/`tn_expectation` 经 `_statevector_tensor` 转发；`partial_amplitude` 的 `bitstrings` 分支把两参转发给 `single_amplitude`。docstring 各加一行参数说明（中文）。不改任何其他逻辑。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `PYTHONPATH=. pytest tests/simulator -q && PYTHONPATH=. pytest -q`
Expected: 全部 PASS；全量基线 1284 passed + 新增测试, 2 skipped

- [ ] **Step 5: 提交**

```bash
git add aicir/simulator/__init__.py tests/simulator/test_api_cotengra.py
git commit -m "feat(simulator): 四个公共 API 透传 optimize/memory_limit"
```

---

### Task 5: 文档 + NPU demo 切片检查

**Files:**
- Modify: `aicir/simulator/README.md`（§4 重写）
- Modify: `CHANGELOG.md`（2026-07-03 节追加条目）
- Modify: `compare.md`（cotengra 差距条目改写：§3 能力矩阵、§4、§19、§20）
- Modify: `demos/demo_npu_tensor.py`（追加切片检查节）

**Interfaces:**
- Consumes: Task 3/4 的最终 API。

- [ ] **Step 1: 重写 simulator README §4**

「收缩路径与可选依赖」改为三级来源 + 切片语义：

- `optimize="auto"`（默认）：网络较大且装了 cotengra → `HyperOptimizer`；否则 `opt_einsum`（已装）→ 内置贪心。
- `optimize="cotengra"|"opt_einsum"|"greedy"` 强制指定，对应库缺失抛 `ImportError`。
- `memory_limit=`（最大中间张量元素数）：设定后由 cotengra 规划切片；执行侧对每个切片赋值固定指标、串行收缩、结果累加——**执行仍走 `backend.tensordot`，torch/NPU 上可微性保留**（求和保梯度）。需 `pip install -e ".[tn]"`。
- 切片与 `optimize="opt_einsum"/"greedy"` 组合抛 `ValueError`。

- [ ] **Step 2: CHANGELOG 条目（追加到既有 2026-07-03 节的 Added / 无则新建 Added 小节）**

```markdown
- **`aicir.simulator` 接入 cotengra（新 `tn` extra）**：`optimize="auto"|"cotengra"|"opt_einsum"|"greedy"`
  选择收缩路径来源；`memory_limit=` 设定中间张量内存预算后由 cotengra 规划切片，执行侧逐切片
  固定指标（新 backend 原语 `take`）、同一 pairwise `tensordot` 收缩、`add` 累加——NPU 复数
  分解与 torch autograd 全程保留（`_NpuTakeFn`/`_NpuAddFn` 规避 aclnnAdd DT_COMPLEX64）。
  四个公共函数（`tn_statevector`/`single_amplitude`/`partial_amplitude`/`tn_expectation`）
  透传两参数。
```

- [ ] **Step 3: compare.md 改写**

- §3 能力矩阵「张量网络模拟」行：aicir 侧加「cotengra 路径 + 切片（可选 `tn` extra），且在 NPU 上保持可微」。
- §4 aicir 小节与结论：删去「不如 cotengra 精细」表述，改为两者同样可用 cotengra；aicir 独有差异保留（NPU + 可微 + 切片执行仍走后端抽象）。
- §19「仅 WuYueSDK 具备」中删除「cotengra 级张量网络路径优化」条目；§20 选型表「超大规模低纠缠线路」行改为「均可（两者都用 cotengra 规划；aicir 另支持切片下保持可微/NPU）」。

- [ ] **Step 4: demo_npu_tensor.py 追加切片检查**

`run_checks` 内、期望值梯度检查之后追加（cotengra 缺失时打印 skip 不算失败）：

```python
    # 检查 4：切片收缩 parity（cotengra 装了才跑）
    try:
        import cotengra  # noqa: F401
        sliced = tn_statevector(circ, backend=bk, optimize="cotengra", memory_limit=4)
        ok4 = _allclose(bk, sliced.data, full.data)
        print(f"[check ] sliced contraction: {'OK' if ok4 else 'MISMATCH'}")
        all_ok = all_ok and ok4
    except ImportError:
        print("[check ] sliced contraction: SKIP (cotengra 未安装)")
```

（变量名 `circ`/`bk`/`full`/`_allclose`/`all_ok` 以该文件实际命名为准，实现时读文件对齐；意图：切片态 vs 全收缩态一致即过。）

- [ ] **Step 5: 全量回归 + demo CPU 冒烟**

Run: `PYTHONPATH=. pytest -q && python demos/demo_npu_tensor.py --allow-cpu-fallback`
Expected: 测试全绿；demo `ALL PASSED`（含新检查 OK 或 SKIP）

- [ ] **Step 6: 提交**

```bash
git add aicir/simulator/README.md CHANGELOG.md compare.md demos/demo_npu_tensor.py
git commit -m "docs+demo: cotengra 路径/切片文档、compare.md 差距更新、NPU demo 切片检查"
```

---

## Self-Review

**Spec coverage:** 依赖打包（Task 3 Step 1）；组件 1 take/add（Task 1/2，NPU 包装 Task 2）；组件 2 contract 扩展含错误表 4 情形（Task 3；unknown optimize ✅、memory_limit+greedy ✅、cotengra 缺失 ImportError 在 `_plan` ✅、open-index 断言 ✅）；组件 3 API 透传（Task 4）；测试 1-6（原语 Task 1/2、cotengra parity Task 3、强制切片+防假通过 Task 3、切片 autograd Task 4、guards Task 3、NPU demo Task 5）；文档三处 + compare.md（Task 5）。缺口：spec 错误表中「memory_limit 设定但 cotengra 缺失 → ImportError」无法在装了 cotengra 的本机直接测——与既有 `test_pipeline_guard` 同策略，可在 Task 3 测试中加 monkeypatch 隐藏 cotengra 的用例；留给实现者作为可选加强，不设为必须（guard 逻辑路径已由 `_plan` 代码结构覆盖）。

**Deviation:** Global Constraints 中已显式标注 auto 阈值（≥24 张量）偏离 spec 的「auto 即 cotengra 优先」，需执行前人确认。

**Placeholder scan:** Task 3 Step 2 草稿段落已用「最终版」替换标注；Task 5 Step 4 的变量名对齐说明是读文件指令非占位。无 TBD/TODO。

**Type consistency:** `_cotengra_path(indices, open_indices, memory_limit) -> (path, sliced)`：Task 3 定义与测试一致；`contract`/四个 API 的 kwarg 名 `optimize`/`memory_limit` 全程一致；`take(a, axis, index)` 参数序 Task 1 定义 = Task 2 override = Task 3 调用（`backend.take(t, tup.index(lab), val)`）一致。

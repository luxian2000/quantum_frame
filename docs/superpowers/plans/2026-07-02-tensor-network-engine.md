# 精确张量网络模拟引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 aicir 新增精确张量网络模拟引擎，交付张量网络 / 单振幅 / 部分振幅三种模拟方法，真正在 NPUBackend 上执行且期望值可微。

**Architecture:** 在 `Backend` 抽象上新增收缩原语（`tensordot/transpose/reshape/conj`），NPU 的 `tensordot` 复用既有 autograd-safe 复数 `matmul`（permute→reshape→matmul→reshape），其余用 real/imag 分解。`aicir/simulator/` 把 `Circuit` 构建成带整数标签的张量网络，用 opt_einsum（可选）或内置贪心求收缩路径，逐对经 `backend.tensordot` 执行。单/部分振幅是同一构建器的不同边界条件。

**Tech Stack:** Python, numpy（唯一硬依赖），torch（可选，GPU/NPU），opt_einsum（可选，收缩路径）。

## Global Constraints

- 从仓库根运行，`PYTHONPATH=.`；测试 `PYTHONPATH=. pytest`。
- numpy 是唯一硬依赖；torch / opt_einsum 全部可选，测试用 `pytest.importorskip(...)` 跳过。
- 注释/文档/README 用中文，匹配周边风格。
- 比特序默认 msb（qubit 0 为最高有效位），比特索引与既有态矢量引擎完全一致——正确性以对齐既有 `State`/`Measure` 的测试强制。
- TN 引擎仅纯态、无噪声；含噪/密度矩阵路径不在范围。
- 既有 `State`/`Measure`/`Circuit` 默认行为与签名不变（纯新增）。
- 每个任务结束提交一次（frequent commits）。

## 局部矩阵约定（所有任务共享，务必遵守）

- 门局部矩阵 `M`（`2^k × 2^k`）作用于比特轴序 `axes = [a0, ..., a_{k-1}]`。
- 局部基下标 `local_index` 的比特映射：`a0` 是 `local_index` 的**最高位**（`bit = (local_index >> (k-1-pos)) & 1`）。
- 全局态矢量下标 `i` 中，qubit `q` 的比特在位置 `n-1-q`（qubit 0 为 MSB）。
- 因此把 `M` reshape 成 `(2,)*k + (2,)*k`：前 `k` 轴为**输出腿**（对应 `a0..a_{k-1}`），后 `k` 轴为**输入腿**（同序）。
- 输出边界按 qubit 升序排列开放腿并 C-order 展平 → 与既有态矢量分量一一对应（qubit 0 为 MSB）。

## 文件结构

- 修改 `aicir/backends/base.py` — 新增 4 个收缩原语（默认抛 NotImplementedError）。
- 修改 `aicir/backends/numpy_backend.py` / `gpu_backend.py` / `npu_backend.py` — 实现原语。
- 新增 `aicir/backends/_contract.py` — `tensordot_via_matmul` 设备无关辅助（NPU 用）。
- 修改 `aicir/core/gates.py` — 新增公共 `gate_tensors(gate, backend)`。
- 新增 `aicir/simulator/__init__.py` — 公共入口 `tn_statevector/single_amplitude/partial_amplitude/tn_expectation`。
- 新增 `aicir/simulator/network.py` — `build_network(...)`。
- 新增 `aicir/simulator/contract.py` — `contract(...)` + 路径。
- 修改 `aicir/measure/measure.py` — `Measure.run(..., method=...)`。
- 修改 `aicir/__init__.py` — 顶层再导出。
- 新增 `aicir/simulator/README.md`、更新 `CHANGELOG.md`。
- 新增 `demos/demo_npu_tensor.py`、`tests/simulator/test_*.py`。

---

### Task 1: Backend 收缩原语（base + numpy）

**Files:**
- Modify: `aicir/backends/base.py`
- Modify: `aicir/backends/numpy_backend.py`
- Test: `tests/backends/test_contract_primitives.py`

**Interfaces:**
- Produces: `Backend.tensordot(a, b, axes)`、`Backend.transpose(a, axes)`、`Backend.reshape(a, shape)`、`Backend.conj(a)`。`axes` 为 `(list_a, list_b)`；`tensordot` 结果轴序 = a 的自由轴（原序）后接 b 的自由轴（原序），与 `numpy.tensordot` 一致。

- [ ] **Step 1: 写失败测试**

```python
# tests/backends/test_contract_primitives.py
import numpy as np
from aicir.backends import NumpyBackend


def test_numpy_tensordot_matches_numpy():
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_numpy_transpose_reshape_conj():
    bk = NumpyBackend()
    a = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4)
    ca = bk.cast(a)
    assert np.allclose(bk.to_numpy(bk.transpose(ca, (2, 0, 1))), np.transpose(a, (2, 0, 1)), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.reshape(ca, (6, 4))), a.reshape(6, 4), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.conj(ca)), np.conj(a), atol=1e-5)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_contract_primitives.py -q`
Expected: FAIL（`AttributeError: 'NumpyBackend' object has no attribute 'tensordot'`）

- [ ] **Step 3: base.py 加默认原语**

在 `aicir/backends/base.py` 的 `Backend` 类内（`tensor_product` 之前）加入：

```python
    # ──────────────────────── 张量网络收缩原语 ──────────────────────
    def tensordot(self, a, b, axes):
        """沿 axes=(list_a, list_b) 收缩两张量；轴序同 numpy.tensordot。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 tensordot")

    def transpose(self, a, axes):
        """按 axes 置换张量轴。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 transpose")

    def reshape(self, a, shape):
        """把张量变形为 shape。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 reshape")

    def conj(self, a):
        """逐元素复共轭。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 conj")
```

- [ ] **Step 4: numpy_backend.py 实现**

在 `aicir/backends/numpy_backend.py` 的 `real` 方法后加入：

```python
    def tensordot(self, a, b, axes):
        out = np.tensordot(np.asarray(a), np.asarray(b), axes=axes)
        return out.astype(self._dtype)

    def transpose(self, a, axes):
        return np.transpose(np.asarray(a), axes)

    def reshape(self, a, shape):
        return np.reshape(np.asarray(a), tuple(shape))

    def conj(self, a):
        return np.conj(np.asarray(a))
```

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_contract_primitives.py -q`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add aicir/backends/base.py aicir/backends/numpy_backend.py tests/backends/test_contract_primitives.py
git commit -m "feat(backend): 新增张量收缩原语 tensordot/transpose/reshape/conj (numpy)"
```

---

### Task 2: 收缩原语（gpu + npu）+ 设备无关 matmul-tensordot

**Files:**
- Create: `aicir/backends/_contract.py`
- Modify: `aicir/backends/gpu_backend.py`
- Modify: `aicir/backends/npu_backend.py`
- Test: `tests/backends/test_contract_primitives.py`（追加）

**Interfaces:**
- Consumes: Task 1 的原语契约。
- Produces: `aicir.backends._contract.tensordot_via_matmul(backend, a, b, axes)` — 仅用 `backend.transpose/reshape/matmul` 实现 tensordot，设备无关；NPUBackend.tensordot 在复数时走此路径（复用 autograd-safe matmul）。

- [ ] **Step 1: 写失败测试（追加到同文件）**

```python
def test_tensordot_via_matmul_matches_numpy():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_tensordot_via_matmul_outer_product():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.array([1.0, 2.0], dtype=np.complex64)
    b = np.array([3.0, 4.0, 5.0], dtype=np.complex64)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([], []))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([], [])), atol=1e-5)


def test_gpu_tensordot_matches_numpy():
    import pytest
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-4)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_contract_primitives.py -q`
Expected: FAIL（`ModuleNotFoundError: aicir.backends._contract`）

- [ ] **Step 3: 创建 `aicir/backends/_contract.py`**

```python
"""设备无关的 tensordot：仅用 backend.transpose/reshape/matmul 实现。

NPUBackend 复数张量走此路径，复用其 autograd-safe 复数 matmul，从而收缩全程
无跨节点 complex64 累加，且保留自动微分图。"""

from __future__ import annotations

import math


def _prod(dims) -> int:
    out = 1
    for d in dims:
        out *= int(d)
    return out


def tensordot_via_matmul(backend, a, b, axes):
    ax_a = [int(x) for x in axes[0]]
    ax_b = [int(x) for x in axes[1]]
    nda, ndb = len(a.shape), len(b.shape)
    ax_a = [x % nda for x in ax_a]
    ax_b = [x % ndb for x in ax_b]
    free_a = [i for i in range(nda) if i not in ax_a]
    free_b = [i for i in range(ndb) if i not in ax_b]

    a2 = backend.transpose(a, free_a + ax_a)
    b2 = backend.transpose(b, ax_b + free_b)

    fa = [int(a.shape[i]) for i in free_a]
    ca = [int(a.shape[i]) for i in ax_a]
    fb = [int(b.shape[i]) for i in free_b]

    a2 = backend.reshape(a2, (_prod(fa), _prod(ca)))
    b2 = backend.reshape(b2, (_prod(ca), _prod(fb)))
    out = backend.matmul(a2, b2)
    return backend.reshape(out, tuple(fa) + tuple(fb))
```

- [ ] **Step 4: gpu_backend.py 实现原语**

在 `aicir/backends/gpu_backend.py` 的 `real` 方法后加入：

```python
    def tensordot(self, a, b, axes):
        return torch.tensordot(a, b, dims=(list(axes[0]), list(axes[1])))

    def transpose(self, a, axes):
        return a.permute(*[int(x) for x in axes])

    def reshape(self, a, shape):
        return a.reshape(tuple(int(s) for s in shape))

    def conj(self, a):
        return torch.conj(a)
```

（确认文件顶部已 `import torch`；GPUBackend 已有 torch 导入。）

- [ ] **Step 5: npu_backend.py 覆写原语（real/imag 安全）**

在 `aicir/backends/npu_backend.py` 的 `NPUBackend` 类内（`kron` 方法后）加入：

```python
    def tensordot(self, a, b, axes):
        if self._is_npu_complex(a) or self._is_npu_complex(b):
            from ._contract import tensordot_via_matmul
            return tensordot_via_matmul(self, a, b, axes)
        return super().tensordot(a, b, axes)

    def transpose(self, a, axes):
        if self._is_npu_complex(a):
            perm = [int(x) for x in axes]
            return torch.complex(torch.real(a).permute(*perm), torch.imag(a).permute(*perm))
        return super().transpose(a, axes)

    def reshape(self, a, shape):
        shape = tuple(int(s) for s in shape)
        if self._is_npu_complex(a):
            return torch.complex(torch.real(a).reshape(shape), torch.imag(a).reshape(shape))
        return super().reshape(a, shape)

    def conj(self, a):
        if self._is_npu_complex(a):
            return torch.complex(torch.real(a), -torch.imag(a))
        return super().conj(a)
```

- [ ] **Step 6: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_contract_primitives.py -q`
Expected: PASS（全部 passed；无 torch 时 gpu 用例 skip）

- [ ] **Step 7: 提交**

```bash
git add aicir/backends/_contract.py aicir/backends/gpu_backend.py aicir/backends/npu_backend.py tests/backends/test_contract_primitives.py
git commit -m "feat(backend): gpu/npu 收缩原语 + 设备无关 tensordot_via_matmul"
```

---

### Task 3: `gate_tensors` 门→张量转换（gates.py）

**Files:**
- Modify: `aicir/core/gates.py`
- Test: `tests/simulator/test_gate_tensors.py`

**Interfaces:**
- Consumes: 既有 `_gate_local_matrix`、`_multi_target_subgates`、`_unitary_parameter_matrix`、`_local_target_qubits`、`normalize_gate`、`canonical_gate_name`。
- Produces: `aicir.core.gates.gate_tensors(gate, backend) -> list[tuple[matrix, axes]]`。返回门在其作用比特上的 `2^k×2^k` 后端矩阵与比特轴序列表（多目标受控门展开为多项；`identity`/`measure` 返回 `[]`）。无法转张量的门抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_gate_tensors.py
import numpy as np
from aicir import NumpyBackend, cnot, ry
from aicir.core.gates import gate_tensors


def test_single_qubit_gate_tensor():
    bk = NumpyBackend()
    (matrix, axes), = gate_tensors(ry(0.5, 0), bk)
    assert axes == [0]
    expected = np.array([[np.cos(0.25), -np.sin(0.25)], [np.sin(0.25), np.cos(0.25)]])
    assert np.allclose(bk.to_numpy(matrix), expected, atol=1e-6)


def test_cnot_gate_tensor_axes_and_shape():
    bk = NumpyBackend()
    tensors = gate_tensors(cnot(1, [0]), bk)
    assert len(tensors) == 1
    matrix, axes = tensors[0]
    # 控制在前、目标在后：axes = controls + targets
    assert axes == [0, 1]
    assert bk.to_numpy(matrix).shape == (4, 4)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_gate_tensors.py -q`
Expected: FAIL（`ImportError: cannot import name 'gate_tensors'`）

- [ ] **Step 3: 在 gates.py 末尾（`gate_to_matrix` 之后）加入**

```python
def gate_tensors(gate, backend):
    """把门转为张量列表 ``[(matrix, axes), ...]``（TN 构建单一来源）。

    - 多目标受控门（``cx([t1,t2],[c])`` 等）展开为逐目标项；
    - ``identity``/``measure`` 返回 ``[]``（不产生节点）；
    - ``unitary`` 自定义门用其参数矩阵；其余走 ``_gate_local_matrix``；
    - 无法局部展开的门（未注册/含未绑定参数）抛 ``ValueError``。
    ``matrix`` 为 ``2^k×2^k`` 后端张量，``axes`` 为其作用比特轴序（controls+targets）。
    """
    gate = normalize_gate(gate)
    subgates = _multi_target_subgates(gate)
    if subgates is not None:
        result = []
        for subgate in subgates:
            result.extend(gate_tensors(subgate, backend))
        return result

    gate_type = canonical_gate_name(gate["type"])
    if gate_type in ("identity", "measure"):
        return []

    if gate_type == "unitary":
        matrix = _unitary_parameter_matrix(gate.get("parameter"), backend)
        return [(matrix, _local_target_qubits(gate))]

    local, axes, _ = _gate_local_matrix(gate, gate_type, backend)
    if local is None:
        raise ValueError(f"门 {gate_type!r} 无法转为张量（未注册/含未绑定参数/measure）")
    return [(local, list(axes))]
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_gate_tensors.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/core/gates.py tests/simulator/test_gate_tensors.py
git commit -m "feat(gates): 新增 gate_tensors 门→张量转换（TN 构建单一来源）"
```

---

### Task 4: TN 构建器（network.py）

**Files:**
- Create: `aicir/simulator/network.py`
- Create: `aicir/simulator/__init__.py`（本任务仅占位，Task 6 填充公共 API）
- Test: `tests/simulator/test_network.py`

**Interfaces:**
- Consumes: `aicir.core.gates.gate_tensors`。
- Produces: `aicir.simulator.network.build_network(circuit, backend, *, input_bits=None, output_spec=None) -> (tensors, indices, open_indices)`。`tensors` 为后端张量列表；`indices` 为与之对齐的整数标签元组列表；`open_indices` 为按 qubit 升序的开放腿标签元组。`output_spec[q]`：`None`=开放，`0/1`=接 `⟨bit|`。`input_bits[q]` 默认 0。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_network.py
from aicir import Circuit, NumpyBackend, ry
from aicir.simulator.network import build_network


def test_build_network_statevector_shapes():
    bk = NumpyBackend()
    c = Circuit(ry(0.3, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(bk_circuit := c, bk, output_spec=[None])
    # 1 个输入 |0> 向量 + 1 个门张量；输出开放
    assert len(tensors) == len(indices) == 2
    assert len(open_idx) == 1
    # 每条腿维度均为 2
    for t, ids in zip(tensors, indices):
        assert tuple(t.shape) == (2,) * len(ids)


def test_build_network_single_amplitude_no_open():
    bk = NumpyBackend()
    c = Circuit(ry(0.3, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(c, bk, output_spec=[0])
    assert open_idx == ()
    # 输入向量 + 门 + 输出 bra 向量
    assert len(tensors) == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_network.py -q`
Expected: FAIL（`ModuleNotFoundError: aicir.simulator`）

- [ ] **Step 3: 创建空 `aicir/simulator/__init__.py`**

```python
"""aicir 精确张量网络模拟引擎。"""
```

- [ ] **Step 4: 创建 `aicir/simulator/network.py`**

```python
"""把 Circuit 构建成带整数标签的张量网络（见 plan「局部矩阵约定」）。"""

from __future__ import annotations

import numpy as np

from ..core.gates import gate_tensors


def _basis_vector(bit: int):
    return np.array([1.0, 0.0] if int(bit) == 0 else [0.0, 1.0], dtype=np.complex64)


def build_network(circuit, backend, *, input_bits=None, output_spec=None):
    n = int(circuit.n_qubits)
    if input_bits is None:
        input_bits = [0] * n
    if output_spec is None:
        output_spec = [None] * n
    if len(input_bits) != n or len(output_spec) != n:
        raise ValueError("input_bits / output_spec 长度必须等于 n_qubits")

    tensors, indices = [], []
    counter = 0

    def fresh():
        nonlocal counter
        counter += 1
        return counter

    # 输入边界：每比特一个 |bit> 向量
    wire = [0] * n
    for q in range(n):
        vid = fresh()
        tensors.append(backend.cast(_basis_vector(input_bits[q])))
        indices.append((vid,))
        wire[q] = vid

    # 门节点：matrix reshape 成 (2,)*k(out) + (2,)*k(in)
    for gate in circuit.gates:
        for matrix, axes in gate_tensors(gate, backend):
            k = len(axes)
            node = backend.reshape(backend.cast(matrix), (2,) * (2 * k))
            out_ids = [fresh() for _ in range(k)]
            in_ids = [wire[a] for a in axes]
            tensors.append(node)
            indices.append(tuple(out_ids) + tuple(in_ids))
            for j, a in enumerate(axes):
                wire[a] = out_ids[j]

    # 输出边界：None 开放；0/1 接 <bit|
    open_indices = []
    for q in range(n):
        spec = output_spec[q]
        if spec is None:
            open_indices.append(wire[q])
        else:
            tensors.append(backend.cast(_basis_vector(int(spec))))
            indices.append((wire[q],))

    return tensors, indices, tuple(open_indices)
```

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_network.py -q`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add aicir/simulator/__init__.py aicir/simulator/network.py tests/simulator/test_network.py
git commit -m "feat(simulator): TN 构建器 build_network（含输入/输出边界）"
```

---

### Task 5: 收缩器（contract.py）+ 路径

**Files:**
- Create: `aicir/simulator/contract.py`
- Test: `tests/simulator/test_contract.py`

**Interfaces:**
- Consumes: `build_network` 的 `(tensors, indices, open_indices)`；`Backend.tensordot/transpose`。
- Produces: `aicir.simulator.contract.contract(tensors, indices, open_indices, backend) -> 后端张量`（shape `(2,)*len(open_indices)`，`open_indices` 为空时为 0 维标量）。内部 `_greedy_path(indices)` 与 opt_einsum 路径二选一。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_contract.py
import numpy as np
from aicir import Circuit, NumpyBackend, ry
from aicir.simulator.network import build_network
from aicir.simulator.contract import contract, _greedy_path


def test_contract_single_ry_statevector():
    bk = NumpyBackend()
    c = Circuit(ry(0.7, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(c, bk, output_spec=[None])
    result = contract(tensors, indices, open_idx, bk)
    vec = bk.to_numpy(result).reshape(-1)
    assert np.allclose(vec, [np.cos(0.35), np.sin(0.35)], atol=1e-6)


def test_greedy_path_reduces_to_single_tensor():
    # 3 个张量链式共享标签 -> 2 步收缩
    indices = [(1,), (1, 2), (2,)]
    path = _greedy_path(indices)
    assert len(path) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_contract.py -q`
Expected: FAIL（`ModuleNotFoundError: aicir.simulator.contract`）

- [ ] **Step 3: 创建 `aicir/simulator/contract.py`**

```python
"""张量网络收缩：求路径（opt_einsum 可选 / 内置贪心），逐对经 backend.tensordot 执行。"""

from __future__ import annotations


def _pair_contract(t1, id1, t2, id2, backend):
    shared = [x for x in id1 if x in id2]
    a1 = [id1.index(x) for x in shared]
    a2 = [id2.index(x) for x in shared]
    out = backend.tensordot(t1, t2, (a1, a2))
    new_ids = tuple(x for x in id1 if x not in shared) + tuple(x for x in id2 if x not in shared)
    return out, new_ids


def _greedy_path(indices):
    """贪心成对路径：优先共享标签最多、结果秩最小的一对。

    返回 (i, j) 列表，位置指向**收缩中不断缩小**的列表：每步移除 i、j，把结果追加到末尾。
    """
    idx = [set(t) for t in indices]
    path = []
    while len(idx) > 1:
        best = None
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                shared = idx[i] & idx[j]
                result_rank = len(idx[i] ^ idx[j]) if shared else len(idx[i]) + len(idx[j])
                key = (-len(shared), result_rank)
                if best is None or key < best[0]:
                    best = (key, i, j)
        _, i, j = best
        merged = idx[i] ^ idx[j]
        for k in sorted((i, j), reverse=True):
            idx.pop(k)
        idx.append(merged)
        path.append((i, j))
    return path


def _opt_einsum_path(indices, open_indices):
    import opt_einsum

    ids = sorted({x for tup in indices for x in tup})
    sym = {x: opt_einsum.get_symbol(i) for i, x in enumerate(ids)}
    subs = ",".join("".join(sym[x] for x in tup) for tup in indices)
    out = "".join(sym[x] for x in open_indices)
    eq = f"{subs}->{out}"
    shapes = [tuple(2 for _ in tup) for tup in indices]
    path, _info = opt_einsum.contract_path(eq, *shapes, shapes=True, optimize="auto")
    return list(path)


def _contraction_path(indices, open_indices):
    try:
        return _opt_einsum_path(indices, open_indices)
    except ImportError:
        return _greedy_path(indices)


def contract(tensors, indices, open_indices, backend):
    tens = list(tensors)
    idx = [tuple(t) for t in indices]
    for step in _contraction_path(idx, open_indices):
        i, j = step[0], step[1]
        t, ids = _pair_contract(tens[i], idx[i], tens[j], idx[j], backend)
        for k in sorted((i, j), reverse=True):
            tens.pop(k)
            idx.pop(k)
        tens.append(t)
        idx.append(ids)

    result, ids = tens[0], idx[0]
    if open_indices:
        perm = [ids.index(x) for x in open_indices]
        if perm != list(range(len(perm))):
            result = backend.transpose(result, perm)
    return result
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_contract.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/simulator/contract.py tests/simulator/test_contract.py
git commit -m "feat(simulator): TN 收缩器 contract（opt_einsum 可选 + 贪心回退）"
```

---

### Task 6: 公共入口 + 顶层再导出 + 对齐既有引擎

**Files:**
- Modify: `aicir/simulator/__init__.py`
- Modify: `aicir/__init__.py`
- Test: `tests/simulator/test_engine_parity.py`

**Interfaces:**
- Consumes: `build_network`、`contract`；`State`；`Backend`。
- Produces（均顶层再导出）：
  - `tn_statevector(circuit, *, backend=None) -> State`
  - `single_amplitude(circuit, bitstring, *, backend=None) -> complex`
  - `partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None) -> np.ndarray`
  - 内部 `_statevector_tensor(circuit, backend)`（供 Task 7 复用，保梯度）。
  - `bitstring` 接受 `"0101"` 或可迭代 0/1，长度须等于 `n_qubits`（qubit 0 为最左/MSB）。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_engine_parity.py
import numpy as np
import pytest
from aicir import (Circuit, NumpyBackend, State, cnot, ry, rzz,
                   single_amplitude, partial_amplitude, tn_statevector)


def _ref_state(circuit, bk):
    return State.zero_state(circuit.n_qubits, bk).evolve(circuit.unitary(backend=bk)).to_numpy()


def _demo_circuit():
    return Circuit(ry(0.4, 0), cnot(1, [0]), ry(0.9, 1), rzz(0.3, 0, 1), n_qubits=2)


def test_tn_statevector_matches_reference():
    bk = NumpyBackend()
    c = _demo_circuit()
    assert np.allclose(tn_statevector(c, backend=bk).to_numpy(), _ref_state(c, bk), atol=1e-5)


def test_single_amplitude_matches_reference():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)
    for i, bits in enumerate(["00", "01", "10", "11"]):
        assert np.isclose(single_amplitude(c, bits, backend=bk), ref[i], atol=1e-5)


def test_partial_amplitude_open_qubits():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)  # qubit0 MSB: index = b0*2 + b1
    # 固定 qubit0=0，开放 qubit1 -> [<00|psi>, <01|psi>]
    vec = partial_amplitude(c, open_qubits=[1], backend=bk)
    assert np.allclose(vec, [ref[0], ref[1]], atol=1e-5)


def test_partial_amplitude_bitstrings():
    bk = NumpyBackend()
    c = _demo_circuit()
    ref = _ref_state(c, bk)
    vec = partial_amplitude(c, bitstrings=["11", "00"], backend=bk)
    assert np.allclose(vec, [ref[3], ref[0]], atol=1e-5)


def test_partial_amplitude_requires_exactly_one_mode():
    bk = NumpyBackend()
    c = _demo_circuit()
    with pytest.raises(ValueError):
        partial_amplitude(c, backend=bk)
    with pytest.raises(ValueError):
        partial_amplitude(c, open_qubits=[0], bitstrings=["00"], backend=bk)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_engine_parity.py -q`
Expected: FAIL（`ImportError: cannot import name 'tn_statevector' from 'aicir'`）

- [ ] **Step 3: 填充 `aicir/simulator/__init__.py`**

```python
"""aicir 精确张量网络模拟引擎：张量网络 / 单振幅 / 部分振幅。"""

from __future__ import annotations

import numpy as np

from ..core.state import State
from .contract import contract
from .network import build_network


def _resolve_backend(circuit, backend):
    if backend is not None:
        return backend
    bk = getattr(circuit, "backend", None)
    if bk is not None:
        return bk
    from ..backends import NumpyBackend
    return NumpyBackend()


def _parse_bits(bitstring, n):
    if isinstance(bitstring, str):
        bits = [int(ch) for ch in bitstring]
    else:
        bits = [int(b) for b in bitstring]
    if len(bits) != n or any(b not in (0, 1) for b in bits):
        raise ValueError(f"bitstring 必须为长度 {n} 的 0/1 串")
    return bits


def _statevector_tensor(circuit, backend):
    n = int(circuit.n_qubits)
    tensors, indices, open_idx = build_network(circuit, backend, output_spec=[None] * n)
    result = contract(tensors, indices, open_idx, backend)
    return backend.reshape(result, (1 << n, 1))


def tn_statevector(circuit, *, backend=None):
    """经张量网络收缩求整段末态，返回 State（向量形态）。"""
    backend = _resolve_backend(circuit, backend)
    data = _statevector_tensor(circuit, backend)
    return State(data, int(circuit.n_qubits), backend)


def single_amplitude(circuit, bitstring, *, backend=None):
    """求单个基态振幅 ⟨bitstring|U|0⟩（标量收缩，不构造全态矢量）。"""
    backend = _resolve_backend(circuit, backend)
    n = int(circuit.n_qubits)
    bits = _parse_bits(bitstring, n)
    tensors, indices, open_idx = build_network(circuit, backend, output_spec=bits)
    result = contract(tensors, indices, open_idx, backend)
    return complex(np.asarray(backend.to_numpy(result)).reshape(()))


def partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None):
    """求部分振幅。二选一：
    - open_qubits=[...]：这些比特开放、其余输出固定 |0>，返回按开放比特升序（首个为 MSB）
      排列的 2^len 振幅向量；
    - bitstrings=[...]：枚举给定基态，返回其振幅数组。
    """
    if (open_qubits is None) == (bitstrings is None):
        raise ValueError("open_qubits 与 bitstrings 必须且只能提供其一")
    backend = _resolve_backend(circuit, backend)
    n = int(circuit.n_qubits)
    if open_qubits is not None:
        openset = {int(q) for q in open_qubits}
        spec = [None if q in openset else 0 for q in range(n)]
        tensors, indices, open_idx = build_network(circuit, backend, output_spec=spec)
        result = contract(tensors, indices, open_idx, backend)
        return np.asarray(backend.to_numpy(result)).reshape(-1)
    return np.array([single_amplitude(circuit, b, backend=backend) for b in bitstrings])
```

- [ ] **Step 4: 顶层再导出（`aicir/__init__.py`）**

在 `aicir/__init__.py` 找到已有的子系统导入区（如 `from .measure import ...` 附近）加入：

```python
from .simulator import (
    partial_amplitude,
    single_amplitude,
    tn_statevector,
)
```

并把这三个名字加入该文件的 `__all__`（若存在）。

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_engine_parity.py -q`
Expected: PASS（6 passed）

- [ ] **Step 6: gpu 后端对齐（追加到同文件）**

```python
def test_tn_statevector_matches_reference_gpu():
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    c = _demo_circuit()
    assert np.allclose(tn_statevector(c, backend=bk).to_numpy(), _ref_state(c, bk), atol=1e-4)
```

Run: `PYTHONPATH=. pytest tests/simulator/test_engine_parity.py -q`
Expected: PASS（7 passed，无 torch 时最后一个 skip）

- [ ] **Step 7: 提交**

```bash
git add aicir/simulator/__init__.py aicir/__init__.py tests/simulator/test_engine_parity.py
git commit -m "feat(simulator): 公共入口 tn_statevector/single_amplitude/partial_amplitude + 对齐既有引擎"
```

---

### Task 7: 可微期望 `tn_expectation`

**Files:**
- Modify: `aicir/simulator/__init__.py`
- Modify: `aicir/__init__.py`
- Test: `tests/simulator/test_tn_autograd.py`

**Interfaces:**
- Consumes: `_statevector_tensor`；`Backend.expectation_sv`；`Hamiltonian.to_matrix`。
- Produces: `tn_expectation(circuit, observable, *, backend=None) -> 后端标量`（torch/NPU 后端上对参数门可微）。`observable` 为 `Hamiltonian` 或算符矩阵。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_tn_autograd.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir import Circuit, Hamiltonian, ry
from aicir.backends import GPUBackend
from aicir.qml import psr
from aicir.simulator import tn_expectation


def test_tn_expectation_value_matches_reference():
    bk = GPUBackend(device="cpu")
    c = Circuit(ry(0.6, 0), n_qubits=1)
    val = float(tn_expectation(c, Hamiltonian([("Z", 1.0)]), backend=bk))
    assert np.isclose(val, np.cos(0.6), atol=1e-5)


def test_tn_expectation_is_differentiable():
    bk = GPUBackend(device="cpu")
    theta = torch.tensor(0.6, dtype=torch.float32, requires_grad=True)

    def build(t):
        return Circuit(ry(t, 0), n_qubits=1)

    energy = tn_expectation(build(theta), Hamiltonian([("Z", 1.0)]), backend=bk)
    energy.backward()
    # 解析梯度 -sin(theta)，交叉核对 psr
    ref = float(psr(lambda p: float(tn_expectation(build(float(p[0])), Hamiltonian([("Z", 1.0)]), backend=bk)),
                    np.array([0.6]))[0])
    assert np.isclose(float(theta.grad), -np.sin(0.6), atol=1e-4)
    assert np.isclose(float(theta.grad), ref, atol=1e-4)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_tn_autograd.py -q`
Expected: FAIL（`ImportError: cannot import name 'tn_expectation'`）

- [ ] **Step 3: 在 `aicir/simulator/__init__.py` 末尾加入**

```python
def tn_expectation(circuit, observable, *, backend=None):
    """经张量网络收缩求期望值 ⟨ψ|O|ψ⟩。torch/NPU 后端上对参数门可微。"""
    backend = _resolve_backend(circuit, backend)
    psi = _statevector_tensor(circuit, backend)
    if hasattr(observable, "to_matrix"):
        operator = observable.to_matrix(backend)
    else:
        operator = backend.cast(observable)
    return backend.expectation_sv(psi, operator)
```

- [ ] **Step 4: 顶层再导出**

在 `aicir/__init__.py` 的 simulator 导入处补上 `tn_expectation`，并加入 `__all__`：

```python
from .simulator import (
    partial_amplitude,
    single_amplitude,
    tn_expectation,
    tn_statevector,
)
```

- [ ] **Step 5: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_tn_autograd.py -q`
Expected: PASS（2 passed；无 torch 时整文件 skip）

- [ ] **Step 6: 提交**

```bash
git add aicir/simulator/__init__.py aicir/__init__.py tests/simulator/test_tn_autograd.py
git commit -m "feat(simulator): 可微 tn_expectation（torch/NPU 后端）"
```

---

### Task 8: `Measure.run(method="tensor")` 集成

**Files:**
- Modify: `aicir/measure/measure.py`
- Test: `tests/simulator/test_measure_tensor.py`

**Interfaces:**
- Consumes: `tn_statevector`；既有 `Measure.run` 语义。
- Produces: `Measure.run(..., method="statevector"|"tensor")`。`method="tensor"` 经 TN 求末态后交既有概率/期望机制；含噪声或线路内嵌 measure 时报错。默认 `"statevector"`，既有行为不变。

- [ ] **Step 1: 写失败测试**

```python
# tests/simulator/test_measure_tensor.py
import numpy as np
import pytest
from aicir import Circuit, Hamiltonian, Measure, NumpyBackend, cnot, ry


def _circuit():
    return Circuit(ry(0.4, 0), cnot(1, [0]), ry(0.9, 1), n_qubits=2)


def test_measure_tensor_matches_statevector_probs():
    bk = NumpyBackend()
    c = _circuit()
    ref = Measure(bk).run(c, shots=None, method="statevector")
    tn = Measure(bk).run(c, shots=None, method="tensor")
    assert np.allclose(ref.state.probabilities(), tn.state.probabilities(), atol=1e-5)


def test_measure_tensor_expectation_matches():
    bk = NumpyBackend()
    c = _circuit()
    H = {"H": Hamiltonian([("ZI", 1.0)]).to_matrix(bk)}
    ref = Measure(bk).run(c, shots=None, observables=H, method="statevector")
    tn = Measure(bk).run(c, shots=None, observables=H, method="tensor")
    assert np.isclose(ref.expectation_values["H"], tn.expectation_values["H"], atol=1e-5)


def test_measure_tensor_rejects_bad_method():
    bk = NumpyBackend()
    with pytest.raises(ValueError):
        Measure(bk).run(_circuit(), shots=None, method="bogus")
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_measure_tensor.py -q`
Expected: FAIL（`TypeError: run() got an unexpected keyword argument 'method'`）

- [ ] **Step 3: 修改 `Measure.run` 签名与分支**

在 `aicir/measure/measure.py` 的 `run` 定义把签名末尾的 `return_state=True) -> Result:` 改为：

```python
            observables=None, return_state=True, method="statevector") -> Result:
```

在方法体开头（`n = int(circuit.n_qubits)` 与 `backend = self._resolve_backend(circuit)` 之后）插入：

```python
        if method not in ("statevector", "tensor"):
            raise ValueError(f"method 必须是 statevector/tensor，收到 {method!r}")
        if method == "tensor":
            if getattr(circuit, "noise_model", None) is not None:
                raise ValueError("method='tensor' 仅支持纯态，无法用于含噪线路")
            if any(_is_measure(g) for g in circuit_instructions(circuit)):
                raise ValueError("method='tensor' 不支持线路内嵌 measure 标记")
            from ..simulator import tn_statevector
            psi = tn_statevector(circuit, backend=backend)
            from ..core.circuit import Circuit as _Circuit
            stripped = _Circuit(n_qubits=n)
            return self.run(
                stripped, shots=shots, measure_qubits=measure_qubits, snap=snap,
                sm=sm, seed=seed, initial_state=psi, observables=observables,
                return_state=return_state, method="statevector",
            )
```

（`_is_measure` 与 `circuit_instructions` 已在该模块导入；若 `snap` 需求存在，`method='tensor'` 下因 stripped 电路无操作而不生效——在 README 注明。）

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_measure_tensor.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 回归既有测量测试**

Run: `PYTHONPATH=. pytest tests/ -q -k "measure"`
Expected: PASS（既有测量用例不回归）

- [ ] **Step 6: 提交**

```bash
git add aicir/measure/measure.py tests/simulator/test_measure_tensor.py
git commit -m "feat(measure): Measure.run 支持 method='tensor'（TN 末态复用既有测量机制）"
```

---

### Task 9: 文档 + NPU 测试脚本 + 全量回归

**Files:**
- Create: `aicir/simulator/README.md`
- Modify: `CHANGELOG.md`
- Create: `demos/demo_npu_tensor.py`
- Test: `tests/simulator/test_npu_demo_importable.py`

**Interfaces:**
- Consumes: 全部公共 API。
- Produces: 可远程运行的 `demos/demo_npu_tensor.py`（`python demos/demo_npu_tensor.py [--allow-cpu-fallback]`）。

- [ ] **Step 1: 写失败测试（脚本可导入 + 冒烟）**

```python
# tests/simulator/test_npu_demo_importable.py
import importlib.util
import pathlib


def test_demo_npu_tensor_importable():
    path = pathlib.Path(__file__).resolve().parents[2] / "demos" / "demo_npu_tensor.py"
    spec = importlib.util.spec_from_file_location("demo_npu_tensor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 提供可复用的核对函数，CPU 上亦可跑
    assert hasattr(module, "run_checks")
    module.run_checks(allow_cpu_fallback=True)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=. pytest tests/simulator/test_npu_demo_importable.py -q`
Expected: FAIL（找不到 `demos/demo_npu_tensor.py`）

- [ ] **Step 3: 创建 `demos/demo_npu_tensor.py`**

```python
"""NPU 张量网络引擎远程验证脚本。

用法：
    python demos/demo_npu_tensor.py                    # 严格要求 NPU
    python demos/demo_npu_tensor.py --allow-cpu-fallback  # 无卡开发

核对项：
    1) NPU 上 TN 全态矢量 vs NPU 态矢量引擎一致；
    2) NPU 上 single_amplitude / partial_amplitude 正确；
    3) NPU 上 tn_expectation 可微（反传得到 .grad）；
并打印后端 name / 设备 / 是否复数（触发 real/imag 分解路径）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aicir import Circuit, Hamiltonian, State, cnot, ry, rzz
from aicir.backends.npu_backend import NPUBackend, is_npu_available
from aicir.simulator import partial_amplitude, single_amplitude, tn_expectation, tn_statevector


def _demo_circuit(theta=0.4):
    return Circuit(ry(theta, 0), cnot(1, [0]), ry(0.9, 1), rzz(0.3, 0, 1), n_qubits=2)


def run_checks(allow_cpu_fallback: bool = False) -> bool:
    bk = NPUBackend(fallback_to_cpu=allow_cpu_fallback)
    print(f"[backend] {bk.name}")
    print(f"[device ] {bk._device}  npu_available={is_npu_available()}")

    c = _demo_circuit()
    ref = State.zero_state(c.n_qubits, bk).evolve(c.unitary(backend=bk)).to_numpy()

    # 1) 全态矢量一致
    tn = tn_statevector(c, backend=bk).to_numpy()
    ok_sv = np.allclose(tn, ref, atol=1e-3)
    print(f"[check ] tn_statevector vs statevector: {'OK' if ok_sv else 'FAIL'}")

    # 2) 单 / 部分振幅
    ok_single = np.isclose(single_amplitude(c, "11", backend=bk), ref[3], atol=1e-3)
    part = partial_amplitude(c, open_qubits=[1], backend=bk)
    ok_partial = np.allclose(part, [ref[0], ref[1]], atol=1e-3)
    print(f"[check ] single_amplitude: {'OK' if ok_single else 'FAIL'}")
    print(f"[check ] partial_amplitude: {'OK' if ok_partial else 'FAIL'}")

    # 3) 可微
    ok_grad = True
    try:
        import torch

        theta = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
        energy = tn_expectation(_demo_circuit(theta), Hamiltonian([("ZI", 1.0)]), backend=bk)
        energy.backward()
        grad = float(theta.grad)
        ok_grad = np.isclose(grad, -np.sin(0.4), atol=1e-2)
        print(f"[check ] tn_expectation grad={grad:.5f} vs -sin(0.4)={-np.sin(0.4):.5f}: "
              f"{'OK' if ok_grad else 'FAIL'}")
    except ImportError:
        print("[check ] torch 缺失，跳过可微检查")

    passed = ok_sv and ok_single and ok_partial and ok_grad
    print(f"[result] {'ALL PASSED' if passed else 'FAILED'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_fallback and not is_npu_available():
        print("NPU 不可用；如需在 CPU 上开发请加 --allow-cpu-fallback", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if run_checks(allow_cpu_fallback=args.allow_cpu_fallback) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=. pytest tests/simulator/test_npu_demo_importable.py -q`
Expected: PASS（1 passed；CPU-fallback 路径核对通过）

- [ ] **Step 5: 创建 `aicir/simulator/README.md`**

写明：三种方法用途、`tn_statevector/single_amplitude/partial_amplitude/tn_expectation` 与 `Measure.run(method="tensor")` 用法、比特序（msb，qubit0 MSB）与开放比特排序约定、仅纯态无噪声、opt_einsum 可选、`method='tensor'` 下 `snap` 不生效、NPU 远程脚本 `demos/demo_npu_tensor.py` 用法。（内容用中文，风格对齐其他子包 README。）

- [ ] **Step 6: 更新 `CHANGELOG.md`**

在 `## 2026-07-02` 的 `### Added` 下追加一条：

```markdown
- **`aicir.simulator` 精确张量网络模拟引擎：`tn_statevector` / `single_amplitude` /
  `partial_amplitude` / `tn_expectation`，并为 `Measure.run` 增加 `method="tensor"`。**
  收缩建立在新的 `Backend` 原语（`tensordot/transpose/reshape/conj`）之上，NPU 的
  `tensordot` 复用 autograd-safe 复数 matmul（real/imag 分解），期望值在 torch/NPU 后端
  可微；收缩路径用 opt_einsum（可选）或内置贪心。仅纯态、无噪声。配套
  `demos/demo_npu_tensor.py`（远程 NPU 验证）与 `aicir/simulator/README.md`。MPS 截断另立 Spec 2。
```

- [ ] **Step 7: 全量回归**

Run: `PYTHONPATH=. pytest tests/simulator tests/backends -q && PYTHONPATH=. pytest -q`
Expected: PASS（新用例全过，既有套件不回归）

- [ ] **Step 8: 提交**

```bash
git add aicir/simulator/README.md CHANGELOG.md demos/demo_npu_tensor.py tests/simulator/test_npu_demo_importable.py
git commit -m "docs+demo: simulator README、CHANGELOG、NPU 远程验证脚本 demo_npu_tensor"
```

---

## Self-Review

**Spec 覆盖：**
- 张量网络模拟 → Task 4/5/6（`tn_statevector`）✓
- 单振幅 → Task 6（`single_amplitude`）✓
- 部分振幅 → Task 6（`partial_amplitude`，open_qubits + bitstrings 两模式）✓
- 真正 NPU on-device → Task 2（`tensordot_via_matmul` + real/imag 原语）✓
- 可微 → Task 7（`tn_expectation`）✓
- opt_einsum 可选 + 贪心回退 → Task 5 ✓
- Measure/State 集成 → Task 8（`method="tensor"`）✓
- 约定对齐（msb）→ Task 6 对齐测试 ✓
- 仅纯态/无噪声 → Task 8 报错守卫 ✓
- NPU 远程测试脚本（交付物）→ Task 9 ✓
- README + CHANGELOG + 顶层再导出 → Task 6/7/9 ✓

**占位符扫描：** 无 TBD/TODO；每个代码步骤含完整代码与命令。

**类型/命名一致性：** `gate_tensors`（Task 3）→ `build_network`（Task 4）→ `contract`（Task 5）→ 公共入口（Task 6/7）→ `Measure.run(method=)`（Task 8）签名前后一致；`_statevector_tensor` 在 Task 6 定义、Task 7 复用；`tensordot` axes 契约（`(list_a, list_b)`）在 Task 1/2/5 一致。

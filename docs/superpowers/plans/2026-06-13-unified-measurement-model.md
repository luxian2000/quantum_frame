# 统一量子测量模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 README §4 的"机制一/机制二"合并为单一测量模型，并实现到代码：线路内**非破坏性联合 Pauli 投影测量** + **末端逐比特 Z 读出**，`reset` 为重置信道，采用**逐操作轨迹引擎**（噪声完整纳入），`output/counts/prob` 改为按操作下标的方法。

**Architecture:** 新增 `projector.py`（投影/重置/末端测量纯函数，作用于 `State`，走 numpy 主机计算）、`trajectory.py`（单条轨迹的逐操作执行）、`aggregate.py`（`sm="avg"` 聚合）；重写 `measure.py`（`Measure.run` 编排）与 `result.py`（方法式 `Result`）；扩展 `Measurement` IR（`basis`/`id`）与 `measure()` 工厂；`Circuit.unitary()` 对非酉操作报错。下游 `estimator/primitives/VQA/IO/demos` 同步迁移，最终全测试通过。

**Tech Stack:** Python，numpy（核心），`aicir.core.State`/`Backend` 抽象，pytest（`PYTHONPATH=. pytest`）。`torch`/`scipy`/`matplotlib` 可选（用 `pytest.importorskip`）。

**权威设计来源：** [docs/superpowers/specs/2026-06-13-unified-measurement-model-design.md](../specs/2026-06-13-unified-measurement-model-design.md)（含对上传规则的有意偏差：`shots=0≡None` 且覆盖 `tm` 不做末端测量；`basis` 命名；末端输出保留输入顺序；仅 `sm="avg"`；噪声纳入；QASM 不支持即报错）。

**关键约定（务必遵守）：**
- 比特序：`State` 默认 `bit_order="msb"`，qubit `q` 对应 flat index 的第 `(n-1-q)` 位（`mask = 1 << (n-1-q)`），reshape `[2]*n` 后 qubit `q` 是张量第 `q` 轴。
- 投影/坍缩/重置一律走 numpy 主机计算（`backend.to_numpy` → 计算 → `backend.cast`），与现有 `_collapse_*`/`_reset_*` 风格一致（测量本就不可微）。
- 每个任务 TDD：先写失败测试 → 跑失败 → 最小实现 → 跑通过 → 提交。命令一律 `PYTHONPATH=. pytest ...`。

---

## 文件结构

**新增：**
- `aicir/measure/projector.py` — 纯函数：单比特门施加、基变换、联合宇称概率与投影、重置信道、末端逐比特 Z 测量。
- `aicir/measure/trajectory.py` — `TrajectoryResult` 数据类 + `run_trajectory(...)` 单条轨迹执行。
- `aicir/measure/aggregate.py` — `aggregate_avg(...)` 把多条轨迹折叠为 `Result` 字段。
- 测试：`tests/measure/test_projector.py`、`tests/measure/test_trajectory.py`、`tests/measure/test_unified_run.py`、`tests/measure/test_result_api.py`。

**重写：** `aicir/measure/measure.py`、`aicir/measure/result.py`。

**修改：** `aicir/ir/measurement.py`、`aicir/core/circuit.py`、`aicir/measure/estimator.py`、`aicir/primitives/{sampler,estimator}.py`、`aicir/vqc/{VQE,QAOA,SSVQE,VQD}.py`、`aicir/optimization/qubo/qaoa.py`、`aicir/transpile/passmanager.py`、`aicir/optimizer/circuit.py`、`aicir/qas/demos/_np_ising_utils.py`、`aicir/core/io/*`、`demos/*`、`README.md`、`CHANGELOG.md`。

---

## Task 1: `Measurement` IR 增加 `basis` 与 `id` 字段

**Files:**
- Modify: `aicir/ir/measurement.py`
- Test: `tests/ir/test_measurement_basis_id.py` (create)

- [ ] **Step 1: 写失败测试**

创建 `tests/ir/test_measurement_basis_id.py`：

```python
from aicir.ir.measurement import Measurement


def test_defaults_basis_z_id_none():
    m = Measurement((0, 1))
    assert m.basis == "Z"
    assert m.id is None


def test_basis_normalized_uppercase_and_validated():
    assert Measurement((0,), basis="x").basis == "X"
    import pytest
    with pytest.raises(ValueError):
        Measurement((0,), basis="W")


def test_to_dict_round_trip_includes_basis_and_id():
    m = Measurement((0, 2), basis="Y", id="m0")
    d = m.to_dict()
    assert d["type"] == "measure"
    assert d["qubits"] == [0, 2]
    assert d["basis"] == "Y"
    assert d["id"] == "m0"
    assert Measurement.from_dict(d) == m


def test_from_dict_backward_compatible_without_basis_id():
    m = Measurement.from_dict({"type": "measure", "qubits": [1]})
    assert m.basis == "Z"
    assert m.id is None


def test_reset_keeps_basis_default_and_no_id_field_emitted():
    r = Measurement((0,), measurement_type="reset")
    d = r.to_dict()
    assert d["type"] == "reset"
    # reset 不携带 basis/id 语义；basis 默认 Z 不必输出，id 为 None 不输出
    assert "id" not in d
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/ir/test_measurement_basis_id.py -q`
Expected: FAIL（`Measurement` 无 `basis`/`id`）。

- [ ] **Step 3: 实现**

在 `aicir/ir/measurement.py`：`_KNOWN_MEASUREMENT_KEYS` 增加 `"basis"`、`"id"`；给 `@dataclass` 增加字段并在 `__post_init__` 校验；改 `from_dict`/`to_dict`/`__eq__`。

```python
_KNOWN_MEASUREMENT_KEYS = {
    "type", "target_qubit", "qubits", "return_type",
    "classical_bit", "classical_bits", "clbits",
    "basis", "id",
}

_VALID_BASES = {"X", "Y", "Z"}


@dataclass(frozen=True)
class Measurement(LegacyGateView):
    qubits: tuple[int, ...] = ()
    measurement_type: str = "measure"
    return_type: str = "counts"
    classical_bits: tuple[int, ...] = ()
    basis: str = "Z"
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        measurement_type = str(self.measurement_type).strip()
        if not measurement_type:
            raise ValueError("measurement_type cannot be empty")
        return_type = str(self.return_type).strip()
        if not return_type:
            raise ValueError("return_type cannot be empty")
        basis = str(self.basis).strip().upper()
        if basis not in _VALID_BASES:
            raise ValueError(f"measure basis 必须是 X/Y/Z 之一，收到 {self.basis!r}")
        measure_id = None if self.id is None else str(self.id)

        qubits = _as_int_tuple(self.qubits, label="qubits")
        classical_bits = _as_int_tuple(self.classical_bits, label="classical_bits")
        if classical_bits and qubits and len(classical_bits) != len(qubits):
            raise ValueError("classical_bits length must match qubits length")

        object.__setattr__(self, "measurement_type", measurement_type)
        object.__setattr__(self, "return_type", return_type)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "classical_bits", classical_bits)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "id", measure_id)
        object.__setattr__(self, "metadata", dict(self.metadata))
```

`from_dict` 在构造前解析（放在 `metadata` 推导之前，并让这两个键不落入 metadata）：

```python
        basis = str(gate.get("basis", "Z"))
        measure_id = gate.get("id", None)
        metadata = {
            key: value
            for key, value in gate.items()
            if key not in _KNOWN_MEASUREMENT_KEYS
        }
        return cls(
            qubits=qubits,
            measurement_type=measurement_type,
            return_type=str(gate.get("return_type", "counts")),
            classical_bits=classical_bits,
            basis=basis,
            id=measure_id,
            metadata=metadata,
        )
```

`to_dict`（`reset` 不输出 basis/id；`measure` 仅在非默认时输出 basis、在有 id 时输出 id）：

```python
    def to_dict(self) -> dict[str, Any]:
        gate: dict[str, Any] = {
            "type": self.measurement_type,
            "qubits": list(self.qubits),
        }
        if self.return_type != "counts":
            gate["return_type"] = self.return_type
        if self.classical_bits:
            gate["classical_bits"] = list(self.classical_bits)
        if self.measurement_type != "reset":
            if self.basis != "Z":
                gate["basis"] = self.basis
            if self.id is not None:
                gate["id"] = self.id
        for key, value in self.metadata.items():
            gate[key] = value
        return gate
```

`__eq__` 的字段元组同时加入 `self.basis, self.id`（两侧一致）。

> 注：上面 `to_dict` 默认 `basis=="Z"` 时**不**输出 `basis` 键，但测试 `test_to_dict_round_trip_includes_basis_and_id` 用的是 `basis="Y"`，会输出；`from_dict` 缺省回落 `"Z"`，往返一致。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/ir/test_measurement_basis_id.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/ir/measurement.py tests/ir/test_measurement_basis_id.py
git commit -m "feat(ir): Measurement 增加 basis/id 字段与 JSON 往返"
```

---

## Task 2: `measure()` 工厂支持 `basis`/`id`；`reset()` 删除前置说明

**Files:**
- Modify: `aicir/core/circuit.py:694-738`
- Test: `tests/circuit/test_measure_factory_basis_id.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import pytest
from aicir import Circuit, measure, reset, hadamard


def test_measure_basis_id_flow_to_dict():
    cir = Circuit(hadamard(0), measure(0, 1, basis="X", id="m0"), n_qubits=2)
    d = cir.gates[-1]
    assert d["type"] == "measure"
    assert d["qubits"] == [0, 1]
    assert d["basis"] == "X"
    assert d["id"] == "m0"


def test_measure_default_basis_z_no_id():
    cir = Circuit(measure(0), n_qubits=1)
    d = cir.gates[-1]
    assert d.get("basis", "Z") == "Z"
    assert d.get("id") is None


def test_measure_iterable_form_with_basis():
    cir = Circuit(measure([0, 1], basis="y"), n_qubits=2)
    assert cir.gates[-1]["basis"] == "Y"


def test_reset_factory_still_works():
    cir = Circuit(reset(0, 1), n_qubits=2)
    assert cir.gates[-1]["type"] == "reset"
    assert cir.gates[-1]["qubits"] == [0, 1]
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/circuit/test_measure_factory_basis_id.py -q`
Expected: FAIL（`measure()` 不接受 `basis`/`id`）。

- [ ] **Step 3: 实现**

`aicir/core/circuit.py` 改 `measure`：

```python
def measure(*qubits, basis="Z", id=None):
    """线路内联合 Pauli 投影测量标记（非破坏性，保留比特）。

    measure(0)                 # 单比特 Z 测量
    measure(0, 1, basis="X")   # 联合 X⊗X 投影测量
    measure([0, 1], id="m0")   # 可迭代形式 + 结果标识符
    measure()                  # 空 = 运行时读取全部比特

    basis 默认 "Z"（X/Y/Z）；id 可选、用于 result.output("m0")。
    """
    return Measurement(_flat_marker_qubits(qubits), basis=basis, id=id)
```

改 `reset` 的 docstring，删除"目标比特此前必须已 measure、且两者之间不得有门"的描述：

```python
def reset(*qubits):
    """线路内重置信道标记：把指定比特重置为 |0>。

    参数形式与 measure 相同：reset(0)、reset(0, 1)、reset([0, 1])、reset()。
    无前置条件——可出现在线路任意位置（见统一测量模型设计文档）。
    """
    return Measurement(_flat_marker_qubits(qubits), measurement_type="reset")
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/circuit/test_measure_factory_basis_id.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/core/circuit.py tests/circuit/test_measure_factory_basis_id.py
git commit -m "feat(core): measure() 支持 basis/id；reset 去除前置约束"
```

---

## Task 3: `Circuit.unitary()/matrix()` 对非酉操作报错 + `ignore_nonunitary`

**Files:**
- Modify: `aicir/core/circuit.py:503-529`
- Test: `tests/circuit/test_unitary_nonunitary.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import pytest
import numpy as np
from aicir import Circuit, hadamard, cnot, measure, reset


def test_unitary_raises_on_measure_by_default():
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    with pytest.raises(ValueError):
        cir.unitary()


def test_unitary_raises_on_reset_by_default():
    cir = Circuit(hadamard(0), reset(0), n_qubits=1)
    with pytest.raises(ValueError):
        cir.unitary()


def test_unitary_ignore_nonunitary_drops_markers():
    cir = Circuit(hadamard(0), measure(0), reset(0), n_qubits=1)
    u = cir.unitary(ignore_nonunitary=True)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    assert np.allclose(np.asarray(u), h, atol=1e-6)


def test_pure_circuit_unitary_unchanged():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    assert np.asarray(cir.unitary()).shape == (4, 4)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/circuit/test_unitary_nonunitary.py -q`
Expected: FAIL（measure 被静默跳过、且无 `ignore_nonunitary` 参数）。

- [ ] **Step 3: 实现**

改 `unitary` 签名与循环（`matrix()` 若是 `unitary` 的薄封装则透传该参数）：

```python
    def unitary(self, backend=None, ignore_nonunitary=False):
        parameters = self.parameters
        if parameters:
            names = ", ".join(parameter.name for parameter in parameters)
            raise ValueError(f"Circuit has unbound parameter(s): {names}; call bind_parameters(...) first")

        backend = backend or self._backend
        if not self.gates:
            return identity(self.n_qubits) if backend is None else backend.eye(1 << self.n_qubits)

        gate_qubits = _infer_n_qubits_from_gates(self.gates)
        if gate_qubits > self.n_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {self.n_qubits}")

        circuit_matrix = identity(self.n_qubits) if backend is None else backend.eye(1 << self.n_qubits)
        for gate in self.gates:
            if _is_measure_gate(gate) or _is_reset_gate(gate):
                if ignore_nonunitary:
                    continue
                kind = "measure" if _is_measure_gate(gate) else "reset"
                raise ValueError(
                    f"{kind} 是非酉操作，不能用于 Circuit.unitary()/matrix()；"
                    f"如需仅取酉部分请传 ignore_nonunitary=True"
                )
            gm = gate_to_matrix(gate, self.n_qubits, backend=backend)
            if backend is None:
                circuit_matrix = np.matmul(gm, circuit_matrix)
            else:
                circuit_matrix = backend.matmul(gm, circuit_matrix)
        return circuit_matrix
```

若 `matrix()` 独立定义，给它加 `ignore_nonunitary=False` 并转发给 `unitary`。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/circuit/test_unitary_nonunitary.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/core/circuit.py tests/circuit/test_unitary_nonunitary.py
git commit -m "feat(core): unitary() 对 measure/reset 报错并提供 ignore_nonunitary"
```

---

## Task 4: `projector.py` — 单比特门施加 + 基变换

**Files:**
- Create: `aicir/measure/projector.py`
- Test: `tests/measure/test_projector.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from aicir.core.state import State
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.measure import projector


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_basis_change_x_maps_plus_to_zero():
    # |+> 在 X 基变换 (H) 下变成 |0>
    plus = _sv([1, 1]) ; plus = State(plus.backend.cast((plus.to_numpy() / np.sqrt(2))), 1, plus.backend)
    out = projector.pauli_basis_change(plus, [0], "X", inverse=False)
    v = out.to_numpy().reshape(-1)
    assert np.allclose(v, [1, 0], atol=1e-6)


def test_basis_change_round_trip_identity():
    psi = _sv([0.5, 0.5, 0.5, 0.5])
    fwd = projector.pauli_basis_change(psi, [0, 1], "Y", inverse=False)
    back = projector.pauli_basis_change(fwd, [0, 1], "Y", inverse=True)
    assert np.allclose(back.to_numpy().reshape(-1), psi.to_numpy().reshape(-1), atol=1e-6)


def test_basis_change_z_is_noop():
    psi = _sv([0.6, 0.8])
    out = projector.pauli_basis_change(psi, [0], "Z", inverse=False)
    assert np.allclose(out.to_numpy().reshape(-1), [0.6, 0.8], atol=1e-6)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: FAIL（模块/函数不存在）。

- [ ] **Step 3: 实现**

创建 `aicir/measure/projector.py`：

```python
"""测量投影 / 重置 / 末端读出的后端无关纯函数（numpy 主机计算）。

约定：bit_order="msb"，qubit q 对应 flat index 第 (n-1-q) 位、reshape [2]*n 后第 q 轴。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.state import State

_H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def _apply_1q_sv(psi_col: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    t = psi_col.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1, 1)


def _apply_1q_dm(rho: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    dim = 1 << n
    t = rho.reshape([2] * (2 * n))
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    t = np.tensordot(U.conj(), t, axes=([1], [n + q]))
    t = np.moveaxis(t, 0, n + q)
    return t.reshape(dim, dim)


def _basis_change_seq(basis: str, inverse: bool) -> List[np.ndarray]:
    """返回把单比特从 basis 旋到 Z 所需、按施加顺序排列的单比特门。

    X: 前 H；后 H。
    Y: 前 Sdg 然后 H；后 H 然后 S。
    Z: 无。
    """
    basis = basis.upper()
    if basis == "Z":
        return []
    if basis == "X":
        return [_H]
    if basis == "Y":
        return [_SDG, _H] if not inverse else [_H, _S]
    raise ValueError(f"未知 basis {basis!r}")


def pauli_basis_change(state: State, qubits: Sequence[int], basis: str, inverse: bool) -> State:
    backend = state.backend
    n = state.n_qubits
    seq = _basis_change_seq(basis, inverse)
    if not seq:
        return state
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).astype(np.complex128)
        for q in qubits:
            for U in seq:
                rho = _apply_1q_dm(rho, n, int(q), U)
        return State(backend.cast(rho), n, backend)
    psi = backend.to_numpy(state.data).reshape(-1, 1).astype(np.complex128)
    for q in qubits:
        for U in seq:
            psi = _apply_1q_sv(psi, n, int(q), U)
    return State(backend.cast(psi), n, backend, bit_order=state.bit_order)
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/projector.py tests/measure/test_projector.py
git commit -m "feat(measure): projector 单比特门施加与基变换"
```

---

## Task 5: `projector.py` — 联合宇称概率与投影（联合 Pauli 测量核心）

**Files:**
- Modify: `aicir/measure/projector.py`
- Test: `tests/measure/test_projector.py` (extend)

- [ ] **Step 1: 写失败测试**

向 `tests/measure/test_projector.py` 追加：

```python
def _bell():
    b = NumpyBackend()
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return State(b.cast(v.reshape(-1, 1)), 2, b)


def test_joint_zz_on_bell_is_deterministic_plus_and_keeps_entanglement():
    psi = _bell()
    rng = np.random.default_rng(0)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "Z", rng)
    assert lam == 1  # Bell 态 Z⊗Z 恒 +1
    v = out.to_numpy().reshape(-1)
    # 仍是 (|00>+|11>)/√2，未坍缩到单基态（保持纠缠）
    assert np.allclose(np.abs(v), [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], atol=1e-6)


def test_joint_measure_single_qubit_not_in_pa_eigenstate():
    # |psi> = (|00> + |01> + |10> - |11>)/2  → X⊗X 本征值确定，但单比特不在 X 本征态
    b = NumpyBackend()
    v = np.array([1, 1, 1, -1], dtype=complex) / 2
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(1)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "X", rng)
    assert lam in (1, -1)
    # 约化到 qubit0 的态：若已是 X 本征态则 <X> = ±1；这里应严格在 (-1,1) 之间
    red = out.partial_trace([0]).to_numpy().reshape(2, 2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    exp_x = np.real(np.trace(red @ X))
    assert abs(exp_x) < 1 - 1e-6


def test_joint_probs_match_born():
    b = NumpyBackend()
    v = np.array([np.sqrt(0.3), 0, 0, np.sqrt(0.7)], dtype=complex)  # 00 与 11 均偶宇称
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    p_plus, p_minus = projector.joint_parity_probs(psi, [0, 1], "Z")
    assert np.isclose(p_plus, 1.0, atol=1e-6)
    assert np.isclose(p_minus, 0.0, atol=1e-6)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: FAIL（无 `joint_parity_probs`/`measure_joint_pauli`）。

- [ ] **Step 3: 实现**

向 `projector.py` 追加：

```python
def _parity_mask(n: int, qubits: Sequence[int]) -> int:
    m = 0
    for q in qubits:
        m |= 1 << (n - 1 - int(q))
    return m


def _parities(dim: int, mask: int) -> np.ndarray:
    """返回每个 flat index 在 mask 选中比特上的宇称（0=偶,1=奇）。"""
    idx = np.arange(dim, dtype=np.int64) & mask
    p = idx.copy()
    shift = 32
    while shift:
        p ^= p >> shift
        shift >>= 1
    return (p & 1).astype(np.int64)


def joint_parity_probs(state: State, qubits: Sequence[int], basis: str) -> Tuple[float, float]:
    n = state.n_qubits
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    backend = state.backend
    par = _parities(1 << n, _parity_mask(n, qubits))
    if rotated.is_density:
        rho = backend.to_numpy(rotated.data).reshape(1 << n, 1 << n)
        diag = np.real(np.diag(rho))
        p_plus = float(diag[par == 0].sum())
    else:
        psi = backend.to_numpy(rotated.data).reshape(-1)
        probs = np.abs(psi) ** 2
        p_plus = float(probs[par == 0].sum())
    p_plus = min(max(p_plus, 0.0), 1.0)
    return p_plus, 1.0 - p_plus


def _project_parity_rotated(rotated: State, qubits: Sequence[int], lam: int) -> State:
    """在已旋到 Z 的态上，投影到联合宇称 lam(±1) 子空间并归一化（保持子空间内相干）。"""
    backend = rotated.backend
    n = rotated.n_qubits
    par = _parities(1 << n, _parity_mask(n, qubits))
    keep = (par == (0 if lam == 1 else 1))
    if rotated.is_density:
        rho = backend.to_numpy(rotated.data).reshape(1 << n, 1 << n).copy()
        mask2d = np.outer(keep, keep)
        rho = np.where(mask2d, rho, 0.0)
        tr = np.real(np.trace(rho))
        if tr > 0:
            rho = rho / tr
        return State(backend.cast(rho), n, backend)
    psi = backend.to_numpy(rotated.data).reshape(-1, 1).copy()
    psi[~keep, 0] = 0.0
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    return State(backend.cast(psi), n, backend, bit_order=rotated.bit_order)


def measure_joint_pauli(state: State, qubits: Sequence[int], basis: str, rng) -> Tuple[State, int]:
    """非破坏性联合 Pauli 投影测量：返回 (坍缩后完整态, 本征值 lam∈{+1,-1})。"""
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    p_plus, _ = joint_parity_probs(state, qubits, basis)
    lam = 1 if rng.random() < p_plus else -1
    projected = _project_parity_rotated(rotated, qubits, lam)
    restored = pauli_basis_change(projected, qubits, basis, inverse=True)
    return restored, lam
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/projector.py tests/measure/test_projector.py
git commit -m "feat(measure): 联合 Pauli 非破坏性投影测量（宇称子空间投影）"
```

---

## Task 6: `projector.py` — 重置信道（纠缠目标 → 升级 DM）

**Files:**
- Modify: `aicir/measure/projector.py`
- Test: `tests/measure/test_projector.py` (extend)

- [ ] **Step 1: 写失败测试**

```python
def test_reset_product_qubit_stays_pure():
    # |1> ⊗ |+>  → reset(0) → |0> ⊗ |+>，仍是纯态向量
    b = NumpyBackend()
    v = np.kron([0, 1], np.array([1, 1]) / np.sqrt(2)).astype(complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    out = projector.reset_channel(psi, [0])
    assert not out.is_density
    expect = np.kron([1, 0], np.array([1, 1]) / np.sqrt(2))
    assert np.allclose(out.to_numpy().reshape(-1), expect, atol=1e-6)


def test_reset_entangled_qubit_promotes_to_density_matrix():
    # Bell 态 reset(0) → |0><0| ⊗ I/2（混合态）
    b = NumpyBackend()
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    out = projector.reset_channel(psi, [0])
    assert out.is_density
    rho = out.to_numpy().reshape(4, 4)
    expect = np.zeros((4, 4), dtype=complex)
    expect[0, 0] = 0.5
    expect[1, 1] = 0.5  # |00><00| 与 |01><01| 各 0.5
    assert np.allclose(rho, expect, atol=1e-6)


def test_reset_on_density_matrix_input():
    b = NumpyBackend()
    rho = np.zeros((2, 2), dtype=complex)
    rho[1, 1] = 1.0  # |1><1|
    st = State.from_matrix(rho, 1, b)
    out = projector.reset_channel(st, [0])
    assert np.allclose(out.to_numpy().reshape(2, 2), np.array([[1, 0], [0, 0]]), atol=1e-6)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: FAIL（无 `reset_channel`）。

- [ ] **Step 3: 实现**

向 `projector.py` 追加（DM 路径复用现 measure.py 中 `_reset_density_matrix_state` 的索引公式；这里就地实现，保持 projector 自包含）：

```python
def _replace_bit(index: int, n: int, q: int, bit: int) -> int:
    mask = 1 << (n - 1 - q)
    return (index | mask) if bit else (index & ~mask)


def _reset_dm(rho: np.ndarray, n: int, q: int) -> np.ndarray:
    dim = 1 << n
    out = np.zeros_like(rho)
    mask = 1 << (n - 1 - q)
    rows = [r for r in range(dim) if not (r & mask)]
    for r in rows:
        r1 = _replace_bit(r, n, q, 1)
        for c in rows:
            c1 = _replace_bit(c, n, q, 1)
            out[r, c] = rho[r, c] + rho[r1, c1]
    return out


def _reset_one(state: State, q: int) -> State:
    backend = state.backend
    n = state.n_qubits
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).astype(np.complex128)
        return State(backend.cast(_reset_dm(rho, n, int(q))), n, backend)

    psi = backend.to_numpy(state.data).reshape([2] * n).astype(np.complex128)
    sl0 = [slice(None)] * n; sl0[q] = 0
    sl1 = [slice(None)] * n; sl1[q] = 1
    a = psi[tuple(sl0)].reshape(-1)
    b = psi[tuple(sl1)].reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)

    parallel = False
    if na < 1e-12 or nb < 1e-12:
        parallel = True
    else:
        c = np.vdot(a, b) / (na * na)
        parallel = np.linalg.norm(b - c * a) < 1e-9

    if parallel:
        v = a if na >= nb else b
        v = v / np.linalg.norm(v)
        out = np.zeros([2] * n, dtype=np.complex128)
        out[tuple(sl0)] = v.reshape(out[tuple(sl0)].shape)
        return State(backend.cast(out.reshape(-1, 1)), n, backend, bit_order=state.bit_order)

    flat = psi.reshape(-1, 1)
    rho = (flat @ flat.conj().T)
    rho = _reset_dm(rho, n, int(q))
    return State(backend.cast(rho), n, backend)


def reset_channel(state: State, qubits: Sequence[int]) -> State:
    for q in qubits:
        state = _reset_one(state, int(q))
    return state
```

> 实现注意：`_reset_dm` 里那行 `rho[r, r and 0 or c] if False else ...` 是为避免误改的占位写法，**最终实现应直接写** `out[r, c] = rho[r, c] + rho[r1, c1]`。执行时请用后者。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/projector.py tests/measure/test_projector.py
git commit -m "feat(measure): reset 重置信道（纠缠目标升级密度矩阵）"
```

---

## Task 7: `projector.py` — 末端逐比特 Z 测量（采样子集 + 投影）

**Files:**
- Modify: `aicir/measure/projector.py`
- Test: `tests/measure/test_projector.py` (extend)

- [ ] **Step 1: 写失败测试**

```python
def test_terminal_z_full_register_collapses_to_basis():
    psi = _bell()
    rng = np.random.default_rng(3)
    out, eig = projector.terminal_z_measure(psi, [0, 1], rng)
    v = out.to_numpy().reshape(-1)
    nz = np.flatnonzero(np.abs(v) > 1e-9)
    assert len(nz) == 1  # 全比特测量坍缩到单一基态
    # Bell 只可能 00 或 11 → 本征值乘积 +1，逐比特相等
    assert eig in ([1, 1], [-1, -1])


def test_terminal_z_subset_keeps_other_qubit():
    # |0> ⊗ |+>，只测 qubit0 → 必得 +1，qubit1 仍是 |+>
    b = NumpyBackend()
    v = np.kron([1, 0], np.array([1, 1]) / np.sqrt(2)).astype(complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(4)
    out, eig = projector.terminal_z_measure(psi, [0], rng)
    assert eig == [1]
    red = out.partial_trace([1]).to_numpy().reshape(2, 2)
    plus = np.array([1, 1]) / np.sqrt(2)
    assert np.allclose(red, np.outer(plus, plus.conj()), atol=1e-6)


def test_terminal_order_preserved():
    # 比特0=|0>, 比特1=|1> 的乘积态，measure_qubits=[1,0] → 本征值顺序应为 [z1, z0]=[-1, +1]
    b = NumpyBackend()
    v = np.kron([1, 0], [0, 1]).astype(complex)  # |0>_0 ⊗ |1>_1
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(5)
    _, eig = projector.terminal_z_measure(psi, [1, 0], rng)
    assert eig == [-1, 1]
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: FAIL（无 `terminal_z_measure`）。

- [ ] **Step 3: 实现**

向 `projector.py` 追加：

```python
def _born_sample_index(state: State, rng) -> int:
    backend = state.backend
    n = state.n_qubits
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n)
        probs = np.real(np.diag(rho))
    else:
        psi = backend.to_numpy(state.data).reshape(-1)
        probs = np.abs(psi) ** 2
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    probs = probs / total if total > 0 else np.full_like(probs, 1.0 / probs.size)
    return int(rng.choice(probs.size, p=probs))


def _project_subset_outcome(state: State, qubits: Sequence[int], bits: Sequence[int]) -> State:
    """把指定比特投影到给定 0/1 取值（其余比特保留），归一化。"""
    backend = state.backend
    n = state.n_qubits
    keep = np.ones(1 << n, dtype=bool)
    idx = np.arange(1 << n, dtype=np.int64)
    for q, bit in zip(qubits, bits):
        qmask = 1 << (n - 1 - int(q))
        bitvals = (idx & qmask) >> (n - 1 - int(q))
        keep &= (bitvals == int(bit))
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).copy()
        mask2d = np.outer(keep, keep)
        rho = np.where(mask2d, rho, 0.0)
        tr = np.real(np.trace(rho))
        if tr > 0:
            rho = rho / tr
        return State(backend.cast(rho), n, backend)
    psi = backend.to_numpy(state.data).reshape(-1, 1).copy()
    psi[~keep, 0] = 0.0
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    return State(backend.cast(psi), n, backend, bit_order=state.bit_order)


def terminal_z_measure(state: State, measure_qubits: Sequence[int], rng) -> Tuple[State, List[int]]:
    """对 measure_qubits 逐比特 Z 基测量（输入顺序保留）。

    返回 (坍缩后完整态, 本征值列表[按 measure_qubits 顺序, 取值 ±1])。
    """
    n = state.n_qubits
    x = _born_sample_index(state, rng)
    bits = [(x >> (n - 1 - int(q))) & 1 for q in measure_qubits]
    collapsed = _project_subset_outcome(state, measure_qubits, bits)
    eig = [1 if bit == 0 else -1 for bit in bits]
    return collapsed, eig
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_projector.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/projector.py tests/measure/test_projector.py
git commit -m "feat(measure): 末端逐比特 Z 测量（采样子集并投影，顺序保留）"
```

---

## Task 8: `trajectory.py` — 单条轨迹执行

**Files:**
- Create: `aicir/measure/trajectory.py`
- Test: `tests/measure/test_trajectory.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from aicir import Circuit, hadamard, cnot, measure, reset, pauli_x
from aicir.core.state import State
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.measure.trajectory import run_trajectory


def _init(n):
    b = NumpyBackend()
    return State.zero_state(n, b), b


def test_trajectory_records_incircuit_outcome_and_terminal():
    cir = Circuit(hadamard(0), cnot(1, [0]), measure(0, 1), n_qubits=2)  # ops: 0,1,2
    st, b = _init(2)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=True, measure_qubits=[0, 1],
                        snap_ops=set(), rng=rng, noise_model=None)
    assert tr.incircuit[2] == 1            # Bell 的 Z⊗Z 恒 +1
    assert len(tr.terminal) == 2           # 末端两比特
    assert set(tr.terminal) <= {1, -1}


def test_trajectory_reset_zeroes_qubit():
    cir = Circuit(pauli_x(0), reset(0), n_qubits=1)  # ops 0,1
    st, b = _init(1)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=False, measure_qubits=None,
                        snap_ops={1}, rng=rng, noise_model=None)
    snap = tr.snaps[1].to_numpy().reshape(-1)
    assert np.allclose(snap, [1, 0], atol=1e-6)
    assert tr.terminal is None             # tm=False 不做末端测量


def test_trajectory_snap_after_op_index():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    st, b = _init(2)
    rng = np.random.default_rng(0)
    tr = run_trajectory(cir, st, b, tm=False, measure_qubits=None,
                        snap_ops={0}, rng=rng, noise_model=None)
    v = tr.snaps[0].to_numpy().reshape(-1)
    assert np.allclose(np.abs(v), [1/np.sqrt(2), 0, 1/np.sqrt(2), 0], atol=1e-6)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_trajectory.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

创建 `aicir/measure/trajectory.py`：

```python
"""单条测量轨迹：逐操作执行 cir，处理线路内 measure/reset、snap 与末端测量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from ..core.gates import apply_gate_to_state, gate_to_matrix
from ..core.state import State
from ..ir import (
    circuit_instructions,
    instruction_name,
    instruction_qubits,
)
from . import projector


def _is_measure(gate) -> bool:
    return instruction_name(gate).lower() in {"measure", "measurement"}


def _is_reset(gate) -> bool:
    return instruction_name(gate).lower() == "reset"


def _marker_qubits(gate, n: int) -> List[int]:
    qs = [int(q) for q in instruction_qubits(gate)]
    return qs if qs else list(range(n))


@dataclass
class TrajectoryResult:
    pre: State                                  # ρ_pre（末端测量前的完整态）
    post: State                                 # ρ_post（末端测量后；tm=False 时同 pre）
    incircuit: Dict[int, int] = field(default_factory=dict)   # op_index -> λ(±1)
    terminal: Optional[List[int]] = None        # 末端逐比特 ±1（输入顺序）；tm=False 为 None
    snaps: Dict[int, State] = field(default_factory=dict)     # op_index -> 该操作后完整态


def run_trajectory(circuit, init_state: State, backend, *, tm: bool,
                   measure_qubits: Optional[Sequence[int]], snap_ops: Set[int],
                   rng, noise_model=None) -> TrajectoryResult:
    state = init_state
    n = state.n_qubits
    incircuit: Dict[int, int] = {}
    snaps: Dict[int, State] = {}

    for op_index, gate in enumerate(circuit_instructions(circuit)):
        if _is_measure(gate):
            qubits = _marker_qubits(gate, n)
            basis = str(gate.get("basis", "Z")) if isinstance(gate, dict) else "Z"
            state, lam = projector.measure_joint_pauli(state, qubits, basis, rng)
            incircuit[op_index] = lam
        elif _is_reset(gate):
            qubits = _marker_qubits(gate, n)
            state = projector.reset_channel(state, qubits)
        else:
            new_data = apply_gate_to_state(gate, state.data, n, backend)
            if new_data is None:
                gm = gate_to_matrix(gate, cir_qubits=n, backend=backend)
                state = state.evolve(gm)
            else:
                state = State(new_data, n, backend, bit_order=state.bit_order)
            if noise_model is not None:
                rho_noisy = noise_model.apply(
                    state.to_density_matrix().data if not state.is_density else state.data,
                    n_qubits=n, backend=backend, gate_type=instruction_name(gate),
                )
                state = State(rho_noisy, n, backend)
        if op_index in snap_ops:
            snaps[op_index] = state

    pre = state
    if tm and measure_qubits is not None and len(measure_qubits) > 0:
        post, terminal = projector.terminal_z_measure(state, measure_qubits, rng)
    elif tm and measure_qubits is None:
        post, terminal = projector.terminal_z_measure(state, list(range(n)), rng)
    else:
        post, terminal = pre, None

    return TrajectoryResult(pre=pre, post=post, incircuit=incircuit, terminal=terminal, snaps=snaps)
```

> 说明：`measure_qubits is None` 表示读取全部比特；`measure_qubits == []` 或 `tm=False` 表示不做末端测量（`post==pre`，`terminal=None`）。`Measure.run` 负责把 `shots∈{None,0}` 归一化为"关闭末端测量"后再调用本函数（传 `tm=False`）。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_trajectory.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/trajectory.py tests/measure/test_trajectory.py
git commit -m "feat(measure): 单条轨迹逐操作执行引擎"
```

---

## Task 9: `result.py` — 方法式 `Result`（registry + output/counts/prob/state/final_state/snap/reduce）

**Files:**
- Rewrite: `aicir/measure/result.py`
- Test: `tests/measure/test_result_api.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from aicir.measure.result import Result, MeasureSpec


def _empty_probs(n):
    p = np.zeros(1 << n); p[0] = 1.0
    return p


def test_output_by_index_id_and_terminal():
    specs = [MeasureSpec(op_index=2, id="m0", qubits=[0, 1], basis="Z")]
    r = Result(n_qubits=2, backend_name="numpy", probabilities=_empty_probs(2),
               shots=4, measurement_specs=specs,
               incircuit_outputs={2: np.array([[1], [1], [-1], [1]])},
               terminal_output=np.array([[1, 1], [1, -1], [-1, -1], [1, 1]]),
               terminal_qubits=[0, 1])
    assert r.output(2).shape == (4, 1)
    assert np.array_equal(r.output("m0"), r.output(2))
    assert r.output(-1).shape == (4, 2)


def test_output_invalid_index_raises():
    r = Result(n_qubits=1, backend_name="numpy", probabilities=_empty_probs(1),
               shots=2, measurement_specs=[], incircuit_outputs={}, terminal_output=None,
               terminal_qubits=None)
    with pytest.raises(ValueError):
        r.output(0)
    with pytest.raises(ValueError):
        r.output(-1)


def test_counts_prob_sampling_only():
    specs = [MeasureSpec(op_index=1, id=None, qubits=[0], basis="Z")]
    r = Result(n_qubits=1, backend_name="numpy", probabilities=_empty_probs(1),
               shots=None, measurement_specs=specs, incircuit_outputs={1: 1},
               terminal_output=None, terminal_qubits=None)
    with pytest.raises(RuntimeError):
        r.counts(1)


def test_reduce_partial_trace():
    rho = np.zeros((4, 4), dtype=complex); rho[0, 0] = rho[3, 3] = 0.5  # 1/2(|00><00|+|11><11|)
    r = Result(n_qubits=2, backend_name="numpy", probabilities=_empty_probs(2),
               shots=8, measurement_specs=[], incircuit_outputs={}, terminal_output=None,
               terminal_qubits=None, final_state=rho, final_state_kind="density_matrix")
    red = r.reduce([0], pos="final")
    assert np.allclose(red, np.array([[0.5, 0], [0, 0.5]]), atol=1e-6)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_result_api.py -q`
Expected: FAIL（新 `Result`/`MeasureSpec` 不存在）。

- [ ] **Step 3: 实现**

重写 `aicir/measure/result.py`：

```python
"""统一测量结果对象（统一测量模型，见 README §4 与设计文档）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class MeasureSpec:
    """一个线路内 measure 操作的登记项。"""
    op_index: int
    id: Optional[str]
    qubits: List[int]
    basis: str


def _reduced_density(rho_flat: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    keep = list(keep)
    rho = np.asarray(rho_flat).reshape(1 << n, 1 << n).reshape([2] * (2 * n))
    traced = [q for q in range(n) if q not in set(keep)]
    perm = keep + traced + [n + q for q in keep] + [n + q for q in traced]
    m, k = len(keep), len(traced)
    t = np.transpose(rho, perm).reshape(1 << m, 1 << k, 1 << m, 1 << k)
    return np.einsum("akbk->ab", t)


@dataclass
class Result:
    n_qubits: int
    backend_name: str
    probabilities: np.ndarray
    shots: Optional[int] = None
    measurement_specs: List[MeasureSpec] = field(default_factory=list)
    incircuit_outputs: Dict[int, object] = field(default_factory=dict)   # op_index -> (M,1) 或 标量
    incircuit_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)
    terminal_output: Optional[np.ndarray] = None                          # (M,k) 或 None
    terminal_counts: Optional[Dict[str, int]] = None
    terminal_qubits: Optional[List[int]] = None
    state: Optional[np.ndarray] = None
    final_state: Optional[np.ndarray] = None
    final_state_kind: Optional[str] = None
    expectation_values: Dict[str, float] = field(default_factory=dict)
    expectation_variances: Dict[str, float] = field(default_factory=dict)
    snapshot_states: Dict[int, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    # ---- 测量结果解析 ----
    def _resolve(self, target: Union[int, str]) -> int:
        if isinstance(target, str):
            for spec in self.measurement_specs:
                if spec.id == target:
                    return spec.op_index
            raise ValueError(f"未找到 id={target!r} 的 measure 操作")
        return int(target)

    def output(self, target: Union[int, str]):
        if target == -1:
            if self.terminal_output is None:
                raise ValueError("未执行末端测量：output(-1) 不可用（tm=False / measure_qubits=[] / shots∈{None,0}）")
            return self.terminal_output
        idx = self._resolve(target)
        if idx not in self.incircuit_outputs:
            raise ValueError(f"操作下标 {idx} 不是线路内 measure 操作")
        return self.incircuit_outputs[idx]

    def counts(self, target: Union[int, str]):
        if self.shots is None:
            raise RuntimeError("单轨迹模式（shots=None/0）不支持统计结果")
        if target == -1:
            if self.terminal_counts is None:
                raise ValueError("未执行末端测量：counts(-1) 不可用")
            return dict(self.terminal_counts)
        idx = self._resolve(target)
        if idx not in self.incircuit_counts:
            raise ValueError(f"操作下标 {idx} 不是线路内 measure 操作")
        return dict(self.incircuit_counts[idx])

    def prob(self, target: Union[int, str]):
        counts = self.counts(target)
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    # ---- 末态 / 约化 / 中间态 ----
    def snap(self, op_index: int) -> Optional[np.ndarray]:
        s = self.snapshot_states.get(int(op_index))
        return None if s is None else np.array(s, copy=True)

    def reduce(self, R: Sequence[int], pos: str = "final") -> np.ndarray:
        src = self.final_state if pos == "final" else self.state
        if src is None:
            raise ValueError(f"{pos} 态不可用，无法 reduce")
        arr = np.asarray(src)
        if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:  # 态矢 → 先转 DM
            vec = arr.reshape(-1, 1)
            arr = vec @ vec.conj().T
        return _reduced_density(arr, self.n_qubits, list(R))

    def most_probable(self):
        idx = int(np.argmax(self.probabilities))
        return f"|{idx:0{self.n_qubits}b}>", float(self.probabilities[idx])

    def variance(self, name: str) -> Optional[float]:
        return None if name not in self.expectation_variances else float(self.expectation_variances[name])

    def stddev(self, name: str) -> Optional[float]:
        var = self.variance(name)
        return None if var is None else float(np.sqrt(max(var, 0.0)))

    def summary(self) -> str:
        peak, p = self.most_probable()
        lines = [f"Result(n_qubits={self.n_qubits}, backend={self.backend_name})", f"peak={peak}, prob={p:.6f}"]
        if self.shots is not None:
            lines.append(f"shots={self.shots}")
        if self.expectation_values:
            lines.append(f"expectations={self.expectation_values}")
        return " | ".join(lines)

    def __repr__(self) -> str:
        return self.summary()
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_result_api.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/result.py tests/measure/test_result_api.py
git commit -m "feat(measure): 方法式 Result（output/counts/prob/snap/reduce + registry）"
```

---

## Task 10: `aggregate.py` — 把轨迹折叠为 `Result` 字段（avg）

**Files:**
- Create: `aicir/measure/aggregate.py`
- Test: `tests/measure/test_aggregate.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from aicir.core.state import State
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.measure.trajectory import TrajectoryResult
from aicir.measure.aggregate import aggregate_avg


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_aggregate_two_shots_outputs_stacked_and_state_is_density():
    t0 = TrajectoryResult(pre=_sv([1, 0]), post=_sv([1, 0]), incircuit={1: 1}, terminal=[1])
    t1 = TrajectoryResult(pre=_sv([0, 1]), post=_sv([0, 1]), incircuit={1: -1}, terminal=[-1])
    agg = aggregate_avg([t0, t1], n_qubits=1, measurement_specs=[], terminal_qubits=[0])
    # state 为平均密度矩阵 I/2
    assert agg["state"].shape == (2, 2)
    assert np.allclose(agg["state"], np.array([[0.5, 0], [0, 0.5]]), atol=1e-6)
    assert agg["incircuit_outputs"][1].shape == (2, 1)
    assert agg["terminal_output"].shape == (2, 1)
    assert agg["incircuit_counts"][1] == {1: 1, -1: 1}
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_aggregate.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

创建 `aicir/measure/aggregate.py`：

```python
"""sm="avg" 聚合：把多条 TrajectoryResult 折叠为 Result 的字段字典。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _as_density(state) -> np.ndarray:
    arr = np.asarray(state.to_numpy())
    if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:
        vec = arr.reshape(-1, 1)
        return vec @ vec.conj().T
    return arr


def _bitstr(bits: Sequence[int]) -> str:
    return "".join("0" if b == 1 else "1" for b in bits)  # +1→0, -1→1


def aggregate_avg(trajectories, n_qubits: int, measurement_specs,
                  terminal_qubits: Optional[List[int]]) -> Dict[str, object]:
    M = len(trajectories)
    dim = 1 << n_qubits

    # 平均密度矩阵
    pre_sum = np.zeros((dim, dim), dtype=complex)
    post_sum = np.zeros((dim, dim), dtype=complex)
    for tr in trajectories:
        pre_sum += _as_density(tr.pre)
        post_sum += _as_density(tr.post)
    state = pre_sum / M
    final_state = post_sum / M

    # 线路内 measure 输出 (M,1) 与计数
    incircuit_outputs: Dict[int, np.ndarray] = {}
    incircuit_counts: Dict[int, Dict[int, int]] = {}
    op_indices = [s.op_index for s in measurement_specs]
    for op in op_indices:
        col = np.array([[int(tr.incircuit[op])] for tr in trajectories], dtype=int)
        incircuit_outputs[op] = col
        c: Dict[int, int] = {}
        for tr in trajectories:
            lam = int(tr.incircuit[op]); c[lam] = c.get(lam, 0) + 1
        incircuit_counts[op] = c

    # 末端输出 (M,k) 与计数
    terminal_output = None
    terminal_counts = None
    if terminal_qubits is not None and trajectories and trajectories[0].terminal is not None:
        k = len(terminal_qubits)
        terminal_output = np.array([tr.terminal for tr in trajectories], dtype=int).reshape(M, k)
        terminal_counts = {}
        for tr in trajectories:
            key = _bitstr(tr.terminal)
            terminal_counts[key] = terminal_counts.get(key, 0) + 1

    # snap 平均
    snap_states: Dict[int, np.ndarray] = {}
    snap_keys = set().union(*[set(tr.snaps) for tr in trajectories]) if trajectories else set()
    for t in snap_keys:
        snap_states[t] = sum(_as_density(tr.snaps[t]) for tr in trajectories) / M

    # 概率分布（取平均 ρ_pre 对角）
    probabilities = np.real(np.diag(state)).astype(np.float64)

    return {
        "state": state,
        "final_state": final_state,
        "final_state_kind": "density_matrix",
        "incircuit_outputs": incircuit_outputs,
        "incircuit_counts": incircuit_counts,
        "terminal_output": terminal_output,
        "terminal_counts": terminal_counts,
        "snapshot_states": snap_states,
        "probabilities": probabilities,
    }
```

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_aggregate.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/aggregate.py tests/measure/test_aggregate.py
git commit -m "feat(measure): sm=avg 轨迹聚合"
```

---

## Task 11: `measure.py` — `Measure.run` 编排（签名 + 校验 + shot 策略 + 装配）

**Files:**
- Rewrite: `aicir/measure/measure.py`
- Test: `tests/measure/test_unified_run.py` (create)

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot, measure, reset


def run(cir, **kw):
    return Measure(NumpyBackend()).run(cir, **kw)


def test_shots_none_no_terminal_measurement_state_equals_final():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=None)
    assert np.allclose(r.final_state, r.state, atol=1e-6)
    with pytest.raises(ValueError):
        r.output(-1)


def test_shots0_alias_none():
    cir = Circuit(hadamard(0), n_qubits=1)
    r = run(cir, shots=0)
    assert np.allclose(r.final_state, r.state, atol=1e-6)


def test_shots_m_terminal_shapes_and_density_state():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=64)
    assert r.output(-1).shape == (64, 2)
    assert np.asarray(r.state).shape == (4, 4)        # shots>1 → 密度矩阵
    counts = r.counts(-1)
    assert set(counts) <= {"00", "11"}


def test_incircuit_measure_collapses_and_output_indexed():
    cir = Circuit(hadamard(0), cnot(1, [0]), measure(0, 1), n_qubits=2)  # op2 = measure
    r = run(cir, shots=16, tm=False)
    assert r.output(2).shape == (16, 1)
    assert set(np.unique(r.output(2))) <= {1}          # Bell ZZ 恒 +1


def test_terminal_order_preserved_in_output():
    cir = Circuit(pauli_x_op := __import__("aicir").pauli_x(1), n_qubits=2)
    r = run(cir, shots=1, measure_qubits=[1, 0])
    # qubit1=|1> → z1=-1; qubit0=|0> → z0=+1
    assert r.output(-1).tolist() == [[-1, 1]]


def test_conflict_tm_false_with_measure_qubits():
    cir = Circuit(hadamard(0), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, tm=False, measure_qubits=[0])


def test_conflict_exact_mode_with_explicit_measure_qubits():
    cir = Circuit(hadamard(0), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=None, measure_qubits=[0])


def test_invalid_shots():
    cir = Circuit(hadamard(0), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=-1)
    with pytest.raises(ValueError):
        run(cir, shots=1.5)


def test_seed_reproducible():
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    a = run(cir, shots=32, seed=7, tm=False).output(1)
    b = run(cir, shots=32, seed=7, tm=False).output(1)
    assert np.array_equal(a, b)


def test_duplicate_id_raises():
    cir = Circuit(measure(0, id="m"), measure(0, id="m"), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=4, tm=False)


def test_sm_shot_not_implemented():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    with pytest.raises(NotImplementedError):
        run(cir, shots=4, snap=[0], sm="shot")
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py -q`
Expected: FAIL（旧 `run` 签名/语义不符）。

- [ ] **Step 3: 实现**

重写 `aicir/measure/measure.py`。核心 `Measure` 类（保留 `__init__`、`_resolve_backend`、`_build_initial_state`/`_build_initial_density_matrix`、`observables`→`expectation_values` 计算等可复用片段；删除 `run_density_matrix`/`run_batch`/`_resolve_post_measurement`/`_marginal_counts`/`_collapse_*`/`_reduced_*`/`_assert_reset_allowed`/旧 `_readout_qubits`/`_resolve_readout` 互斥逻辑）：

```python
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..core.state import State
from ..ir import circuit_instructions, has_circuit_instructions, instruction_name, instruction_qubits
from .result import Result, MeasureSpec
from .trajectory import run_trajectory
from .aggregate import aggregate_avg


def _is_measure(g): return instruction_name(g).lower() in {"measure", "measurement"}
def _is_reset(g): return instruction_name(g).lower() == "reset"


class Measure:
    def __init__(self, backend):
        self.backend = backend

    def _resolve_backend(self, circuit):
        return getattr(circuit, "backend", None) or self.backend

    # ---------- 校验 ----------
    @staticmethod
    def _validate_shots(shots):
        if shots is None or shots == 0:
            return None  # exact 模式
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 0:
            raise ValueError(f"shots 必须是正整数、0 或 None，收到 {shots!r}")
        return int(shots)

    def _collect_specs(self, circuit, n) -> List[MeasureSpec]:
        specs: List[MeasureSpec] = []
        ids = set()
        for op_index, gate in enumerate(circuit_instructions(circuit)):
            qs = [int(q) for q in instruction_qubits(gate)] or list(range(n))
            for q in qs:
                if q < 0 or q >= n:
                    raise ValueError(f"{instruction_name(gate)} 含越界比特 {q}（n={n}）")
            if len(set(qs)) != len(qs):
                raise ValueError(f"{instruction_name(gate)} 含重复比特：{qs}")
            if _is_measure(gate):
                mid = gate.get("id") if isinstance(gate, dict) else None
                if mid is not None:
                    if mid in ids:
                        raise ValueError(f"重复的 measure id={mid!r}")
                    ids.add(mid)
                basis = gate.get("basis", "Z") if isinstance(gate, dict) else "Z"
                specs.append(MeasureSpec(op_index=op_index, id=mid, qubits=qs, basis=str(basis)))
        return specs

    @staticmethod
    def _normalize_measure_qubits(mq, n):
        if isinstance(mq, (int, np.integer)):
            mq = [int(mq)]
        out = [int(q) for q in mq]
        for q in out:
            if q < 0 or q >= n:
                raise ValueError(f"measure_qubits 含越界比特 {q}（n={n}）")
        if len(set(out)) != len(out):
            raise ValueError(f"measure_qubits 含重复比特：{out}")
        return out  # 保留输入顺序，不排序

    @staticmethod
    def _normalize_snap(snap, n_ops):
        if snap is None:
            return set()
        if isinstance(snap, (int, np.integer)) and not isinstance(snap, bool):
            snap = [snap]
        out = set()
        for t in snap:
            t = int(t)
            if t < 0 or t >= n_ops:
                raise ValueError(f"snap 含越界操作下标 {t}（操作数={n_ops}）")
            out.add(t)
        return out

    # ---------- 主入口 ----------
    def run(self, circuit, shots=1, measure_qubits=None, snap=None,
            tm=True, sm="avg", seed=None, *,
            initial_state=None, observables=None, return_state=True) -> Result:
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        n = int(circuit.n_qubits)
        if n <= 0:
            raise ValueError("n_qubits 必须为正整数")
        backend = self._resolve_backend(circuit)

        norm_shots = self._validate_shots(shots)             # None=exact
        exact = norm_shots is None
        n_ops = len(list(circuit_instructions(circuit))) if has_circuit_instructions(circuit) else 0
        specs = self._collect_specs(circuit, n)
        snap_ops = self._normalize_snap(snap, n_ops)

        if sm not in ("avg", "shot", "cond"):
            raise ValueError(f"sm 必须是 avg/shot/cond，收到 {sm!r}")
        if sm in ("shot", "cond") and snap_ops:
            raise NotImplementedError(f"sm={sm!r} 暂未实现（仅支持 avg）")

        # 末端测量解析（exact 模式覆盖 tm；与显式 measure_qubits 冲突报错）
        mq_explicit = measure_qubits is not None
        if mq_explicit:
            norm_mq = self._normalize_measure_qubits(measure_qubits, n)
        else:
            norm_mq = None
        if not tm and mq_explicit and len(norm_mq) > 0:
            raise ValueError("tm=False 与非空 measure_qubits 冲突")
        if exact and mq_explicit and len(norm_mq) > 0:
            raise ValueError("shots∈{None,0}（exact 模式）覆盖 tm、不做末端测量，"
                             "与显式 measure_qubits 冲突；如需末端读出请用 shots≥1")

        do_terminal = tm and not exact and not (mq_explicit and len(norm_mq) == 0)
        terminal_qubits = (norm_mq if (mq_explicit and len(norm_mq) > 0) else list(range(n))) if do_terminal else None

        noise_model = getattr(circuit, "noise_model", None)
        seed_seq = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()

        def fresh_state():
            if initial_state is None:
                return State.zero_state(n, backend)
            if isinstance(initial_state, State):
                return initial_state
            return State(initial_state, n, backend)

        has_incircuit = any(True for g in circuit_instructions(circuit) if _is_measure(g)) if n_ops else False
        M = 1 if exact else norm_shots

        rng = np.random.default_rng(seed_seq)
        trajectories = []
        if has_incircuit or noise_model is not None:
            for _ in range(M):
                trajectories.append(run_trajectory(
                    circuit, fresh_state(), backend, tm=do_terminal,
                    measure_qubits=terminal_qubits, snap_ops=snap_ops, rng=rng, noise_model=noise_model))
        else:
            # 无线路中途随机源：ρ_pre 算一次，末端采样 M 次
            base = run_trajectory(circuit, fresh_state(), backend, tm=False,
                                  measure_qubits=None, snap_ops=snap_ops, rng=rng, noise_model=None)
            from .projector import terminal_z_measure
            for _ in range(M):
                if do_terminal:
                    post, terminal = terminal_z_measure(base.pre, terminal_qubits, rng)
                else:
                    post, terminal = base.pre, None
                trajectories.append(type(base)(pre=base.pre, post=post, incircuit={},
                                               terminal=terminal, snaps=base.snaps))

        result = self._build_result(trajectories, n, backend, norm_shots, exact, specs,
                                     terminal_qubits, do_terminal, observables, return_state)
        return result

    def _build_result(self, trajectories, n, backend, norm_shots, exact, specs,
                      terminal_qubits, do_terminal, observables, return_state) -> Result:
        if exact or norm_shots == 1:
            tr = trajectories[0]
            state = np.asarray(tr.pre.to_numpy())
            final = np.asarray(tr.post.to_numpy())
            incircuit_outputs = ({op: tr.incircuit[op] for op in (s.op_index for s in specs)}
                                 if exact else
                                 {op: np.array([[tr.incircuit[op]]]) for op in (s.op_index for s in specs)})
            terminal_output = None
            if do_terminal and tr.terminal is not None:
                terminal_output = (np.array(tr.terminal) if exact
                                   else np.array(tr.terminal).reshape(1, -1))
            snap_states = {t: np.asarray(s.to_numpy()) for t, s in tr.snaps.items()}
            probabilities = np.asarray(tr.pre.probabilities()).reshape(-1).astype(np.float64) \
                if hasattr(tr.pre, "probabilities") else np.abs(state.reshape(-1)) ** 2
            incircuit_counts = {}
            terminal_counts = None
            if not exact:  # shots=1 仍可统计
                incircuit_counts = {op: {int(tr.incircuit[op]): 1} for op in (s.op_index for s in specs)}
                if terminal_output is not None:
                    key = "".join("0" if b == 1 else "1" for b in tr.terminal)
                    terminal_counts = {key: 1}
        else:
            agg = aggregate_avg(trajectories, n, specs, terminal_qubits if do_terminal else None)
            state = agg["state"]; final = agg["final_state"]
            incircuit_outputs = agg["incircuit_outputs"]; incircuit_counts = agg["incircuit_counts"]
            terminal_output = agg["terminal_output"]; terminal_counts = agg["terminal_counts"]
            snap_states = agg["snapshot_states"]; probabilities = agg["probabilities"]

        exp_vals, exp_vars = {}, {}
        if observables:
            rho = state if (np.asarray(state).ndim == 2 and state.shape[0] == state.shape[1]) else None
            vec = None if rho is not None else np.asarray(state).reshape(-1, 1)
            for name, op in observables.items():
                op = np.asarray(op)
                if rho is not None:
                    exp_vals[name] = float(np.real(np.trace(rho @ op)))
                else:
                    exp_vals[name] = float(np.real((vec.conj().T @ op @ vec)[0, 0]))

        return Result(
            n_qubits=n, backend_name=type(backend).__name__,
            probabilities=probabilities, shots=norm_shots,
            measurement_specs=specs, incircuit_outputs=incircuit_outputs,
            incircuit_counts=incircuit_counts, terminal_output=terminal_output,
            terminal_counts=terminal_counts, terminal_qubits=terminal_qubits,
            state=(state if return_state else None),
            final_state=(final if return_state else None),
            final_state_kind=("density_matrix" if np.asarray(final).ndim == 2 and final.shape[0] == final.shape[1] else "state_vector") if return_state else None,
            expectation_values=exp_vals, expectation_variances=exp_vars,
            snapshot_states=snap_states,
        )
```

> 边界处理：`TrajectoryResult` 在优化路径里通过 `type(base)(...)` 重建——确保 `trajectory.py` 中 `TrajectoryResult` 字段顺序与此调用一致（`pre, post, incircuit, terminal, snaps`）。
> `exact` 模式 `output(i)` 返回标量（`tr.incircuit[op]`），`shots≥1` 返回 `(M,1)` 数组——与 README §4.1 形状表一致。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py tests/measure -q`
Expected: PASS（新建测试全过；旧 `tests/measure/test_measure*.py` 可能失败，将在 Task 12 起逐个迁移）。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/measure.py tests/measure/test_unified_run.py
git commit -m "feat(measure): Measure.run 统一测量模型编排（轨迹引擎+校验+聚合）"
```

---

## Task 12: 迁移 `estimator.py` 与 `primitives`（counts → counts(-1)）

**Files:**
- Modify: `aicir/measure/estimator.py:422-429`
- Modify: `aicir/primitives/sampler.py:16-33,52`
- Modify: `aicir/primitives/estimator.py`
- Test: `tests/primitives/test_primitives.py`、`tests/measure/test_estimator.py`（更新）

- [ ] **Step 1: 跑现状，定位失败**

Run: `PYTHONPATH=. pytest tests/primitives/test_primitives.py tests/measure/test_estimator.py -q`
Expected: FAIL（`result.counts` 现为方法、且语义变化）。

- [ ] **Step 2: 改 `primitives/sampler.py`**

`sampler.py` 第 16-33 行读取 `result.counts`（字段）→ 改为 `result.counts(-1)`，并对 `shots is None` 兜底（exact 模式无 counts）：

```python
    measured = result.metadata.get("measured_qubits")
    if measured is None:
        measured = tuple(result.terminal_qubits or range(int(result.n_qubits)))
    width = int(round(math.log2(len(result.probabilities)))) if len(result.probabilities) else 0
    probs = {f"|{i:0{width}b}>": float(p) for i, p in enumerate(result.probabilities)}
    try:
        counts = dict(result.counts(-1)) if result.shots else {}
    except (ValueError, RuntimeError):
        counts = {}
    # counts 的 key 现为不含 | > 的比特串；若下游需 |..> 形式，这里统一加壳：
    counts = {f"|{k}>": v for k, v in counts.items()}
```

`sampler.py` 第 52 行的 `measure.run(...)`：确保传入电路**含末端比特**（默认 `measure_qubits=None` 读取全部）；若该 primitive 之前依赖内嵌 `measure()` 门做边缘采样，改为 `measure_qubits=None, tm=True`，并用 `counts(-1)`。

- [ ] **Step 3: 改 `measure/estimator.py`**

第 422-429 行：`result = measure.run(...)` 后 `counts = dict(result.counts or {})` → 改为：

```python
            result = measure.run(circuit, shots=shots, measure_qubits=None)
            counts = {f"|{k}>": v for k, v in result.counts(-1).items()}
```

并核对该函数后续对 counts key 格式（`|bits>` vs `bits`）的依赖，统一为 `|bits>`。

- [ ] **Step 4: 改 `primitives/estimator.py`**

审计其对 `result` 的读取：若读 `expectation_values` → 不变；若读 `counts`/`probabilities` → 同上改为 `counts(-1)` / 保留 `probabilities`。按实际实现对齐（运行测试驱动）。

- [ ] **Step 5: 更新对应测试并跑通过**

更新 `tests/primitives/test_primitives.py`、`tests/measure/test_estimator.py` 中对 `result.counts`（字段）/`result.output`（字段）的旧断言为方法式，并使内嵌 `measure()` 的样例符合"投影坍缩 + counts(-1)"新语义。

Run: `PYTHONPATH=. pytest tests/primitives/test_primitives.py tests/measure/test_estimator.py -q`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add aicir/measure/estimator.py aicir/primitives/sampler.py aicir/primitives/estimator.py \
        tests/primitives/test_primitives.py tests/measure/test_estimator.py
git commit -m "refactor: estimator/primitives 迁移到 counts(-1) 与统一测量模型"
```

---

## Task 13: 迁移 VQA 家族与其余 `.run()`/`unitary()` 调用方

**Files:**
- Modify: `aicir/vqc/VQE.py`、`QAOA.py`、`SSVQE.py`、`VQD.py`
- Modify: `aicir/optimization/qubo/qaoa.py`、`aicir/transpile/passmanager.py`、`aicir/optimizer/circuit.py`、`aicir/qas/demos/_np_ising_utils.py`
- Test: `tests/vqc/test_vqe_orchestration.py`、`tests/transpile/test_validate_pass.py`

- [ ] **Step 1: 跑现状定位**

Run: `PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py tests/transpile/test_validate_pass.py -q`
Expected: FAIL/ERROR 列表，逐项处理。

- [ ] **Step 2: VQA 家族**

`VQE.py` 用 `measurement.expectation_values[...]`（字段保留）→ 通常无需改；核对 `QAOA.py`/`SSVQE.py`/`VQD.py` 是否读 `result.counts`/`result.output`（字段）或依赖 `result.state` 为态矢（`shots>1` 现为 DM）。凡 `shots>1` 且需要态矢的地方，改为 `shots=None`（exact）或显式 `final_state_kind` 分支。

- [ ] **Step 3: `unitary()` 调用方**

`aicir/optimizer/circuit.py`、`aicir/transpile/passmanager.py`、`aicir/optimization/qubo/qaoa.py`、`aicir/qas/demos/_np_ising_utils.py`：搜索 `.unitary(`，对**可能含 measure/reset** 的电路传 `ignore_nonunitary=True`；纯电路不变。

```bash
PYTHONPATH=. python - <<'PY'
import subprocess
print(subprocess.run(["grep","-rn","--include=*.py",".unitary(","aicir"],capture_output=True,text=True).stdout)
PY
```

- [ ] **Step 4: 更新测试并跑通过**

更新相关测试的断言到新 API。

Run: `PYTHONPATH=. pytest tests/vqc tests/transpile tests/optimization -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/vqc aicir/optimization aicir/transpile aicir/optimizer aicir/qas tests/vqc tests/transpile
git commit -m "refactor: VQA 家族与 unitary 调用方适配统一测量模型"
```

---

## Task 14: IO — JSON `basis`/`id` 往返；QASM 对不可表达 measure 报错

**Files:**
- Modify: `aicir/core/io/` 下 QASM 导出（定位含 measure 导出的文件）、JSON 序列化
- Test: `tests/circuit/io/test_qiskit_interop.py`、`test_wuyue_interop.py`、`tests/core/io/test_json_roundtrip.py`（若存在；否则在 `tests/circuit/io/` 新建 `test_measure_io.py`）

- [ ] **Step 1: 写失败测试**

新建 `tests/circuit/io/test_measure_io.py`：

```python
import pytest
from aicir import Circuit, measure, hadamard
from aicir.core.io import circuit_to_json, json_to_circuit  # 按实际导出名调整


def test_json_round_trip_preserves_basis_id():
    cir = Circuit(hadamard(0), measure(0, 1, basis="X", id="m0"), n_qubits=2)
    back = json_to_circuit(circuit_to_json(cir))
    g = back.gates[-1]
    assert g["basis"] == "X" and g["id"] == "m0"


def test_qasm_export_raises_on_joint_measure():
    from aicir.core.io import circuit_to_qasm  # 按实际导出名调整
    cir = Circuit(hadamard(0), measure(0, 1, basis="X"), n_qubits=2)
    with pytest.raises(NotImplementedError):
        circuit_to_qasm(cir)
```

- [ ] **Step 2: 跑失败**

Run: `PYTHONPATH=. pytest tests/circuit/io/test_measure_io.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

- JSON：序列化基于 gate 字典，已自动包含 `basis`/`id`（Task 1）——确认 `circuit_to_json`/`json_to_circuit` 不丢弃未知键；若有白名单需补上 `basis`/`id`。
- QASM 导出：在写 measure 处加判定——若 `len(qubits)>1` 或 `gate.get("basis","Z")!="Z"` 或 `gate.get("id") is not None` → `raise NotImplementedError("联合/非Z/带id 的 measure 无法导出标准 QASM；请用 JSON")`；普通单比特 Z measure 仍正常导出。

- [ ] **Step 4: 跑通过**

Run: `PYTHONPATH=. pytest tests/circuit/io/test_measure_io.py tests/circuit/io -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add aicir/core/io tests/circuit/io/test_measure_io.py
git commit -m "feat(io): JSON 往返 basis/id；QASM 对不可表达 measure 报错"
```

---

## Task 15: 更新 demos

**Files:**
- Modify: `demos/demo_1.py`、`demos/grover_demo.py`、`demos/qft_demo.py`、`demos/reset_demo.py`、`demos/snap_demo.py`、`demos/test_1.py`、`demos/demo_npu.py`、`demos/BeH2/BeH2_npu.py`、`demos/H2O/H2O_npu.py`

- [ ] **Step 1: 逐个运行，定位破坏**

```bash
for f in demos/demo_1.py demos/grover_demo.py demos/qft_demo.py demos/reset_demo.py demos/snap_demo.py demos/test_1.py; do
  echo "== $f =="; PYTHONPATH=. python "$f" || true
done
```

- [ ] **Step 2: 修正调用**

- `result.output` → `result.output(i)`/`result.output(-1)`；`result.counts` → `result.counts(-1)`。
- `reset_demo.py`：去掉"reset 前必须 measure"的旧约束示例；展示无前置 reset、以及 reset 纠缠比特得到混合态。
- `snap_demo.py`：`result.snap(t)` 的 `t` 现为**操作下标**（含 measure/reset），按新语义改示例与注释。
- 内嵌 `measure()` 的 demo：说明现在是投影坍缩。

- [ ] **Step 3: 验证**

```bash
for f in demos/demo_1.py demos/grover_demo.py demos/qft_demo.py demos/reset_demo.py demos/snap_demo.py demos/test_1.py; do
  echo "== $f =="; PYTHONPATH=. python "$f" >/dev/null && echo OK || echo FAIL
done
```
Expected: 全 OK（NPU demos 需硬件，跳过实际运行，仅静态核对 API）。

- [ ] **Step 4: 提交**

```bash
git add demos
git commit -m "docs(demos): 适配统一测量模型（方法式输出/投影 measure/无前置 reset）"
```

---

## Task 16: README §4 整节重构 + CHANGELOG + 全测试绿

**Files:**
- Modify: `README.md:527-740`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 重写 README §4**

按设计文档 §7.1 的新结构替换 `README.md` 第 527-740 行：4 量子测量（统一模型导言）/ 4.1 运行接口与参数 / 4.2 线路与操作序列约定 / 4.3 线路内 measure（联合 Pauli 投影测量，basis/id，output(i)/counts(i)/prob(i)，Bell→ZZ 示例）/ 4.4 reset 重置信道 / 4.5 末端测量（tm、逐比特 Z、输入顺序保留、output(-1)）/ 4.6 shots 语义（None/0 exact 覆盖 tm；M 独立重跑）/ 4.7 state/final_state/snap/reduce（形状表，sm=avg，注 shot/cond 待实现）/ 4.8 期望值与 observables / 4.9 从 State 直接测量 / 4.10 Result 字段/方法速查（方法 vs 字段）/ 4.11 Sampler/Estimator primitives / 4.12 输入检查与报错。删除"机制一/机制二/互斥"全部表述。所有代码示例须可运行。

- [ ] **Step 2: 核对 README 示例可运行**

把 README §4 关键示例抽成 `tests/docs/test_readme_measure_examples.py`（doctest 风格或显式断言），运行确认。

```bash
PYTHONPATH=. pytest tests/docs/test_readme_measure_examples.py -q
```

- [ ] **Step 3: 更新 CHANGELOG**

`CHANGELOG.md` 顶部加 `2026-06-13` 条目，列出设计文档 §7.2 的全部破坏性变更（统一模型；投影 measure(basis,id)；reset 信道无前置；新增 tm/sm/seed；output/counts/prob 方法化；state/final_state 语义 + shots>1 DM；shots=0≡None 且覆盖 tm 不做末端测量；unitary() 报错 + ignore_nonunitary；QASM 报错；JSON basis/id 往返；sm shot/cond 待实现）。

- [ ] **Step 4: 全量测试 + 清理**

```bash
PYTHONPATH=. pytest -q
```
Expected: 全绿。若有遗漏的旧测试引用旧 API（`result.counts` 字段、`run_density_matrix`、`run_batch`、机制互斥报错等），逐个迁移到新 API 或删除已不适用的用例（在提交信息中说明）。

- [ ] **Step 5: 提交**

```bash
git add README.md CHANGELOG.md tests/docs
git commit -m "docs: README §4 重构为统一测量模型 + CHANGELOG"
```

---

## Self-Review 结论（写作时自检）

- **Spec coverage**：投影联合测量(Task5)、basis/id(Task1-2)、reset 信道(Task6)、unitary 报错(Task3)、末端逐比特+顺序(Task7,11)、shots None/0 覆盖 tm(Task11)、shots>1 DM(Task10-11)、output/counts/prob 方法化(Task9)、snap 操作下标+avg(Task8-11)、reduce(Task9)、seed(Task11)、校验 56-60+shots(Task11)、noise 纳入(Task8,11)、JSON/QASM(Task14)、下游迁移(Task12-13,15)、README/CHANGELOG(Task16) —— 均有对应任务。
- **Placeholder 扫描**：Task6 中 `_reset_dm` 有一处**故意标注**的占位写法，已在该任务 Step3 明确要求最终用 `out[r, c] = rho[r, c] + rho[r1, c1]`，非遗留 TODO。
- **类型/命名一致**：`projector.measure_joint_pauli/reset_channel/terminal_z_measure`、`TrajectoryResult(pre,post,incircuit,terminal,snaps)`、`Result.output/counts/prob/snap/reduce` + `MeasureSpec(op_index,id,qubits,basis)`、`aggregate_avg(...)` 在各任务间签名一致。
- **顺序**：核心(1-11)先建后切，再迁移下游(12-15)，最后文档与全绿(16)，每步可独立提交。

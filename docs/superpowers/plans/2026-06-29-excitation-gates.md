# Particle-conserving excitation gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native particle-conserving `single_excitation` (2-qubit) and `double_excitation` (4-qubit) gates, a correct four-term parameter-shift rule for them, and an H2 VQE validation proving they create electron correlation.

**Architecture:** Each gate follows the existing `rxx`/`rzz` pattern — a numpy matrix builder + a graph-preserving torch backend kernel in `aicir/core/gates.py`, a factory in `aicir/core/circuit.py`, a `GateSpec` in `aicir/gates/registry.py`, and a top-level re-export. The 4-qubit gate is placed on arbitrary qubits via the existing `_apply_local_matrix_to_state`. A standalone `psr4` four-term rule is added to `aicir/qml/deriv.py`, plus a `GateSpec.shift_rule` metadata field. Validation runs VQE on the 4-qubit JW H2 preset.

**Tech Stack:** Python, NumPy, PyTorch (autograd backend; gate kernels are torch-graph-preserving), pytest. Run from repo root with `PYTHONPATH=.`.

## Global Constraints

- Gates are **general/reusable** — not chemistry-tagged. This plan does NOT touch supernet, adds no chemistry HF helper, and wires nothing into any search pool.
- Both gates conserve particle number (commute with `N = Σ(I−Z_i)/2`).
- Gate names: `single_excitation` (alias `givens`), `double_excitation`. Two-word names, no `xxx_yyy_zzz`.
- `generator=None` for both (their generators aren't single Pauli strings) → not auto-classified as standard 2-term-PSR.
- `qasm_name=None` for both (no standard OpenQASM excitation gate; QASM export out of scope).
- The torch kernels MUST build each grad-bearing complex cell as a **fresh** tensor (the `_rxx_backend` rule) so Ascend NPU's real-only autograd add works; real `cos`/`sin` may be shared, constants (`z`/`one`) may be reused.
- Four-term-rule coefficients are **verified against autograd** (the autograd gradient is ground truth) — the autograd-equality test is authoritative.
- Comments/docstrings in Chinese to match surrounding code.
- Run tests from repo root with `PYTHONPATH=.`.
- Out of scope: supernet hook, chemistry helper, pool wiring, four-term auto-dispatch into VQE/QAS, NEXT.md §7 `GateSpec.matrix` migration.

Spec: `docs/superpowers/specs/2026-06-29-excitation-gates-design.md`

H2 reference (verified): `h2_jw` is 4-qubit, 15 terms, exact ground energy `-1.8572750091552734` Ha.

---

## File Structure

- **Modify** `aicir/gates/spec.py` — add `shift_rule: str | None = None` field.
- **Modify** `aicir/gates/registry.py` — registry helper `gate_shift_rule(name)`; register the two excitation `GateSpec`s.
- **Modify** `aicir/gates/__init__.py` — export `gate_shift_rule`.
- **Modify** `aicir/ir/operation.py` — add `"single_excitation"` to `_PAIR_QUBIT_GATES`.
- **Modify** `aicir/core/gates.py` — numpy builders `_single_excitation`/`_double_excitation`, torch kernels `_single_excitation_backend`/`_double_excitation_backend`, and dispatch branches in `apply_gate_to_state` + both `gate_to_matrix` paths.
- **Modify** `aicir/core/circuit.py` — factories `single_excitation`/`double_excitation`.
- **Modify** `aicir/__init__.py` — re-export both factories.
- **Modify** `aicir/qml/deriv.py` — `psr4` four-term rule.
- **Create** `tests/gates/test_excitation_gates.py`, `tests/qml/test_psr4.py`, `tests/test_h2_excitation_vqe.py`.

---

## Task 1: `GateSpec.shift_rule` metadata field

**Files:**
- Modify: `aicir/gates/spec.py` (GateSpec dataclass + `__post_init__`)
- Modify: `aicir/gates/registry.py` (add `gate_shift_rule` helper)
- Modify: `aicir/gates/__init__.py` (export it)
- Test: `tests/gates/test_excitation_gates.py`

**Interfaces:**
- Produces: `GateSpec.shift_rule: str | None = None`; `gate_shift_rule(name: str) -> str | None` returning the canonical gate's `shift_rule` (or `None` if unregistered).

- [ ] **Step 1: Write the failing test**

Create `tests/gates/test_excitation_gates.py`:

```python
"""粒子数守恒激发门 + shift_rule 元数据测试。"""

from aicir.gates import get_gate_spec, gate_shift_rule


def test_shift_rule_defaults_to_none():
    assert get_gate_spec("rx").shift_rule is None
    assert get_gate_spec("rzz").shift_rule is None


def test_gate_shift_rule_helper_reads_registry():
    # 未注册门返回 None；标准门 None
    assert gate_shift_rule("rx") is None
    assert gate_shift_rule("not_a_gate") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: FAIL — `ImportError: cannot import name 'gate_shift_rule'`.

- [ ] **Step 3: Add the field to `GateSpec`**

In `aicir/gates/spec.py`, add after the `generator` field:

```python
    generator: str | None = None
    shift_rule: str | None = None
    decomposition: Callable[..., Any] | None = None
```

Add a docstring line in the attribute list (after the `generator` entry):

```python
    - ``shift_rule``：参数移位规则类别（``"two_term"`` / ``"four_term"``）；
      ``None`` 表示未指定（标准 Pauli 旋转走默认两项规则）。供后续按门选择移位规则。
```

In `__post_init__`, after the `num_params` check, add:

```python
        if self.shift_rule is not None and self.shift_rule not in ("two_term", "four_term"):
            raise ValueError("shift_rule must be 'two_term', 'four_term', or None")
```

- [ ] **Step 4: Add the registry helper**

In `aicir/gates/registry.py`, after `gate_generator`, add:

```python
def gate_shift_rule(name: str) -> str | None:
    """返回该门的参数移位规则类别（``two_term``/``four_term``/``None``）。"""
    spec = get_gate_spec(name)
    return spec.shift_rule if spec is not None else None
```

In `aicir/gates/__init__.py`, add `gate_shift_rule` to the imports from `.registry` and to `__all__`.

- [ ] **Step 5: Run tests + full gates suite**

Run: `PYTHONPATH=. pytest tests/gates/ -q`
Expected: PASS (new tests pass; no regressions).

- [ ] **Step 6: Commit**

```bash
git add aicir/gates/spec.py aicir/gates/registry.py aicir/gates/__init__.py tests/gates/test_excitation_gates.py
git commit -m "feat(gates): add GateSpec.shift_rule metadata + gate_shift_rule helper"
```

---

## Task 2: `single_excitation` gate

**Files:**
- Modify: `aicir/core/gates.py` (numpy `_single_excitation`, torch `_single_excitation_backend`, dispatch in `apply_gate_to_state` and both `gate_to_matrix` paths)
- Modify: `aicir/core/circuit.py` (factory)
- Modify: `aicir/ir/operation.py` (`_PAIR_QUBIT_GATES`)
- Modify: `aicir/gates/registry.py` (GateSpec)
- Modify: `aicir/__init__.py` (re-export)
- Test: `tests/gates/test_excitation_gates.py`

**Interfaces:**
- Consumes: `GateSpec.shift_rule` (Task 1).
- Produces: `single_excitation(theta, qubit_1=0, qubit_2=1) -> Operation`; gate dict `{"type":"single_excitation","qubit_1":..,"qubit_2":..,"parameter":theta}`; numpy `_single_excitation(theta)` 4×4. Alias `givens`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gates/test_excitation_gates.py`:

```python
import math
import numpy as np
from aicir import single_excitation, NumpyBackend
from aicir.core.gates import gate_to_matrix
from aicir.core.circuit import Circuit


def _num_op(n):
    # 粒子数算符 N = sum_i (I - Z_i)/2，对角线为各基态的 popcount
    return np.diag([bin(i).count("1") for i in range(1 << n)]).astype(complex)


def test_single_excitation_matrix_is_real_givens():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 0.7}, cir_qubits=2, backend=NumpyBackend())
    c, s = math.cos(0.35), math.sin(0.35)
    expected = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=complex)
    assert np.allclose(m, expected)


def test_single_excitation_is_unitary():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 1.3}, cir_qubits=2, backend=NumpyBackend())
    assert np.allclose(m.conj().T @ m, np.eye(4))


def test_single_excitation_conserves_particle_number():
    m = gate_to_matrix({"type": "single_excitation", "qubit_1": 0, "qubit_2": 1,
                        "parameter": 0.9}, cir_qubits=2, backend=NumpyBackend())
    N = _num_op(2)
    assert np.allclose(m.conj().T @ N @ m, N)  # [G, N] = 0


def test_givens_alias_and_factory():
    from aicir.gates import canonical_gate_name
    assert canonical_gate_name("givens") == "single_excitation"
    op = single_excitation(0.5, qubit_1=1, qubit_2=2)
    d = op.to_dict()
    assert d["type"] == "single_excitation" and d["qubit_1"] == 1 and d["qubit_2"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: FAIL — `ImportError` for `single_excitation` / unsupported gate type.

- [ ] **Step 3: Add the numpy matrix builder**

In `aicir/core/gates.py`, after `_rxx`, add:

```python
def _single_excitation(theta, qubit_1=0, qubit_2=1):
    _ = (qubit_1, qubit_2)
    t = float(theta)
    c = math.cos(t / 2.0)
    s = math.sin(t / 2.0)
    return np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, c, -s, 0.0 + 0.0j],
            [0.0 + 0.0j, s, c, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=_CDTYPE,
    )
```

- [ ] **Step 4: Add the torch backend kernel**

After `_rxx_backend`, add:

```python
def _single_excitation_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_single_excitation(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    one_r = torch.ones((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    cos = torch.cos(t / 2.0)
    sin = torch.sin(t / 2.0)
    neg_sin = -sin
    # 每个含梯度复数 cell 必须是新张量（见 _rxx_backend 说明）；实数 cos/sin 可共享，
    # 常量 z/one 可重用。
    z = _torch_complex(zero, complex_dtype=dtype)
    one = _torch_complex(one_r, complex_dtype=dtype)
    return _torch_base_matrix(
        [
            [one, z, z, z],
            [z, _torch_complex(cos, complex_dtype=dtype), _torch_complex(neg_sin, complex_dtype=dtype), z],
            [z, _torch_complex(sin, complex_dtype=dtype), _torch_complex(cos, complex_dtype=dtype), z],
            [z, z, z, one],
        ],
        dtype,
        device,
    )
```

- [ ] **Step 5: Wire the dispatch (3 sites in `aicir/core/gates.py`)**

In `apply_gate_to_state`, immediately after the `if gate_type in {"rzz", "rxx"}:` block, add:

```python
    if gate_type == "single_excitation":
        parameter = gate.get("parameter")
        local = _single_excitation_backend(parameter, backend)
        cache_key = None if _contains_torch_tensor(parameter) else (gate_type, _parameter_cache_key(parameter))
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            [gate["qubit_1"], gate["qubit_2"]],
            n_qubits,
            backend,
        )
```

In the numpy `gate_to_matrix` path, after the `elif gate_type in {"rzz", "rxx"}:` block, add:

```python
        elif gate_type == "single_excitation":
            return _expand_local_matrix_to_full(
                _single_excitation(gate_parameter),
                [gate["qubit_1"], gate["qubit_2"]],
                int(cir_qubits),
            )
```

In the backend `gate_to_matrix` path, after its `elif gate_type in {"rzz", "rxx"}:` block, add:

```python
        elif gate_type == "single_excitation":
            return _expand_local_matrix_to_full(
                _single_excitation_backend(gate_parameter, backend),
                [gate["qubit_1"], gate["qubit_2"]],
                int(cir_qubits),
                backend=backend,
            )
```

- [ ] **Step 6: Add the factory + pair-gate serialization + GateSpec + re-export**

In `aicir/core/circuit.py`, after the `rxx` factory, add:

```python
def single_excitation(theta, qubit_1=0, qubit_2=1):
    return Operation("single_excitation", qubits=(qubit_1, qubit_2), params=(theta,))


givens = single_excitation
```

In `aicir/ir/operation.py`, change `_PAIR_QUBIT_GATES`:

```python
_PAIR_QUBIT_GATES = {"swap", "rzz", "rxx", "single_excitation"}
```

In `aicir/gates/registry.py`, after the `rxx` registration, add:

```python
    GateSpec("single_excitation", 2, 1, aliases=("givens",), qasm_name=None, shift_rule="four_term"),
```

In `aicir/__init__.py`, re-export `single_excitation` and `givens` next to `rxx`/`rzz` (imports + `__all__`).

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aicir/core/gates.py aicir/core/circuit.py aicir/ir/operation.py aicir/gates/registry.py aicir/__init__.py tests/gates/test_excitation_gates.py
git commit -m "feat(gates): add single_excitation (Givens) gate"
```

---

## Task 3: `double_excitation` gate

**Files:**
- Modify: `aicir/core/gates.py` (numpy `_double_excitation`, torch `_double_excitation_backend`, dispatch in `apply_gate_to_state` + both `gate_to_matrix` paths)
- Modify: `aicir/core/circuit.py` (factory)
- Modify: `aicir/gates/registry.py` (GateSpec)
- Modify: `aicir/__init__.py` (re-export)
- Test: `tests/gates/test_excitation_gates.py`

**Interfaces:**
- Produces: `double_excitation(theta, q1=0,q2=1,q3=2,q4=3) -> Operation`; gate dict `{"type":"double_excitation","qubits":[q1,q2,q3,q4],"parameter":theta}`; numpy `_double_excitation(theta)` 16×16 (couples basis indices 3=`|0011>` and 12=`|1100>`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/gates/test_excitation_gates.py`:

```python
from aicir import double_excitation


def test_double_excitation_matrix_couples_0011_and_1100():
    m = gate_to_matrix({"type": "double_excitation", "qubits": [0, 1, 2, 3],
                        "parameter": 0.8}, cir_qubits=4, backend=NumpyBackend())
    c, s = math.cos(0.4), math.sin(0.4)
    expected = np.eye(16, dtype=complex)
    expected[3, 3] = c
    expected[3, 12] = -s
    expected[12, 3] = s
    expected[12, 12] = c
    assert np.allclose(m, expected)


def test_double_excitation_is_unitary_and_conserves_N():
    m = gate_to_matrix({"type": "double_excitation", "qubits": [0, 1, 2, 3],
                        "parameter": 1.1}, cir_qubits=4, backend=NumpyBackend())
    assert np.allclose(m.conj().T @ m, np.eye(16))
    N = _num_op(4)
    assert np.allclose(m.conj().T @ N @ m, N)


def test_double_excitation_factory_serializes_qubits_list():
    d = double_excitation(0.3, 0, 1, 2, 3).to_dict()
    assert d["type"] == "double_excitation" and d["qubits"] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: FAIL — `ImportError` / unsupported gate type.

- [ ] **Step 3: Add the numpy builder**

In `aicir/core/gates.py`, after `_single_excitation`, add:

```python
def _double_excitation(theta, q1=0, q2=1, q3=2, q4=3):
    _ = (q1, q2, q3, q4)
    t = float(theta)
    c = math.cos(t / 2.0)
    s = math.sin(t / 2.0)
    m = np.eye(16, dtype=_CDTYPE)
    m[3, 3] = c
    m[3, 12] = -s
    m[12, 3] = s
    m[12, 12] = c
    return m
```

- [ ] **Step 4: Add the torch backend kernel**

After `_single_excitation_backend`, add:

```python
def _double_excitation_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_double_excitation(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    one_r = torch.ones((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    cos = torch.cos(t / 2.0)
    sin = torch.sin(t / 2.0)
    neg_sin = -sin
    z = _torch_complex(zero, complex_dtype=dtype)
    one = _torch_complex(one_r, complex_dtype=dtype)
    # 16×16 单位阵，仅 (3,3),(3,12),(12,3),(12,12) 为含梯度新张量（fresh cell 规则）。
    special = {
        (3, 3): lambda: _torch_complex(cos, complex_dtype=dtype),
        (3, 12): lambda: _torch_complex(neg_sin, complex_dtype=dtype),
        (12, 3): lambda: _torch_complex(sin, complex_dtype=dtype),
        (12, 12): lambda: _torch_complex(cos, complex_dtype=dtype),
    }
    rows = []
    for i in range(16):
        row = []
        for j in range(16):
            if (i, j) in special:
                row.append(special[(i, j)]())
            elif i == j:
                row.append(one)
            else:
                row.append(z)
        rows.append(row)
    return _torch_base_matrix(rows, dtype, device)
```

- [ ] **Step 5: Wire the dispatch (3 sites)**

In `apply_gate_to_state`, after the `single_excitation` block, add:

```python
    if gate_type == "double_excitation":
        parameter = gate.get("parameter")
        local = _double_excitation_backend(parameter, backend)
        cache_key = None if _contains_torch_tensor(parameter) else (gate_type, _parameter_cache_key(parameter))
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            list(gate["qubits"]),
            n_qubits,
            backend,
        )
```

In the numpy `gate_to_matrix` path, after the `single_excitation` branch:

```python
        elif gate_type == "double_excitation":
            return _expand_local_matrix_to_full(
                _double_excitation(gate_parameter),
                list(gate["qubits"]),
                int(cir_qubits),
            )
```

In the backend `gate_to_matrix` path, after its `single_excitation` branch:

```python
        elif gate_type == "double_excitation":
            return _expand_local_matrix_to_full(
                _double_excitation_backend(gate_parameter, backend),
                list(gate["qubits"]),
                int(cir_qubits),
                backend=backend,
            )
```

- [ ] **Step 6: Add factory + GateSpec + re-export**

In `aicir/core/circuit.py`, after `single_excitation`:

```python
def double_excitation(theta, qubit_1=0, qubit_2=1, qubit_3=2, qubit_4=3):
    return Operation("double_excitation", qubits=(qubit_1, qubit_2, qubit_3, qubit_4), params=(theta,))
```

In `aicir/gates/registry.py`, after the `single_excitation` registration:

```python
    GateSpec("double_excitation", 4, 1, qasm_name=None, shift_rule="four_term"),
```

In `aicir/__init__.py`, re-export `double_excitation` (imports + `__all__`).

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aicir/core/gates.py aicir/core/circuit.py aicir/gates/registry.py aicir/__init__.py tests/gates/test_excitation_gates.py
git commit -m "feat(gates): add double_excitation gate"
```

---

## Task 4: autograd-gradient check for the gates

**Files:**
- Test: `tests/gates/test_excitation_gates.py`

**Interfaces:**
- Consumes: `single_excitation`, `double_excitation` factories; `GPUBackend`.

This task adds no production code — it proves the torch kernels preserve the autograd graph (a prerequisite for the validation and a guard on Tasks 2-3). If a kernel fails to backprop, fix the kernel here.

- [ ] **Step 1: Write the autograd tests**

Append to `tests/gates/test_excitation_gates.py`:

```python
import pytest


def _energy_grad_autograd_vs_fd(circuit_fn, theta0, n_qubits):
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import GPUBackend
    from aicir.core.gates import gate_to_matrix
    backend = GPUBackend(device="cpu")

    def energy(theta_t):
        circuit = circuit_fn(theta_t)
        state = backend.zeros_state(n_qubits)
        for g in circuit.gates:
            from aicir.core.gates import apply_gate_to_state
            state = apply_gate_to_state(g, state, n_qubits, backend)
        # H = Z on qubit 0
        import numpy as np
        z0 = np.kron(np.diag([1.0, -1.0]), np.eye(1 << (n_qubits - 1)))
        H = backend.cast(z0.astype(np.complex64))
        return backend.expectation_sv(state, H)

    theta_t = torch.tensor(float(theta0), requires_grad=True)
    e = energy(theta_t)
    e.real.backward()
    grad_ad = float(theta_t.grad)

    eps = 1e-4
    ep = float(energy(torch.tensor(theta0 + eps)).real.detach())
    em = float(energy(torch.tensor(theta0 - eps)).real.detach())
    grad_fd = (ep - em) / (2 * eps)
    assert abs(grad_ad - grad_fd) < 1e-3


def test_single_excitation_autograd_grad_matches_fd():
    from aicir.core.circuit import Circuit
    _energy_grad_autograd_vs_fd(
        lambda th: Circuit(single_excitation(th, 0, 1), n_qubits=2), 0.6, 2)


def test_double_excitation_autograd_grad_matches_fd():
    from aicir.core.circuit import Circuit
    _energy_grad_autograd_vs_fd(
        lambda th: Circuit(double_excitation(th, 0, 1, 2, 3), n_qubits=4), 0.7, 4)
```

- [ ] **Step 2: Run the tests**

Run: `PYTHONPATH=. pytest tests/gates/test_excitation_gates.py -q`
Expected: PASS (if a gate starts in the all-zeros state the gradient may be 0 — that is still a valid AD-vs-FD match; the test only asserts they agree). If FAIL because grad is `None`, the kernel broke the graph — fix the kernel from Task 2/3.

- [ ] **Step 3: Commit**

```bash
git add tests/gates/test_excitation_gates.py
git commit -m "test(gates): autograd-vs-FD gradient check for excitation gates"
```

---

## Task 5: four-term parameter-shift rule `psr4`

**Files:**
- Modify: `aicir/qml/deriv.py` (add `psr4`)
- Test: `tests/qml/test_psr4.py`

**Interfaces:**
- Produces: `psr4(fn: Callable[[np.ndarray], float], params, *, shifts=(np.pi/2, 3*np.pi/2), coefficients=None) -> np.ndarray`. Default coefficients are the excitation-gate four-term rule `c1=(√2+1)/(4√2)`, `c2=(√2−1)/(4√2)`, applied as `grad = c1*(f(θ+s1)−f(θ−s1)) − c2*(f(θ+s2)−f(θ−s2))` per parameter.
- The autograd gradient is the authoritative oracle: if the default constants disagree with autograd in the test, correct the constants until they match.

- [ ] **Step 1: Write the failing test**

Create `tests/qml/test_psr4.py`:

```python
"""四项参数移位规则 psr4：对激发门梯度与 autograd 一致，两项 psr 不一致。"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir import single_excitation, double_excitation
from aicir.backends.gpu_backend import GPUBackend
from aicir.core.circuit import Circuit
from aicir.core.gates import apply_gate_to_state
from aicir.qml.deriv import psr, psr4


def _energy_fn(gate_factory, n_qubits, init_x):
    backend = GPUBackend(device="cpu")
    z0 = np.kron(np.diag([1.0, -1.0]), np.eye(1 << (n_qubits - 1))).astype(np.complex64)
    H = backend.cast(z0)

    def run(theta_value, grad=False):
        th = torch.tensor(float(theta_value[0]), requires_grad=grad)
        state = backend.zeros_state(n_qubits)
        for q in init_x:  # 预置 X 翻转，使梯度非平凡
            from aicir.core.circuit import pauli_x
            state = apply_gate_to_state(pauli_x(q).to_dict(), state, n_qubits, backend)
        for g in Circuit(gate_factory(th), n_qubits=n_qubits).gates:
            state = apply_gate_to_state(g, state, n_qubits, backend)
        e = backend.expectation_sv(state, H).real
        return th, e

    def fn(theta_value):
        _, e = run(theta_value)
        return float(e.detach())

    def autograd_grad(theta_value):
        th, e = run(theta_value, grad=True)
        e.backward()
        return np.array([float(th.grad)])

    return fn, autograd_grad


def test_psr4_matches_autograd_single_excitation():
    fn, autograd_grad = _energy_fn(lambda th: single_excitation(th, 0, 1), 2, init_x=(1,))
    theta = np.array([0.6])
    assert np.allclose(psr4(fn, theta), autograd_grad(theta), atol=1e-4)


def test_psr4_matches_autograd_double_excitation():
    fn, autograd_grad = _energy_fn(lambda th: double_excitation(th, 0, 1, 2, 3), 4, init_x=(2, 3))
    theta = np.array([0.7])
    assert np.allclose(psr4(fn, theta), autograd_grad(theta), atol=1e-4)


def test_two_term_psr_is_wrong_for_excitation():
    fn, autograd_grad = _energy_fn(lambda th: single_excitation(th, 0, 1), 2, init_x=(1,))
    theta = np.array([0.6])
    # 标准两项规则对激发门不正确，应与 autograd 不一致
    assert not np.allclose(psr(fn, theta), autograd_grad(theta), atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/qml/test_psr4.py -q`
Expected: FAIL — `ImportError: cannot import name 'psr4'`.

- [ ] **Step 3: Implement `psr4`**

In `aicir/qml/deriv.py`, add (near `psr`):

```python
_FOUR_TERM_C1 = (math.sqrt(2.0) + 1.0) / (4.0 * math.sqrt(2.0))
_FOUR_TERM_C2 = (math.sqrt(2.0) - 1.0) / (4.0 * math.sqrt(2.0))


def psr4(
    fn: Callable[[np.ndarray], float],
    params: Any,
    *,
    shifts: tuple[float, float] = (np.pi / 2.0, 3.0 * np.pi / 2.0),
    coefficients: tuple[float, float] | None = None,
) -> np.ndarray:
    """四项参数移位规则（激发门等 {-1,0,1} 谱生成元）。

    grad = c1[f(θ+s1) − f(θ−s1)] − c2[f(θ+s2) − f(θ−s2)]，逐参数计算。
    默认 (s1,s2)=(π/2, 3π/2)，(c1,c2)=((√2+1)/4√2, (√2−1)/4√2)。
    autograd 为权威校验：若与 autograd 不符，应据其修正系数。
    """
    if math is None:  # pragma: no cover
        raise RuntimeError
    s1, s2 = float(shifts[0]), float(shifts[1])
    c1, c2 = coefficients if coefficients is not None else (_FOUR_TERM_C1, _FOUR_TERM_C2)
    theta = np.asarray(params, dtype=float)
    grad = np.zeros_like(theta, dtype=float)
    for index in np.ndindex(theta.shape):
        base = theta.copy()
        def shifted(delta):
            p = base.copy()
            p[index] = base[index] + delta
            return float(fn(p))
        grad[index] = c1 * (shifted(s1) - shifted(-s1)) - c2 * (shifted(s2) - shifted(-s2))
    return grad
```

Ensure `import math` is present at the top of `aicir/qml/deriv.py` (add if missing).

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=. pytest tests/qml/test_psr4.py -q`
Expected: PASS. If the two excitation tests fail by a constant factor, the default `c1`/`c2` are off — adjust them so the autograd-equality tests pass (autograd is authoritative), then re-run.

- [ ] **Step 5: Commit**

```bash
git add aicir/qml/deriv.py tests/qml/test_psr4.py
git commit -m "feat(qml): add psr4 four-term parameter-shift rule for excitation gates"
```

---

## Task 6: H2 correlation validation

**Files:**
- Test: `tests/test_h2_excitation_vqe.py`

**Interfaces:**
- Consumes: `double_excitation`, `aicir.chemistry.molecule_hamiltonian("h2_jw")`, `GPUBackend`.

- [ ] **Step 1: Write the validation test**

Create `tests/test_h2_excitation_vqe.py`:

```python
"""H2 关联验证：HF + double_excitation 经 autograd 优化达化学精度。

证明激发门能产生真实电子关联（仅单激发/无关联无法达此精度）。
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir import double_excitation, NumpyBackend
from aicir.backends.gpu_backend import GPUBackend
from aicir.chemistry import molecule_hamiltonian, molecule_matrix
from aicir.core.circuit import Circuit, pauli_x
from aicir.core.gates import apply_gate_to_state

CHEMICAL_ACCURACY = 1.6e-3
EXACT_GROUND = -1.8572750091552734


def _hf_occupied_qubits(ham_matrix):
    # 在 2 电子扇区里选与精确基态重叠最大的 HF 行列式（避开 JW 排序约定的歧义）
    w, v = np.linalg.eigh(ham_matrix)
    ground = v[:, 0]
    weight2 = [i for i in range(16) if bin(i).count("1") == 2]
    best = max(weight2, key=lambda i: abs(ground[i]))
    return [q for q in range(4) if (best >> (3 - q)) & 1]


def test_h2_double_excitation_reaches_chemical_accuracy():
    ham = molecule_hamiltonian("h2_jw")
    Hmat = molecule_matrix("h2_jw", backend=NumpyBackend())
    occupied = _hf_occupied_qubits(Hmat)

    backend = GPUBackend(device="cpu")
    Ht = backend.cast(Hmat.astype(np.complex64))

    def energy(theta_t):
        state = backend.zeros_state(4)
        for q in occupied:
            state = apply_gate_to_state(pauli_x(q).to_dict(), state, 4, backend)
        for g in Circuit(double_excitation(theta_t, 0, 1, 2, 3), n_qubits=4).gates:
            state = apply_gate_to_state(g, state, 4, backend)
        return backend.expectation_sv(state, Ht).real

    theta = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        e = energy(theta)
        e.backward()
        opt.step()

    final = float(energy(theta).detach())
    assert abs(final - EXACT_GROUND) < CHEMICAL_ACCURACY, f"E={final}, exact={EXACT_GROUND}"
```

- [ ] **Step 2: Run the validation**

Run: `PYTHONPATH=. pytest tests/test_h2_excitation_vqe.py -q`
Expected: PASS — final energy within 1.6e-3 Ha of −1.8572750. If it converges to the HF energy (no correlation), the `double_excitation` coupling is wrong — revisit the basis-index coupling in Task 3.

- [ ] **Step 3: Commit**

```bash
git add tests/test_h2_excitation_vqe.py
git commit -m "test: H2 VQE with double_excitation reaches chemical accuracy"
```

---

## Self-Review notes

- **Spec coverage:** Unit 1 single_excitation → Task 2; Unit 2 double_excitation → Task 3; Unit 3 four-term rule + shift_rule metadata → Task 1 (field/helper) + Task 5 (psr4); Unit 4 H2 validation → Task 6; autograd-kernel testing (spec Testing #2) → Task 4; gate-matrix/particle-conservation/registry tests (spec Testing #1,#3) → Tasks 2-3.
- **Placeholder scan:** none — all steps carry concrete code/commands.
- **Type consistency:** `single_excitation`/`double_excitation`/`givens`/`psr4`/`gate_shift_rule`/`shift_rule` used identically across tasks; gate-dict keys (`qubit_1`/`qubit_2` for single, `qubits` list for double) consistent with the `_PAIR_QUBIT_GATES` change and the dispatch reads.
- **Known risk flagged inline:** psr4 default constants are verified against autograd (Task 5 Step 4); double_excitation basis coupling verified by the H2 validation (Task 6 Step 2).

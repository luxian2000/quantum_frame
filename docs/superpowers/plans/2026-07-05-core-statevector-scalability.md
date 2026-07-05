# Core Statevector Scalability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend exact pure-state simulation scalability by routing local gates through bounded-memory statevector kernels instead of full circuit matrices or whole-state transpose temporaries.

**Architecture:** Keep `Circuit.unitary()` as a small-circuit diagnostic API. Improve the core execution path used by `apply_gate_to_state()` and `Measure.run(method="statevector")` by adding an optional backend hook for local statevector application, implementing a chunked NumPy kernel for one- and two-qubit gates, and preserving the existing generic fallback for Torch, NPU, arbitrary multi-qubit unitaries, density matrices, and custom unsupported gates.

**Tech Stack:** Python 3.10+, NumPy, optional PyTorch, pytest, existing `aicir.core.gates`, `aicir.backends`, and `aicir.measure` APIs.

## Global Constraints

- Do not change public semantics of `Circuit.unitary()` or `Circuit.matrix()`.
- Do not route density-matrix/noise execution through the new pure-state kernel.
- Preserve Torch autograd behavior covered by `tests/gates/test_matrix_autograd.py`.
- Preserve NPU flat local application behavior for real NPU devices.
- New backend hook must be optional so custom backends that only implement `Backend` continue working.
- Tests must avoid fragile wall-clock assertions; use parity, routing, and bounded-chunk behavior instead.
- Keep this plan limited to exact statevector scaling for local gates. Tensor-network sampling, MPS, stabilizer simulation, and noisy quantum trajectories need separate plans.

---

## Scope Check

This plan covers one independently testable subsystem: exact pure-state local-gate execution. It does not implement a tensor-network sampler, MPS engine, stabilizer tableau backend, or noisy trajectory simulator. Those are separate engines with different correctness contracts and should not be mixed into this first implementation.

## File Structure

- Modify `aicir/backends/base.py`: add an optional `apply_statevector_local(...)` hook with a default `None` return.
- Modify `aicir/backends/numpy_backend.py`: implement a chunked local statevector kernel for one- and two-qubit local matrices.
- Modify `aicir/core/gates.py`: call the optional backend hook before the generic reshape/transpose local apply path, while preserving the NPU flat path priority.
- Add `tests/backends/test_numpy_statevector_local.py`: focused unit tests for the NumPy backend hook.
- Add `tests/circuit/test_statevector_scalability.py`: integration tests proving supported pure-state circuit execution uses local kernels and matches existing exact results.
- Modify `aicir/backends/README.md`: document the optional backend hook and the dense-unitary boundary.

### Task 1: Add NumPy Local Kernel Tests First

**Files:**
- Create: `tests/backends/test_numpy_statevector_local.py`
- Test target: `aicir.backends.numpy_backend.NumpyBackend.apply_statevector_local`

**Interfaces:**
- Consumes: current `NumpyBackend` class.
- Produces: failing tests that define the new optional method signature:
  `apply_statevector_local(self, state, local_matrix, axes, n_qubits) -> ndarray | None`

- [ ] **Step 1: Write the failing backend tests**

Create `tests/backends/test_numpy_statevector_local.py`:

```python
import numpy as np

from aicir.backends.numpy_backend import NumpyBackend


def _random_state(n_qubits: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec = (vec / np.linalg.norm(vec)).astype(np.complex64)
    return vec.reshape(-1, 1)


def _random_unitary(dim: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = phases / np.where(np.abs(phases) == 0, 1.0, np.abs(phases))
    return (q * phases).astype(np.complex64)


def _reference_apply(state: np.ndarray, local: np.ndarray, axes: tuple[int, ...], n_qubits: int) -> np.ndarray:
    axes = tuple(int(axis) for axis in axes)
    dim_local = 1 << len(axes)
    out = np.empty_like(state.reshape(-1))
    flat = state.reshape(-1)
    for basis in range(1 << n_qubits):
        bits = [(basis >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_col = 0
        for axis in axes:
            local_col = (local_col << 1) | bits[axis]
        value = 0.0 + 0.0j
        for local_row in range(dim_local):
            row_bits = bits.copy()
            for pos, axis in enumerate(axes):
                row_bits[axis] = (local_row >> (len(axes) - 1 - pos)) & 1
            row_index = 0
            for bit in row_bits:
                row_index = (row_index << 1) | bit
            value += local[local_col, local_row] * flat[row_index]
        out[basis] = value
    return out.reshape(-1, 1)


def test_apply_statevector_local_one_qubit_matches_reference_with_small_chunks():
    backend = NumpyBackend()
    backend._statevector_chunk_size = 5
    state = _random_state(5)
    local = _random_unitary(2)

    actual = backend.apply_statevector_local(state, local, axes=(2,), n_qubits=5)
    expected = _reference_apply(state, local, axes=(2,), n_qubits=5)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (32, 1)


def test_apply_statevector_local_two_qubit_matches_reference_non_adjacent_axes():
    backend = NumpyBackend()
    backend._statevector_chunk_size = 7
    state = _random_state(5, seed=23)
    local = _random_unitary(4, seed=29)

    actual = backend.apply_statevector_local(state, local, axes=(3, 0), n_qubits=5)
    expected = _reference_apply(state, local, axes=(3, 0), n_qubits=5)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (32, 1)


def test_apply_statevector_local_returns_none_for_three_qubit_gate():
    backend = NumpyBackend()
    state = _random_state(4)
    local = np.eye(8, dtype=np.complex64)

    assert backend.apply_statevector_local(state, local, axes=(0, 1, 2), n_qubits=4) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/backends/test_numpy_statevector_local.py -q
```

Expected: FAIL with `AttributeError: 'NumpyBackend' object has no attribute 'apply_statevector_local'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/backends/test_numpy_statevector_local.py
git commit -m "test: define numpy local statevector kernel contract"
```

### Task 2: Add Optional Backend Hook And NumPy Chunked Kernel

**Files:**
- Modify: `aicir/backends/base.py:203-228`
- Modify: `aicir/backends/numpy_backend.py:108-111`
- Test: `tests/backends/test_numpy_statevector_local.py`

**Interfaces:**
- Consumes: test contract from Task 1.
- Produces:
  `Backend.apply_statevector_local(self, state, local_matrix, axes, n_qubits) -> object | None`
  and `NumpyBackend.apply_statevector_local(...)`.

- [ ] **Step 1: Add the optional hook to `Backend`**

In `aicir/backends/base.py`, insert this method after `expectation_dm` and before tensor-network primitives:

```python
    def apply_statevector_local(self, state, local_matrix, axes, n_qubits: int):
        """Optionally apply a local gate directly to a pure statevector.

        Backends that can perform bounded-memory local statevector updates may
        return a new ``(2**n_qubits, 1)`` state tensor. Returning ``None`` means
        the caller should use the generic local-matrix fallback.
        """
        return None
```

- [ ] **Step 2: Add chunk helpers to `NumpyBackend`**

In `aicir/backends/numpy_backend.py`, add these private helpers near the module top, after `_CDTYPE = np.complex64`:

```python
def _local_offsets(axes, n_qubits: int):
    offsets = []
    for local_index in range(1 << len(axes)):
        offset = 0
        for local_pos, axis in enumerate(axes):
            bit = (local_index >> (len(axes) - 1 - local_pos)) & 1
            offset |= bit << (int(n_qubits) - 1 - int(axis))
        offsets.append(np.int64(offset))
    return offsets


def _zero_target_bases(start: int, stop: int, axes, n_qubits: int) -> np.ndarray:
    basis = np.arange(start, stop, dtype=np.int64)
    mask = np.ones(basis.shape, dtype=bool)
    for axis in axes:
        shift = int(n_qubits) - 1 - int(axis)
        mask &= ((basis >> shift) & 1) == 0
    return basis[mask]
```

- [ ] **Step 3: Implement `NumpyBackend.apply_statevector_local`**

Add this method to `NumpyBackend` after `apply_unitary`:

```python
    def apply_statevector_local(self, state, local_matrix, axes, n_qubits: int):
        axes = tuple(int(axis) for axis in axes)
        n_qubits = int(n_qubits)
        if len(axes) == 0:
            return state
        if len(axes) > 2:
            return None
        if len(set(axes)) != len(axes):
            raise ValueError("局部门作用的量子比特不能重复")
        if any(axis < 0 or axis >= n_qubits for axis in axes):
            raise ValueError("局部门作用的量子比特索引超出范围")

        dim_local = 1 << len(axes)
        local = np.asarray(local_matrix, dtype=self._dtype)
        if local.shape != (dim_local, dim_local):
            raise ValueError(
                f"local gate matrix shape {local.shape} does not match {len(axes)} target qubit(s)"
            )

        flat = np.asarray(state).reshape(-1)
        dim = 1 << n_qubits
        if flat.size != dim:
            raise ValueError(f"state length {flat.size} does not match n_qubits={n_qubits}")

        out = np.empty_like(flat)
        chunk_size = int(getattr(self, "_statevector_chunk_size", 1 << 20))
        chunk_size = max(chunk_size, 1)
        offsets = _local_offsets(axes, n_qubits)

        for start in range(0, dim, chunk_size):
            stop = min(start + chunk_size, dim)
            bases = _zero_target_bases(start, stop, axes, n_qubits)
            if bases.size == 0:
                continue
            gathered_indices = [bases | offset for offset in offsets]
            gathered = np.stack([flat[index] for index in gathered_indices], axis=0)
            updated = local @ gathered
            for index, values in zip(gathered_indices, updated):
                out[index] = values

        return out.reshape(dim, 1)
```

- [ ] **Step 4: Run backend tests**

Run:

```bash
pytest tests/backends/test_numpy_statevector_local.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing backend and gate tests that touch local state application**

Run:

```bash
pytest tests/gates/test_matrix_autograd.py tests/backends/test_take_add.py tests/circuit/test_circuit_backend_unitary.py -q
```

Expected: PASS. The autograd test must continue passing because the new NumPy hook is not used by Torch backends.

- [ ] **Step 6: Commit the backend hook**

```bash
git add aicir/backends/base.py aicir/backends/numpy_backend.py tests/backends/test_numpy_statevector_local.py
git commit -m "feat: add chunked numpy local statevector kernel"
```

### Task 3: Route Local Gate Application Through The Optional Hook

**Files:**
- Modify: `aicir/core/gates.py:705-720`
- Add: `tests/circuit/test_statevector_scalability.py`
- Test: `tests/circuit/test_statevector_scalability.py`

**Interfaces:**
- Consumes: `Backend.apply_statevector_local(...)`.
- Produces: `_apply_local_matrix_to_state(...)` uses the hook when available, except real NPU flat path remains first for large NPU statevectors.

- [ ] **Step 1: Write routing tests**

Create `tests/circuit/test_statevector_scalability.py`:

```python
import numpy as np

from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, rx, ry, rzz
from aicir.core import State
from aicir.core.gates import apply_gate_to_state


class RecordingNumpyBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        self.local_calls = []

    def apply_statevector_local(self, state, local_matrix, axes, n_qubits):
        self.local_calls.append((tuple(int(axis) for axis in axes), int(n_qubits)))
        return super().apply_statevector_local(state, local_matrix, axes, n_qubits)


def test_apply_gate_to_state_uses_backend_local_hook_for_one_and_two_qubit_gates():
    backend = RecordingNumpyBackend()
    state = State.zero_state(4, backend).data

    state = apply_gate_to_state(hadamard(0), state, 4, backend)
    state = apply_gate_to_state(cnot(3, [1]), state, 4, backend)
    state = apply_gate_to_state(rzz(0.25, 0, 2), state, 4, backend)

    assert backend.local_calls == [((0,), 4), ((3, 1), 4), ((0, 2), 4)]


def test_measure_run_statevector_uses_chunked_hook_and_matches_unitary_reference():
    backend = RecordingNumpyBackend()
    circuit = Circuit(
        hadamard(0),
        ry(0.3, 1),
        cnot(2, [0]),
        rzz(0.2, 1, 3),
        rx(-0.4, 2),
        n_qubits=4,
        backend=backend,
    )

    result = Measure(backend).run(circuit, shots=None, return_state=True)
    actual = result.final_state.to_numpy()

    reference_backend = NumpyBackend()
    reference = State.zero_state(4, reference_backend).evolve(
        circuit.unitary(backend=reference_backend)
    ).to_numpy()

    np.testing.assert_allclose(actual, reference, atol=1e-6)
    assert len(backend.local_calls) == 5
```

- [ ] **Step 2: Run routing tests to verify they fail**

Run:

```bash
pytest tests/circuit/test_statevector_scalability.py -q
```

Expected: FAIL because `_apply_local_matrix_to_state` does not call `apply_statevector_local` yet.

- [ ] **Step 3: Modify `_apply_local_matrix_to_state`**

In `aicir/core/gates.py`, change the start of `_apply_local_matrix_to_state` after the shape validation block:

```python
    if _should_use_flat_local_apply(backend, n_qubits):
        return _apply_local_matrix_to_state_flat(state, local_matrix, axes, n_qubits, backend)

    apply_statevector_local = getattr(backend, "apply_statevector_local", None)
    if callable(apply_statevector_local):
        updated = apply_statevector_local(state, local_matrix, axes, n_qubits)
        if updated is not None:
            return updated
```

Keep the existing grouped reshape/transpose fallback immediately after this new block.

- [ ] **Step 4: Run routing tests**

Run:

```bash
pytest tests/circuit/test_statevector_scalability.py -q
```

Expected: PASS.

- [ ] **Step 5: Run tests covering existing local matrix behavior**

Run:

```bash
pytest tests/circuit/test_circuit_backend_unitary.py tests/gates/test_matrix_dispatch_consistency.py tests/gates/test_matrix_autograd.py tests/measure/test_unified_run.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit routing change**

```bash
git add aicir/core/gates.py tests/circuit/test_statevector_scalability.py
git commit -m "feat: route pure state local gates through backend kernel"
```

### Task 4: Add A No-Global-Matrix Integration Guard

**Files:**
- Modify: `tests/circuit/test_statevector_scalability.py`
- Test: `tests/circuit/test_statevector_scalability.py::test_statevector_measure_path_does_not_call_gate_to_matrix_for_supported_local_gates`

**Interfaces:**
- Consumes: routing from Task 3.
- Produces: regression guard that supported pure-state gates in `Measure.run` do not call `gate_to_matrix`.

- [ ] **Step 1: Add the guard test**

Append this test to `tests/circuit/test_statevector_scalability.py`:

```python
def test_statevector_measure_path_does_not_call_gate_to_matrix_for_supported_local_gates(monkeypatch):
    import aicir.measure.trajectory as trajectory

    def forbidden_gate_to_matrix(*args, **kwargs):
        raise AssertionError("pure statevector local gates must not build global gate matrices")

    monkeypatch.setattr(trajectory, "gate_to_matrix", forbidden_gate_to_matrix)

    backend = NumpyBackend()
    circuit = Circuit(
        hadamard(0),
        rx(0.1, 1),
        ry(-0.2, 2),
        cnot(3, [0]),
        rzz(0.4, 1, 3),
        n_qubits=4,
        backend=backend,
    )

    result = Measure(backend).run(circuit, shots=None, return_state=True)

    assert result.final_state.n_qubits == 4
    assert np.isclose(result.final_state.norm(), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run the guard test**

Run:

```bash
pytest tests/circuit/test_statevector_scalability.py::test_statevector_measure_path_does_not_call_gate_to_matrix_for_supported_local_gates -q
```

Expected: PASS.

- [ ] **Step 3: Run all statevector scalability tests**

Run:

```bash
pytest tests/backends/test_numpy_statevector_local.py tests/circuit/test_statevector_scalability.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the guard**

```bash
git add tests/circuit/test_statevector_scalability.py
git commit -m "test: guard statevector path against global matrix fallback"
```

### Task 5: Document The Scalable Statevector Boundary

**Files:**
- Modify: `aicir/backends/README.md`
- Test: `pytest tests/docs/test_readme_measure_examples.py -q`

**Interfaces:**
- Consumes: implemented backend hook and routing behavior.
- Produces: documentation that tells users which path scales and which path is dense.

- [ ] **Step 1: Add backend documentation**

In `aicir/backends/README.md`, add this section near the backend API discussion:

````markdown
### Local Statevector Kernel Hook

Pure-state circuit execution should prefer local statevector updates over full
`2^n x 2^n` unitary construction. Backends may implement:

```python
apply_statevector_local(state, local_matrix, axes, n_qubits)
```

The method receives a `(2**n_qubits, 1)` statevector, a local gate matrix, and
the target/control axes in the local matrix order. It returns a new statevector
or `None` to request the generic fallback.

`NumpyBackend` implements this hook for one- and two-qubit local matrices using
bounded chunks. `Circuit.unitary()` remains a dense diagnostic API and is not
the scalable execution path for large circuits.
````

- [ ] **Step 2: Run documentation-adjacent tests**

Run:

```bash
pytest tests/docs/test_readme_measure_examples.py tests/backends/test_numpy_statevector_local.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit documentation**

```bash
git add aicir/backends/README.md
git commit -m "docs: document local statevector backend hook"
```

### Task 6: Final Verification

**Files:**
- No source changes.
- Verify full suite.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified implementation ready for review.

- [ ] **Step 1: Run focused simulator, circuit, backend, and gate tests**

Run:

```bash
pytest tests/backends/test_numpy_statevector_local.py tests/circuit/test_statevector_scalability.py tests/simulator/test_engine_parity.py tests/measure/test_unified_run.py tests/gates/test_matrix_autograd.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run:

```bash
pytest -q
```

Expected: PASS, with the existing warning profile no worse than before this work.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff --stat HEAD~5..HEAD
git diff HEAD~5..HEAD -- aicir/backends/base.py aicir/backends/numpy_backend.py aicir/core/gates.py tests/backends/test_numpy_statevector_local.py tests/circuit/test_statevector_scalability.py aicir/backends/README.md
```

Expected: only the planned files changed; no unrelated formatting churn.

- [ ] **Step 4: Commit final verification note if any verification-only documentation was added**

If no files changed during verification, do not create an empty commit.

## Self-Review

- Spec coverage: The plan covers exact pure-state local gate scalability, optional backend API, NumPy bounded-chunk implementation, routing through existing core simulation, regression guards, and documentation.
- Scope: The plan intentionally excludes tensor-network sampling, MPS, stabilizer, and noisy trajectories because they are independent engines.
- Type consistency: The same hook name and signature are used throughout: `apply_statevector_local(self, state, local_matrix, axes, n_qubits)`.
- Placeholder scan: The plan contains concrete file paths, commands, test bodies, and implementation bodies for each code-changing task.

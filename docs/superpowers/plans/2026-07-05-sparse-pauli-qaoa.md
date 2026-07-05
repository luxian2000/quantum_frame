# Sparse/Pauli QAOA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dense `2^n × 2^n` matrix and the pure-Python `2^n` loop from exact gate-level QAOA energy, and add an opt-in analytically-exact parameter-shift gradient, without changing existing QAOA public behavior or defaults.

**Architecture:** Keep `BasicQAOA` and its `Hamiltonian`-based gate-level path. (1) Add a sparse per-term expectation `Σ_j c_j ⟨ψ|P_j|ψ⟩` that applies each Pauli string to the statevector via `apply_gate_to_state` (reusing the local-gate kernel) instead of building `problem_hamiltonian.to_matrix(...)`. (2) Route both the diagonal and non-diagonal exact (`shots=None`) energy through that helper. (3) Refactor `build_circuit` onto an explicit gate "tape" so the same gate sequence can be materialized with any single gate's argument shifted, then add a decomposed per-gate parameter-shift gradient (chain rule over each gate argument) that is exact for the Trotterized circuit. Finite-difference stays the default gradient; analytic is opt-in.

**Tech Stack:** Python 3.10+, NumPy, pytest, existing `aicir.core.circuit` gate factories, `aicir.core.gates.apply_gate_to_state`, `aicir.core.operators.Hamiltonian`, `aicir.measure.Measure`.

## Global Constraints

- Do not change the public signature or default behavior of `BasicQAOA.__init__`, `BasicQAOA.run`, `BasicQAOA.energy`, `BasicQAOA.build_circuit`, or `run_qaoa`.
- Finite-difference (`finite_difference_gradient`) stays the default gradient path; analytic gradient is opt-in via a new `grad_method` argument that defaults to `"fd"`.
- Do not touch the legacy dense-matrix-input path (`problem_hamiltonian` passed as a raw `np.ndarray`, `self._gate_level is False`) — `ansatz_state`/`_exp_hermitian` remain the compatibility fallback for that input form only.
- Do not change shots-based (`shots is not None`) energy or sampling behavior.
- Rotation-gate convention is `rx/ry/rz/rzz(t) = exp(-i t/2 · P)` (generator eigenvalues ±½); the parameter-shift rule on a gate *argument* `t` uses shift `π/2` and coefficient `0.5`.
- Comments/docstrings in this file are Chinese to match `aicir/vqc/QAOA.py`.
- Run everything from repo root with `PYTHONPATH=.`.

---

## Scope Check

This plan covers one independently testable subsystem: exact gate-level QAOA energy/gradient math in `aicir/vqc/QAOA.py`. It does **not** implement an MPS/tensor sampler, shots-based non-diagonal energy, custom mixers, or any change to the legacy dense-matrix-input path. Those are separate concerns.

## File Structure

- Modify `aicir/vqc/QAOA.py`:
  - Add `_GateRecord` dataclass + tape builders (`_qaoa_tape`, `_circuit_from_tape`, `_trotter_slice_records`, `_pauli_evolution_records`, `_basis_change_records`, `_basis_uncompute_records`) as module-level helpers.
  - Add `BasicQAOA._sparse_cost_expectation`, `BasicQAOA.analytic_gradient`, `BasicQAOA._tape_energy`.
  - Refactor `BasicQAOA.build_circuit` to materialize from the tape.
  - Rewrite the exact (`shots=None`) branch of `BasicQAOA.energy` for the gate-level path.
  - Extend `BasicQAOA._gradient` and `BasicQAOA.run` with an opt-in `grad_method`.
- Add `tests/vqc/test_qaoa_sparse_energy.py`: sparse expectation + exact-energy parity.
- Add `tests/vqc/test_qaoa_analytic_gradient.py`: tape parity + analytic-vs-FD gradient parity.
- Modify `CHANGELOG.md`: dated entry for the interface additions.

---

### Task 1: Sparse Per-Term Cost Expectation Helper

**Files:**
- Modify: `aicir/vqc/QAOA.py` (add `BasicQAOA._sparse_cost_expectation` after `bitstring_energy`, before `ansatz_state`, around `aicir/vqc/QAOA.py:416`)
- Test: `tests/vqc/test_qaoa_sparse_energy.py`

**Interfaces:**
- Consumes: `self._cost_terms` (`tuple[_PauliCostTerm, ...]`, each with `.paulis: tuple[str,...]`, `.qubits: tuple[int,...]`, `.coefficient: float`), `self._cost_offset: float`, `self.n_qubits: int`, `aicir.core.gates.apply_gate_to_state`, `aicir.core.circuit.pauli_x/pauli_y/pauli_z`.
- Produces: `BasicQAOA._sparse_cost_expectation(self, state, backend) -> float` — exact `⟨ψ|H_C|ψ⟩` where `state` is a backend-native statevector tensor (as returned by `State.data`).

- [ ] **Step 1: Write the failing test**

Create `tests/vqc/test_qaoa_sparse_energy.py`:

```python
import numpy as np

from aicir import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.vqc.QAOA import BasicQAOA


def _dense_reference(qaoa, params, backend):
    result = qaoa.measure(params, backend=backend, return_state=True)
    operator = qaoa.problem_hamiltonian.to_matrix(backend)
    return float(result.final_state.expectation(operator))


def test_sparse_cost_expectation_matches_dense_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=7)
    params = qaoa.initial_params()

    result = qaoa.measure(params, backend=backend, return_state=True)
    sparse = qaoa._sparse_cost_expectation(result.final_state.data, backend)
    dense = _dense_reference(qaoa, params, backend)

    assert np.isclose(sparse, dense, atol=1e-6)


def test_sparse_cost_expectation_matches_dense_non_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("X", [0], 0.5), ("YZ", [0, 1], -0.25), ("ZZ", 1.0)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, seed=3)
    params = qaoa.initial_params()

    result = qaoa.measure(params, backend=backend, return_state=True)
    sparse = qaoa._sparse_cost_expectation(result.final_state.data, backend)
    dense = _dense_reference(qaoa, params, backend)

    assert np.isclose(sparse, dense, atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_sparse_energy.py -q`
Expected: FAIL with `AttributeError: 'BasicQAOA' object has no attribute '_sparse_cost_expectation'`.

- [ ] **Step 3: Add the import for the Pauli factories**

In `aicir/vqc/QAOA.py`, change the existing gate import line (`aicir/vqc/QAOA.py:15`):

```python
from ..core.circuit import Circuit, cnot, hadamard, pauli_x, pauli_y, pauli_z, rx, ry, rz, rzz
```

- [ ] **Step 4: Implement `_sparse_cost_expectation`**

In `aicir/vqc/QAOA.py`, add this method to `BasicQAOA` immediately after `bitstring_energy` (after `aicir/vqc/QAOA.py:416`):

```python
    def _sparse_cost_expectation(self, state, backend) -> float:
        """稀疏计算 ⟨ψ|H_C|ψ⟩ = Σ_j c_j ⟨ψ|P_j|ψ⟩，逐 Pauli 串作用到态矢量，避免构造 2^n×2^n 稠密矩阵。

        参数:
            state:   后端原生态矢量张量（如 ``State.data``）。
            backend: 计算后端。
        返回:
            实数期望值（含常数偏移 ``self._cost_offset``）。
        """
        from ..core.gates import apply_gate_to_state

        factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
        psi = np.asarray(backend.to_numpy(state), dtype=np.complex128).reshape(-1)
        value = float(self._cost_offset)
        for term in self._cost_terms:
            transformed = state
            for pauli, qubit in zip(term.paulis, term.qubits):
                transformed = apply_gate_to_state(
                    factory[pauli](int(qubit)), transformed, self.n_qubits, backend
                )
            t_np = np.asarray(backend.to_numpy(transformed), dtype=np.complex128).reshape(-1)
            value += term.coefficient * float(np.vdot(psi, t_np).real)
        return float(value)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_sparse_energy.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add aicir/vqc/QAOA.py tests/vqc/test_qaoa_sparse_energy.py
git commit -m "feat(qaoa): 稀疏逐项计算代价哈密顿量期望，避免稠密矩阵"
```

---

### Task 2: Route Exact Gate-Level Energy Through The Sparse Helper

**Files:**
- Modify: `aicir/vqc/QAOA.py` — the gate-level branch of `BasicQAOA.energy` (`aicir/vqc/QAOA.py:447-467`)
- Test: `tests/vqc/test_qaoa_sparse_energy.py`

**Interfaces:**
- Consumes: `BasicQAOA._sparse_cost_expectation` from Task 1.
- Produces: unchanged `BasicQAOA.energy(...)` signature; exact (`shots=None`) gate-level energy for both diagonal and non-diagonal Hamiltonians is computed via the sparse helper (no `to_matrix`, no `2^n` Python loop).

- [ ] **Step 1: Write the failing regression test**

Append to `tests/vqc/test_qaoa_sparse_energy.py`:

```python
def test_energy_exact_does_not_build_dense_matrix(monkeypatch):
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("XIX", 0.3)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, seed=5)
    params = qaoa.initial_params()

    def forbidden_to_matrix(*args, **kwargs):
        raise AssertionError("exact gate-level energy must not build a dense Hamiltonian matrix")

    monkeypatch.setattr(qaoa.problem_hamiltonian, "to_matrix", forbidden_to_matrix)

    energy = qaoa.energy(params, backend=backend)
    assert np.isfinite(energy)


def test_energy_exact_matches_previous_dense_values_diagonal_and_non_diagonal():
    backend = NumpyBackend()
    for terms in (
        [("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)],
        [("XIX", 0.3), ("ZZI", -1.0), ("IYY", 0.4)],
    ):
        hamiltonian = Hamiltonian(n_qubits=3, terms=terms)
        qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=11)
        params = qaoa.initial_params()

        result = qaoa.measure(params, backend=backend, return_state=True)
        dense = float(result.final_state.expectation(qaoa.problem_hamiltonian.to_matrix(backend)))
        actual = qaoa.energy(params, backend=backend)

        assert np.isclose(actual, dense, atol=1e-6)
```

- [ ] **Step 2: Run the new tests to verify the guard fails**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_sparse_energy.py::test_energy_exact_does_not_build_dense_matrix -q`
Expected: FAIL with the `AssertionError` from `forbidden_to_matrix` (current non-diagonal exact path calls `to_matrix`).

- [ ] **Step 3: Rewrite the gate-level branch of `energy`**

In `aicir/vqc/QAOA.py`, replace the current gate-level block inside `energy` (`aicir/vqc/QAOA.py:447-467`, from `if self._gate_level:` down to the line ending `return float(result.final_state.expectation(operator))`) with:

```python
        if self._gate_level:
            if shots is not None:
                if not self._diagonal_cost:
                    raise ValueError(
                        "shots-based energy for non-diagonal Hamiltonians requires a Pauli estimator; "
                        "use shots=None for exact energy"
                    )
                counts = self.sample(params, shots=int(shots), backend=backend, seed=seed, method=method)
                total = sum(counts.values()) or 1
                return float(sum(self.bitstring_energy(bitstring) * count for bitstring, count in counts.items()) / total)
            # 精确能量（shots=None）：稀疏逐项期望，避免稠密矩阵与 2^n Python 循环
            active_backend = NumpyBackend() if backend is None else backend
            result = self.measure(params, backend=active_backend, method=method, return_state=True)
            return self._sparse_cost_expectation(result.final_state.data, active_backend)
```

Leave the trailing non-gate-level branch (`if shots is not None: raise ...` and the `ansatz_state` dense path at `aicir/vqc/QAOA.py:468-472`) unchanged.

- [ ] **Step 4: Run the full sparse-energy test file**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_sparse_energy.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Run the existing QAOA suite for regressions**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_canonical.py tests/vqc/test_qaoa_qfun.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/vqc/QAOA.py tests/vqc/test_qaoa_sparse_energy.py
git commit -m "feat(qaoa): 精确门级能量走稀疏期望路径，去除稠密矩阵与 2^n 循环"
```

---

### Task 3: Refactor `build_circuit` Onto A Gate Tape

**Files:**
- Modify: `aicir/vqc/QAOA.py` — add tape helpers as module-level functions after `_append_trotter_slice` (after `aicir/vqc/QAOA.py:181`); refactor `BasicQAOA.build_circuit` (`aicir/vqc/QAOA.py:322-343`).
- Test: `tests/vqc/test_qaoa_analytic_gradient.py`

**Interfaces:**
- Consumes: `self._cost_terms`, `self.trotter_steps`, `self.trotter_order`, `self.p`, `self.n_qubits`, `self.split_params`.
- Produces:
  - `_GateRecord(name: str, qubits: tuple[int, ...], arg: float | None, owner: int | None, dcoeff: float | None)` dataclass.
  - `BasicQAOA._qaoa_tape(self, params) -> list[_GateRecord]` — ordered records for every emitted gate. `owner` is the flat parameter index the gate's argument depends on (`0..p-1` for γ layers, `p..2p-1` for β layers) or `None` for fixed gates; `dcoeff = d(arg)/d(theta_owner)` for variational gates.
  - `_circuit_from_tape(records, n_qubits, backend=None) -> Circuit`.
  - `build_circuit(...)` produces a circuit unitarily identical to the pre-refactor implementation for non-degenerate parameters.

- [ ] **Step 1: Write the failing parity test**

Create `tests/vqc/test_qaoa_analytic_gradient.py`:

```python
import numpy as np

from aicir import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.vqc.QAOA import BasicQAOA


def _unitary(circuit, backend):
    return np.asarray(circuit.unitary(backend=backend), dtype=np.complex128)


def test_build_circuit_tape_matches_reference_unitary():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("XIX", 0.3)])
    for order in (1, 2):
        qaoa = BasicQAOA(
            problem_hamiltonian=hamiltonian, p=2, trotter_steps=2, trotter_order=order, seed=13
        )
        params = qaoa.initial_params()
        circuit = qaoa.build_circuit(params, backend=backend)
        # 从磁带重建的线路应与 build_circuit 一致（build_circuit 本身即走磁带）
        from aicir.vqc.QAOA import _circuit_from_tape

        rebuilt = _circuit_from_tape(qaoa._qaoa_tape(params), qaoa.n_qubits, backend)
        np.testing.assert_allclose(_unitary(circuit, backend), _unitary(rebuilt, backend), atol=1e-6)


def test_qaoa_tape_owner_indices_cover_all_parameters():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("X", [0], 0.4)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=1)
    params = qaoa.initial_params()
    owners = {rec.owner for rec in qaoa._qaoa_tape(params) if rec.owner is not None}
    assert owners == set(range(qaoa.n_params))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_analytic_gradient.py -q`
Expected: FAIL with `ImportError: cannot import name '_circuit_from_tape'` / `AttributeError: '_qaoa_tape'`.

- [ ] **Step 3: Add the `_GateRecord` dataclass and tape builders**

In `aicir/vqc/QAOA.py`, insert this block immediately after `_append_trotter_slice` (after `aicir/vqc/QAOA.py:181`, before the `@dataclass class QAOAResult`):

```python
@dataclass(frozen=True)
class _GateRecord:
    """QAOA 线路磁带的单门记录。

    name:   门类型（"h"/"rx"/"ry"/"rz"/"rzz"/"cx"）。
    qubits: 作用比特（cx 为 (pivot, control)，rzz 为 (q0, q1)）。
    arg:    旋转门参数；h/cx 为 None。
    owner:  变分参数的扁平索引（γ 层为 0..p-1，β 层为 p..2p-1）；固定门为 None。
    dcoeff: d(arg)/d(theta_owner)；固定门为 None。
    """

    name: str
    qubits: tuple[int, ...]
    arg: float | None
    owner: int | None
    dcoeff: float | None


def _basis_change_records(records: list[_GateRecord], pauli: str, qubit: int) -> None:
    if pauli == "X":
        records.append(_GateRecord("h", (qubit,), None, None, None))
    elif pauli == "Y":
        records.append(_GateRecord("rz", (qubit,), -np.pi / 2.0, None, None))
        records.append(_GateRecord("h", (qubit,), None, None, None))


def _basis_uncompute_records(records: list[_GateRecord], pauli: str, qubit: int) -> None:
    if pauli == "X":
        records.append(_GateRecord("h", (qubit,), None, None, None))
    elif pauli == "Y":
        records.append(_GateRecord("h", (qubit,), None, None, None))
        records.append(_GateRecord("rz", (qubit,), np.pi / 2.0, None, None))


def _pauli_evolution_records(
    records: list[_GateRecord], term: _PauliCostTerm, rotation: float, drotation: float, owner: int
) -> None:
    """记录 exp(-i angle P) 的门序列，rotation = 2*angle，drotation = d(rotation)/d(theta_owner)。"""

    if len(term.qubits) == 1:
        pauli = term.paulis[0]
        qubit = term.qubits[0]
        name = {"X": "rx", "Y": "ry"}.get(pauli, "rz")
        records.append(_GateRecord(name, (qubit,), float(rotation), owner, float(drotation)))
        return
    if len(term.qubits) == 2 and term.paulis == ("Z", "Z"):
        records.append(
            _GateRecord("rzz", (term.qubits[0], term.qubits[1]), float(rotation), owner, float(drotation))
        )
        return

    for pauli, qubit in zip(term.paulis, term.qubits):
        _basis_change_records(records, pauli, qubit)

    pivot = term.qubits[-1]
    for control in term.qubits[:-1]:
        records.append(_GateRecord("cx", (pivot, control), None, None, None))
    records.append(_GateRecord("rz", (pivot,), float(rotation), owner, float(drotation)))
    for control in reversed(term.qubits[:-1]):
        records.append(_GateRecord("cx", (pivot, control), None, None, None))

    for pauli, qubit in reversed(tuple(zip(term.paulis, term.qubits))):
        _basis_uncompute_records(records, pauli, qubit)


def _trotter_slice_records(
    records: list[_GateRecord],
    terms: tuple[_PauliCostTerm, ...],
    gamma: float,
    owner: int,
    trotter_steps: int,
    trotter_order: int,
) -> None:
    if not terms:
        return
    gamma_step = gamma / trotter_steps

    def emit(term: _PauliCostTerm, mult: float) -> None:
        rotation = 2.0 * mult * gamma_step * term.coefficient
        drotation = 2.0 * mult * term.coefficient / trotter_steps
        _pauli_evolution_records(records, term, rotation, drotation, owner)

    if trotter_order == 1 or len(terms) == 1:
        for term in terms:
            emit(term, 1.0)
        return

    for term in terms[:-1]:
        emit(term, 0.5)
    emit(terms[-1], 1.0)
    for term in reversed(terms[:-1]):
        emit(term, 0.5)


def _circuit_from_tape(records: list[_GateRecord], n_qubits: int, backend: Any = None) -> Circuit:
    circuit = Circuit(n_qubits=n_qubits, backend=backend)
    for rec in records:
        if rec.name == "h":
            circuit.append(hadamard(rec.qubits[0]))
        elif rec.name == "cx":
            circuit.append(cnot(rec.qubits[0], [rec.qubits[1]]))
        elif rec.name == "rx":
            circuit.append(rx(rec.arg, rec.qubits[0]))
        elif rec.name == "ry":
            circuit.append(ry(rec.arg, rec.qubits[0]))
        elif rec.name == "rz":
            circuit.append(rz(rec.arg, rec.qubits[0]))
        elif rec.name == "rzz":
            circuit.append(rzz(rec.arg, rec.qubits[0], rec.qubits[1]))
        else:
            raise ValueError(f"未知磁带门类型 {rec.name!r}")
    return circuit
```

- [ ] **Step 4: Add `_qaoa_tape` and refactor `build_circuit`**

In `aicir/vqc/QAOA.py`, replace the body of `build_circuit` (`aicir/vqc/QAOA.py:322-343`) with a tape delegation and add `_qaoa_tape` directly above it:

```python
    def _qaoa_tape(self, params: np.ndarray) -> list[_GateRecord]:
        """构造 QAOA 线路的门磁带（单一事实来源，build_circuit 与解析梯度共用）。"""

        gammas, betas = self.split_params(params)
        records: list[_GateRecord] = []
        for qubit in range(self.n_qubits):
            records.append(_GateRecord("h", (qubit,), None, None, None))

        for layer in range(self.p):
            gamma = float(gammas[layer])
            for _ in range(self.trotter_steps):
                _trotter_slice_records(
                    records, self._cost_terms, gamma, layer, self.trotter_steps, self.trotter_order
                )
            beta = float(betas[layer])
            for qubit in range(self.n_qubits):
                records.append(_GateRecord("rx", (qubit,), 2.0 * beta, self.p + layer, 2.0))
        return records

    def build_circuit(self, params: np.ndarray, *, backend: Any = None) -> Circuit:
        """Build the canonical gate-level QAOA circuit for ``params``."""

        if self.cost is not None:
            raise ValueError("build_circuit is unavailable when BasicQAOA delegates to an external cost")
        if not self._gate_level:
            raise ValueError("build_circuit requires an aicir Hamiltonian problem_hamiltonian")

        return _circuit_from_tape(self._qaoa_tape(params), self.n_qubits, backend)
```

- [ ] **Step 5: Run the parity tests**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_analytic_gradient.py -q`
Expected: PASS.

- [ ] **Step 6: Run the QAOA suite to confirm no behavior change**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_canonical.py tests/vqc/test_qaoa_qfun.py tests/vqc/test_qaoa_sparse_energy.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aicir/vqc/QAOA.py tests/vqc/test_qaoa_analytic_gradient.py
git commit -m "refactor(qaoa): build_circuit 走门磁带，单一事实来源"
```

---

### Task 4: Opt-In Analytic Parameter-Shift Gradient

**Files:**
- Modify: `aicir/vqc/QAOA.py` — add `BasicQAOA.analytic_gradient` and `BasicQAOA._tape_energy` (after `finite_difference_gradient`, after `aicir/vqc/QAOA.py:491`); extend `BasicQAOA._gradient` (`aicir/vqc/QAOA.py:474-478`) and `BasicQAOA.run` (`aicir/vqc/QAOA.py:493-591`) with `grad_method`.
- Test: `tests/vqc/test_qaoa_analytic_gradient.py`

**Interfaces:**
- Consumes: `_qaoa_tape`, `_circuit_from_tape`, `_sparse_cost_expectation`, `_GateRecord`, `Measure`.
- Produces:
  - `BasicQAOA.analytic_gradient(self, params, *, backend=None, method="statevector") -> np.ndarray` — analytically exact gradient of the exact (`shots=None`) gate-level energy, shape `(n_params,)`.
  - `BasicQAOA._tape_energy(self, records, gate_index, delta, backend, method) -> float`.
  - `BasicQAOA._gradient(self, params, *, grad_method="fd", **energy_kwargs)` and `BasicQAOA.run(..., grad_method="fd")` — FD stays the default; `grad_method in {"analytic", "psr"}` selects the analytic path.

- [ ] **Step 1: Write the failing gradient tests**

Append to `tests/vqc/test_qaoa_analytic_gradient.py`:

```python
def _fd_reference(qaoa, params, backend):
    return qaoa.finite_difference_gradient(params, eps=1e-5, backend=backend)


def test_analytic_gradient_matches_fd_diagonal():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", 0.5), ("ZIZ", 0.25)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=17)
    params = qaoa.initial_params()

    analytic = qaoa.analytic_gradient(params, backend=backend)
    fd = _fd_reference(qaoa, params, backend)

    np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_analytic_gradient_matches_fd_non_diagonal_and_trotterized():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("XX", 0.6), ("YZ", [0, 1], -0.3), ("ZZ", 1.0)])
    for order in (1, 2):
        qaoa = BasicQAOA(
            problem_hamiltonian=hamiltonian, p=2, trotter_steps=3, trotter_order=order, seed=19
        )
        params = qaoa.initial_params()

        analytic = qaoa.analytic_gradient(params, backend=backend)
        fd = _fd_reference(qaoa, params, backend)

        np.testing.assert_allclose(analytic, fd, atol=1e-4)


def test_run_grad_method_analytic_reaches_same_optimum_as_fd():
    backend = NumpyBackend()
    hamiltonian = Hamiltonian(n_qubits=3, terms=[("ZZI", -1.0), ("IZZ", -1.0)])
    init = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).initial_params()

    fd_run = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).run(
        max_iters=40, lr=0.1, init_params=init, backend=backend
    )
    an_run = BasicQAOA(problem_hamiltonian=hamiltonian, p=2, seed=23).run(
        max_iters=40, lr=0.1, init_params=init, backend=backend, grad_method="analytic"
    )

    assert np.isclose(fd_run.energy, an_run.energy, atol=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_analytic_gradient.py -q`
Expected: FAIL with `AttributeError: 'BasicQAOA' object has no attribute 'analytic_gradient'` (and `run` rejecting `grad_method`).

- [ ] **Step 3: Add `analytic_gradient` and `_tape_energy`**

In `aicir/vqc/QAOA.py`, add these two methods to `BasicQAOA` immediately after `finite_difference_gradient` (after `aicir/vqc/QAOA.py:491`):

```python
    def _tape_energy(self, records, gate_index: int, delta: float, backend, method: str) -> float:
        """将 records[gate_index] 的参数平移 delta 后重建线路并返回精确能量。"""

        rec = records[gate_index]
        shifted = list(records)
        shifted[gate_index] = _GateRecord(rec.name, rec.qubits, rec.arg + delta, rec.owner, rec.dcoeff)
        circuit = _circuit_from_tape(shifted, self.n_qubits, backend)
        result = Measure(backend).run(
            circuit, shots=None, measure_qubits=(), return_state=True, method=method
        )
        return self._sparse_cost_expectation(result.final_state.data, backend)

    def analytic_gradient(
        self, params: np.ndarray, *, backend: Any = None, method: str = "statevector"
    ) -> np.ndarray:
        """精确能量的解析梯度：对磁带中每个变分门做逐门参数移位（生成元谱 ±½，移位 π/2），
        再按链式法则 dE/dθ_k = Σ_g (d arg_g/dθ_k)·½[E(arg_g+π/2)−E(arg_g−π/2)] 聚合。

        对被评估的（Trotter 化）线路是解析精确的。仅支持门级 Hamiltonian 路径。
        """

        if not self._gate_level:
            raise ValueError("analytic_gradient requires an aicir Hamiltonian problem_hamiltonian")

        active_backend = NumpyBackend() if backend is None else backend
        theta = np.asarray(params, dtype=float).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"params size {theta.size} does not match expected {self.n_params}")

        records = self._qaoa_tape(theta)
        grad = np.zeros(self.n_params, dtype=float)
        shift = np.pi / 2.0
        for gate_index, rec in enumerate(records):
            if rec.owner is None:
                continue
            e_plus = self._tape_energy(records, gate_index, shift, active_backend, method)
            e_minus = self._tape_energy(records, gate_index, -shift, active_backend, method)
            grad[rec.owner] += rec.dcoeff * 0.5 * (e_plus - e_minus)
        return grad
```

- [ ] **Step 4: Thread `grad_method` through `_gradient`**

In `aicir/vqc/QAOA.py`, replace `_gradient` (`aicir/vqc/QAOA.py:474-478`) with:

```python
    def _gradient(self, params: np.ndarray, *, grad_method: str = "fd", **energy_kwargs: Any) -> np.ndarray:
        if self.cost is not None:
            flat = np.asarray(params, dtype=float).reshape(-1)
            return np.asarray(self.cost.grad(flat), dtype=float).reshape(flat.shape)
        if grad_method in ("analytic", "psr"):
            return self.analytic_gradient(
                params,
                backend=energy_kwargs.get("backend"),
                method=energy_kwargs.get("method", "statevector"),
            )
        if grad_method != "fd":
            raise ValueError(f"grad_method 必须是 fd/analytic/psr，收到 {grad_method!r}")
        return self.finite_difference_gradient(params, **energy_kwargs)
```

- [ ] **Step 5: Thread `grad_method` through `run`**

In `aicir/vqc/QAOA.py`, add `grad_method: str = "fd"` to the `run` keyword-only arguments (after `method: str = "statevector"` at `aicir/vqc/QAOA.py:504`):

```python
        method: str = "statevector",
        grad_method: str = "fd",
```

Then, in the manual gradient-descent loop of `run`, change the gradient call (`aicir/vqc/QAOA.py:572`) from:

```python
            grad = self._gradient(params, **energy_kwargs)
```

to:

```python
            grad = self._gradient(params, grad_method=grad_method, **energy_kwargs)
```

(The optimizer-driven branch at `aicir/vqc/QAOA.py:521-560` uses only the objective and is unaffected.)

- [ ] **Step 6: Run the gradient tests**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_analytic_gradient.py -q`
Expected: PASS.

- [ ] **Step 7: Run the full QAOA suite**

Run: `PYTHONPATH=. pytest tests/vqc/test_qaoa_canonical.py tests/vqc/test_qaoa_qfun.py tests/vqc/test_qaoa_sparse_energy.py tests/vqc/test_qaoa_analytic_gradient.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aicir/vqc/QAOA.py tests/vqc/test_qaoa_analytic_gradient.py
git commit -m "feat(qaoa): 新增可选解析参数移位梯度（逐门分解，链式法则聚合）"
```

---

### Task 5: Final Verification And Changelog

**Files:**
- Modify: `CHANGELOG.md`
- Verify: full suite.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: dated changelog entry; verified full suite.

- [ ] **Step 1: Add a changelog entry**

In `CHANGELOG.md`, add under a new dated heading (match the file's existing format; place at the top of the entries):

```markdown
## 2026-07-05

### QAOA 稀疏化与解析梯度
- 门级 `BasicQAOA` 精确能量（`shots=None`）改走稀疏逐项期望 `Σ_j c_j⟨ψ|P_j|ψ⟩`，移除稠密 `to_matrix` 与 `2^n` Python 循环（对角/非对角统一）。
- `build_circuit` 重构为门磁带（`_qaoa_tape`/`_circuit_from_tape`），作为前向与梯度的单一事实来源。
- 新增可选解析参数移位梯度：`BasicQAOA.analytic_gradient(...)` 与 `run(..., grad_method="analytic")`（逐门 π/2 移位 + 链式法则聚合，对 Trotter 化线路解析精确）。默认梯度仍为有限差分（`grad_method="fd"`）。
```

- [ ] **Step 2: Run the focused suite**

Run:

```bash
PYTHONPATH=. pytest tests/vqc/test_qaoa_canonical.py tests/vqc/test_qaoa_qfun.py tests/vqc/test_qaoa_sparse_energy.py tests/vqc/test_qaoa_analytic_gradient.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH=. pytest -q`
Expected: PASS, with the warning profile no worse than before this work.

- [ ] **Step 4: Inspect changed files**

Run:

```bash
git diff --stat HEAD~4..HEAD
```

Expected: only `aicir/vqc/QAOA.py`, the two new test files, and `CHANGELOG.md` changed.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(qaoa): 记录稀疏能量与解析梯度变更"
```

## Self-Review

- **Spec coverage:** Energy fixes — Task 1 (sparse per-term expectation) + Task 2 (wire exact diagonal & non-diagonal energy, kill dense `to_matrix` and `2^n` Python loop). Analytic gradient — Task 3 (tape refactor, single source of truth) + Task 4 (`analytic_gradient` opt-in via `grad_method`, FD stays default). Both scoped items covered.
- **Scope:** Excludes MPS/tensor sampler, shots-based non-diagonal energy, custom mixers, and the legacy dense-matrix-input `ansatz_state`/`_exp_hermitian` path — all intentionally out of scope per Global Constraints.
- **Placeholder scan:** Every code-changing step shows complete code; test steps show full test bodies; commands include expected output. No TBD/TODO.
- **Type consistency:** `_GateRecord(name, qubits, arg, owner, dcoeff)` is defined in Task 3 and consumed identically in Tasks 3–4. `_sparse_cost_expectation(self, state, backend)` defined in Task 1, consumed in Tasks 2 and 4. `analytic_gradient(self, params, *, backend, method)` and `_tape_energy(self, records, gate_index, delta, backend, method)` defined and consumed in Task 4. `grad_method` default `"fd"` consistent across `_gradient` and `run`. Owner-index convention (γ → `0..p-1`, β → `p..2p-1`) matches `split_params` (`params = [gammas, betas]`).
- **Correctness note:** The analytic gradient differentiates exactly the (Trotterized) circuit that `build_circuit`/`energy` evaluate, so it agrees with the exact energy's own gradient — there is no Trotter-vs-exact mismatch between the forward value and its gradient. The FD test oracle (`atol=1e-4`) guards this.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-sparse-pauli-qaoa.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

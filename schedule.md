# aicir Implementation Schedule

Derived from `design.md` (Sections 6 and 9). This document breaks every phase
into concrete, assignable work items and sorts them by **serial execution**
(must run in order — they are dependency bottlenecks) versus **parallel
execution** (independent tracks that different staff can run concurrently).

Each item lists its files, dependency, and a done-when verification command.

---

## 0. Dependency Graph (at a glance)

```text
Phase 1 (Planner Contracts)  ── SERIAL ROOT, blocks everything
   │
   ├── Track A ── Phase 2 (Batched QML) ──serial──> Phase 3 (QAS prefilters)
   │
   ├── Track B ── Phase 4 (QEC contracts) ──serial──> Phase 5 (Stabilizer engines)
   │              Phase 6 (Noisy scaling)  ──parallel with 4/5──
   │
   └── Track C ── Phase 7 (Tensor-network/MPS)  ──independent, lowest priority──
```

| Stage | Type | Depends on | Owner |
| :--- | :--- | :--- | :--- |
| Phase 1 | **Serial root** | None | Core architect(s) |
| Phase 2 → Phase 3 | Serial within Track A | Phase 1 | Team A (QML/QAS) |
| Phase 4 → Phase 5 | Serial within Track B | Phase 1 | Team B (QEC) |
| Phase 6 | Parallel inside Track B | Phase 1 | Team B (Noise) |
| Phase 7 | Independent | Phase 1 | Team C (Optional) |

Tracks A, B, and C run **fully in parallel** once Phase 1 lands.

---

## 1. SERIAL EXECUTION — Foundational Core

### Phase 1 — Planner Contracts & Core API (global bottleneck)

> Introduces the routing mechanism with **no behavior change**. Nothing in the
> parallel tracks can integrate cleanly until this is merged.

Files: `aicir/primitives/results.py`, `aicir/primitives/estimator.py`,
`aicir/primitives/sampler.py`, new `aicir/primitives/planner.py`,
`tests/primitives/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 1.1 | Define immutable `ExecutionPlan` dataclass (`task`, `engine`, `backend_name`, `shots`, `supports_grad`, `noisy`, `batched`, `warnings`) | — |
| 1.2 | Add planner metadata to `EstimateResult.metadata` and `SampleResult.metadata` without breaking existing fields | 1.1 |
| 1.3 | Implement `select_execution_plan(circuit, *, task="auto", backend=None, shots=None, noise_model=None, engine="auto")` | 1.1 |
| 1.4 | Route `SVEstimator`, `ShotEstimator`, `ShotSampler` through the planner **for metadata only** — execution stays behaviorally identical | 1.2, 1.3 |
| 1.5 | Tests proving existing primitive outputs are unchanged except for added metadata | 1.4 |

These five items are themselves serial (1.1 → 1.2/1.3 → 1.4 → 1.5).

Verification:

```bash
env PYTHONPATH=. pytest tests/primitives
```

**Gate:** Phase 1 must be green before forking the parallel tracks below.

---

## 2. PARALLEL EXECUTION — Independent Tracks (start after Phase 1)

### Track A — QML & QAS (Tensor & Batching focus) — Team A

#### Phase 2 — Strengthen batched QML and variational execution
*Serial prerequisite for Phase 3.*

Files: `aicir/core/batch.py`, `aicir/qml/deriv.py`, `aicir/qml/diff/registry.py`,
`aicir/primitives/estimator.py`, tests under `tests/core/`, `tests/qml/`,
`tests/primitives/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 2.1 | Helper functions to evolve a `Circuit` through `BatchSV` when the gate set is supported | Phase 1 |
| 2.2 | Clear fallback metadata when a circuit cannot use `BatchSV` | 2.1 |
| 2.3 | Grouped parameter batches for fixed-topology VQE/QML evaluation | 2.1 |
| 2.4 | Keep `NumpyBackend` / `GPUBackend` / `NPUBackend` unchanged for non-batched calls (regression guard) | 2.1 |
| 2.5 | Tests comparing `BatchSV` output vs. normal state vector path for small circuits | 2.1–2.4 |

```bash
env PYTHONPATH=. pytest tests/core tests/qml tests/primitives
```

#### Phase 3 — QAS prefilters, cache hooks, grouped evaluation
*Depends on Phase 2 (reuses the batched evaluation path).*

Files: `aicir/qas/evaluator.py`, `aicir/qas/architecture_search.py`,
`aicir/qas/_utils.py`, `aicir/metrics/`, `aicir/primitives/estimator.py`,
new `tests/qas/test_qas_evaluation_planning.py`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 3.1 | Stable circuit-structure hash from `CircuitIR` ops/qubits/controls/gate names, excluding trainable param values when weight sharing is intended | Phase 2 |
| 3.2 | Optional structural prefilters: depth, two-qubit count, entangler topology, trainability proxies, hardware efficiency | Phase 2 |
| 3.3 | Cache interface for candidate scores and reusable prefix/suffix states | 3.1 |
| 3.4 | Grouped evaluation for candidates with compatible topology | 3.2, 3.3 |
| 3.5 | Ensure MoG_VQE and existing QAS algorithms use the same evaluator path | 3.4 |

```bash
env PYTHONPATH=. pytest tests/qas tests/metrics
```

---

### Track B — QEC & Noise (Stabilizers & Trajectories focus) — Team B

> Track B contains its own serial chain (Phase 4 → Phase 5) plus one item
> (Phase 6) that runs in parallel with that chain. With enough staff, split
> Team B into **B1 (QEC engines: Phase 4→5)** and **B2 (Noise: Phase 6)**.

#### Phase 4 — QEC result contracts & Clifford analysis
*Serial prerequisite for Phase 5.*

Files: new `aicir/qec/`, `aicir/ir/accessors.py`, `aicir/gates/registry.py`,
`aicir/primitives/results.py`, `aicir/measure/measure.py`, `tests/qec/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 4.1 | `QECResult` with `syndrome_history`, `logical_error_rate`, `decoder_metadata`, `pauli_frame`, `metadata` | Phase 1 |
| 4.2 | Clifford gate classification helpers using gate registry + typed IR accessors | Phase 1 |
| 4.3 | Pauli-measurement and reset capability checks | 4.2 |
| 4.4 | Tests for classification of Clifford / non-Clifford / measurement / reset circuits | 4.1–4.3 |

```bash
env PYTHONPATH=. pytest tests/qec tests/ir tests/gates
```

#### Phase 5 — Stabilizer/tableau & Pauli-frame engines
*Depends on Phase 4.*

Files: new `aicir/qec/stabilizer.py`, new `aicir/qec/pauli_frame.py`,
new `aicir/qec/sampler.py`, `aicir/primitives/sampler.py` (if a generic facade
is added), `tests/qec/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 5.1 | Tableau state init, Clifford gate updates, Pauli measurement, reset, shot sampling | Phase 4 |
| 5.2 | Pauli-frame correction tracking, separate from physical gate application | Phase 4 |
| 5.3 | QEC sampler facade returning `QECResult` | 5.1, 5.2, 4.1 |
| 5.4 | Cross-check small Clifford circuits against state vector / density-matrix path | 5.1 |
| 5.5 | Planner dispatch for `task="qec"` when circuit is Clifford-compatible | 5.3, Phase 1 |

```bash
env PYTHONPATH=. pytest tests/qec tests/measure
```

#### Phase 6 — Noisy scaling paths
*Parallel with Phases 4/5 (only depends on Phase 1).*

Files: `aicir/noise/`, `aicir/measure/trajectory.py`,
new `aicir/qec/noise.py`, tests under `tests/noise/`, `tests/measure/`,
`tests/qec/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 6.1 | Keep density matrix simulation as the exact small-system reference (guard) | Phase 1 |
| 6.2 | Add/extend trajectory execution for larger noisy circuits | Phase 1 |
| 6.3 | Share `NoiseModel` definitions across density matrix, trajectory, and QEC paths | 6.2 |
| 6.4 | Planner rules for density-matrix limits and trajectory fallback | 6.2, Phase 1 |
| 6.5 | Statistical tests with deterministic seeds and tolerances | 6.1–6.4 |

> Note: item 6.3 touches `aicir/qec/noise.py`; coordinate the shared
> `NoiseModel` surface with B1 once Phase 4's `aicir/qec/` package exists.

```bash
env PYTHONPATH=. pytest tests/noise tests/measure tests/qec
```

---

### Track C — Optional Engines — Team C

#### Phase 7 — Tensor-network/MPS engine
*Independent, lowest priority. Only depends on Phase 1's planner module.*

Files: new `aicir/tensor_network/`, the planner module from Phase 1,
tests under `tests/tensor_network/`

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 7.1 | Opt-in MPS support for 1D local circuits | Phase 1 |
| 7.2 | Exact comparisons against state vector for small circuits | 7.1 |
| 7.3 | Do **not** enable automatic planner selection until benchmarks justify it | 7.1 |
| 7.4 | Keep optional dependencies optional | 7.1 |

```bash
env PYTHONPATH=. pytest tests/tensor_network
```

---

## 3. Scheduling Summary

| When | Serial work | Parallel work |
| :--- | :--- | :--- |
| Sprint 0 | **Phase 1** (1.1→1.5) by core architects | — |
| Sprint 1+ | Phase 2→3 (Team A); Phase 4→5 (Team B1) | Phase 6 (Team B2); Phase 7 (Team C) |

**Critical path:** Phase 1 → (longest of: Phase 2→3 *or* Phase 4→5). Phases 6
and 7 should never be on the critical path; staff them only after the dependent
serial chains (A and B1) are covered.

**Cross-cutting (per `design.md` §7 Public API Policy):** any item that changes
a public surface must also update `README.md`, `CHANGELOG.md`, and the relevant
submodule README, and must preserve the listed public symbols and
`Circuit.gates` compatibility. Keep optional dependencies optional throughout.

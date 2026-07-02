# quantum_frame / aicir Hybrid Simulator Design

This file combines the architectural conclusions from `design_gemini.md`,
`design_gpt.md`, and `design_claude.md` into one implementation-oriented design
for `aicir`.

## 1. Design Decision

No single simulator scheme is optimal for quantum machine learning (QML),
quantum architecture search (QAS), and quantum error correction (QEC). These
workloads stress different bottlenecks:

| Workload | Main variable | Main bottleneck | Best execution style |
| --- | --- | --- | --- |
| QML | Continuous parameters over mostly fixed circuits | Repeated differentiable forward/backward passes | Batched tensor/state vector engine with GPU/NPU autograd |
| QAS | Circuit topology and ansatz structure | Evaluating, mutating, ranking, and caching many candidate circuits | Typed IR + primitives + structural prefilters + batched estimator |
| QEC | Measurements, resets, Pauli noise, syndrome history | Large Clifford-heavy circuits and high shot throughput | Stabilizer/tableau + Pauli-frame sampler, with density/trajectory validation |

The core design is therefore:

```text
Circuit / typed IR is the single source of truth.
Primitives are the stable execution API.
A planner selects the best specialized engine for the task.
QNode-style callables remain optional QML ergonomics, not the core abstraction.
```

This matches the current direction of `aicir`: `Circuit` and `CircuitIR` already
exist, `aicir.primitives` provides `SVEstimator`, `ShotEstimator`, and
`ShotSampler`, `BatchSV` provides a batched state vector path, and backends are
already separated into CPU/GPU/NPU implementations.

## 2. Main Alternatives

### 2.1 QNode-style callable

A QNode wraps circuit construction, device/backend, measurement, shots, and
differentiation into one callable.

Advantages:

- Excellent ergonomics for simple QML and variational models.
- Natural to expose as a layer in a PyTorch-style training loop.
- Can hide gradient selection through `aicir.qml.diff.select_diff`.

Disadvantages:

- Couples circuit definition, execution, and differentiation.
- Poor fit for QAS because circuit topology is the search variable.
- Not sufficient for QEC dynamic circuits, syndrome histories, and hardware
  control-flow paths.

Use in `aicir`:

- Add only as a thin optional frontend after primitives and planner behavior are
  stable.
- Do not make lower layers depend on it.

### 2.2 Primitives: Estimator and Sampler

The circuit remains data. Execution happens through `Estimator` and `Sampler`
objects.

Advantages:

- Clean separation of concerns.
- Batch-first interface for many circuits, observables, or shots.
- Best public surface for QAS, VQE-style benchmarking, and QEC shot collection.
- Already matches `aicir.primitives`.

Disadvantages:

- More verbose than a QNode for the simplest QML examples.
- Needs a good parameter-binding/batching path to reach peak QML throughput.

Use in `aicir`:

- Keep `SVEstimator`, `ShotEstimator`, and `ShotSampler` as the primary
  algorithm execution API.
- Add planner metadata to `EstimateResult` and `SampleResult` so callers know
  which engine and plan were used.

### 2.3 Typed IR plus transforms

The circuit is represented as typed data, and optimization, differentiation,
metrics, transpilation, and execution are transforms over that data.

Advantages:

- Best introspection model for QAS metrics, QEC classifiers, gate counts,
  Clifford detection, and transpilation.
- Lets `CircuitIR` compile/run on many engines without rebuilding user code.
- Preserves compatibility with current `Circuit.gates` while allowing typed
  internal consumers through `aicir.ir`.

Disadvantages:

- Requires clear interfaces and careful migration discipline.
- Slightly more complex than a single callable for first-time QML users.

Use in `aicir`:

- Treat `CircuitIR` as the internal planning input.
- Preserve `Circuit`, `Circuit.operations`, `Circuit.ir`, and legacy
  `Circuit.gates` compatibility.
- Put structure analysis in small modules rather than embedding it into
  primitives.

### 2.4 Pure tensor/autodiff graph

The simulation is expressed as tensor operations so the ML framework records the
whole computation graph.

Advantages:

- Best performance path for differentiable QML on GPU/NPU.
- Exact backpropagation for supported gates.
- Works naturally with hybrid classical neural networks.
- `BatchSV` already implements an NPU-safe real/imag batched path.

Disadvantages:

- Rebuilding graphs is costly when QAS changes topology constantly.
- Simulator-only; it does not map directly to shot-based hardware execution.
- Memory grows with stored intermediate states.

Use in `aicir`:

- Make this the preferred QML engine for fixed-topology differentiable circuits.
- Keep parameter-shift, finite difference, SPSA, and adjoint paths for cases
  where autograd is unavailable or inappropriate.

### 2.5 AOT/JIT compilation

Circuits or dynamic programs are compiled ahead of repeated execution, for
example into OpenQASM 3.0, QIR, MLIR, or compiled internal kernels.

Advantages:

- Reduces repeated interpreter overhead.
- Important for hardware-facing QEC with mid-circuit measurement and classical
  feed-forward.
- Can eventually support gate fusion, scheduling, and layout.

Disadvantages:

- Highest engineering cost.
- Compilation overhead is wasteful when QAS topology changes every iteration.
- Differentiation through compiled code requires extra design.

Use in `aicir`:

- Keep as a later execution target behind the planner and `aicir.transpile`.
- Prioritize simulator engines first; compile paths should consume the same IR.

## 3. Simulation Engines

| Engine | Advantages | Disadvantages | Best use |
| --- | --- | --- | --- |
| state vector | Exact, simple, good for noiseless VQE/QAOA/QML/QAS | `O(2^n)` memory, poor for large QEC | Default exact noiseless simulator |
| Batched state vector | High throughput for data/parameter batches, maps to GPU/NPU, already started as `BatchSV` | More complex memory layout, best when batch members share structure | QML training and grouped QAS evaluation |
| Density matrix | Exact mixed-state and channel simulation | `O(4^n)` memory, small systems only | Noise correctness checks and small noisy circuits |
| Quantum trajectories | Scales noisy simulation better than density matrix, parallel over trajectories | Stochastic variance and harder debugging | Medium noisy QML/QAS and approximate QEC noise studies |
| Stabilizer/tableau | Polynomial scaling for Clifford circuits, ideal for syndrome extraction | Cannot represent arbitrary non-Clifford rotations directly | QEC core simulator |
| Pauli frame | Cheap tracking of Pauli corrections and logical frames | Requires clear measurement/reset semantics | QEC decoding and correction loops |
| Tensor network/MPS | Can exceed state vector size for low-entanglement local circuits | Performance depends on entanglement and contraction order | Optional large-qubit QML/QAS engine |

## 4. Integrated Architecture

The recommended architecture is a layered stack, not a single monolithic
simulator class:

```text
Frontend APIs
  - Circuit construction
  - optional QNode-like QML wrapper
  - QAS runners
  - QEC code/circuit builders

Typed IR
  - aicir.core.Circuit
  - aicir.ir.CircuitIR
  - Operation / Measurement / Observable
  - legacy Circuit.gates compatibility

Analysis and planning
  - gate and Clifford classification
  - noise classification
  - topology, depth, and hardware metrics
  - structural QAS prefilters
  - engine selection

Primitives
  - SVEstimator
  - ShotEstimator
  - ShotSampler
  - future BatchedEstimator / QECSampler facade if needed

Engines
  - NumpyBackend / GPUBackend / NPUBackend state vector
  - BatchSV batched state vector
  - density matrix path through State / Measure / noise
  - future TrajectoryEngine
  - future StabilizerEngine and PauliFrameEngine
  - optional TensorNetworkEngine

Result layer
  - EstimateResult
  - SampleResult
  - future QECResult with syndrome and logical-error fields
```

The planner should choose deterministic rules first and allow explicit override:

```text
if task == "qec" and circuit is Clifford-compatible:
    use StabilizerEngine / PauliFrameEngine
elif noise_model is not None and n_qubits <= density_matrix_limit:
    use density matrix
elif noise_model is not None:
    use TrajectoryEngine
elif task in {"qml", "vqe", "qaoa"} and topology is fixed:
    use BatchSV or GPUBackend/NPUBackend state vector with autograd when possible
elif task == "qas":
    run structural prefilters, then SVEstimator or grouped BatchSV
elif circuit is local and expected entanglement is low:
    optionally use TensorNetworkEngine
else:
    use SVEstimator / state vector backend
```

Example override shape:

```python
estimator = SVEstimator(backend=backend)
sampler = ShotSampler(backend=backend, shots=4096)

# Future planner facade:
# estimator = Estimator(engine="auto", task="qas")
# sampler = Sampler(engine="stabilizer", task="qec")
```

## 5. Workload-specific Schemes

### 5.1 QML

Recommended scheme:

```text
Circuit / CircuitIR with parameters
  -> parameter binding or batched parameter input
  -> BatchSV or GPUBackend/NPUBackend state vector
  -> differentiable expectation
  -> autograd, adjoint, parameter shift, or QNG transform
```

Implementation direction:

- Prefer `BatchSV` for batched differentiable models on GPU/NPU.
- Keep `aicir.qml.deriv.auto` for tensor-backed autograd.
- Keep `psr`, `spsr`, `spsa`, `fd`, `ad`, and QNG methods as explicit
  transforms for non-autograd or geometry-aware training.
- Add an optional QNode-like wrapper only after primitives and planner metadata
  are stable.

### 5.2 QAS

Recommended scheme:

```text
Candidate generator
  -> Circuit / CircuitIR
  -> structural prefilter
  -> topology/hash cache
  -> SVEstimator, ShotEstimator, or grouped BatchSV
  -> multi-objective ranking / search policy
```

Implementation direction:

- Build on `aicir.qas.ArchitectureSearch`, `ArchitectureEvaluator`,
  `aicir.metrics`, and `aicir.primitives`.
- Prefer primitives over one QNode per candidate.
- Add circuit hashing and prefix/suffix cache hooks for repeated candidates.
- Group candidates by topology when batched evaluation is possible.
- Keep MoG_VQE, CRLQAS, PPR_DQL, PPO_RB, and VQA_QAS as algorithm frontends
  that consume the same evaluator/primitives layer.

### 5.3 QEC

Recommended scheme:

```text
QEC circuit / code builder
  -> CircuitIR with measurement/reset/classical metadata
  -> Clifford and Pauli-noise classifier
  -> StabilizerEngine + PauliFrameEngine
  -> ShotSampler-style syndrome sampling
  -> decoder and logical-error statistics
```

Implementation direction:

- Add a stabilizer/tableau simulator for Clifford circuits and Pauli
  measurements.
- Add a Pauli-frame representation for correction tracking.
- Keep density matrix execution for small exact noisy validation.
- Add quantum trajectories for larger noisy studies that cannot stay in the
  stabilizer regime.
- Treat hardware/AOT dynamic-circuit compilation as a later target behind
  `aicir.transpile` and OpenQASM 3.0/QIR export.

## 6. Implementation Steps Based on Current aicir

### Phase 1: Add planner contracts without changing behavior

Files:

- `aicir/primitives/results.py`
- `aicir/primitives/estimator.py`
- `aicir/primitives/sampler.py`
- new `aicir/primitives/planner.py`
- tests under `tests/primitives/`

Steps:

1. Add a small immutable `ExecutionPlan` dataclass with fields such as
   `task`, `engine`, `backend_name`, `shots`, `supports_grad`, `noisy`,
   `batched`, and `warnings`.
2. Add planner metadata to `EstimateResult.metadata` and
   `SampleResult.metadata` without breaking existing fields.
3. Implement an initial `select_execution_plan(circuit, *, task="auto",
   backend=None, shots=None, noise_model=None, engine="auto")`.
4. Route existing `SVEstimator`, `ShotEstimator`, and `ShotSampler`
   through this planner only for metadata first; execution should remain
   behaviorally identical.
5. Add tests proving existing primitive outputs do not change except for added
   metadata.

Verification:

```bash
env PYTHONPATH=. pytest tests/primitives
```

### Phase 2: Strengthen batched QML and variational execution

Files:

- `aicir/core/batch.py`
- `aicir/qml/deriv.py`
- `aicir/qml/diff/registry.py`
- `aicir/primitives/estimator.py`
- tests under `tests/core/`, `tests/qml/`, and `tests/primitives/`

Steps:

1. Add helper functions that evolve a `Circuit` through `BatchSV` when the gate
   set is supported.
2. Add clear fallback metadata when a circuit cannot use `BatchSV`.
3. Support grouped parameter batches for fixed-topology VQE/QML evaluation.
4. Keep `NumpyBackend`, `GPUBackend`, and `NPUBackend` behavior unchanged for
   non-batched calls.
5. Add tests comparing `BatchSV` outputs with the normal state vector path for
   small circuits.

Verification:

```bash
env PYTHONPATH=. pytest tests/core tests/qml tests/primitives
```

### Phase 3: Add QAS prefilters, cache hooks, and grouped evaluation

Files:

- `aicir/qas/evaluator.py`
- `aicir/qas/architecture_search.py`
- `aicir/qas/_utils.py`
- `aicir/metrics/`
- `aicir/primitives/estimator.py`
- new `tests/qas/test_qas_evaluation_planning.py`

Steps:

1. Add a stable circuit-structure hash based on `CircuitIR` operations,
   qubits, controls, and gate names, excluding trainable parameter values when
   weight sharing is intended.
2. Add optional structural prefilters for depth, two-qubit count, entangler
   topology, trainability proxies, and hardware efficiency.
3. Add a cache interface for candidate scores and reusable prefix/suffix states.
4. Add grouped evaluation for candidates with compatible topology.
5. Ensure MoG_VQE and existing QAS algorithms can use the same evaluator path.

Verification:

```bash
env PYTHONPATH=. pytest tests/qas tests/metrics
```

### Phase 4: Add QEC result contracts and Clifford analysis

Files:

- new `aicir/qec/`
- `aicir/ir/accessors.py`
- `aicir/gates/registry.py`
- `aicir/primitives/results.py`
- `aicir/measure/measure.py`
- tests under `tests/qec/`

Steps:

1. Add `QECResult` with `syndrome_history`, `logical_error_rate`,
   `decoder_metadata`, `pauli_frame`, and `metadata`.
2. Add Clifford gate classification helpers using the gate registry and typed
   IR accessors.
3. Add Pauli-measurement and reset capability checks.
4. Add tests for classification of Clifford, non-Clifford, measurement, and
   reset circuits.

Verification:

```bash
env PYTHONPATH=. pytest tests/qec tests/ir tests/gates
```

### Phase 5: Implement stabilizer/tableau and Pauli-frame engines

Files:

- new `aicir/qec/stabilizer.py`
- new `aicir/qec/pauli_frame.py`
- new `aicir/qec/sampler.py`
- `aicir/primitives/sampler.py` if a generic facade is added
- tests under `tests/qec/`

Steps:

1. Implement tableau state initialization, Clifford gate updates, Pauli
   measurement, reset, and shot sampling.
2. Implement Pauli-frame correction tracking separately from physical gate
   application.
3. Provide a QEC sampler facade that returns `QECResult`.
4. Cross-check small Clifford circuits against the existing state vector or
   density-matrix path.
5. Add planner dispatch for `task="qec"` when the circuit is
   Clifford-compatible.

Verification:

```bash
env PYTHONPATH=. pytest tests/qec tests/measure
```

### Phase 6: Add noisy scaling paths

Files:

- `aicir/noise/`
- `aicir/measure/trajectory.py`
- new `aicir/qec/noise.py` for QEC-specific Pauli-noise helpers
- tests under `tests/noise/`, `tests/measure/`, and `tests/qec/`

Steps:

1. Keep density matrix simulation as the exact small-system reference.
2. Add or extend trajectory execution for larger noisy circuits.
3. Share `NoiseModel` definitions across density matrix, trajectory, and QEC
   paths.
4. Add planner rules for density-matrix limits and trajectory fallback.
5. Add statistical tests with deterministic seeds and tolerances.

Verification:

```bash
env PYTHONPATH=. pytest tests/noise tests/measure tests/qec
```

### Phase 7: Add optional tensor-network/MPS engine

Files:

- new `aicir/tensor_network/`
- planner module from Phase 1
- tests under `tests/tensor_network/`

Steps:

1. Start with opt-in MPS support for 1D local circuits.
2. Add exact comparisons against state vector for small circuits.
3. Do not enable automatic planner selection until benchmarks justify it.
4. Keep optional dependencies optional.

Verification:

```bash
env PYTHONPATH=. pytest tests/tensor_network
```

## 7. Public API Policy

- Preserve `Circuit`, `CircuitIR`, `Operation`, `Measurement`, `Observable`,
  `State`, `Measure`, `SVEstimator`, `ShotEstimator`, `ShotSampler`,
  `NumpyBackend`, `GPUBackend`, `NPUBackend`, and `BatchSV`.
- Preserve `Circuit.gates` compatibility while internal consumers continue
  moving toward typed IR accessors.
- Add new engines behind primitives/planner facades instead of requiring users
  to instantiate low-level simulation internals.
- Keep optional dependencies optional. Tensor network, QIR, MLIR, Qiskit, or
  PennyLane integrations must not become mandatory for core simulation.
- Update `README.md`, `CHANGELOG.md`, and relevant submodule READMEs when a
  public API changes. A design document alone does not require a changelog entry.

## 8. Final Recommendation

Use a hybrid architecture:

```text
QML -> batched differentiable state vector on GPU/NPU, with explicit gradient transforms
QAS -> typed IR + primitives + structural prefilter + caching + grouped evaluation
QEC -> stabilizer/tableau + Pauli frame, with density/trajectory validation
```

The integration point is:

```text
one typed IR
one primitive/result API
one automatic but overrideable execution planner
multiple specialized simulation engines
```

This keeps `aicir` pragmatic: QML gets the fast tensor path, QAS gets circuits
as mutable searchable data, and QEC gets the polynomial-time representation it
needs instead of forcing every workload through a dense state vector simulator.

## 9. Implementation Schedule

This breaks every phase in Section 6 into concrete, assignable work items and
sorts them by **Serial Execution** (must run in order — dependency bottlenecks)
versus **Parallel Execution** (independent tracks for concurrent teams). Each
item lists its dependency; per-phase files and verification commands are in
Section 6.

### 9.0 Dependency Graph

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
| Phase 1 | Serial root | None | Core architect(s) |
| Phase 2 → Phase 3 | Serial within Track A | Phase 1 | Team A (QML/QAS) |
| Phase 4 → Phase 5 | Serial within Track B | Phase 1 | Team B (QEC) |
| Phase 6 | Parallel inside Track B | Phase 1 | Team B (Noise) |
| Phase 7 | Independent | Phase 1 | Team C (Optional) |

Tracks A, B, and C run fully in parallel once Phase 1 lands.

### 9.1 Serial Execution — Foundational Core

Must be completed before the parallel tracks can integrate.

**Phase 1: Planner Contracts & Core API (global bottleneck)**
*Introduces the routing mechanism with no behavior change. The five items are
themselves serial (1.1 → 1.2/1.3 → 1.4 → 1.5).*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 1.1 | Define immutable `ExecutionPlan` dataclass (`task`, `engine`, `backend_name`, `shots`, `supports_grad`, `noisy`, `batched`, `warnings`) | — |
| 1.2 | Add planner metadata to `EstimateResult.metadata` and `SampleResult.metadata` without breaking existing fields | 1.1 |
| 1.3 | Implement `select_execution_plan(circuit, *, task="auto", backend=None, shots=None, noise_model=None, engine="auto")` | 1.1 |
| 1.4 | Route `SVEstimator`, `ShotEstimator`, `ShotSampler` through the planner **for metadata only** — execution stays behaviorally identical | 1.2, 1.3 |
| 1.5 | Tests proving existing primitive outputs are unchanged except for added metadata | 1.4 |

**Gate:** Phase 1 must be green (`pytest tests/primitives`) before forking the
parallel tracks below.

### 9.2 Parallel Execution Tracks

Once Phase 1 is complete, staff split into three independent tracks.

#### Track A — QML & QAS (Tensor & Batching focus) — Team A

**Phase 2: Strengthen batched QML and variational execution**
*Serial prerequisite for Phase 3.*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 2.1 | Helper functions to evolve a `Circuit` through `BatchSV` when the gate set is supported | Phase 1 |
| 2.2 | Clear fallback metadata when a circuit cannot use `BatchSV` | 2.1 |
| 2.3 | Grouped parameter batches for fixed-topology VQE/QML evaluation | 2.1 |
| 2.4 | Keep `NumpyBackend` / `GPUBackend` / `NPUBackend` unchanged for non-batched calls (regression guard) | 2.1 |
| 2.5 | Tests comparing `BatchSV` output vs. normal state vector path for small circuits | 2.1–2.4 |

**Phase 3: QAS prefilters, cache hooks, grouped evaluation**
*Depends on Phase 2 (reuses the batched evaluation path).*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 3.1 | Stable circuit-structure hash from `CircuitIR` ops/qubits/controls/gate names, excluding trainable param values when weight sharing is intended | Phase 2 |
| 3.2 | Optional structural prefilters: depth, two-qubit count, entangler topology, trainability proxies, hardware efficiency | Phase 2 |
| 3.3 | Cache interface for candidate scores and reusable prefix/suffix states | 3.1 |
| 3.4 | Grouped evaluation for candidates with compatible topology | 3.2, 3.3 |
| 3.5 | Ensure MoG_VQE and existing QAS algorithms use the same evaluator path | 3.4 |

#### Track B — QEC & Noise (Stabilizers & Trajectories focus) — Team B

*Track B has its own serial chain (Phase 4 → Phase 5) plus Phase 6 in parallel
with that chain. With enough staff, split into B1 (engines: Phase 4→5) and
B2 (noise: Phase 6).*

**Phase 4: QEC result contracts & Clifford analysis**
*Serial prerequisite for Phase 5.*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 4.1 | `QECResult` with `syndrome_history`, `logical_error_rate`, `decoder_metadata`, `pauli_frame`, `metadata` | Phase 1 |
| 4.2 | Clifford gate classification helpers using gate registry + typed IR accessors | Phase 1 |
| 4.3 | Pauli-measurement and reset capability checks | 4.2 |
| 4.4 | Tests for classification of Clifford / non-Clifford / measurement / reset circuits | 4.1–4.3 |

**Phase 5: Stabilizer/tableau & Pauli-frame engines**
*Depends on Phase 4.*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 5.1 | Tableau state init, Clifford gate updates, Pauli measurement, reset, shot sampling | Phase 4 |
| 5.2 | Pauli-frame correction tracking, separate from physical gate application | Phase 4 |
| 5.3 | QEC sampler facade returning `QECResult` | 5.1, 5.2, 4.1 |
| 5.4 | Cross-check small Clifford circuits against state vector / density-matrix path | 5.1 |
| 5.5 | Planner dispatch for `task="qec"` when circuit is Clifford-compatible | 5.3, Phase 1 |

**Phase 6: Noisy scaling paths**
*Parallel with Phases 4/5 (only depends on Phase 1).*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 6.1 | Keep density matrix simulation as the exact small-system reference (guard) | Phase 1 |
| 6.2 | Add/extend trajectory execution for larger noisy circuits | Phase 1 |
| 6.3 | Share `NoiseModel` definitions across density matrix, trajectory, and QEC paths | 6.2 |
| 6.4 | Planner rules for density-matrix limits and trajectory fallback | 6.2, Phase 1 |
| 6.5 | Statistical tests with deterministic seeds and tolerances | 6.1–6.4 |

> Item 6.3 touches `aicir/qec/noise.py`; coordinate the shared `NoiseModel`
> surface with B1 once Phase 4's `aicir/qec/` package exists.

#### Track C — Optional Engines — Team C

**Phase 7: Tensor-network/MPS engine**
*Independent, lowest priority. Only depends on Phase 1's planner module.*

| Item | Task | Depends on |
| :--- | :--- | :--- |
| 7.1 | Opt-in MPS support for 1D local circuits | Phase 1 |
| 7.2 | Exact comparisons against state vector for small circuits | 7.1 |
| 7.3 | Do **not** enable automatic planner selection until benchmarks justify it | 7.1 |
| 7.4 | Keep optional dependencies optional | 7.1 |

### 9.3 Scheduling Summary

| When | Serial work | Parallel work |
| :--- | :--- | :--- |
| Sprint 0 | **Phase 1** (1.1→1.5) by core architects | — |
| Sprint 1+ | Phase 2→3 (Team A); Phase 4→5 (Team B1) | Phase 6 (Team B2); Phase 7 (Team C) |

**Critical path:** Phase 1 → (longest of: Phase 2→3 *or* Phase 4→5). Phases 6
and 7 should never sit on the critical path; staff them only after the dependent
serial chains (A and B1) are covered.

**Cross-cutting (per Section 7):** any item that changes a public surface must
also update `README.md`, `CHANGELOG.md`, and the relevant submodule README, and
must preserve the listed public symbols and `Circuit.gates` compatibility. Keep
optional dependencies optional throughout.

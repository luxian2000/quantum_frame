# aicir Execution-Architecture Design: Schemes, Trade-offs, and Integration

This document records the main architectural schemes for exposing and executing
quantum circuits in `aicir`, weighs their advantages and disadvantages, maps the
best-fit scheme to each of the three core workloads — Quantum Machine Learning
(QML), Quantum Architecture Search (QAS), and Quantum Error Correction (QEC) —
and proposes a single layered architecture that integrates the advantages of all
of them.

**Core thesis:** No single scheme is best across QML, QAS, and QEC, because the
three workloads stress different performance axes. A `QNode` fuses three
concerns into one object — *circuit-as-data*, *execution (device)*, and
*differentiation (gradient rule)*. The winning design is to **keep these three
concerns orthogonal** over a shared IR, so each workload composes the
combination it needs while reusing common infrastructure.

```text
        Circuit-as-data (IR)   ×   Execution (Backend / Primitive)   ×   Differentiation (transform)
```

---

## 1. Main Architectural Alternatives

### 1.1 QNode (PennyLane style)

Wraps `(quantum function + device + measurement + gradient rule)` into one
differentiable callable.

- **Advantages**
  - Best ergonomics for the common case: one object, minimal boilerplate.
  - Natural to drop into a PyTorch/JAX graph as a layer (hybrid models).
  - A convenient place to auto-select the differentiation method.
- **Disadvantages**
  - Couples circuit definition to device and gradient rule → hard to "compile
    once, run on many backends."
  - Structure is fixed at node-construction time, which is the *wrong* model
    when the circuit structure itself is the variable (QAS).
  - Monolithic: heavy machinery (adjoint, QFIM, transpilation) ends up routed
    *around* the node anyway (e.g. PennyLane's `metric_tensor` and Catalyst are
    transforms, not node features).

### 1.2 Primitives (Qiskit style)

Separate the passive `Circuit` data structure from execution engines
(`Estimator` for expectation values, `Sampler` for shots). Binding happens
per *call*: `run(circuits, observables, params)`.

- **Advantages**
  - Clean separation of concerns; circuit is compiled/optimized once and reused.
  - Batch-first: evaluate many circuits / collect many shots in one call —
    ideal for throughput on simulators and hardware.
  - Standard, hardware/runtime-friendly interface.
- **Disadvantages**
  - More verbose for a single small QML task.
  - Less "native feel" embedded inside an autodiff graph.

### 1.3 Tape / IR-as-data + Transforms (MLIR / Catalyst / TKET style)

The circuit is an explicit IR object; differentiation, optimization,
transpilation, and execution are all *transforms* (passes) over that IR.

- **Advantages**
  - Introspection is first-class: adjoint generators, QFIM, gate counts,
    structural metrics all read directly off the IR.
  - Composable passes; compiler-level optimization and large speedups possible.
  - Decouples "what the circuit is" from "how it is differentiated/run."
- **Disadvantages**
  - Less turnkey for newcomers than a QNode for a single circuit.
  - Requires a well-designed IR up front.

### 1.4 Autodiff-native / Backpropagation through the Unitary (JAX / Torch style)

Gates are tensor operations; the autodiff engine's graph *is* the differentiable
record. No node, no tape needed for gradients.

- **Advantages**
  - Fastest wall-clock on GPU/NPU for moderate sizes (fused ops, one backward
    pass); exact gradients.
  - Composes seamlessly with classical layers for hybrid models.
  - Zero custom differentiation machinery.
- **Disadvantages**
  - Simulator-only; no shot-based or hardware gradients.
  - Memory cost O(depth · 2ⁿ) (stores intermediate states for backprop).
  - Does not provide the QFIM / quantum-geometry quantities for free.

### 1.5 Pure functional callables (`fn: params → scalar`)

The circuit is just a function; differentiation is a transform on that function
(`psr(fn, params)`).

- **Advantages**
  - Maximal composability, no hidden state; works even on non-circuit
    objectives.
  - Backend-agnostic; trivial to test.
- **Disadvantages**
  - Cannot introspect structure (no adjoint/QFIM) without a separate handle to
    the circuit.
  - Duplication risk when two related callables describe the same circuit
    (e.g. `fn` for the loss and `state_fn` for the QFIM in `qng`).

### 1.6 Summary table

| Scheme | Best at | Weakest at |
|---|---|---|
| QNode | single-circuit ergonomics, hybrid layers | structural search, compile-once/run-many |
| Primitives | batch throughput, hardware/runtime | single-task verbosity, autodiff embedding |
| IR + transforms | introspection, compiler passes, decoupling | turnkey simplicity |
| Autodiff-native | GPU/NPU speed, hybrid models, exactness | hardware/shots, memory, no QFIM |
| Functional callables | composability, minimal machinery | no structure introspection |

---

## 2. Best-Fit Scheme per Workload

The three workloads vary *different things* and therefore have *different
bottlenecks*.

| Workload | What varies | Bottleneck | Best-fit scheme(s) |
|---|---|---|---|
| **QML** | continuous parameters (fixed structure) | repeated gradients over the same circuit; hybrid classical layers | **Autodiff-native** (`auto`) for GPU/NPU speed; **adjoint** (`ad`) for memory; **QNG** for convergence |
| **QAS** | the circuit **structure** itself | construct / mutate / evaluate *many* circuits; caching & weight sharing | **IR-as-data** + batched **Estimator** + supernet weight-sharing |
| **QEC** | qubit count & shots (mostly Clifford) | state-vector is exponential; need stabilizer sim + shot throughput | **Stabilizer (tableau) backend** + **Sampler** primitive |

### 2.1 QML — autodiff-native and adjoint win

Structure is fixed; gradients are taken thousands of times. `auto` computes the
full gradient in one backward pass (fastest on GPU/NPU, but O(depth · 2ⁿ)
memory). `ad` is O(P) gate-applications with only O(1) extra state storage (the
memory-efficient winner) and is backend-agnostic. Both beat `psr`'s O(P²) cost
for the full gradient. `qng` (QFIM preconditioning) accelerates convergence in
flat/ill-conditioned landscapes. A QNode here is *fine* but adds nothing the
transforms don't already provide.

### 2.2 QAS — IR-as-data is mandatory; QNode is a poor fit

The circuit is the search variable, so it must be first-class, manipulable data
you can mutate, hash, cache, and batch. A QNode fixes structure at construction
— exactly the wrong abstraction. The real performance levers are **weight
sharing** (a supernet scores many architectures with one shared parameter set —
already implemented in `aicir/qas/VQA_QAS.py`) and **batched evaluation** of
candidates via an Estimator primitive. Structural metrics (expressibility,
trainability, gate counts) read directly off the IR.

### 2.3 QEC — a different regime entirely

Stabilizer / Clifford circuits simulate in **polynomial** time via a tableau
backend (≈ O(n²) per operation) versus **exponential** O(2ⁿ) for state-vector —
the difference between simulating ~1000 qubits and stalling near ~30. QEC also
needs high-throughput **shot sampling** for syndrome extraction plus a decoder.
Differentiation is largely irrelevant *except* for ML-based decoders, which are
ordinary classical networks trained with standard autodiff — and therefore
reuse the QML path.

---

## 3. Integration Strategy — One Layered Architecture

Integrate the advantages by **decoupling the three concerns** a QNode fuses, on
top of a single IR. Each layer can be specialized for performance without
disturbing the others.

```text
┌─────────────────────────────────────────────────────────────────────┐
│ (Optional) thin QNode-like binder  — QML ergonomics only, never load-bearing │
├─────────────────────────────────────────────────────────────────────┤
│ Differentiation transforms (à la carte over IR)                       │
│   psr · spsr · spsa · mpsr · fd · ad · auto · qng/bdqng/kqng/dqng …    │
├─────────────────────────────────────────────────────────────────────┤
│ Primitives  —  Estimator (batched ⟨O⟩)  ·  Sampler (batched shots)    │
├─────────────────────────────────────────────────────────────────────┤
│ Execution backends (Backend ABC)                                      │
│   state-vector: Numpy · GPU(Torch) · NPU      stabilizer: tableau     │
├─────────────────────────────────────────────────────────────────────┤
│ Circuit IR (the single source of truth — the "tape")                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer responsibilities**

1. **Circuit IR** — the single source of truth for all three workloads.
   Differentiation, search, transpilation, and metrics are functions *of* the
   IR, not properties baked into a callable.
2. **Backends (executors)** behind the existing `Backend` ABC. State-vector
   backends (Numpy / GPU / NPU) serve QML/QAS; a new **stabilizer/tableau
   backend** unlocks QEC scale. Upper layers do not change.
3. **Primitives** — a batched **Estimator** (expectation values) and **Sampler**
   (shots) as the shared throughput layer for QAS (many candidates) and QEC
   (many syndrome shots).
4. **Differentiation transforms** — kept à la carte over the IR (QML and
   ML-decoders consume them; QEC mostly ignores them).
5. **Optional thin binder** — purely for QML ergonomics and to remove the
   `fn`/`state_fn` duplication in `qng`. Never required by the lower layers.

**Per-workload composition over the shared stack**

```text
QML  →  Circuit ── ad / auto + qng ──────────── GPU / NPU state-vector
QAS  →  Circuit-as-data ── supernet weight-share ── batched Estimator
QEC  →  Circuit ── (no diff) ── stabilizer backend + Sampler ── ML-decoder reuses QML path
```

---

## 4. Where aicir Stands and the Gaps to Close

aicir already implements most of this layered design:

- **Circuit IR** — dict/`Operation`-based tape (`aicir/core`).
- **Backend ABC** — `NumpyBackend`, `GPUBackend` (Torch), `NPUBackend`
  (`aicir/channel/backends`).
- **Differentiation transforms** — `psr · spsr · spsa · mpsr · fd · ad · auto ·
  qng · bdqng · kqng · dqng · rotosolve` (`aicir/qml/deriv.py`), all
  NPU-compatible.
- **QAS weight sharing** — supernet in `aicir/qas/VQA_QAS.py`.
- **Noise** — density-matrix path (`aicir/channel/noise`).

**Two gaps remain to fully integrate the advantages:**

1. **Stabilizer (tableau) backend + `Sampler` primitive** → unlocks QEC at
   hundreds/thousands of qubits and high-throughput syndrome sampling. Slots in
   behind the existing `Backend` ABC; the QML/QAS paths are untouched.
2. **Batched `Estimator` primitive** → the throughput layer QAS and QEC share
   (evaluate N circuits / collect M shots per call).

Both fit behind interfaces that already exist, so neither disturbs the QML path.
An optional thin **binder** (circuit-builder + observable + backend exposing
`.expectation(params)` and `.state(params)`) is a nice-to-have that gives
newcomers QNode-like ergonomics and lets `qng` derive both the loss and the
state from one circuit source — without re-coupling device and gradient rule.

---

## 5. Recommendation (Phased)

1. **Keep the transform-on-IR core** — it is already "better than a monolithic
   QNode" for a framework: composable, backend-pluggable, NPU-friendly.
2. **Add the stabilizer backend + Sampler** — the single highest-leverage
   addition; it is what makes QEC tractable and is otherwise impossible in the
   state-vector regime.
3. **Add a batched Estimator** — accelerates QAS screening and QEC shot
   collection through one shared API.
4. **Add an optional binder last** — pure ergonomics; never let lower layers
   depend on it.

**Conclusion:** Best performance comes from *per-layer specialization on a shared
IR*, not from choosing one global scheme. QML wants autodiff/adjoint (+QNG),
QAS wants IR-as-data with batched evaluation and weight sharing, and QEC wants a
stabilizer backend with a Sampler. Decoupling circuit / execution /
differentiation lets all three coexist and lets each pick its fastest path —
which is precisely the architecture aicir is already converging toward.

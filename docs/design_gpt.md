# quantum_frame Simulator Design: Hybrid Execution and Simulation Scheme

This document records the main alternatives for a quantum simulator whose core workloads are quantum machine learning (QML), quantum architecture search (QAS), and quantum error correction (QEC). The conclusion is that no single execution or simulation scheme dominates all workloads. The best design is a shared typed IR and primitives API with a simulation planner that dispatches to specialized engines.

## 1. Design Goals

`quantum_frame` should support three different performance profiles:

- QML: fixed or slowly changing parameterized circuits, repeated forward/backward passes, GPU/NPU acceleration, and differentiable objectives.
- QAS: many candidate circuits, frequent topology changes, structural metrics, fast approximate screening, and optional VQE-style energy evaluation.
- QEC: Clifford-heavy circuits, Pauli noise, syndrome measurement, reset, conditional correction, and eventually dynamic control flow.

The core design principle is:

```text
Circuit / typed IR stays independent from execution strategy.
Execution strategy is chosen by the planner based on workload, circuit features, and requested result.
```

## 2. Main API and Execution Alternatives

### 2.1 QNode-Style Callable

A QNode-style interface wraps circuit construction, backend, observable, shots, and differentiation into one callable.

Advantages:

- Excellent user experience for QML.
- Natural PyTorch/JAX-like model integration.
- Good place to hide gradient-rule selection.
- Convenient for rapid prototyping.

Disadvantages:

- Couples circuit definition to execution policy.
- Less natural for QAS, where topology changes constantly.
- Can obscure compilation, caching, and backend reuse.
- Not a sufficient internal abstraction for QEC dynamic circuits.

Best use:

- Optional QML frontend, not the core simulator abstraction.

### 2.2 Primitives Pattern: Estimator and Sampler

The circuit is a passive object. Execution happens through `Estimator`, `Sampler`, `Measure`, or task-specific runners.

Advantages:

- Clean separation between circuit data and execution.
- Works well with many backends and many observables.
- Natural for QAS: candidate circuits can be generated, scored, cached, and discarded cheaply.
- Matches existing `aicir.primitives` direction.

Disadvantages:

- Slightly more verbose than a QNode for simple QML models.
- Needs explicit parameter binding or a parameter-binding batch interface for best QML performance.

Best use:

- Primary public execution API for algorithms and benchmarks.

### 2.3 Pure Tensor Graph

The whole circuit evaluation is represented as tensor operations in a framework such as PyTorch, preserving the autograd graph end to end.

Advantages:

- Best fit for differentiable QML and VQE training.
- Supports GPU/NPU acceleration.
- Supports automatic differentiation for supported gates.
- Avoids parameter-shift overhead when the backend can differentiate directly.

Disadvantages:

- Dynamic topology changes require rebuilding the graph.
- Large Python-side graph construction can become expensive.
- Not portable to real QPUs.
- Not ideal for Clifford/QEC workloads that have much faster specialized representations.

Best use:

- QML and fixed-ansatz variational algorithms.

### 2.4 AOT/JIT Compilation

Circuits or dynamic programs are compiled before execution, for example to OpenQASM 3.0, QIR, MLIR, or a compiled internal kernel.

Advantages:

- Reduces repeated interpreter overhead.
- Essential for hardware-facing dynamic circuits and fast classical feed-forward.
- Can enable gate fusion, scheduling, layout, and control-flow lowering.

Disadvantages:

- High engineering complexity.
- Compilation overhead is wasteful when topology changes every iteration.
- Differentiation through compiled code requires careful integration.

Best use:

- QEC hardware path, repeated fixed circuits, and high-throughput simulation kernels.

## 3. Main Simulation Engine Alternatives

### 3.1 Statevector Simulation

Stores a pure state vector of size `2^n`.

Advantages:

- Best default for noiseless algorithms.
- Simple and exact.
- Good for VQE, QAOA, QML, and small-to-medium QAS evaluation.
- Works with CPU, GPU, and NPU backends.

Disadvantages:

- Memory grows as `O(2^n)`.
- Noise requires either density matrix or trajectory methods.
- Poor fit for large QEC circuits where stabilizer methods are available.

Best use:

- Default exact simulator for noiseless QML/QAS.

### 3.2 Batched Statevector Simulation

Stores and evolves many statevectors together, usually for parameter batches, data batches, or candidate batches.

Advantages:

- Highest practical throughput for QML mini-batches.
- Useful for QAS candidate evaluation and parameter sweeps.
- Maps well to GPU/NPU devices.
- Can share gate kernels across a batch.

Disadvantages:

- More complex memory layout.
- Batch members must have compatible circuit structure for maximum efficiency.
- Less useful for highly irregular QAS candidates unless grouped by topology.

Best use:

- QML training and high-throughput QAS evaluation.

### 3.3 Density Matrix Simulation

Stores a full density matrix of size `4^n`.

Advantages:

- Exact simulation of mixed states and noise channels.
- Natural for reset, decoherence, readout noise, and non-unitary channels.
- Directly supports exact expectation values under noise.

Disadvantages:

- Memory and runtime scale as `O(4^n)`.
- Quickly becomes infeasible beyond small systems.
- Overkill for noiseless QML/QAS.

Best use:

- Small noisy circuits and correctness validation for noise models.

### 3.4 Quantum Trajectory / Monte Carlo Wavefunction

Samples noisy evolution as many stochastic pure-state trajectories.

Advantages:

- More scalable than density matrix for larger noisy systems.
- Reuses statevector infrastructure.
- Naturally parallel across trajectories.
- Gives statistical error bars.

Disadvantages:

- Stochastic estimator; requires many trajectories for low variance.
- Harder to debug than exact density matrix.
- Not always efficient for rare-event noise or high-precision observables.

Best use:

- Medium-size noisy QML/QAS experiments and approximate QEC noise studies.

### 3.5 Stabilizer / Tableau / Pauli-Frame Simulation

Represents Clifford circuits and Pauli measurements compactly instead of storing amplitudes.

Advantages:

- Orders of magnitude faster than statevector for Clifford circuits.
- Scales to many more qubits.
- Ideal for QEC syndrome extraction, stabilizer codes, Pauli noise, and Pauli-frame tracking.
- Supports repeated measurement and correction naturally.

Disadvantages:

- Cannot directly represent arbitrary non-Clifford rotations.
- Hybrid Clifford + non-Clifford simulation requires fallback, decomposition, or approximation.
- Different internal representation from statevector/density matrix.

Best use:

- QEC core engine.

### 3.6 Tensor Network / MPS Simulation

Represents states or circuits as tensor networks, often exploiting locality and low entanglement.

Advantages:

- Can exceed statevector qubit counts for low-entanglement circuits.
- Good for 1D and local 2D circuits.
- Useful for shallow circuits and structured ansatz families.

Disadvantages:

- Performance depends strongly on entanglement and contraction order.
- Harder to integrate with arbitrary mid-circuit control and noise.
- Engineering complexity is higher than statevector or stabilizer engines.

Best use:

- Optional large-qubit engine for low-entanglement QML/QAS circuits.

## 4. Best Scheme by Workload

| Workload | Primary scheme | Secondary scheme | Avoid as default |
| --- | --- | --- | --- |
| QML | Batched tensor/statevector with autograd | QNode-style frontend over primitives | Density matrix unless noise is required |
| QAS | Primitives + statevector/structural scoring + caching | Batched candidate evaluation, tensor network for low-entanglement candidates | One monolithic QNode per candidate |
| QEC | Stabilizer/tableau/Pauli frame | Density matrix for small exact noisy validation; trajectories for larger noisy validation | Dense statevector for Clifford-heavy codes |

### 4.1 QML Recommendation

Use a tensor-backed batched statevector engine as the performance path.

Recommended stack:

```text
QML model frontend
  -> CircuitIR with parameters
  -> parameter-binding batch
  -> GPUBackend / NPUBackend / BatchSV
  -> differentiable expectation
  -> autograd or selected gradient method
```

Why:

- QML repeatedly evaluates the same topology.
- Batch throughput matters more than topology flexibility.
- Autograd and device acceleration are decisive.

### 4.2 QAS Recommendation

Use primitives over lightweight `Circuit` objects, with a planner that chooses between structural scoring, statevector evaluation, and batched evaluation.

Recommended stack:

```text
QAS candidate generator
  -> Circuit / CircuitIR
  -> structural prefilter
  -> cached prefix/suffix simulation when possible
  -> StatevectorEstimator or task-specific evaluator
  -> multi-objective ranking / search policy
```

Why:

- QAS changes topology often.
- Rebuilding a tensor graph for every candidate is expensive.
- Many candidates can be rejected by structural metrics before full simulation.
- Candidate groups with the same topology can still use batched parameter evaluation.

### 4.3 QEC Recommendation

Use a stabilizer/tableau and Pauli-frame simulator as the primary QEC engine.

Recommended stack:

```text
QEC circuit / code description
  -> dynamic CircuitIR with measurement/reset/classical bits
  -> Clifford and Pauli-noise classifier
  -> StabilizerEngine / PauliFrameEngine
  -> syndrome history and logical error statistics
```

Why:

- Most QEC circuits are Clifford-heavy.
- Syndrome extraction and Pauli corrections are naturally represented by tableaus and Pauli frames.
- Dense amplitude simulation wastes memory and runtime for these circuits.

Use density matrix only for small exact checks, and trajectories for larger noisy studies.

## 5. Integration Strategy: Combine the Advantages

The integration point should not be a single simulator class. It should be a planner and a set of engines behind one stable API.

### 5.1 Common Layers

```text
Frontend APIs
  - Circuit construction
  - optional QNode-like QML wrapper
  - QAS runners
  - QEC code/circuit builders

Typed IR
  - gates
  - parameters
  - observables
  - measurement/reset
  - future classical conditions

Analysis and optimization
  - gate classification
  - Clifford/non-Clifford classification
  - noise classification
  - topology and depth metrics
  - gate fusion and canonicalization

Simulation planner
  - selects engine
  - creates execution plan
  - chooses gradient/noise/sampling path

Engines
  - statevector
  - batched statevector
  - density matrix
  - trajectory
  - stabilizer/tableau/Pauli frame
  - tensor network

Result layer
  - state/result objects
  - estimator result
  - sampler result
  - QEC syndrome/logical error result
```

### 5.2 Dispatch Rules

The planner should use deterministic rules first:

```text
if task == "qec" and circuit is Clifford-compatible:
    use StabilizerEngine or PauliFrameEngine

elif noise_model is not None and n_qubits <= density_matrix_limit:
    use DensityMatrixEngine

elif noise_model is not None:
    use TrajectoryEngine

elif task in {"qml", "vqe", "qaoa"} and topology is fixed:
    use BatchedStatevectorEngine with GPU/NPU autograd if available

elif task == "qas":
    use structural prefilter first
    then use StatevectorEstimator or grouped BatchedStatevectorEngine

elif circuit has local geometry and low expected entanglement:
    optionally use TensorNetworkEngine

else:
    use StatevectorEngine
```

The planner must be overrideable:

```python
estimator = Estimator(engine="statevector")
estimator = Estimator(engine="batched_statevector")
estimator = Estimator(engine="stabilizer")
estimator = Estimator(engine="auto")
```

### 5.3 Shared Result Contract

All engines should report through consistent result objects:

- `state`: final state when meaningful.
- `counts` / `probs`: sampling output.
- `expectation_values`: observable estimates.
- `variances`: shot or trajectory variance.
- `metadata`: engine name, precision, shots, trajectories, compilation plan, warnings.

QEC needs an additional result shape:

- `syndrome_history`
- `logical_error_rate`
- `decoder_metadata`
- `pauli_frame`

## 6. Implementation Priorities for quantum_frame

### Phase 1: Strengthen the Current Default

- Keep `Circuit`, `CircuitIR`, `Backend`, `Measure`, `Estimator`, and `Sampler` as the stable public surface.
- Improve batched statevector paths for QML and repeated variational evaluation.
- Add explicit planner metadata to primitive results.
- Add structural prefilters and prefix caching hooks for QAS.

### Phase 2: Add QEC-Specific Engine

- Add Clifford gate classification in the typed IR.
- Implement a stabilizer/tableau backend.
- Add Pauli-frame tracking.
- Add QEC result objects for syndrome and logical error reporting.

### Phase 3: Add Noisy Scaling Paths

- Keep density matrix for small exact noisy simulation.
- Add trajectory simulation for larger noisy circuits.
- Share noise model definitions across density and trajectory engines.

### Phase 4: Add Optional Tensor Network Engine

- Start with MPS for 1D local circuits.
- Use it as an opt-in engine first.
- Add planner heuristics only after benchmark evidence.

## 7. Final Recommendation

The best performance scheme is:

```text
QML: batched differentiable statevector on GPU/NPU
QAS: primitives + structural prefilter + statevector/batched evaluation + caching
QEC: stabilizer/tableau + Pauli-frame engine, with density/trajectory validation
```

The best integration scheme is:

```text
one typed IR
one primitive/result API
multiple specialized engines
an automatic but overrideable simulation planner
```

This design preserves the simplicity of the current `aicir` API while allowing each domain to use the representation that actually matches its computational structure.

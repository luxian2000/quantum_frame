# Distributed Native Autograd and Multi-NPU Performance Design

**Date:** 2026-07-31

**Status:** Approved design

## 1. Goal

Extend `aicir.distributed` from a forward-only sharded simulator into a
native-autograd, multi-NPU simulator without weakening its strict NPU
contract.

The implementation must:

- retain one public execution entrypoint, `DistSimulator.run()`;
- support every power-of-two process count, `world_size = 2^p`;
- use distributed parameter-shift for shift-rule parameters and CPU float64
  central finite differences for unconstrained state, density-factor, and
  Stinespring parameters;
- enable native PyTorch/torch_npu autograd only after complete 2-, 4-, and
  8-NPU release gates pass;
- preserve sharded statevector and density-matrix execution;
- support differentiable gates, initial states, probabilities,
  expectations, built-in noise parameters, and arbitrary Kraus channels
  represented by a physical Stinespring parameterization;
- prohibit silent CPU or parameter-shift fallback;
- make native autograd faster than the applicable explicit oracle for the
  fixed workloads with at least 32 trainable parameters.

This design concerns classical multi-NPU parallelism for one simulated
quantum system. It does not add quantum-network nodes, LOCC, EPR links, or
remote-QPU protocol semantics.

## 2. Scope

### 2.1 Included

- Exact sharded statevector simulation.
- Exact sharded density-matrix simulation.
- Non-identity logical-to-storage layouts.
- Local and cross-shard gates.
- Parameterized unitary gates and custom unitary parameters.
- Root-owned and already-sharded trainable initial states.
- Physically parameterized trainable density matrices.
- Differentiable local probabilities.
- Differentiable Pauli, Hamiltonian, and local dense expectations.
- Built-in differentiable noise-channel parameters.
- Arbitrary Kraus channels produced from a differentiable Stinespring
  isometry.
- Replicated parameters and optimizer state on every rank.
- Gradient synchronization, checkpoint recomputation, P2P buffer reuse, and
  measured communication/computation overlap.
- Strict 2-, 4-, and 8-NPU probes and evidence artifacts.

### 2.2 Excluded

- Gradients of discrete samples, counts, or collapse results.
- Gradients through measurement-fed classical control flow.
- Native MPS or tensor-network autograd. Those paths continue to use
  parameter-shift.
- Circuit cutting or quantum-network simulation.
- Multi-node release qualification. Multi-node HCCL remains a separate
  follow-on slice after the single-node 2/4/8-NPU gate.
- Parameter sharding. Circuit parameters are small relative to the quantum
  state and remain replicated.
- A fixed speedup multiplier. Native autograd must be faster than the
  applicable explicit oracle, but the release gate does not require `1.5x`
  or another hardware-fragile ratio.

## 3. Public API

### 3.1 Single simulator entrypoint

`DistSimulator.run()` remains the public entrypoint:

```python
result = simulator.run(
    circuit,
    initial_state=initial_state,
    observables={"energy": hamiltonian},
)
loss = result.expectations["energy"]
loss.backward()
optimizer.step()
```

The simulator selects its engine collectively:

```text
no trainable input
    -> current complex64 forward engine

at least one trainable input
    -> paired-real distributed autograd engine
```

Every rank must make the same routing decision. A mismatch is rejected
before the first state collective.

### 3.2 Trainable physical objects

Real gate angles continue to use Torch scalar tensors:

```python
theta = torch.nn.Parameter(
    torch.tensor(0.1, dtype=torch.float32, device="npu")
)
```

Complex physical objects use paired-real parameter containers:

```python
PureStateParam(real, imag)
DensityParam(factor_real, factor_imag)
StinespringParam(real, imag, input_dim, environment_dim)
```

Their physical values are:

```text
psi = z / ||z||
rho = L L^H / Tr(L L^H)
Kraus operators = blocks of a Stinespring isometry
```

Ordinary NumPy arrays and non-trainable complex tensors retain their current
forward-only semantics.

A direct `requires_grad=True` complex leaf is enabled only if the real NPU
capability probe proves its leaf-gradient path safe. Otherwise all ranks
raise one exact error directing the caller to a paired-real parameter
container.

### 3.3 Checkpoint policy

Gradient execution adds:

```python
simulator.run(..., grad_checkpoint="none")
simulator.run(..., grad_checkpoint="auto")
simulator.run(..., grad_checkpoint=16)
```

- `"none"` saves all intermediate distributed states.
- A positive integer saves every N circuit layers and recomputes each
  interval during backward.
- `"auto"` selects an interval from the active NPU memory budget and keeps
  measured peak allocation below 80% of currently available device memory.

Statevector and density-matrix estimators use separate memory formulas.

### 3.4 Result semantics

In gradient mode:

- `result.state` retains the distributed computation graph.
- `result.state.local_data` is a compatibility complex view created through
  one atomic combine boundary.
- `result.local_probabilities` is a differentiable local float tensor.
- `result.expectations[name]` is a replicated differentiable float scalar
  tensor.

The following methods are explicit detach/materialization boundaries:

```python
result.state.gather()
result.state.to_numpy()
result.gather_probabilities()
```

They must not be used to construct a loss.

`sample`, counts, and collapse remain valid in ordinary forward mode. If a
caller attempts to make them part of a gradient execution, every rank raises
the same error before sampling begins.

## 4. NPU-Safe Internal Representation

`CLAUDE.md` records real Ascend failures for complex64 matmul, add,
in-place add, multiply, scalar division, indexing, reduction, QR/SVD, and
fan-out gradient accumulation. It also records the eight-dimensional
`aclnnComplex` limit and the absence of a Gloo backend for NPU tensors.
These are hard design constraints.

### 4.1 Paired-real graph

The native-autograd engine does not propagate raw complex operations:

```text
public complex value
    -> atomic split
    -> (float32 real, float32 imag)
    -> all distributed forward/backward work
    -> atomic combine
    -> public complex view
```

All gradient accumulation occurs on float32 tensors. No distributed
collective transports a complex dtype.

### 4.2 Prohibited implementation patterns

The autograd path must not use:

- raw complex `torch.matmul`;
- raw complex add, multiply, or scalar divide;
- complex advanced indexing;
- complex trace or sum;
- complex HCCL collectives;
- complex QR or SVD;
- rank-n complex reshapes such as `(2,) * n`;
- `.contiguous()` on high-rank complex work tensors;
- a complex tensor fan-out that forces Ascend to accumulate complex
  gradients.

### 4.3 Required replacements

- Matrix products use explicit real/imag formulas.
- Adds, multiplies, and divides use NPU-safe paired-real kernels.
- Diagonal and indexed reads operate on real and imaginary views.
- Traces and probability normalization remain real.
- Qubit permutations use flat bit-index maps and `index_select`/`gather`.
- Work tensors remain one- or two-dimensional where possible.
- Stinespring isometries use custom paired-real Householder reflections;
  they do not call complex QR/SVD.
- HCCL receives independent contiguous float32 real and imaginary buffers.

## 5. Component Boundaries

Add an internal package:

```text
aicir/distributed/autograd/
├── __init__.py
├── _pair.py
├── _collectives.py
├── _vector.py
├── _density.py
├── _reducers.py
├── _channels.py
├── _checkpoint.py
├── _parameters.py
└── _contracts.py
```

Responsibilities:

- `_pair.py`: atomic complex-to-pair and pair-to-complex boundaries.
- `_collectives.py`: differentiable real-valued P2P, all-reduce, scatter,
  and gather.
- `_vector.py`: statevector gate forward/backward kernels.
- `_density.py`: density-matrix gate forward/backward kernels.
- `_reducers.py`: probability and expectation forward/backward kernels.
- `_channels.py`: built-in channels and Stinespring/Kraus kernels.
- `_checkpoint.py`: checkpoint planning and deterministic recomputation.
- `_parameters.py`: parameter structure validation, gradient buckets, and
  replicated optimizer synchronization.
- `_contracts.py`: collective-safe preflight and failure synchronization.

Existing integration points:

```text
aicir/distributed/simulator.py
aicir/distributed/state.py
aicir/distributed/result.py
aicir/distributed/communication.py
aicir/distributed/backend.py
```

The implementation must not place all backward logic in
`simulator.py` or `gates.py`.

## 6. Distributed Backward Semantics

### 6.1 Replicated loss convention

Every rank executes:

```python
loss.backward()
optimizer.step()
```

Global scalar outputs are identical on all ranks. The backward of a
replicated all-reduced scalar uses:

```text
all_reduce(grad_output) / world_size
```

This prevents a replicated loss from multiplying gradients by
`world_size`.

### 6.2 Parameter ownership

Circuit, noise, and Stinespring parameters are replicated. Each rank
computes its local contribution, packs gradients into float32 buckets, and
performs one bucketed all-reduce.

Real parameters use one buffer. Complex physical parameters use independent
real and imaginary buffers. Every rank then performs the same optimizer
step and must retain identical optimizer state.

Initial-state parameters are the ownership exception:

- a root-owned full state/factor is optimized only on root and scattered
  again on the next run;
- a sharded `DistState` factor is optimized shard-locally on every rank;
- neither form enters the replicated parameter-gradient bucket.

Optimizer and parameter equality checks apply to replicated circuit,
noise, and Stinespring parameters. Initial-state checks instead validate
the declared root-owned or sharded ownership contract.

### 6.3 Initial-state ownership

- A root-owned full trainable state is split into real/imag shards.
  Backward gathers real/imag shard gradients to root. Only root receives the
  full input gradient.
- A trainable `DistState` retains local gradients on every rank.
- Root-owned and sharded modes cannot be mixed between ranks.

### 6.4 P2P backward

- Forward and backward use separate tag namespaces.
- Backward traverses gates in reverse operation order.
- Peer, shape, dtype, operation index, layout, and tag must agree across
  ranks before transport.
- The receive buffer cannot be read until its asynchronous HCCL work handle
  completes.
- A rank-local error is synchronized before any rank enters the next
  collective.

### 6.5 No silent fallback

Parameter-shift is a separate explicit gradient method and correctness
oracle. Native backward never silently switches to parameter-shift or CPU.
An unsupported native primitive produces one synchronized error on every
rank.

## 7. Physical Parameterizations

### 7.1 Pure state

Given paired-real raw vector `z`, compute:

```text
norm_squared = sum(z_real^2 + z_imag^2)
psi_real = z_real / sqrt(norm_squared)
psi_imag = z_imag / sqrt(norm_squared)
```

The zero-norm case is rejected collectively.

### 7.2 Density matrix

Given paired-real factor `L`, compute `L L^H` using real-valued matrix
products and normalize by its real trace. The public density matrix is
Hermitian, positive semidefinite, and trace one by construction.

### 7.3 Arbitrary Kraus channel

`StinespringParam` maps unconstrained paired-real parameters to a rectangular
isometry using a sequence of custom Householder reflections. The first
`input_dim` columns form the isometry. Environment blocks form the Kraus
operators.

The forward and backward implementation uses only real-valued primitives.
The release gate checks:

```text
sum_i K_i^H K_i = I
```

with maximum error `1e-5`.

## 8. Memory and Communication Performance

### 8.1 Optimization order

Correctness precedes optimization:

1. synchronous P2P backward;
2. reusable real/imag send and receive buffers;
3. asynchronous HCCL P2P;
4. double buffering;
5. overlap of the next exchange with current local computation;
6. packed parameter-gradient all-reduce.

The engine must not execute a collective for a local gate or one all-reduce
per scalar parameter.

### 8.2 Recomputation

Checkpoint recomputation uses deterministic gate and communication order.
It must recreate the same stochastic channel realization or analytic
density update. Sampling is outside the differentiable path.

### 8.3 Metrics

Record:

- forward, backward, and optimizer wall time;
- median and p95 total gradient time;
- peak NPU allocation and reservation;
- recomputation overhead;
- forward/backward P2P call count;
- HCCL bytes by real and imaginary stream;
- buffer reuse count;
- time waiting on communication;
- strong-scaling and weak-scaling efficiency.

## 9. Validation Strategy

### 9.1 Local tests

```text
tests/distributed/autograd/
├── test_pair.py
├── test_collectives_multiprocess.py
├── test_vector_grad.py
├── test_density_grad.py
├── test_channel_grad.py
├── test_stinespring_grad.py
├── test_optimizer_sync_multiprocess.py
├── test_checkpoint.py
└── test_contracts_multiprocess.py
```

CPU float64 finite difference and distributed parameter-shift are separate
oracles. Parameter-shift is used only when the parameterized generator or
channel has a valid shift rule. Raw pure-state, density-factor, and
Stinespring parameters use central finite differences. Tests must exercise
non-identity layouts and every distributed axis.

### 9.2 Real NPU probe

Add:

```text
scripts/npu/distributed_autograd_probe.py
scripts/npu/distributed_autograd.sh
```

Probe sections:

1. `environment`
2. `statevector`
3. `density`
4. `gates`
5. `probability`
6. `observable`
7. `noise`
8. `stinespring`
9. `communication`
10. `optimizer`
11. `performance`
12. `memory`
13. `contract`

The probe must verify root-owned and sharded initial states, non-identity
layout, local and cross-shard gates, all distributed axes, probability
Jacobians, Pauli/Hamiltonian/dense expectations, built-in noise, arbitrary
Stinespring channels, P2P backward counts, optimizer synchronization,
performance, memory stability, and exact synchronized errors.

### 9.3 Fixed benchmark matrix

| Path | Qubits | Depth | Trainable parameters |
|---|---:|---:|---:|
| Statevector | 24 | 64 | 32 and 128 |
| Density matrix | 12 | 32 | 32 |
| Built-in noise | 12 | 32 | 32 |
| Stinespring | 10 | 16 | 32 |

For each path and world size:

- 5 warmups;
- 30 measured runs;
- explicit NPU synchronization around timing;
- median, p95, and peak-memory reporting;
- comparison of complete gradient wall time, not an isolated kernel.

### 9.4 Correctness release gates

Every 2-, 4-, and 8-NPU run must satisfy:

- maximum error against CPU float64 `<= 1e-4`;
- maximum error against the applicable parameter-shift or central
  finite-difference oracle `<= 1e-4`;
- replicated rank gradients agree within `1e-6`;
- density outputs remain Hermitian, PSD, and trace one;
- Stinespring completeness error `<= 1e-5`;
- all transport uses real tensors;
- backend is HCCL;
- `fallback_to_cpu` is false;
- no rank diverges into a different collective sequence;
- `failed_invariants` is empty.

### 9.5 Performance release gate

For every fixed workload with at least 32 trainable parameters and for each
world size independently, compare against the applicable explicit oracle:

```text
shift-rule parameters:
    native_autograd_median < parameter_shift_median

raw state/density/Stinespring parameters:
    native_autograd_median < finite_difference_median
```

A failure at world size 2, 4, or 8 blocks that path from release. The design
does not require a fixed speedup ratio.

### 9.6 Stability release gate

Each path runs 100 optimizer iterations:

- peak memory must not grow monotonically;
- memory growth from the stable phase to iteration 100 must be at most 1%;
- replicated parameters and their optimizer state must agree on all ranks;
- root-owned and sharded initial-state parameters must preserve their
  declared ownership and global physical constraints;
- no HCCL work handle may remain unfinished;
- no P2P tag may collide;
- process-group teardown must complete.

## 10. Evidence Contract

Run:

```bash
torchrun --nproc-per-node=2 \
  scripts/npu/distributed_autograd_probe.py --section all
torchrun --nproc-per-node=4 \
  scripts/npu/distributed_autograd_probe.py --section all
torchrun --nproc-per-node=8 \
  scripts/npu/distributed_autograd_probe.py --section all
```

Each independent run emits one rank-0 JSON report with:

- tested Git commit;
- complete launch command and exit code;
- world size and rank-to-device mapping;
- torch, torch_npu, and CANN identity, or explicit `unknown`;
- HCCL backend;
- section metrics and statuses;
- CPU-fallback status;
- raw report SHA-256.

Evidence is stored under:

```text
docs/evidence/distributed-autograd/<commit>/
├── world2.json
├── world4.json
├── world8.json
└── manifest.json
```

`manifest.json` verifies the three independent report digests and emits:

```json
{"release_gate": "PASS"}
```

only when all three runs pass. Missing 8-NPU resources leave the release
gate blocked; they are not converted into a passing or skipped result.

## 11. Delivery Order

1. Freeze distributed parameter-shift behavior and its NPU probe.
2. Implement paired-real types and local statevector backward.
3. Implement differentiable real-valued collectives.
4. Implement cross-shard gate backward.
5. Implement probability and expectation reducers.
6. Implement trainable root-owned and sharded initial states.
7. Implement density-matrix backward.
8. Implement built-in channel gradients.
9. Implement Stinespring/Kraus gradients.
10. Implement checkpoint recomputation.
11. Add buffer reuse, asynchronous P2P, and measured overlap.
12. Add replicated optimizer and gradient-bucket validation.
13. Run the complete 2/4/8-NPU release gate.
14. Enable automatic autograd routing in `DistSimulator.run()` only after
    the evidence manifest passes.

Before step 14, the public simulator retains its current exact forward-only
autograd rejection. Partial native-autograd support is not exposed.

## 12. Design Decisions

- Use the existing `DistSimulator.run()` instead of a separate training
  simulator.
- Use paired-real internal graphs rather than raw complex64 autograd.
- Replicate parameters and optimizer state; shard only the quantum state.
- Use parameter-shift for valid shift-rule parameters and central finite
  differences for raw physical parameters; neither is a silent fallback.
- Support exact statevector and density-matrix engines; exclude MPS native
  autograd.
- Differentiate physical continuous outputs; keep sampling and collapse
  non-differentiable.
- Parameterize arbitrary Kraus channels through Stinespring isometries.
- Officially support `world_size = 2^p`; require 2/4/8 NPU evidence for
  release.
- Require native autograd to beat the applicable explicit oracle at 32 or
  more parameters.
- Keep multi-node qualification separate from this single-node release.

# Distributed NPU State Sharding Design

**Date:** 2026-07-29

**Status:** Approved in conversation; awaiting review of this written specification

## Purpose

Add a distributed execution mode to AICIR that partitions one statevector or
density matrix across multiple Ascend NPU devices. The implementation must
increase the aggregate state capacity beyond one NPU without changing the
meaning of any existing `NPUBackend`, `State`, `Measure.run()`, or `Result`
API.

The first release is a forward-simulation feature. It supports deterministic
unitary and density-matrix evolution, structured expectation values, exact
probabilities, and terminal computational-basis sampling. It does not support
distributed automatic differentiation.

## Fixed Requirements

- Use one process per NPU and launch with `torchrun`.
- Use HCCL for real Ascend execution and Gloo for CPU reference tests.
- Require `world_size = 2**p` for an integer `p >= 0`.
- Require `n_qubits >= p`.
- Support `complex64` only in the first release.
- Never fall back from distributed execution to CPU or to a full
  `2**n_qubits` square unitary.
- Keep every existing single-device and task-parallel API unchanged.
- Add a separate, explicitly named distributed API.
- Keep all logical qubits. Rank bits locate storage; they are not consumed
  logical qubits.
- Use a static, circuit-aware logical-to-storage-axis layout in the first
  release. Do not dynamically redistribute axes during execution.
- Treat correctness and memory distribution as acceptance criteria. Do not
  claim NPU speedup without a separate benchmark.

## Non-goals

The first release does not provide:

- distributed automatic differentiation;
- mid-circuit measurement;
- reset instructions;
- measurement-dependent `if` or `while` control flow;
- mid-circuit X- or Y-basis projection;
- stochastic pure-state noise trajectories;
- arbitrary non-power-of-two device counts;
- dynamic logical-to-storage-axis remapping;
- arbitrary full-system dense observables;
- implicit gathering of a complete state or probability vector.

## Public API

The new API is exported only from `aicir.distributed`:

```python
from aicir.distributed import (
    DistNPUBackend,
    DistState,
    DistSimulator,
    DistResult,
)

backend = DistNPUBackend.from_env()
simulator = DistSimulator(backend)
result = simulator.run(circuit)
```

The common construction may be shortened without aliases:

```python
simulator = DistSimulator.from_env()
result = simulator.run(circuit)
```

These are the real class names, not aliases for generic names. The package
must not expose `Backend = DistNPUBackend`, `State = DistState`, or equivalent
mixed naming. The new types are not re-exported from the top-level `aicir`
namespace.

### `DistNPUBackend`

`DistNPUBackend` is responsible for:

- reading `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, rendezvous, and process-group
  settings from the `torchrun` environment;
- binding one process to `npu:LOCAL_RANK`;
- initializing or validating the HCCL process group;
- reusing the existing `NPUBackend` Ascend complex-operation workarounds;
- validating device-count and dtype constraints;
- managing communication, global reductions, and device-capacity checks.

State sharding and the existing batch/task parallel mode are mutually
exclusive within one process group. A state-sharding backend must not use
`should_run_batch_index()` or `gather_indexed_results()` to divide a circuit
batch.

### `DistState`

`DistState` owns one local device tensor and immutable distribution metadata.
Its public read-only properties are:

- `local_data`;
- `local_shape`;
- `global_shape`;
- `n_qubits`;
- `kind`, equal to `"vector"` or `"matrix"`;
- `is_density`;
- `bit_order`;
- `rank`;
- `world_size`.

It does not expose a `data` property whose meaning could be confused with the
complete tensor returned by the existing `State.data`.

Complete materialization is explicit:

```python
full_state = state.gather(root=0)
array = state.to_numpy(root=0)
```

`gather()` returns an existing single-device `State` on the selected root;
`to_numpy()` returns a NumPy array there. Every other rank receives `None`.
Neither representation, formatting, measurement, nor result construction may
implicitly gather the full state.

### `DistSimulator`

`DistSimulator` is the only first-release circuit-execution entry point. It:

- validates the circuit and distributed configuration before state allocation;
- creates or shards an initial state;
- computes a deterministic static layout;
- plans each gate as local or communicating;
- executes all ranks in the same instruction and communication order;
- applies deterministic density-matrix noise;
- computes requested probabilities, expectations, and terminal samples;
- returns `DistResult`.

The initial signature is:

```python
result = simulator.run(
    circuit,
    *,
    initial_state=None,
    initial_density_matrix=None,
    observables=None,
    shots=None,
    measure_qubits=(),
    collapse=False,
    seed=None,
    layout=None,
    return_state=True,
    return_probabilities=True,
)
```

`initial_state` and `initial_density_matrix` are mutually exclusive.
`shots=None` is exact mode. `shots` must otherwise be a positive integer.
`collapse=True` is allowed only when `shots == 1`; multi-shot execution has no
single post-measurement state. With `collapse=False`, `result.state` is the
pre-measurement final state.

An initial state has two accepted distribution modes:

- Every rank passes its matching `DistState` local shard.
- Rank zero passes one complete existing `State`, NumPy array, or compatible
  array-like object; all other ranks pass `None`. Rank zero broadcasts the
  initialization mode and scatters the data.

If rank zero and another rank both pass complete initial data, validation
fails. When every rank passes `None`, the simulator constructs the distributed
zero state without first constructing a complete state. The same rules apply
to `initial_density_matrix`, except that its complete root input is scattered
by row.

### `DistResult`

`DistResult` exposes:

- `state: DistState | None`;
- `local_probabilities`, or `None` when probabilities were not requested;
- `expectations`, containing globally reduced scalar values on every rank;
- `counts`, populated only on rank zero;
- `rank`;
- `world_size`;
- `is_root`, equal to `rank == 0`;
- `gather_probabilities(root=0)`.

`gather_probabilities()` is an explicit potentially large operation. It
returns the complete logical-order probability vector only on the selected
root and returns `None` elsewhere.

## Internal Boundaries

The following implementation types remain private:

- `_ShardSpec`: global and local shapes, rank ranges, and storage-axis mapping;
- `_Communicator`: HCCL/Gloo point-to-point and collective operations;
- `_GatePlanner`: local/communicating gate classification and partner plans;
- `_Layout`: logical-to-storage-axis mapping and inverse mapping;
- `_VectorKernel`: sharded statevector evolution;
- `_MatrixKernel`: row-sharded density-matrix evolution;
- `_Reducer`: distributed norms, probabilities, expectations, and samples.

The public classes must not expose process-group handles or require callers to
construct these internal objects.

## Process and Rank Model

`RANK` is the unique process identifier in the complete process group.
`LOCAL_RANK` selects the NPU on the current host. They must never be used
interchangeably.

For `world_size = 2**p`, rank identifies one contiguous range of global
storage indices. It does not identify an additional logical qubit and does
not reduce the simulated Hilbert space.

For an `n`-qubit statevector:

```text
global dimension = 2**n
local dimension  = 2**(n-p)
global start     = rank * local_dimension
global stop      = global_start + local_dimension
```

All ranks together retain exactly `2**n` amplitudes.

## State Layout

### Statevector

Each rank stores a contiguous complex tensor of shape:

```text
(2**(n-p), 1)
```

The `p` distributed storage axes are encoded by the binary rank. The other
`n-p` storage axes are represented inside the local tensor.

### Density matrix

The density matrix uses contiguous row sharding. Each rank stores:

```text
(2**(n-p), 2**n)
```

The row index is distributed in the same way as a statevector index. Every
rank retains the complete column dimension. Across all ranks, the complete
`2**n` by `2**n` matrix is represented exactly once.

### Logical-to-storage layout

No fixed logical qubits are permanently assigned to rank bits. `_Layout`
maps each logical qubit to a storage axis. A gate is resolved through this
mapping before local or distributed execution is selected.

The automatic first-release layout is deterministic:

1. Start with an empty set `D` of logical qubits assigned to distributed
   storage axes.
2. For a candidate set, score every unitary instruction by `2**r - 1`, where
   `r` is the number of that instruction's distinct target and control qubits
   contained in `D`. Sum this value over the circuit.
3. Add the logical qubit whose inclusion gives the smallest total score.
4. Break equal-score choices by the smaller logical qubit index.
5. Repeat until `len(D) == p`.
6. Map the selected logical qubits to distributed storage axes in ascending
   logical-qubit order and all remaining logical qubits to local storage axes
   in ascending order.
7. Keep the mapping fixed for the complete run.

All ranks compute the same mapping and verify its digest before allocation.
An explicit `layout=` must be a complete bijection over `range(n_qubits)`.
Full state, full probabilities, and bit strings are converted back to logical
qubit order before being returned.

## Gate Execution

### Local gates

If all target storage axes are local, `_VectorKernel` reuses the existing NPU
local-gate kernels. No communication call is permitted for that instruction.

### Gates involving distributed storage axes

If a gate touches `r` distributed storage axes, the relevant ranks form a
logical partner set of size `2**r`. Partner ranks are derived by XORing the
rank bits associated with those storage axes.

Each rank computes only the output rows belonging to its own shard:

1. Allocate one local output buffer and one peer receive buffer.
2. Apply the contribution from the local input shard.
3. Visit partner masks in a deterministic ascending order.
4. Exchange one peer shard at a time.
5. Multiply by the corresponding local-matrix block and accumulate into the
   local output.
6. Wait for communication completion before reusing buffers or entering a
   different process group.

This streaming rule avoids materializing `2**r` complete input shards at
once. Communication grows exponentially with the number of distributed axes
inside a single gate, but no logical qubit is removed.

The distributed path accepts only gates that have a finite local matrix and
target-axis description. It never calls `gate_to_matrix` to form a complete
system unitary. One-, two-, and three-qubit registered gates are required
coverage; a larger custom local unitary is accepted only when its local
matrix and communication workspace pass capacity validation.

### Communication implementation

`_Communicator` provides a backend-neutral exchange operation. CPU
multiprocess tests implement it with Gloo. Real NPU execution uses HCCL.

The implementation must probe the installed `torch_npu` and HCCL runtime for
the chosen point-to-point primitive. PyTorch API presence alone is not proof
of HCCL support. The preferred implementation is deterministic paired
nonblocking send/receive with an explicit wait. If the installed HCCL stack
cannot perform that operation, the implementation may use a verified
subgroup collective only when it preserves the one-peer-buffer memory bound.
If neither path is available, initialization fails; it does not fall back to
CPU.

The probe must also verify whether the selected HCCL primitive accepts
`complex64`. If it does not, `_Communicator` sends real and imaginary
`float32` components in a fixed order and reconstructs `complex64` on the
receiving NPU. This is an NPU-to-NPU representation workaround, not CPU
fallback.

Collectives and process-group operations must occur in globally consistent
order on every rank.

## Density-matrix Evolution

For each unitary:

```text
rho' = U rho U†
```

execution is split into two passes:

1. `U rho`: act on the row index. This uses the statevector-style distributed
   row kernel and treats the full column dimension as a batch dimension.
2. `(U rho) U†`: act on the column index. Because each local row block has all
   columns, this pass is entirely local.

No rank constructs a complete density matrix.

## Deterministic Noise

The distributed simulator supports Kraus evolution:

```text
rho' = sum_i K_i rho K_i†
```

Existing noise-channel behavior remains unchanged. Channel implementations
gain a protected local-Kraus hook that returns finite local matrices and
logical target qubits. Existing full-system `kraus_operators()` behavior
continues to serve the single-device path.

For each local Kraus operator, `_MatrixKernel` performs communicating row
action, local column action, and local block accumulation. A channel without a
local Kraus representation is rejected before execution.

When noise promotes a statevector to a density matrix, each rank constructs
only its density-matrix rows. A complete statevector may be gathered
temporarily across NPU ranks for the outer product because, under
`n_qubits >= p`, its `2**n` elements are no larger than one local
density-matrix block containing `2**(2*n-p)` elements. The capacity check
must include this temporary vector and the local density block.

The first release does not approximate a noisy density matrix with random
pure-state trajectories.

## Probabilities and Normalization

For a statevector, each rank computes the squared magnitudes of its local
amplitudes. For a density matrix, each rank extracts only the diagonal entries
whose global row indices belong to its row block. A scalar all-reduce produces
the global normalization.

The normalized local vector is stored as `DistResult.local_probabilities`.
No complete probability vector is created unless
`gather_probabilities(root=...)` is called.

Norm and trace checks use scalar reductions. They must not gather a complete
state.

## Expectations

The first release accepts:

- `aicir.core.operators.PauliString`;
- Pauli observables represented by `aicir.ir.Observable.pauli(...)`;
- finite local dense operators accompanied by explicit logical target qubits;
- sums of the preceding structured terms.

It rejects an unstructured full-system dense matrix. Each rank computes its
local contribution and `_Reducer` applies an all-reduce sum. The final scalar
is available and equal on every rank.

## Terminal Z-basis Sampling

Terminal computational-basis sampling avoids gathering `2**n`
probabilities:

1. Each rank computes its local total probability mass.
2. The root obtains the `world_size` scalar masses.
3. The root assigns each shot to a rank according to those masses.
4. Each rank samples only from its normalized local conditional
   distribution.
5. Sampled local indices are converted to global storage indices, restored
   to logical bit order, and sent to the root.
6. The root builds `counts`.

Communication volume is proportional to `world_size + shots`, excluding
fixed control metadata.

`measure_qubits=()` means all logical qubits. A non-empty sequence selects a
logical-qubit subset and preserves its input order. `collapse=True` with one
shot masks each local state block, all-reduces the normalization scalar, and
returns the normalized distributed post-measurement state. `collapse=True`
with more than one shot is rejected.

Rank zero is the counts root. It initializes the global sampling generator
from `seed`, broadcasts shot assignments, and each participating rank derives
its local generator from `(seed, rank)`. Identical seeds, circuits, and
layouts therefore reproduce identical counts independently of process
scheduling.

## Validation and Failure Handling

Before allocating a state, every rank validates:

- process-group availability and rank metadata;
- power-of-two world size;
- unique local device binding;
- supported dtype;
- qubit and local-shape constraints;
- circuit instruction support;
- absence of automatic-differentiation inputs;
- initial-state shape and mutual exclusivity;
- local Kraus availability;
- layout validity;
- local state, communication buffer, and gate workspace capacity.

All ranks compare a digest covering the circuit structure, qubit count,
execution options, initial-state mode, and layout. A mismatch terminates
before gate communication starts.

Unsupported instructions, missing HCCL communication support, memory
shortage, or configuration mismatch are hard errors. Diagnostics include the
global rank, local rank, instruction index, logical target qubits, storage
axes, and partner ranks where applicable. Distributed failure handling must
terminate the job rather than leave surviving ranks blocked.

## Test Strategy

### CPU/Gloo multiprocess tests

Run subprocess tests with world sizes two and four. Compare against the
single-device NumPy reference for:

- zero and user-provided initial statevectors;
- initial density matrices;
- local shape and global range metadata;
- local one-, two-, and three-qubit gates;
- gates involving one and multiple distributed storage axes;
- Bell, GHZ, and seeded random shallow circuits;
- density-matrix left and right action;
- single- and two-qubit Kraus channels;
- pure-to-density promotion;
- local and explicitly gathered probabilities;
- Pauli strings, Pauli observables, and sums of terms;
- terminal full-register and subset sampling;
- one-shot distributed collapse;
- automatic and explicit layouts;
- logical-order restoration.

Instrumentation tests must prove that:

- a local gate performs no communication;
- a communicating gate visits exactly its planned partner ranks;
- normal execution does not gather a complete state;
- density-matrix execution never constructs the complete matrix on one rank.

Failure tests cover:

- non-power-of-two world size;
- `n_qubits < log2(world_size)`;
- rank configuration or circuit digest mismatch;
- invalid or non-bijective layout;
- unsupported instruction, observable, or channel;
- `requires_grad=True`;
- invalid collapse/shot combinations;
- insufficient capacity.

### Numerical tolerances

- Statevectors and density matrices: `rtol=1e-5`, `atol=1e-6`.
- Probabilities and expectations: `atol=1e-5`.
- Statevector normalization error: at most `1e-5`.
- Density-matrix trace error: at most `1e-5`.

Sampling tests use seeded statistically robust bounds rather than exact count
equality.

### Real Ascend validation

Run focused suites with:

```bash
torchrun --nproc-per-node=2 ...
torchrun --nproc-per-node=4 ...
```

The tests require:

- `fallback_to_cpu=False`;
- every local tensor on `npu:LOCAL_RANK`;
- verified HCCL communication without deadlock;
- correct local and communicating gates;
- correct density evolution, noise, reductions, and sampling;
- no hidden full-state or full-density allocation;
- local state storage close to `1/world_size` of the corresponding complete
  representation, excluding declared bounded work buffers.

Passing these tests establishes numerical correctness, distributed storage,
and real multi-NPU execution. It does not by itself establish speedup.

## Documentation and Compatibility

The implementation must add:

- an `aicir.distributed` API reference;
- `torchrun` examples for two and four NPUs;
- a supported-feature and rejected-feature table;
- memory formulas for statevectors and row-sharded density matrices;
- an explanation of global rank, local rank, logical qubits, and storage axes;
- explicit warnings around state/probability gathering;
- a CHANGELOG entry for the new public API.

Existing imports and test behavior must remain unchanged.

## Acceptance Criteria

The feature is accepted when:

1. Existing tests pass without API or semantic changes.
2. Gloo world-size-two and world-size-four distributed tests pass.
3. Real two-NPU and four-NPU HCCL tests pass with CPU fallback disabled.
4. Statevector and density-matrix results meet the numerical tolerances.
5. Local gates produce no communication.
6. Communicating gates use the deterministic partner plan without deadlock.
7. No normal execution path implicitly gathers a complete state, density
   matrix, or probability vector.
8. Unsupported features fail before distributed gate execution.
9. Per-rank persistent state storage is approximately
   `1/world_size` of the complete representation.
10. Documentation accurately distinguishes memory scaling, correctness, and
    unproven performance speedup.

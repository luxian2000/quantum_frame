# Distributed API README Manual Design

## Goal

Rewrite `aicir/distributed/README.md` as a complete functional manual rather
than a quick-start page. The manual must describe how users construct and
compose distributed backends, states, circuits, simulations, noise,
observables, sampling, results, and layouts using only the public
`aicir.distributed` API.

## Organization

The document follows the runtime lifecycle:

1. Feature model and prerequisites.
2. Launching one process per NPU.
3. Constructing `DistNPUBackend`.
4. Constructing a distributed quantum state.
5. Constructing a distributed quantum circuit.
6. Constructing and calling `DistSimulator`.
7. Running statevector simulations.
8. Running density-matrix simulations.
9. Applying deterministic Kraus noise.
10. Computing structured observables.
11. Terminal Z-basis sampling and collapse.
12. Reading `DistResult` and explicitly gathering data.
13. Reusing a `DistState` in a later run.
14. Automatic and explicit logical-to-storage layouts.
15. Storage and communication costs.
16. Supported and rejected features.
17. Strict NPU validation and troubleshooting.
18. Public API reference.

Each functional section includes a focused executable example or a precise
parameter/return-value table. One end-to-end program ties the sections
together without replacing their individual explanations.

## Public Construction Boundary

The manual documents exactly three supported ways to obtain a `DistState`:

1. Omit both initial-state arguments so `DistSimulator.run()` creates the
   distributed zero state.
2. Supply a complete statevector or density matrix on rank 0 and `None` on
   every other rank; the simulator converts logical order to storage order
   and scatters row-contiguous shards.
3. Supply a matching `DistState` on every rank, normally the state returned
   by a previous distributed run, and preserve its layout.

`DistState.from_local()` and `DistState.zero()` require internal `_ShardSpec`
or `_Layout` metadata and are not presented as user construction APIs.
Internal types beginning with `_` are not imported in examples.

## Circuit Semantics

Circuits continue to use the existing public `Circuit` and gate constructors.
There is no separate distributed circuit type. Distribution is an execution
property selected by `DistSimulator`; every rank must construct the same
circuit and pass matching run options.

The manual distinguishes:

- local gates, whose target storage axes are entirely within a rank;
- communicating gates, whose target storage axes include rank-distributed
  axes;
- logical qubits, which are never consumed by rank metadata;
- storage axes, which layout may permute to reduce communication.

## Results and Materialization

The manual states which values exist on every rank and which are root-only:

- structured expectation scalars are equal on every rank;
- `DistState` and local probabilities remain sharded;
- terminal counts exist only on rank 0;
- `gather()`, `to_numpy()`, and `gather_probabilities()` are collective,
  explicit full-materialization boundaries and must be called by every rank.

`return_state` and `return_probabilities` are documented independently.

## Accuracy and Scope

Examples must match the implemented signatures and tests. The manual must
not claim:

- autograd support;
- arbitrary user construction from local shards;
- mid-circuit measurement, reset, or classical control;
- random-trajectory noise;
- implicit CPU fallback;
- real-HCCL validation or multi-NPU speedup unless a strict probe was
  actually run on the target Ascend system.

The existing root `README.md` 7.2 link remains the canonical index entry.

## Verification

Before committing the rewritten manual:

- inspect all public signatures against current source;
- scan examples for imports of internal distributed symbols;
- run Python syntax checks for extracted code examples where practical;
- run `git diff --check`;
- confirm root README 7.2 still points to
  `aicir/distributed/README.md`;
- confirm no stale `docs/distributed.md` link remains.

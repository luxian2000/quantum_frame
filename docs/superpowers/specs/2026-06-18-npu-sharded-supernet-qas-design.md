# NPU-sharded supernet QAS — design

Date: 2026-06-18

## Goal

Speed up `supernet_qas` (the VQA ansatz search behind `demos/BeH2/BeH2.py`) by
sharding its three expensive phases — **supernet training**, **architecture
ranking**, and **finetune** — across multiple Ascend NPUs. Sharding activates
**only** on a distributed NPU run; every other path (CPU, CUDA, single NPU,
non-distributed) keeps today's exact behavior.

Two training-parallelization strategies are offered as selectable modes:
`safe` (numerically equivalent to single-card) and `aggressive` (data-parallel
over architectures, ~`world_size`× faster, different optimization dynamics).

## Activation predicate

A run is "sharded" iff **all** hold:

1. the trainer's backend is an `NPUBackend`, and
2. `torch.distributed` is initialized with `world_size > 1`.

The launcher (`torchrun`) initializes the process group; `BeH2_npu.py` already
does this via `NPUBackend.from_distributed_env(...)`. When the predicate is
false, all new code paths are bypassed and the existing single-process logic
runs unchanged.

**Seed invariant:** all ranks use the *same* `config.seed`. Combined with the
trainer's existing `random/numpy/torch` seeding in `__init__`, this guarantees
identical initial shared weights and an identical deterministic candidate
sampling sequence across ranks — the foundation the sharding relies on.

**Weight-sync invariant:** all ranks hold identical shared weights at the start
of every training step. Preserved by a per-step broadcast (safe) or all_reduce
(aggressive). Only real-valued tensors cross device collectives; Python objects
(records, circuits-as-gate-dicts) move via `all_gather_object`. No complex-dtype
collectives are used (NPU lacks complex kernels).

## New module: `aicir/qas/primitives/sharding.py`

Pure, backend-agnostic helpers with no supernet knowledge:

- `shard_context(backend) -> ShardContext` — returns `(is_sharded: bool,
  rank: int, world_size: int)`. Implements the activation predicate. For a
  non-sharded run returns `(False, 0, 1)`.
- `owned_indices(n, rank, world_size) -> list[int]` — strided split
  `list(range(rank, n, world_size))`. The union over ranks covers `range(n)`
  exactly once (tested).
- `all_gather(obj) -> list[Any]` — wraps `torch.distributed.all_gather_object`;
  returns one entry per rank in rank order. No-op (returns `[obj]`) when not
  sharded.
- `all_reduce_mean(tensors) -> None` — in-place sum-then-divide-by-world_size
  over a list of real tensors (aggressive-mode gradient averaging).
- `broadcast_parameters(params, src=0) -> None` — broadcast a mapping of
  real-valued parameter tensors from `src` to all ranks (safe-mode weight sync).

Each helper degrades to a trivial local operation when not in a sharded run, so
callers need no branching beyond reading `shard_context`.

## Config surface

- `SupernetConfig.shard_mode: str = "safe"` — `"safe"` or `"aggressive"`.
  Validated in `_validate_config`. Only consulted on a sharded NPU run.
- `supernet_qas(..., mode: str = "safe")` — forwards to `shard_mode`.
- `demos/BeH2/BeH2_npu.py` gains `--mode {safe,aggressive}` (default `safe`)
  and uses the **same seed on all ranks** (replacing the previous
  per-rank-seed, best-of-N behavior, which is incompatible with in-search
  sharding).

## Phase 1 — Ranking (`rank_architectures`, mode-independent)

All ranks sample the full `ranking_num` candidate list (deterministic, cheap,
keeps RNG in lockstep) but call `select_supernet` (the expensive sims) only on
their `owned_indices(ranking_num, rank, world_size)` slice. Then `all_gather`
the partial records, merge, and sort identically on every rank. Output is
identical to single-card ranking; speedup ≈ `world_size`.

## Phase 2 — Training (`optimize_supernet`)

### safe mode
Per step:
1. All ranks `sample_architecture()` (in sync).
2. Shard the W `select_supernet` forward evals: rank evaluates supernet ids in
   `owned_indices(supernet_num, rank, world_size)`.
3. `all_gather` partial losses → full W-loss vector on all ranks →
   `selected_id = argmin`.
4. Rank 0 runs the single grad step (forward+backward + `optimizer.step`) on the
   selected supernet, then `broadcast_parameters` the updated shared weights.
5. All ranks now hold identical weights for the next step.

Numerically equivalent to single-card (same algorithm, parallel evaluation +
deterministic sync). Per-step sims drop from `W + 1` to
`ceil(W / world_size) + 1`. Speedup bounded by W.

### aggressive mode
Per step:
1. Deterministically sample `world_size` architectures (identical list on all
   ranks); rank r takes `arch[r]`.
2. Rank r runs `select_supernet` (full W) for `arch[r]` → `selected_id_r`, then
   computes the grad on that supernet.
3. Build a full shared-parameter gradient vector (zeros where this rank's
   architecture did not touch a parameter).
4. `all_reduce_mean` the gradient vector across ranks.
5. `all_gather` the set of selected supernet ids; every rank applies the reduced
   grads via `optimizer.step()` for exactly those supernets. Adam state stays in
   sync because every rank applies the identical reduced grads to identical
   weights with identical optimizer state.

Effective batch = `world_size` architectures/step, so `supernet_steps` can be
lowered ~`world_size`× for comparable coverage. Results differ from single-card
by design (larger effective batch, different assignment dynamics).

## Phase 3 — Finetune (mode-independent)

Top-K parallel finetune with `K = world_size`:
- Rank r finetunes the r-th ranked architecture (skips if
  `r >= len(ranking_records)`).
- `all_gather` each rank's `(finetune_score, best_circuit gate-dicts + n_qubits,
  numeric parameters, selected_supernet_id, architecture_indices)`.
- Choose the global-best by `finetune_score` (lower = better) for all tasks.
- Rank 0 reconstructs the winning `Circuit` and computes `_final_metrics`.

Single-card behavior is unchanged (top-1). The `SupernetResult` returned on
every rank reflects the global-best architecture so downstream code is
rank-agnostic.

## Touched files

- `aicir/qas/primitives/sharding.py` — new module (helpers above).
- `aicir/qas/algorithms/supernet.py` — `shard_mode` config + validation;
  sharded branches in `rank_architectures`, `optimize_supernet`, and the
  finetune/selection block of `train`; `mode` kwarg on `supernet_qas`.
- `demos/BeH2/BeH2_npu.py` — `--mode`, same-seed-all-ranks, pass `mode` through.
- `aicir/qas/README.md` — document the sharded NPU run, `--mode safe/aggressive`,
  activation predicate, and the per-phase behavior/speedup. (Required deliverable.)
- `CHANGELOG.md` — dated entry for the new interface.
- Tests under `tests/qas/`.

## Testing

- **Unit (no NPU):** `owned_indices` partitions `range(n)` exactly once for
  several `(n, world_size)`; `shard_context` predicate returns `(False,0,1)`
  off-distributed.
- **CPU gloo, world_size=2 (`init_process_group("gloo")`, spawned procs):**
  safe mode reproduces single-card ranking order and finetune energy within
  float tolerance on a tiny 2–3 qubit Hamiltonian.
- **Aggressive mode:** runs end-to-end under the gloo world and converges
  (energy below the untrained baseline); not asserted bit-equal to single-card.
- Existing single-process supernet tests must remain green (no behavior change
  off the sharded path).

## Out of scope

- Sharding a single state vector across devices (this is task/data parallel).
- Speeding up `optimize_supernet`'s single grad step itself (only the W-eval
  loop is parallelized in safe mode).
- Evolutionary ranking and noisy QAS (still `NotImplementedError`).

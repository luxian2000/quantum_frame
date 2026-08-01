# Task 8 Checkpoint Recomputation Report

## Scope

The public `DistSimulator.run()` continues to validate `grad_checkpoint` while
retaining the forward-only autograd rejection.  Checkpoint execution remains
behind the private paired-real hook.

## Implemented contracts

- Policies `none`, `auto`, and positive integer intervals are validated.
- Auto memory selection accounts for local paired-real state bytes, temporary
  buffers, a 20% safety margin, and 80% of available memory.  NPU memory-query
  failure selects interval `1` with `memory_source="conservative"`; it never
  substitutes host memory.
- Ranks agree on both checkpoint interval and memory source before data P2P.
- Replay retains original plans and operation indices.  Tests cover vector and
  density paths with non-identity layouts on world sizes 2 and 4, including
  analytic BitFlip noise.
- Allocator peak measurement is only reset by `_measure_paired_real`, the
  probe-only explicit measurement boundary.  Each policy is isolated by the
  exact sequence `synchronize -> reset_peak -> workload -> synchronize ->
  read_peak`, so asynchronous work from a previous policy cannot contaminate
  the next peak.  Normal execution never resets global allocator peak state.
  Unavailable allocator support reports `peak_allocation_status="BLOCKED"`
  and no fabricated peak value.
- Density output no longer forces eager contiguous local copies.  The P2P
  receive boundary allocates its required contiguous destination directly from
  the source tensor shape, dtype, and device; a non-contiguous send is
  materialized exactly once.  A hostile mocked transport verifies contiguous
  receive storage plus forward values and backward gradients.

## Verification

Ran successfully:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_checkpoint.py \
  tests/distributed/autograd/test_checkpoint_multiprocess.py \
  tests/distributed/test_simulator_multiprocess.py -q
PYTHONPATH=. pytest tests/distributed -vv -s
python -m py_compile \
  aicir/distributed/autograd/_checkpoint.py \
  aicir/distributed/simulator.py \
  aicir/distributed/communication.py \
  scripts/npu/distributed_autograd_probe.py
git diff --check
```

The focused command reported 38 passing tests.  The full distributed suite
reported 238 passing tests in 56.66 seconds.  No NPU hardware probe was run;
this report makes no live-NPU correctness or performance claim.

One earlier aggregate `pytest tests/distributed -q` session made no progress
after 37 dots for more than six minutes.  It was terminated with SIGTERM only
after inspecting its parent, resource-tracker, and spawned child processes;
the session exited 143 and no targeted process remained.  The candidate 38th
test passed alone, its complete 21-test multiprocess file passed in 17.65
seconds, and the subsequent clean full run above passed.  The stall therefore
was not reproduced by the final verification run.

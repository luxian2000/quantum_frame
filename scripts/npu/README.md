# NPU test scripts

These scripts group existing AICIR tests into layers that are practical on
Ascend/NPU machines.

## Common commands

```sh
scripts/npu/smoke.sh --strict-npu
scripts/npu/typed_ir.sh --strict-npu
scripts/npu/deriv.sh --strict-npu
scripts/npu/backend.sh --strict-npu
scripts/npu/ops.sh --strict-npu
scripts/npu/run_all.sh --strict-npu
scripts/npu/multi_card.sh --nproc-per-node 4 --devices 0,5,6,7
```

Use `--dry-run` to inspect commands without executing them:

```sh
scripts/npu/run_all.sh --strict-npu --dry-run
```

Use `--pytest-arg` to forward extra pytest flags:

```sh
scripts/npu/qml.sh --strict-npu --pytest-arg -vv --pytest-arg --tb=short
```

## Suites

- `smoke`: fast backend sanity plus backend-bound circuit smoke path.
- `backend`: complete NPUBackend behavior and distributed helpers.
- `ops`: NPU-safe complex decompositions and gradient regressions.
- `capacity`: capability probe, capacity guards, sharding helpers.
- `typed_ir`: typed `CircuitIR`/`Operation`/`Measurement`, dict interop, JSON/QASM, metrics, transpile, and NPU execution.
- `circuit`: circuit execution, measurement, typed gates, JSON/QASM I/O.
- `deriv`: typed-IR derivative paths on NPU, with focus on `qml.ad`, direct `qml.auto`, `psr`/`fd`, NPU-safe backend autograd, and estimator parameter binding.
- `qml`: gradient, qlayer, parameter-shift, estimator paths.
- `tensor`: tensor network simulator and cotengra-facing paths.
- `qas`: QAS/VQE workloads likely to stress NPU batch and gradient paths.
- `demos`: demo and molecule smoke tests before long NPU jobs.

## Multi-card NPU probe

Use this after the single-card strict sweep passes. It launches
`python -m torch.distributed.run` and initializes the process group through
`NPUBackend.from_distributed_env(...)`. In strict mode the process group must be
HCCL and every rank must resolve to an `npu` device.

For a 4-card host whose physical NPU ids are non-contiguous, pass the visible
device list explicitly:

```sh
scripts/npu/multi_card.sh \
  --nproc-per-node 4 \
  --devices 0,5,6,7 \
  --section all
```

For a quick HCCL smoke before the heavier supernet probe:

```sh
scripts/npu/multi_card.sh --nproc-per-node 4 --devices 0,5,6,7 --section collectives
```

Use `--dry-run` to inspect the generated command. The probe sections are:

- `collectives`: rank/device mapping, `shard_context`, object `all_gather`,
  real-tensor `broadcast_parameters`, and real-tensor `all_reduce_mean`.
- `typed-ir`: the typed IR hardware cases from `typed_ir_deriv_probe.py` on
  every rank.
- `deriv`: the typed deriv hardware cases from `typed_ir_deriv_probe.py` on
  every rank.
- `supernet`: small real multi-rank `supernet_qas` runs in both `safe` and
  `aggressive` sharding modes, checking that final energies agree across ranks.
- `all`: all sections above.

## Typed IR / deriv probe

For a quick strict hardware probe without the larger pytest sweep:

```sh
scripts/npu/typed_ir_deriv_probe.sh --section typed-ir
scripts/npu/typed_ir_deriv_probe.sh --section deriv
scripts/npu/typed_ir_deriv_probe.sh --section all
```

The probe defaults to strict NPU. It fails before running cases when
`is_npu_available()` is false or `NPUBackend` does not resolve to an `npu`
device. Use `--allow-cpu-fallback` only for local development.

`typed-ir` covers:

- `Circuit.gates` returning typed `Operation` objects while `legacy_gates`
  remains dict-compatible.
- `CircuitIR` conversion, JSON/QASM round-trip, metrics, transpile, and direct
  `gate_to_matrix(Operation, backend=NPUBackend)`.
- `Measure.run` on typed IR, including statevector and density-matrix inputs.
- Typed `Observable.hamiltonian(...)` with `qml.ad`.

`deriv` covers:

- Direct `qml.auto` on an NPU-backed typed `Operation` energy graph, compared
  against `qml.psr`.
- `qml.ad` on typed `CircuitIR`, compared against the analytic `RY` / `<Z>`
  gradient.
- `StatevectorEstimator.gradient(..., method="psr"|"fd")` over a typed-gate
  parameter-binding template.

Generic full-matrix complex autograd in `tests/gates/test_matrix_autograd.py`
is still run for the CPU/fallback backend contract. On a real Ascend NPU that
test intentionally excludes the `NPUBackend` parametrization because torch_npu
cannot backward through arbitrary complex fan-out graphs (`aclnnInplaceAdd` does
not support `DT_COMPLEX64`). Real-NPU deriv coverage is provided by the probe
above plus the NPUBackend custom-autograd tests selected by `scripts/npu/deriv.sh`.

Without `--strict-npu`, the suites still run in environments where current tests
use CPU fallback or mocked NPU paths. With `--strict-npu`, the runner first
checks `aicir.backends.npu_backend.is_npu_available()` and stops before pytest if
the real NPU runtime is unavailable.

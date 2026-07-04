# NPU test scripts

These scripts group existing AICIR tests into layers that are practical on
Ascend/NPU machines.

## Common commands

```sh
scripts/npu/smoke.sh --strict-npu
scripts/npu/backend.sh --strict-npu
scripts/npu/ops.sh --strict-npu
scripts/npu/run_all.sh --strict-npu
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
- `circuit`: circuit execution, measurement, typed gates, JSON/QASM I/O.
- `qml`: gradient, qlayer, parameter-shift, estimator paths.
- `tensor`: tensor network simulator and cotengra-facing paths.
- `qas`: QAS/VQE workloads likely to stress NPU batch and gradient paths.
- `demos`: demo and molecule smoke tests before long NPU jobs.

Without `--strict-npu`, the suites still run in environments where current tests
use CPU fallback or mocked NPU paths. With `--strict-npu`, the runner first
checks `aicir.backends.npu_backend.is_npu_available()` and stops before pytest if
the real NPU runtime is unavailable.

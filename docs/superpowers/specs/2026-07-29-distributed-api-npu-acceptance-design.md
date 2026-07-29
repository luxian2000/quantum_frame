# Distributed API NPU Acceptance Design

## 1. Purpose

This design defines the complete real-Ascend acceptance probe for the first
release of `aicir.distributed`.

The accepted release is forward-only. It supports sharded statevectors and
density matrices, gate evolution, deterministic local Kraus noise,
probabilities, structured expectations, terminal sampling and collapse, and
explicit result aggregation. Distributed automatic differentiation is not part
of this release; inputs requesting automatic differentiation must fail before
simulation with a clear error.

The acceptance probe establishes functional correctness. It does not establish
performance, multi-node behavior, fault tolerance, or distributed training
correctness.

## 2. Current Baseline

`scripts/npu/distributed_state_probe.py` remains the minimal smoke probe. It has
passed on real Ascend hardware with `world_size=2` and `world_size=4`, without
CPU fallback.

The smoke probe covers device binding, HCCL communication, a sharded
statevector, a row-sharded density matrix, amplitude-damping noise,
probabilities, one Pauli expectation, sampling, and result gathering.

It does not independently verify the complete public API. In particular, its
`local_gate` and `communicating_gate` invariants are currently declared rather
than derived from observed P2P calls.

## 3. Component Boundary

The existing smoke probe remains unchanged except for corrections needed to
keep it runnable. Complete API acceptance is implemented as a new entry point:

```text
scripts/npu/distributed_api_probe.py
```

The new probe supports:

```text
--section all
--section state
--section layout
--section continuation
--section noise
--section observable
--section measure
--section result
--section communication
--section contract
```

`all` runs every section in a single initialized HCCL process group. Each
section is an isolated function that returns a structured result and does not
print independently. Rank 0 emits one final JSON object. Every rank receives
the final pass/fail value, destroys the process group, and exits nonzero when
any required invariant fails.

Probe-only communication instrumentation wraps the communicator's internal
`_exchange_tensor` method. It does not add public debugging methods to
`DistNPUBackend` or any other distributed API.

The only production behavior change in this acceptance slice is the missing
forward-only contract: `DistSimulator` must reject automatic-differentiation
inputs before executing or scattering them.

## 4. Global Preconditions

Every probe run requires:

- `world_size` equal to 2 or 4;
- one process per NPU;
- `rank == local_rank == device.index`;
- HCCL as the process-group backend;
- `fallback_to_cpu=False`;
- `torch.complex64` distributed state storage;
- identical circuit, layout, and options on every rank.

The launch environment must preserve the CANN Python path:

```bash
source /usr/local/Ascend/cann/set_env.sh
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_api_probe.py --section all
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_api_probe.py --section all
```

The installation-specific legacy `set_env.sh` path remains acceptable where
the current CANN installation uses it.

## 5. Section Specifications

### 5.1 State construction

The `state` section verifies:

1. implicit distributed zero-state construction;
2. root-owned full statevector scatter;
3. root-owned full density-matrix scatter;
4. global gather and logical-order restoration;
5. statevector norm, density trace, and local shard sizes.

The density input must be a valid complex density matrix with at least one
nonzero off-diagonal entry. Shape-only and diagonal-only checks are
insufficient.

### 5.2 Non-identity explicit layout

The `layout` section uses a non-identity `logical_to_storage` permutation.
It runs the same circuit under automatic and explicit layouts. After explicit
logical-order restoration, the gathered statevectors, probabilities, and
expectations must agree.

For `world_size=4`, the case must exercise two distributed storage axes.

### 5.3 DistState continuation

The `continuation` section executes a circuit in two stages:

1. run the prefix and retain its returned `DistState`;
2. pass that `DistState` directly into a second `run`;
3. compare with one run of the concatenated circuit.

The same check is performed for vector and density state kinds. The probe must
not gather and re-scatter the intermediate `DistState`.

### 5.4 Kraus noise

The `noise` section covers:

- amplitude damping;
- bit flip;
- phase flip;
- depolarizing noise represented by its full Kraus set;
- at least one circuit where multiple noise channels are applied in sequence.

Each case compares the gathered density matrix with an independently calculated
rank-0 NumPy reference. It also verifies trace and probabilities.

### 5.5 Observables

The `observable` section evaluates in one run:

- a `PauliString`;
- a multi-term `Hamiltonian`;
- a local dense observable.

Every expectation is compared with a rank-0 NumPy reference using the reducer
tolerance. A result merely being finite is not sufficient.

### 5.6 Sampling and collapse

The `measure` section verifies:

- full-register Z sampling;
- subset Z sampling with the documented bit ordering;
- `shots=1, collapse=True`;
- consistency between the returned count and collapsed state;
- normalization and support of the collapsed subspace.

When only a subset is measured, unmeasured qubits may remain in superposition.
The acceptance condition is projection onto the measured subspace, not
collapse to one full-register basis vector.

### 5.7 Result combinations

The `result` section covers all four combinations of:

```text
return_state = True / False
return_probabilities = True / False
```

Disabled fields must be `None`. Enabled distributed data must remain local
until the caller explicitly invokes the documented gather operation.

### 5.8 Communication evidence

The `communication` section runs local and distributed gates separately while
recording `_exchange_tensor` calls.

Acceptance requires:

- a local gate produces zero P2P-call delta;
- a distributed gate produces a positive P2P-call delta;
- peers are within `[0, world_size)`;
- no rank communicates with itself;
- real and imaginary transports use paired tags;
- every rank reports its observed peers;
- the 4-NPU run exercises the peer topology required by both distributed axes.

Collectives used for digest agreement, reductions, or result gathering are
reported separately and do not count as evidence that a gate used the P2P
kernel.

### 5.9 Error contracts

The `contract` section verifies expected failures for:

- invalid explicit layouts;
- invalid initial-state shapes;
- inconsistent root/rank initial-state modes;
- unsupported mid-circuit measurement, reset, and classical control flow;
- automatic-differentiation inputs.

Automatic-differentiation rejection covers:

- a `DistState` whose local tensor requires gradients;
- a root-owned statevector requiring gradients;
- a root-owned density matrix requiring gradients;
- a numeric gate parameter requiring gradients;
- a custom-unitary matrix requiring gradients.

The common public error is:

```text
DistSimulator 首期仅支持前向模拟，不支持自动微分
```

An expected error is reported as `EXPECTED_ERROR` or
`UNSUPPORTED_AS_DESIGNED`, not as a failed section.

## 6. Numerical Thresholds

The probe retains the established thresholds:

```text
statevector or density element error <= 1e-6
norm, trace, probability, and expectation error <= 1e-5
```

Sampling checks use support and total-shot invariants. They do not require an
exact random histogram.

All reference calculations are allowed on rank 0 using NumPy. This does not
constitute simulator CPU fallback because distributed evolution remains strict
NPU execution.

## 7. Report and Exit Contract

Rank 0 emits exactly one JSON object:

```json
{
  "passed": true,
  "world_size": 4,
  "fallback_to_cpu": false,
  "sections": {
    "layout": {
      "passed": true,
      "status": "PASS",
      "metrics": {}
    },
    "contract": {
      "passed": true,
      "status": "PASS",
      "metrics": {
        "autograd": "UNSUPPORTED_AS_DESIGNED"
      }
    }
  },
  "failed_invariants": []
}
```

The process exits nonzero if a supported operation fails, a numerical threshold
is exceeded, CPU fallback occurs, communication evidence is inconsistent, or
an expected error is not raised.

## 8. Test Strategy

Local tests are written before probe implementation and must initially fail.
They verify:

- the new entry point and section names;
- strict `fallback_to_cpu=False`;
- preserved CANN `PYTHONPATH` in documented launch commands;
- the JSON schema and nonzero-exit contract;
- communication counters being derived rather than literal booleans;
- explicit automatic-differentiation rejection in `DistSimulator`.

Existing distributed CPU multiprocess tests remain the regression suite.
They do not replace 2-NPU and 4-NPU hardware acceptance.

## 9. Completion Criteria

This acceptance slice is complete only when:

1. local focused and full distributed tests pass;
2. the repository-wide test suite passes;
3. the 2-NPU full API probe reports `passed=true`;
4. the 4-NPU full API probe reports `passed=true`;
5. no run permits CPU fallback;
6. the results and remaining limitations are recorded in the distributed
   documentation.

After these criteria are met, a separate implementation plan will define
future distributed automatic differentiation, performance optimization,
multi-node support, fault handling, and broader scalability work.

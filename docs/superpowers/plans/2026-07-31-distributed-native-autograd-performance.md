# Distributed Native Autograd and Multi-NPU Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict-NPU native autograd to the exact sharded statevector and density-matrix simulator, then prove correctness, stability, and performance on 2, 4, and 8 NPUs.

**Architecture:** `DistSimulator.run()` keeps one public entrypoint and routes trainable execution into a paired-real autograd engine. Custom float32 autograd primitives implement collectives, state evolution, density evolution, reducers, physical state/channel parameterizations, checkpoint recomputation, and bucketed gradient synchronization without raw complex64 NPU operations. Distributed parameter-shift and CPU float64 finite differences remain explicit correctness oracles; no runtime path silently falls back.

**Tech Stack:** Python 3.10+, PyTorch, torch_npu, HCCL, NumPy, pytest, `torch.multiprocessing`, existing `aicir.qml.deriv.psr`, existing `aicir.distributed` layout/gate planner.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-31-distributed-native-autograd-performance-design.md`.
- Public support is `world_size = 2^p`; the release manifest requires independent 2-, 4-, and 8-NPU passes.
- The current forward-only public rejection stays enabled until Task 11 completes.
- Native autograd covers exact distributed statevectors and density matrices, not MPS/tensor-network execution.
- Samples, counts, collapse, reset, and measurement-fed control flow remain non-differentiable.
- Never use raw complex64 NPU matmul, add, multiply, scalar division, advanced indexing, reduction, QR, SVD, or collective.
- Never create rank-n complex work tensors; flat index maps replace `(2,) * n` reshapes.
- All HCCL payloads and all gradient accumulation buffers are float32.
- Parameter-shift is explicit and never a silent fallback.
- Direct trainable complex leaves remain rejected until their strict-NPU capability section passes.
- Fixed native-autograd gradient error is at most `1e-4`; replicated rank-gradient disagreement is at most `1e-6`.
- Native autograd must be faster than the applicable explicit oracle for every fixed workload with at least 32 parameters on each of 2, 4, and 8 NPUs: parameter-shift for valid shift-rule parameters and central finite differences for raw state, density-factor, and Stinespring parameters.
- Each stability workload runs 100 optimizer iterations with stable-phase memory growth at most 1%.
- Every new multi-process test uses bounded join, terminate, and kill cleanup.
- User-facing API changes update `aicir/distributed/README.md`, top-level distributed exports, and `CHANGELOG.md`.

---

### Task 1: Freeze Distributed Gradient Oracles and Probe Schema

**Files:**
- Create: `aicir/distributed/grad.py`
- Create: `tests/distributed/autograd/test_oracles.py`
- Create: `tests/distributed/autograd/test_oracles_multiprocess.py`
- Create: `scripts/npu/distributed_autograd_probe.py`
- Create: `scripts/npu/distributed_autograd.sh`
- Modify: `aicir/distributed/__init__.py`

**Interfaces:**
- Consumes: `aicir.qml.deriv.psr(fn, params, shift=np.pi / 2, coefficient=0.5)`.
- Produces: `parameter_shift_gradient(objective, parameters, *, shift=np.pi / 2, coefficient=0.5) -> np.ndarray`.
- Produces: `parameter_shift_jacobian(objective, parameters, *, shift=np.pi / 2, coefficient=0.5) -> np.ndarray`.
- Produces: `finite_difference_gradient(objective, parameters, *, epsilon=1e-6) -> np.ndarray`.
- Produces: probe CLI `distributed_autograd_probe.py --section SECTION --output-json PATH`, with sections `environment,statevector,density,gates,probability,observable,noise,stinespring,communication,optimizer,performance,memory,contract,all`.

- [ ] **Step 1: Write oracle contract tests**

Add scalar and vector tests:

```python
def test_parameter_shift_gradient_matches_sine():
    theta = np.array([0.2, -0.4], dtype=np.float64)
    actual = parameter_shift_gradient(
        lambda values: np.sin(values).sum(),
        theta,
    )
    np.testing.assert_allclose(actual, np.cos(theta), atol=1e-12)


def test_parameter_shift_jacobian_preserves_output_and_parameter_axes():
    theta = np.array([0.2, -0.4], dtype=np.float64)
    actual = parameter_shift_jacobian(
        lambda values: np.stack(
            [np.sin(values[0]), np.cos(values[1])]
        ),
        theta,
    )
    expected = np.array(
        [[np.cos(theta[0]), 0.0], [0.0, -np.sin(theta[1])]]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_finite_difference_handles_non_shift_rule_parameters():
    theta = np.array([0.3, -0.1], dtype=np.float64)
    actual = finite_difference_gradient(
        lambda values: np.sum(values**3),
        theta,
    )
    np.testing.assert_allclose(actual, 3.0 * theta**2, atol=1e-6)
```

- [ ] **Step 2: Run oracle tests to verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd/test_oracles.py -q
```

Expected: collection fails because `aicir.distributed.grad` does not exist.

- [ ] **Step 3: Implement explicit oracles**

Implement scalar PSR by delegating to the existing source of truth and implement the vector Jacobian without autograd:

```python
def parameter_shift_gradient(
    objective,
    parameters,
    *,
    shift=np.pi / 2,
    coefficient=0.5,
):
    values = np.asarray(parameters, dtype=np.float64)
    return psr(
        lambda point: float(np.asarray(objective(point)).reshape(())),
        values,
        shift=shift,
        coefficient=coefficient,
    )


def parameter_shift_jacobian(
    objective,
    parameters,
    *,
    shift=np.pi / 2,
    coefficient=0.5,
):
    values = np.asarray(parameters, dtype=np.float64)
    baseline = np.asarray(objective(values), dtype=np.float64)
    jacobian = np.empty(
        baseline.shape + values.shape,
        dtype=np.float64,
    )
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += shift
        minus[index] -= shift
        jacobian[(Ellipsis,) + index] = coefficient * (
            np.asarray(objective(plus), dtype=np.float64)
            - np.asarray(objective(minus), dtype=np.float64)
        )
    return jacobian


def finite_difference_gradient(
    objective,
    parameters,
    *,
    epsilon=1e-6,
):
    values = np.asarray(parameters, dtype=np.float64)
    gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (
            float(objective(plus)) - float(objective(minus))
        ) / (2.0 * epsilon)
    return gradient
```

- [ ] **Step 4: Freeze multi-rank oracle ordering**

Add a two-rank Gloo test in which every rank evaluates the same shifted
sequence, records `(parameter_index, sign)`, and asserts identical records
after `all_gather`. Use the repository's bounded spawn cleanup helper.

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_oracles.py \
  tests/distributed/autograd/test_oracles_multiprocess.py -q
```

Expected: all tests pass without a process timeout.

- [ ] **Step 5: Add the strict probe scaffold**

The initial probe must:

- require `torch.npu.is_available()`;
- require `world_size in {2, 4, 8}`;
- require HCCL;
- bind `LOCAL_RANK` to `npu:{LOCAL_RANK}`;
- reject `fallback_to_cpu=True`;
- run every section collectively;
- emit exactly one rank-0 JSON document;
- propagate a failed section to every rank before teardown.

Sections assigned to later tasks report their exact dependency:

```json
{
  "status": "BLOCKED",
  "passed": false,
  "blocked_by_task": 4
}
```

The probe's top-level `passed` is therefore false until Task 11.

- [ ] **Step 6: Add shell launch wrapper**

`scripts/npu/distributed_autograd.sh` must parse:

```text
--nproc-per-node {2,4,8}
--section SECTION
--output-json PATH
```

It launches:

```bash
PYTHONPATH=.:${PYTHONPATH:-} torchrun \
  --nproc-per-node="${NPROC}" \
  scripts/npu/distributed_autograd_probe.py \
  --section "${SECTION}" \
  --output-json "${OUTPUT_JSON}"
```

It does not set `ASCEND_RT_VISIBLE_DEVICES` and does not accept a functional
`--devices` mapping.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd/test_oracles.py \
  tests/distributed/autograd/test_oracles_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
bash -n scripts/npu/distributed_autograd.sh
git diff --check
```

Expected: all commands exit zero.

Commit:

```bash
git add aicir/distributed/grad.py aicir/distributed/__init__.py \
  tests/distributed/autograd/test_oracles.py \
  tests/distributed/autograd/test_oracles_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py \
  scripts/npu/distributed_autograd.sh
git commit -m "test(distributed): add gradient oracle and NPU probe"
```

### Task 2: Add Paired-Real Values and Physical Parameters

**Files:**
- Create: `aicir/distributed/autograd/__init__.py`
- Create: `aicir/distributed/autograd/_pair.py`
- Create: `aicir/distributed/autograd/_parameters.py`
- Create: `tests/distributed/autograd/test_pair.py`
- Create: `tests/distributed/autograd/test_parameters.py`
- Modify: `aicir/distributed/__init__.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: internal `_Pair(real: torch.Tensor, imag: torch.Tensor)`.
- Produces: `_Pair.add`, `mul`, `div_real`, `matmul`, `dagger`, `abs_sq`, `index_select`, and `combine`.
- Produces: public `PureStateParam`, `DensityParam`, and `StinespringParam`.
- Produces: parameter properties `.real`, `.imag`, and `.parameters() -> tuple[torch.Tensor, ...]`.

- [ ] **Step 1: Write paired-real RED tests**

Test forward and backward against CPU complex128 for add, multiply, matrix
multiply, dagger, real division, absolute square, and index selection.

Use a real scalar loss:

```python
loss = pair.abs_sq().sum()
loss.backward()
np.testing.assert_allclose(
    real.grad.numpy(),
    2.0 * real.detach().numpy(),
    atol=1e-6,
)
np.testing.assert_allclose(
    imag.grad.numpy(),
    2.0 * imag.detach().numpy(),
    atol=1e-6,
)
```

- [ ] **Step 2: Run paired-real tests to verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd/test_pair.py -q
```

Expected: import fails because `_pair.py` does not exist.

- [ ] **Step 3: Implement `_Pair` without complex arithmetic**

Use:

```python
@dataclass(frozen=True)
class _Pair:
    real: torch.Tensor
    imag: torch.Tensor

    def __post_init__(self):
        if self.real.dtype != torch.float32:
            raise TypeError("_Pair.real 必须是 torch.float32")
        if self.imag.dtype != torch.float32:
            raise TypeError("_Pair.imag 必须是 torch.float32")
        if self.real.shape != self.imag.shape:
            raise ValueError("_Pair 的 real/imag shape 必须一致")
        if self.real.device != self.imag.device:
            raise ValueError("_Pair 的 real/imag device 必须一致")

    def add(self, other):
        return _Pair(
            self.real + other.real,
            self.imag + other.imag,
        )

    def mul(self, other):
        return _Pair(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def matmul(self, other):
        return _Pair(
            self.real @ other.real - self.imag @ other.imag,
            self.real @ other.imag + self.imag @ other.real,
        )

    def dagger(self):
        return _Pair(self.real.t(), -self.imag.t())

    def div_real(self, denominator):
        return _Pair(
            self.real / denominator,
            self.imag / denominator,
        )

    def abs_sq(self):
        return self.real.square() + self.imag.square()
```

`combine()` uses one custom `torch.autograd.Function`; no internal engine
kernel consumes its complex output.

- [ ] **Step 4: Implement physical parameter containers**

`PureStateParam.normalized_pair()` computes one global real norm and rejects
zero norm. `DensityParam.density_pair()` uses paired-real `L L^H` and real
trace normalization. `StinespringParam` stores dimensions and raw paired
parameters but defers Householder construction to Task 7.

All containers expose only their real-valued leaves through
`.parameters()`.

- [ ] **Step 5: Test physical constraints**

Test:

- pure-state norm equals one;
- density matrix is Hermitian;
- density eigenvalues are non-negative within `1e-6`;
- density trace equals one;
- zero-norm pure state raises exact `ValueError`;
- zero-trace factor raises exact `ValueError`.

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_pair.py \
  tests/distributed/autograd/test_parameters.py -q
```

Expected: all pass.

- [ ] **Step 6: Fill the probe `environment` section**

Record device mapping, backend, dtype capabilities, torch/torch_npu
versions, CANN identity or `"unknown"`, and explicit checks that the paired
operations remain on NPU.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd/test_pair.py \
  tests/distributed/autograd/test_parameters.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd \
  aicir/distributed/__init__.py \
  tests/distributed/autograd/test_pair.py \
  tests/distributed/autograd/test_parameters.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): add paired real parameters"
```

### Task 3: Implement Differentiable Real-Valued Collectives

**Files:**
- Create: `aicir/distributed/autograd/_collectives.py`
- Create: `tests/distributed/autograd/test_collectives_multiprocess.py`
- Modify: `aicir/distributed/communication.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_exchange_pair(pair, *, communicator, peer, operation_index, phase) -> _Pair`.
- Produces: `_replicated_all_reduce(tensor, *, communicator) -> torch.Tensor`.
- Produces: `_scatter_root_pair(pair_or_none, *, communicator, root, local_shape) -> _Pair`.
- Produces: `_gather_root_pair(pair, *, communicator, root) -> _Pair | None`.
- Produces: forward/backward communication counters by dtype, peer, tag, and bytes.

- [ ] **Step 1: Write bounded multi-process RED tests**

Cover world sizes 2 and 4:

- exchange forward values;
- exchange backward values;
- replicated all-reduce gradient normalization;
- root scatter backward gather;
- distinct forward/backward tags;
- float32-only payloads;
- one-rank shape mismatch followed by a successful barrier;
- process cleanup on a deliberately mismatched test.

The replicated scalar test must assert:

```python
local = torch.tensor(
    float(rank + 1),
    requires_grad=True,
)
global_value = _replicated_all_reduce(
    local,
    communicator=communicator,
)
global_value.backward()
assert local.grad.item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_collectives_multiprocess.py -q
```

Expected: import fails because `_collectives.py` does not exist.

- [ ] **Step 3: Add float transport primitives**

Expose internal `_Communicator` methods that accept only real tensors for
autograd transport. Validate `dtype == torch.float32` before any
collective.

Tag allocation is:

```text
tag = operation_index * 8
    + phase_id * 4
    + direction_id * 2
    + component_id
```

where component zero is real and one is imaginary. Forward and backward use
different `phase_id` values.

- [ ] **Step 4: Implement custom backward functions**

`_PairExchangeFn.backward` exchanges `grad_real` and `grad_imag` with the
same peer using backward tags. `_ReplicatedAllReduceFn.backward` computes:

```python
return communicator.all_reduce_sum(grad_output) / communicator.world_size
```

Root scatter backward gathers local real/imag gradients to root. Non-root
ranks return no full-input gradient but still execute the same collectives.

- [ ] **Step 5: Verify world-size coverage**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_collectives_multiprocess.py -q
```

Expected: world-size 2 and 4 cases pass; no complex payload is recorded.

- [ ] **Step 6: Fill the probe `communication` section**

The NPU section verifies:

- each distributed axis causes forward and backward P2P;
- local gates cause zero P2P;
- every payload is float32;
- peers are valid and non-self;
- forward/backward tag sets are disjoint;
- every asynchronous handle is complete before teardown.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_collectives_multiprocess.py \
  tests/distributed/test_communication.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_collectives.py \
  aicir/distributed/communication.py \
  tests/distributed/autograd/test_collectives_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): add differentiable collectives"
```

### Task 4: Implement Statevector Native Backward

**Files:**
- Create: `aicir/distributed/autograd/_vector.py`
- Create: `aicir/distributed/autograd/_reducers.py`
- Create: `tests/distributed/autograd/test_vector_grad.py`
- Create: `tests/distributed/autograd/test_vector_grad_multiprocess.py`
- Modify: `aicir/distributed/gates.py`
- Modify: `aicir/distributed/reducers.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_PairVectorKernel(backend).apply(state_pair, plan, *, operation_index) -> _Pair`.
- Produces: `_PairReducer.probabilities(state_pair, spec) -> torch.Tensor`.
- Produces: `_PairReducer.expectation(state_pair, spec, observable) -> torch.Tensor`.
- Consumes: existing `_GatePlanner` and `_GatePlan`; does not duplicate layout planning.

- [ ] **Step 1: Write local gradient RED tests**

Use RX, RY, RZ, CRX, CRY, CRZ, RZZ, RXX, U2, U3, and a custom unitary.
Compare gradients against CPU float64 and parameter-shift. Include 32
parameters in one circuit.

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_vector_grad.py -q
```

Expected: import fails because `_vector.py` does not exist.

- [ ] **Step 2: Implement paired-real local gate application**

For matrix `M = A + iB` and gathered amplitudes `x + iy`, compute:

```text
out_real = A x - B y
out_imag = A y + B x
```

Use float matrix products and float `scatter_add_` for backward
accumulation. Matrix construction must not reuse one complex tensor across
multiple graph branches.

- [ ] **Step 3: Write cross-shard gradient RED tests**

Parameterize world sizes 2 and 4. For each distributed axis:

- place a parameterized gate on that logical axis;
- compare expectation gradients with parameter-shift;
- compare initial-state local gradients with a gathered CPU reference;
- assert forward and backward P2P counters are positive.

- [ ] **Step 4: Implement cross-shard statevector backward**

Reuse `_GatePlan.peer_mask`, source-rank ordering, and flat index maps.
Exchange paired-real shards through Task 3. Never gather the full state.

- [ ] **Step 5: Implement differentiable probabilities**

Compute local probabilities:

```python
probabilities = real.square() + imag.square()
total = _replicated_all_reduce(
    probabilities.sum(),
    communicator=backend.communicator,
)
probabilities = probabilities / total
```

Test the local Jacobian against distributed parameter-shift with
`max_abs_error <= 1e-4`.

- [ ] **Step 6: Implement observable reducers**

Pauli, Hamiltonian, and local dense observables use paired-real kernels.
The returned expectation is a replicated real scalar. Hamiltonian
accumulation is float32; no complex accumulator is allowed.

- [ ] **Step 7: Fill probe sections**

Implement `statevector`, `gates`, `probability`, and `observable`. Each
section compares native gradients with parameter-shift, covers every
distributed axis, and reports maximum error.

- [ ] **Step 8: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_vector_grad.py \
  tests/distributed/autograd/test_vector_grad_multiprocess.py \
  tests/distributed/test_vector_kernel.py \
  tests/distributed/test_vector_kernel_multiprocess.py \
  tests/distributed/test_reducers_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Expected: all tests pass.

Commit:

```bash
git add aicir/distributed/autograd/_vector.py \
  aicir/distributed/autograd/_reducers.py \
  aicir/distributed/gates.py aicir/distributed/reducers.py \
  tests/distributed/autograd/test_vector_grad.py \
  tests/distributed/autograd/test_vector_grad_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): add statevector backward"
```

### Task 5: Add Trainable Initial States and Differentiable Results

**Files:**
- Create: `tests/distributed/autograd/test_initial_state_grad_multiprocess.py`
- Create: `tests/distributed/autograd/test_result_grad.py`
- Modify: `aicir/distributed/state.py`
- Modify: `aicir/distributed/result.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- `DistState` gains an internal optional paired-real graph while preserving public complex `local_data`.
- `DistResult.expectations` accepts backend scalar tensors.
- Root-owned `PureStateParam` gradients materialize only on root.
- Sharded `DistState` parameters retain rank-local gradients.

- [ ] **Step 1: Write result RED tests**

Assert:

- `result.expectations["energy"].requires_grad` is true;
- `result.local_probabilities.requires_grad` is true;
- `gather`, `to_numpy`, and `gather_probabilities` return detached host
  values;
- converting an expectation to `float` before backward does not retain a
  graph;
- a sampled/count/collapse training request raises one exact error on all
  ranks.

- [ ] **Step 2: Write initial-state RED tests**

Cover:

- root-owned `PureStateParam`;
- sharded paired-real `DistState`;
- non-identity layout;
- root-owned direct complex leaf rejection;
- inconsistent `requires_grad` between ranks;
- error followed by a successful barrier.

- [ ] **Step 3: Extend `DistState`**

Add internal constructors:

```python
@classmethod
def from_pair(
    cls,
    pair,
    *,
    spec,
    backend,
    bit_order="msb",
):
    instance = cls.__new__(cls)
    instance._pair = pair
    instance._local_data = None
    instance._spec = spec
    instance._backend = backend
    instance._bit_order = bit_order
    return instance
```

`local_data` atomically combines the pair. Internal autograd kernels consume
the pair directly. Forward-only states retain the current `_local_data`
storage.

- [ ] **Step 4: Implement root paired scatter**

Rank zero normalizes the paired physical input and scatters float32 real and
imaginary shards. Backward gathers both components to rank zero. Shape and
normalization errors reuse the current bounded synchronized error protocol.

- [ ] **Step 5: Update result annotations and materialization docs**

Do not call `.item()`, `float()`, `.cpu()`, or `.numpy()` on differentiable
expectations inside the simulator. Keep materialization helpers explicitly
detached.

- [ ] **Step 6: Fill probe initial-state coverage**

Add root-owned and sharded statevector cases to `statevector`; add direct
complex leaf and rank mismatch cases to `contract`.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_initial_state_grad_multiprocess.py \
  tests/distributed/autograd/test_result_grad.py \
  tests/distributed/test_state.py \
  tests/distributed/test_state_multiprocess.py \
  tests/distributed/test_simulator_validation.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/state.py aicir/distributed/result.py \
  aicir/distributed/simulator.py \
  tests/distributed/autograd/test_initial_state_grad_multiprocess.py \
  tests/distributed/autograd/test_result_grad.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): preserve distributed result gradients"
```

### Task 6: Implement Density-Matrix Native Backward

**Files:**
- Create: `aicir/distributed/autograd/_density.py`
- Create: `tests/distributed/autograd/test_density_grad.py`
- Create: `tests/distributed/autograd/test_density_grad_multiprocess.py`
- Modify: `aicir/distributed/density.py`
- Modify: `aicir/distributed/autograd/_reducers.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_PairMatrixKernel.apply_unitary(state, plan, *, operation_index) -> DistState`.
- Produces: `_PairMatrixKernel.promote_vector(state) -> DistState`.
- Consumes: `DensityParam.density_pair()`.

- [ ] **Step 1: Write density RED tests**

Cover unitary evolution, direct `DensityParam`, promoted pure states,
non-identity layout, probability gradients, Pauli expectation gradients,
and local dense expectation gradients.

Check:

```text
Hermitian error <= 1e-5
minimum eigenvalue >= -1e-5
trace error <= 1e-5
gradient error <= 1e-4
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_density_grad.py \
  tests/distributed/autograd/test_density_grad_multiprocess.py -q
```

Expected: import fails because `_density.py` does not exist.

- [ ] **Step 3: Implement paired-real `U rho U^H`**

Apply the left and right actions using paired-real matrix kernels. Preserve
the current row-shard specification. Distributed-axis exchanges operate
only on paired float32 buffers.

Diagonal probability reads use:

```python
probabilities = state_pair.real[rows, columns]
```

No complex advanced index is permitted.

- [ ] **Step 4: Implement density reducers**

Trace and normalization use the real diagonal. Observable gradients use
paired-real left/right actions and float32 reduction.

- [ ] **Step 5: Fill the probe `density` section**

Compare native gradients against CPU float64 and parameter-shift for gate
parameters. Compare `DensityParam` factor gradients against CPU float64
finite differences.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_density_grad.py \
  tests/distributed/autograd/test_density_grad_multiprocess.py \
  tests/distributed/test_density_kernel_multiprocess.py \
  tests/distributed/test_reducers_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_density.py \
  aicir/distributed/autograd/_reducers.py \
  aicir/distributed/density.py \
  tests/distributed/autograd/test_density_grad.py \
  tests/distributed/autograd/test_density_grad_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): add density matrix backward"
```

### Task 7: Add Built-In Noise and Stinespring Gradients

**Files:**
- Create: `aicir/distributed/autograd/_channels.py`
- Create: `tests/distributed/autograd/test_channel_grad.py`
- Create: `tests/distributed/autograd/test_stinespring_grad.py`
- Create: `tests/distributed/autograd/test_channel_grad_multiprocess.py`
- Modify: `aicir/distributed/density.py`
- Modify: `aicir/noise/channels.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces differentiable bit flip, phase flip, depolarizing, and amplitude-damping channel parameters.
- Produces: `_householder_isometry(parameter: StinespringParam) -> _Pair`.
- Produces: `_stinespring_kraus(parameter: StinespringParam) -> tuple[_Pair, ...]`.
- Produces: `_PairMatrixKernel.apply_channel(state, channel, *, instruction_index) -> DistState`.

- [ ] **Step 1: Write built-in channel RED tests**

For each channel, compare gradients of probabilities and Pauli expectations
against CPU float64 finite differences. Include channel sequences and
noise-rule selection.

- [ ] **Step 2: Write Stinespring RED tests**

Test:

- isometry column completeness;
- `sum(K_i^H K_i)`;
- trace preservation;
- density positivity;
- raw real and imaginary parameter gradients;
- world-size 2 and 4 distributed-axis coverage.

- [ ] **Step 3: Implement differentiable built-in channels**

Channel matrices and weights remain paired-real. Avoid a complex
`accumulator = accumulator + term`; accumulate real and imaginary density
parts separately.

- [ ] **Step 4: Implement NPU-safe Householder sequence**

For paired vector `v`, compute:

```text
denominator = sum(v_real^2 + v_imag^2) + epsilon
H(x) = x - 2 v (v^H x) / denominator
```

Expand the inner product, multiplication, subtraction, and division into
real-valued formulas. Apply a fixed number of reflections determined by
`output_dim * environment_dim`. Select the first `input_dim` columns and
split environment blocks into Kraus operators.

- [ ] **Step 5: Implement channel backward**

Apply every Kraus term as paired-real `K rho K^H`. Accumulate real and
imaginary outputs independently. Ensure one deterministic Kraus order on
all ranks.

- [ ] **Step 6: Fill `noise` and `stinespring` probe sections**

The report contains channel-specific errors, completeness error, trace
error, positivity error, targeted distributed axes, and maximum raw
parameter gradient error.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_channel_grad.py \
  tests/distributed/autograd/test_stinespring_grad.py \
  tests/distributed/autograd/test_channel_grad_multiprocess.py \
  tests/distributed/test_noise_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_channels.py \
  aicir/distributed/density.py aicir/noise/channels.py \
  tests/distributed/autograd/test_channel_grad.py \
  tests/distributed/autograd/test_stinespring_grad.py \
  tests/distributed/autograd/test_channel_grad_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): add differentiable quantum channels"
```

### Task 8: Add Checkpoint Recomputation

**Files:**
- Create: `aicir/distributed/autograd/_checkpoint.py`
- Create: `tests/distributed/autograd/test_checkpoint.py`
- Create: `tests/distributed/autograd/test_checkpoint_multiprocess.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_CheckpointPolicy.parse(value) -> _CheckpointPolicy`.
- Produces: `_CheckpointPlanner(spec, circuit_depth, available_bytes).interval() -> int`.
- Produces: `_recompute_segment(start_state, plans, start, stop, engine) -> DistState`.
- Adds public `grad_checkpoint: Literal["none", "auto"] | int = "auto"` to `DistSimulator.run()`.

- [ ] **Step 1: Write checkpoint RED tests**

Compare `"none"`, `"auto"`, `1`, `4`, and `16` for identical outputs and
gradients. Test invalid zero, negative, boolean, and unknown string values.
Test non-identity layout and density evolution.

- [ ] **Step 2: Run RED tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_checkpoint.py \
  tests/distributed/autograd/test_checkpoint_multiprocess.py -q
```

Expected: `DistSimulator.run()` rejects the unknown keyword.

- [ ] **Step 3: Implement deterministic checkpoint plans**

`"auto"` estimates saved bytes from `_ShardSpec.local_shape`, paired-real
dtype bytes, forward temporaries, and a 20% safety margin. Select the
smallest interval whose estimate remains below 80% of available NPU memory.

All ranks all-gather the selected interval and reject disagreement before
forward.

- [ ] **Step 4: Implement segment recomputation**

Save boundary states and replay exactly the same gate plans, operation
indices, peers, and tags during backward. Analytic density-channel
execution is deterministic; sampling does not enter this path.

- [ ] **Step 5: Fill probe memory preliminaries**

Record saved-state count, recomputed-gate count, chosen interval, peak
allocation, and gradient error for `"none"`, `"auto"`, and `16`.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_checkpoint.py \
  tests/distributed/autograd/test_checkpoint_multiprocess.py \
  tests/distributed/test_simulator_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_checkpoint.py \
  aicir/distributed/simulator.py \
  tests/distributed/autograd/test_checkpoint.py \
  tests/distributed/autograd/test_checkpoint_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): checkpoint native backward"
```

### Task 9: Add Gradient Buckets and Replicated Optimizer Contracts

**Files:**
- Create: `tests/distributed/autograd/test_optimizer_sync_multiprocess.py`
- Create: `tests/distributed/autograd/test_gradient_bucket.py`
- Modify: `aicir/distributed/autograd/_parameters.py`
- Modify: `aicir/distributed/autograd/_contracts.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_bucket_parameters(parameters, *, communicator) -> tuple[torch.Tensor, ...]`.
- Produces: `_GradientBucketFn`, whose forward returns differentiable
  aliases and whose single backward invocation receives every alias
  gradient, packs one float32 buffer, all-reduces it, and returns unpacked
  gradients to the original leaves.
- Produces: `_bind_trainable_aliases(circuit, mapping) -> Circuit`, which
  rebuilds typed instructions with bucket aliases without mutating the
  caller's circuit.
- Produces: collective-safe parameter structure digest and exact mismatch errors.
- Replicated buckets include circuit, built-in noise, and Stinespring
  parameters. Root-owned and sharded initial-state parameters retain their
  declared ownership and are excluded from these buckets.

- [ ] **Step 1: Write gradient-bucket RED tests**

Assert:

- 32 and 128 scalar parameters use one float32 gradient all-reduce bucket;
- paired-real parameters use separate real and imaginary ranges in the same
  float32 bucket;
- all ranks receive identical gradients;
- root-owned initial parameters are absent from non-root replicated
  buckets;
- sharded initial parameters retain distinct rank-local gradients;
- missing gradient, different parameter order, shape mismatch, dtype
  mismatch, and `requires_grad` mismatch fail synchronously;
- the caller's original parameter leaves receive synchronized gradients;
- no autograd hook or HCCL work handle remains after backward.

- [ ] **Step 2: Implement parameter structure preflight**

Hash ordered fields:

```text
parameter name
shape
dtype
requires_grad
paired component
```

All-gather the digest before forward. A mismatch raises
`ValueError("各 rank 的可训练参数结构不一致")`.

- [ ] **Step 3: Implement one-bucket synchronization**

Before gate planning, pass the ordered replicated parameter leaves through
one `_GradientBucketFn`. Its forward returns tensor aliases with identical
values. Rebuild an internal circuit/channel view using those aliases; do
not mutate caller-owned instructions.

Autograd invokes `_GradientBucketFn.backward` once with every alias
gradient. Replace a missing alias gradient with a same-shaped zero tensor,
pack all float32 values in deterministic order, issue one HCCL all-reduce,
and unpack the result returned for the original leaves. Divide only when
the engine marks a parameter contribution as replicated rather than
rank-local.

This makes synchronization part of `loss.backward()` itself and avoids
per-parameter hooks or post-backward user calls.

- [ ] **Step 4: Test replicated optimizers**

Run 100 SGD steps and 100 Adam steps on world sizes 2 and 4. After every
step, hash replicated circuit/noise/Stinespring parameters and optimizer
state and assert one unique digest. Separately verify root-only optimizer
state for a root-owned initial factor and valid global normalization for
sharded initial factors.

- [ ] **Step 5: Fill the probe `optimizer` section**

Run 100-step SGD and Adam cases with 32 and 128 parameters. Report gradient
all-reduce count, parameter agreement, optimizer-state agreement, and
unfinished work handles.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_gradient_bucket.py \
  tests/distributed/autograd/test_optimizer_sync_multiprocess.py -q
python -m py_compile scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_parameters.py \
  aicir/distributed/autograd/_contracts.py \
  aicir/distributed/simulator.py \
  tests/distributed/autograd/test_gradient_bucket.py \
  tests/distributed/autograd/test_optimizer_sync_multiprocess.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "feat(distributed): synchronize replicated gradients"
```

### Task 10: Optimize P2P Buffers and Communication Overlap

**Files:**
- Create: `scripts/npu/distributed_autograd_benchmark.py`
- Create: `tests/distributed/autograd/test_benchmark_contract.py`
- Create: `tests/distributed/autograd/test_buffer_pool.py`
- Modify: `aicir/distributed/autograd/_collectives.py`
- Modify: `aicir/distributed/communication.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`

**Interfaces:**
- Produces: `_PairBufferPool.acquire(shape, dtype, device, peer, phase) -> _Pair`.
- Produces: `_AsyncPairExchange(pair, real_work, imag_work).wait() -> _Pair`.
- Produces benchmark CLI `distributed_autograd_benchmark.py --communication-mode {baseline,reuse,overlap} --gradient-method {native,parameter_shift,finite_difference} --path {statevector,density,noise,stinespring} --n-qubits INT --depth INT --parameters INT --warmups INT --runs INT --output-json PATH`.

- [ ] **Step 1: Freeze benchmark JSON**

Require:

```json
{
  "communication_mode": "baseline",
  "gradient_method": "native",
  "path": "statevector",
  "world_size": 2,
  "n_qubits": 24,
  "depth": 64,
  "parameters": 32,
  "warmups": 5,
  "runs": 30,
  "forward_ms_median": 0.0,
  "backward_ms_median": 0.0,
  "gradient_ms_median": 0.0,
  "gradient_ms_p95": 0.0,
  "peak_memory_bytes": 1,
  "p2p_bytes": 1,
  "wait_ms": 0.0,
  "buffer_reuse_count": 0,
  "fallback_to_cpu": false
}
```

Tests reject missing fields, non-positive run counts, negative times, an
invalid path/method combination, and a true fallback flag. Shift-rule gate
workloads allow `parameter_shift`; raw state, density-factor, and
Stinespring workloads allow `finite_difference`.

- [ ] **Step 2: Implement synchronized baseline timing**

Use five warmups and 30 measured runs. Call
`torch.npu.synchronize()` before and after every timed forward/backward
gradient. Report complete gradient time.

- [ ] **Step 3: Implement reusable buffers**

Key the pool by shape, dtype, device, peer, and phase. A checked-out buffer
cannot be reused until both real and imaginary work handles finish.

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_buffer_pool.py \
  tests/distributed/autograd/test_benchmark_contract.py -q
```

Expected: all pass.

- [ ] **Step 4: Implement measured overlap**

For a cross-shard gate:

1. launch next paired exchange;
2. compute current local matrix work;
3. wait for both real and imaginary receives;
4. compute the remote-source contribution.

Preserve a synchronous `baseline` mode for numerical and timing comparison.

- [ ] **Step 5: Require numerical parity**

For every communication mode and applicable gradient method:

```text
state error <= 1e-6
gradient error <= 1e-4
```

An optimized mode with worse numerical results is rejected even if faster.

- [ ] **Step 6: Fill probe `performance` communication metrics**

Record baseline, reuse, and overlap timing, bytes, waits, reuse counts, and
gradient errors. Pair each native record with a parameter-shift record for
valid shift-rule parameters or a central finite-difference record for raw
state, density-factor, and Stinespring parameters.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/autograd/test_buffer_pool.py \
  tests/distributed/autograd/test_benchmark_contract.py \
  tests/distributed/autograd/test_collectives_multiprocess.py -q
python -m py_compile \
  scripts/npu/distributed_autograd_benchmark.py \
  scripts/npu/distributed_autograd_probe.py
git diff --check
```

Commit:

```bash
git add aicir/distributed/autograd/_collectives.py \
  aicir/distributed/communication.py \
  tests/distributed/autograd/test_buffer_pool.py \
  tests/distributed/autograd/test_benchmark_contract.py \
  scripts/npu/distributed_autograd_benchmark.py \
  scripts/npu/distributed_autograd_probe.py
git commit -m "perf(distributed): reuse and overlap gradient P2P"
```

### Task 11: Integrate Automatic Routing and Complete the NPU Probe

**Files:**
- Create: `tests/distributed/autograd/test_simulator_autograd_multiprocess.py`
- Create: `tests/distributed/autograd/test_probe_contract.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `aicir/distributed/result.py`
- Modify: `aicir/distributed/_contracts.py`
- Modify: `scripts/npu/distributed_autograd_probe.py`
- Modify: `scripts/npu/distributed_autograd.sh`

**Interfaces:**
- `DistSimulator.run(..., grad_checkpoint="auto")` automatically routes collectively.
- Non-trainable calls retain byte-for-byte forward behavior where currently tested.
- Probe `--section all` contains no pending section.

- [ ] **Step 1: Write routing RED tests**

Test:

- no trainable input calls the existing forward engine;
- one trainable gate calls the paired-real engine;
- trainable root state and trainable `DistState` route consistently;
- one-rank trainable mismatch fails before state transport;
- unsupported operation fails in capability preflight;
- no fallback occurs;
- all existing forward-only result combinations remain unchanged.

- [ ] **Step 2: Replace the blanket rejection with collective routing**

Retain the exact current rejection behind an internal release gate while
tests are incomplete. When every capability required by the circuit is
available, route into the autograd engine. Preflight scans the complete
circuit, observable set, initial-state type, noise model, and run options.

- [ ] **Step 3: Complete all 13 probe sections**

The top-level report includes:

```json
{
  "commit": "full-git-sha",
  "world_size": 2,
  "backend": "hccl",
  "fallback_to_cpu": false,
  "passed": true,
  "failed_invariants": [],
  "sections": {}
}
```

Every section has `status`, `passed`, `metrics`, and `failed_invariants`.
Remove all development-time `BLOCKED` values.

- [ ] **Step 4: Add exact error contracts**

Contract cases include:

- sample in gradient mode;
- counts in gradient mode;
- collapse in gradient mode;
- trainable direct complex leaf if capability remains blocked;
- parameter structure mismatch;
- initial-state ownership mismatch;
- shape/dtype mismatch;
- unsupported gate/channel/observable;
- non-HCCL strict run;
- CPU fallback request;
- invalid checkpoint policy;
- forward/backward tag mismatch injection.

All ranks must report exact exception type, exact message, and one unique
SHA-256 digest.

- [ ] **Step 5: Run local complete verification**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd -q
PYTHONPATH=. pytest tests/distributed -q
python -m py_compile \
  scripts/npu/distributed_autograd_probe.py \
  scripts/npu/distributed_autograd_benchmark.py
bash -n scripts/npu/distributed_autograd.sh
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit**

```bash
git add aicir/distributed/simulator.py \
  aicir/distributed/result.py \
  aicir/distributed/_contracts.py \
  tests/distributed/autograd/test_simulator_autograd_multiprocess.py \
  tests/distributed/autograd/test_probe_contract.py \
  scripts/npu/distributed_autograd_probe.py \
  scripts/npu/distributed_autograd.sh
git commit -m "feat(distributed): enable native autograd routing"
```

### Task 12: Run 2/4/8-NPU Release Gates and Publish Evidence

**Files:**
- Create: `scripts/npu/distributed_autograd_evidence.py`
- Create: `tests/distributed/autograd/test_evidence_contract.py`
- Create after hardware runs: `docs/evidence/distributed-autograd/<commit>/world2.json`
- Create after hardware runs: `docs/evidence/distributed-autograd/<commit>/world4.json`
- Create after hardware runs: `docs/evidence/distributed-autograd/<commit>/world8.json`
- Create after hardware runs: `docs/evidence/distributed-autograd/<commit>/manifest.json`
- Modify: `aicir/distributed/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: evidence CLI `distributed_autograd_evidence.py validate-run REPORT`.
- Produces: evidence CLI `distributed_autograd_evidence.py aggregate WORLD2 WORLD4 WORLD8 --output MANIFEST`.
- `aggregate` requires world-size set `{2, 4, 8}`, matching commit SHA, HCCL, no fallback, all section passes, correctness gates, performance gates, and stability gates.

- [ ] **Step 1: Write evidence RED tests**

Fixtures cover:

- valid independent 2/4/8 reports;
- missing world size;
- duplicated world size;
- commit mismatch;
- failed section;
- fallback flag;
- gradient error above `1e-4`;
- rank disagreement above `1e-6`;
- native median not below the applicable parameter-shift or central
  finite-difference median;
- memory growth above 1%;
- incomplete HCCL work;
- report digest mismatch.

- [ ] **Step 2: Implement strict evidence validation**

Each report records:

```text
commit
command
exit_code
world_size
rank_devices
torch_version
torch_npu_version
cann_version or unknown
backend
fallback_to_cpu
sections
raw_sha256
```

The manifest emits `release_gate="PASS"` only when every required condition
passes. Missing 8-NPU evidence produces `release_gate="BLOCKED"`, never
`SKIPPED`.

- [ ] **Step 3: Run the 2-NPU job**

Run on the remote Ascend platform:

```bash
PYTHONPATH=.:${PYTHONPATH:-} torchrun \
  --nproc-per-node=2 \
  scripts/npu/distributed_autograd_probe.py \
  --section all \
  --output-json world2.json
```

Expected: exit zero, top-level `passed=true`, 13 section statuses `PASS`,
HCCL, no fallback.

- [ ] **Step 4: Run the 4-NPU job**

```bash
PYTHONPATH=.:${PYTHONPATH:-} torchrun \
  --nproc-per-node=4 \
  scripts/npu/distributed_autograd_probe.py \
  --section all \
  --output-json world4.json
```

Expected: same contract with every distributed axis exercised.

- [ ] **Step 5: Run the 8-NPU job**

```bash
PYTHONPATH=.:${PYTHONPATH:-} torchrun \
  --nproc-per-node=8 \
  scripts/npu/distributed_autograd_probe.py \
  --section all \
  --output-json world8.json
```

Expected: same contract. Lack of eight devices blocks release.

- [ ] **Step 6: Aggregate evidence**

Run:

```bash
PYTHONPATH=. python scripts/npu/distributed_autograd_evidence.py \
  aggregate world2.json world4.json world8.json \
  --output manifest.json
```

Expected:

```json
{"release_gate": "PASS"}
```

- [ ] **Step 7: Archive exact reports**

Copy the three validated rank-0 JSON files and manifest into:

```text
docs/evidence/distributed-autograd/<commit>/
```

Do not edit numeric values or remove environment warnings from the archived
records.

- [ ] **Step 8: Update user documentation**

Document:

- trainable `DistSimulator.run()` example;
- `PureStateParam`, `DensityParam`, and `StinespringParam`;
- differentiable and non-differentiable outputs;
- checkpoint modes;
- 2/4/8 commands;
- measured correctness, performance, memory, and environment evidence;
- explicit MPS, sampling, collapse, and multi-node exclusions.

- [ ] **Step 9: Run repository-wide verification**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/autograd -q
PYTHONPATH=. pytest tests/distributed -q
PYTHONPATH=. pytest -q
python -m py_compile \
  scripts/npu/distributed_autograd_probe.py \
  scripts/npu/distributed_autograd_benchmark.py \
  scripts/npu/distributed_autograd_evidence.py
bash -n scripts/npu/distributed_autograd.sh
git diff --check
```

Expected: every command exits zero; existing warnings may remain.

- [ ] **Step 10: Commit evidence and documentation**

```bash
git add scripts/npu/distributed_autograd_evidence.py \
  tests/distributed/autograd/test_evidence_contract.py \
  docs/evidence/distributed-autograd \
  aicir/distributed/README.md CHANGELOG.md
git commit -m "docs(distributed): publish native autograd evidence"
```

## Plan Self-Review Checklist

- [ ] Every approved design requirement maps to at least one task.
- [ ] Parameter-shift remains explicit and uses `aicir.qml.deriv.psr`.
- [ ] Stinespring raw parameters use CPU finite difference plus native
  paired-real backward because a generic parameter-shift rule is not valid.
- [ ] All NPU collectives and gradient accumulation use float32.
- [ ] Direct complex trainable leaves remain capability-gated.
- [ ] Statevector, density, noise, Stinespring, optimizer, performance, and
  memory paths have local and true-NPU tests.
- [ ] 2/4/8 jobs are independent and the manifest aggregates them.
- [ ] Missing 8-NPU resources block release.
- [ ] Public autograd routing is not enabled before complete local coverage.
- [ ] No task claims MPS, sampling, collapse, circuit cutting, quantum
  networking, or multi-node native autograd support.

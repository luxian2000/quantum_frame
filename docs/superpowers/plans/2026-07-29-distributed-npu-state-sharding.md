# Distributed NPU State Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `aicir.distributed` API that shards one statevector or row-shards one density matrix across a power-of-two number of Ascend NPUs without changing existing simulator APIs.

**Architecture:** `DistNPUBackend` owns one process/NPU and a communication wrapper; `DistState` owns only the local tensor plus immutable layout metadata. `DistSimulator` plans a static logical-to-storage layout, routes local gates to existing NPU kernels, streams partner shards for distributed axes, and uses row action plus local column action for density matrices.

**Tech Stack:** Python 3.10+, NumPy, PyTorch `torch.distributed`, Gloo multiprocess tests, Ascend `torch_npu`/HCCL, pytest.

## Global Constraints

- `world_size` must equal `2**p`; require `n_qubits >= p`.
- First release supports `complex64` forward execution only.
- Existing `NPUBackend`, `State`, `Measure.run()`, and `Result` APIs and semantics must not change.
- Public distributed types are exactly `DistNPUBackend`, `DistState`, `DistSimulator`, and `DistResult`.
- Do not add generic public aliases or top-level `aicir` re-exports.
- Do not fall back to CPU, a full global unitary, or implicit full-state gathering.
- Initial release excludes mid-circuit measurement, reset, classical control flow, dynamic layouts, and distributed autograd.
- All ranks execute communication in deterministic instruction and partner order.

---

## File Structure

- Create `aicir/distributed/__init__.py`: explicit public exports only.
- Create `aicir/distributed/layout.py`: immutable `_ShardSpec` and `_Layout`.
- Create `aicir/distributed/communication.py`: `_Communicator` and Gloo/HCCL exchange/reduction primitives.
- Create `aicir/distributed/backend.py`: `DistNPUBackend` construction, validation, device binding, and capacity checks.
- Create `aicir/distributed/state.py`: `DistState`, explicit gather, and root-only NumPy materialization.
- Create `aicir/distributed/gates.py`: `_GatePlan`, `_GatePlanner`, and `_VectorKernel`.
- Create `aicir/distributed/density.py`: `_MatrixKernel` and local Kraus accumulation.
- Create `aicir/distributed/reducers.py`: local probabilities, global scalar reductions, observables, and sampling.
- Create `aicir/distributed/result.py`: `DistResult`.
- Create `aicir/distributed/simulator.py`: validation and end-to-end orchestration.
- Modify `aicir/noise/base.py`: protected local-Kraus capability with an explicit unsupported default.
- Modify `aicir/noise/channels.py`: local Kraus implementations while preserving full embedded operators.
- Create `tests/distributed/`: focused unit and spawned Gloo integration tests.
- Create `scripts/npu/distributed_state_probe.py`: strict two-/four-NPU HCCL validation entry point.
- Create `docs/distributed.md`: public usage, support matrix, memory formulas, and rank/layout explanation.
- Modify `README.md` and `CHANGELOG.md`: discoverability and release record.

---

### Task 1: Distribution Metadata and Public Types

**Files:**
- Create: `aicir/distributed/__init__.py`
- Create: `aicir/distributed/layout.py`
- Create: `tests/distributed/test_layout.py`
- Create: `tests/distributed/test_api.py`

**Interfaces:**
- Produces: `_ShardSpec.build(n_qubits: int, world_size: int, rank: int, kind: str, layout: _Layout) -> _ShardSpec`
- Produces: `_Layout.auto(circuit, n_qubits: int, distributed_axes: int) -> _Layout`
- Produces: `_Layout.explicit(mapping, n_qubits: int, distributed_axes: int) -> _Layout`
- Produces: `logical_to_storage`, `storage_to_logical`, `distributed_logical_qubits`, and `digest()`

- [ ] **Step 1: Write failing metadata and API tests**

```python
def test_shard_spec_vector_and_matrix_shapes():
    layout = _Layout.explicit((0, 1, 2, 3), n_qubits=4, distributed_axes=2)
    vector = _ShardSpec.build(4, 4, 1, "vector", layout)
    matrix = _ShardSpec.build(4, 4, 1, "matrix", layout)
    assert vector.local_shape == (4, 1)
    assert matrix.local_shape == (4, 16)
    assert vector.global_start == 4
    assert vector.global_stop == 8


def test_public_api_uses_explicit_names_only():
    import aicir.distributed as distributed
    assert distributed.__all__ == [
        "DistNPUBackend", "DistState", "DistSimulator", "DistResult"
    ]
    assert not hasattr(distributed, "Backend")
    assert not hasattr(distributed, "State")
```

- [ ] **Step 2: Run tests and verify collection fails**

Run: `PYTHONPATH=. pytest tests/distributed/test_layout.py tests/distributed/test_api.py -q`

Expected: FAIL because `aicir.distributed` does not exist.

- [ ] **Step 3: Implement immutable metadata and deterministic greedy layout**

```python
@dataclass(frozen=True)
class _Layout:
    logical_to_storage: tuple[int, ...]
    distributed_axes: int

    @classmethod
    def explicit(cls, mapping, *, n_qubits, distributed_axes):
        values = tuple(int(x) for x in mapping)
        if sorted(values) != list(range(n_qubits)):
            raise ValueError("layout 必须是 range(n_qubits) 的完整双射")
        return cls(values, int(distributed_axes))

    @property
    def storage_to_logical(self):
        inverse = [0] * len(self.logical_to_storage)
        for logical, storage in enumerate(self.logical_to_storage):
            inverse[storage] = logical
        return tuple(inverse)

    def digest(self):
        return hashlib.sha256(repr(
            (self.logical_to_storage, self.distributed_axes)
        ).encode()).hexdigest()
```

Implement `_Layout.auto()` with the approved greedy score
`sum(2**len(gate_qubits & D) - 1)` and logical-qubit-index tie breaking.
Implement `_ShardSpec` validation for power-of-two world size, rank bounds,
`n_qubits >= log2(world_size)`, and vector/matrix local shapes.

- [ ] **Step 4: Add temporary import-safe public stubs**

```python
from .backend import DistNPUBackend
from .result import DistResult
from .simulator import DistSimulator
from .state import DistState

__all__ = ["DistNPUBackend", "DistState", "DistSimulator", "DistResult"]
```

Create minimal named classes in their target modules so imports are stable;
later tasks replace their bodies without renaming them.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_layout.py tests/distributed/test_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/distributed tests/distributed/test_layout.py tests/distributed/test_api.py
git commit -m "feat(distributed): add shard metadata and explicit API"
```

---

### Task 2: Distributed Backend and Communication

**Files:**
- Modify: `aicir/distributed/backend.py`
- Create: `aicir/distributed/communication.py`
- Create: `tests/distributed/test_backend.py`
- Create: `tests/distributed/test_communication.py`

**Interfaces:**
- Consumes: `_ShardSpec`
- Produces: `DistNPUBackend.from_env(fallback_to_cpu: bool = False, process_group_backend: str | None = None)`
- Produces: `_Communicator.exchange(tensor, peer: int, tag: int)`
- Produces: `_Communicator.all_reduce_sum(tensor)` and root gather/scatter helpers

- [ ] **Step 1: Write failing backend validation tests**

```python
def test_backend_rejects_non_power_of_two_world(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="2 的幂"):
        DistNPUBackend.from_env(
            fallback_to_cpu=True, init_process_group=False
        )
```

Add mock-process-group tests for deterministic partner exchange, scalar
all-reduce, root gather, and `complex64` real/imag transport fallback.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_backend.py tests/distributed/test_communication.py -q`

Expected: FAIL because backend and communicator methods are absent.

- [ ] **Step 3: Implement backend construction**

```python
class DistNPUBackend(NPUBackend):
    @classmethod
    def from_env(cls, *, fallback_to_cpu=False, init_process_group=True,
                 process_group_backend=None):
        ctx = npu_runtime_context_from_env()
        if ctx.world_size & (ctx.world_size - 1):
            raise ValueError("分布式状态分片要求 world_size 是 2 的幂")
        backend = cls(
            device=f"npu:{ctx.local_rank}",
            fallback_to_cpu=fallback_to_cpu,
        )
        backend._initialize_distribution(
            ctx, init_process_group, process_group_backend
        )
        return backend
```

Require `torch.complex64`, expose read-only rank/world/local-rank properties,
and forbid batch-index partition helpers on this subclass.

- [ ] **Step 4: Implement communication wrapper**

Use `torch.distributed.P2POp` plus `batch_isend_irecv` for the preferred
exchange. Wait for every returned work handle. If complex exchange fails the
startup probe, exchange contiguous `float32` real and imaginary buffers in
that fixed order and reconstruct on the source device. Implement injected
process-group functions so unit tests do not require Ascend.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_backend.py tests/distributed/test_communication.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/distributed/backend.py aicir/distributed/communication.py tests/distributed
git commit -m "feat(distributed): add NPU runtime and communication"
```

---

### Task 3: Distributed State Lifecycle

**Files:**
- Modify: `aicir/distributed/state.py`
- Create: `tests/distributed/test_state.py`
- Create: `tests/distributed/test_state_multiprocess.py`

**Interfaces:**
- Consumes: `DistNPUBackend`, `_ShardSpec`, `_Layout`
- Produces: `DistState.zero(...)`, `DistState.from_local(...)`, `gather(root=0)`, `to_numpy(root=0)`
- Produces: `local_probabilities()` for vector and matrix state kinds

- [ ] **Step 1: Write failing state tests**

```python
def test_dist_state_rejects_global_tensor_as_local():
    spec = make_vector_spec(n_qubits=3, world_size=2, rank=0)
    with pytest.raises(ValueError, match="local_shape"):
        DistState.from_local(torch.zeros(8, 1, dtype=torch.complex64), spec, backend)


def test_zero_state_only_rank_zero_owns_unit_amplitude():
    state = DistState.zero(3, backend, layout)
    expected = 1.0 if backend.rank == 0 else 0.0
    assert float(state.local_probabilities().sum()) == expected
```

Spawn a two-rank Gloo test that gathers vector and matrix shards only on rank
zero and checks non-root return values are `None`.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_state.py tests/distributed/test_state_multiprocess.py -q`

Expected: FAIL because `DistState` is still a stub.

- [ ] **Step 3: Implement state validation and zero initialization**

```python
class DistState:
    def __init__(self, local_data, spec, backend, *, bit_order="msb"):
        if tuple(local_data.shape) != spec.local_shape:
            raise ValueError(
                f"local_data shape {tuple(local_data.shape)} != local_shape {spec.local_shape}"
            )
        if getattr(local_data, "requires_grad", False):
            raise ValueError("分布式首期不支持 requires_grad=True")
        self._local_data = local_data
        self._spec = spec
        self._backend = backend
        self._bit_order = bit_order
```

Implement vector/matrix local probability extraction without copying a full
matrix. Implement root-only gather and logical-order restoration.

- [ ] **Step 4: Implement root scatter constructors**

Add internal constructors for root-owned full arrays and already-sharded
`DistState` inputs. Validate mode agreement before communication. Zero-state
construction must allocate only a local tensor.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_state.py tests/distributed/test_state_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/distributed/state.py tests/distributed
git commit -m "feat(distributed): add sharded state lifecycle"
```

---

### Task 4: Gate Planning and Statevector Kernel

**Files:**
- Create: `aicir/distributed/gates.py`
- Create: `tests/distributed/test_gate_planner.py`
- Create: `tests/distributed/test_vector_kernel.py`
- Create: `tests/distributed/test_vector_kernel_multiprocess.py`

**Interfaces:**
- Consumes: `_Layout`, `_Communicator`, existing local gate matrices
- Produces: `_GatePlanner.plan(gate, instruction_index) -> _GatePlan`
- Produces: `_VectorKernel.apply(state: DistState, plan: _GatePlan) -> DistState`

- [ ] **Step 1: Write failing planning and kernel tests**

```python
def test_local_gate_plan_has_no_partners():
    plan = planner.plan(hadamard(3), instruction_index=0)
    assert plan.distributed_storage_axes == ()
    assert plan.partner_masks == ()


def test_distributed_axis_plan_uses_xor_partner():
    plan = planner.plan(pauli_x(0), instruction_index=1)
    assert plan.partner_masks == (2,)
    assert plan.partner_for(rank=1, mask=2) == 3
```

Add two- and four-rank Gloo numerical comparisons for X, H, CX, SWAP,
Toffoli, Bell, GHZ, and a seeded shallow random circuit.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_gate_planner.py tests/distributed/test_vector_kernel.py tests/distributed/test_vector_kernel_multiprocess.py -q`

Expected: FAIL because gate planning and kernels do not exist.

- [ ] **Step 3: Implement gate plans**

Normalize typed instructions with existing IR accessors. Reuse the registry
local matrix source; reject instructions without a finite local matrix. Store
logical targets, mapped storage axes, distributed rank-bit positions, ordered
partner masks, local matrix, and instruction index in an immutable `_GatePlan`.

- [ ] **Step 4: Implement local and streaming distributed kernels**

For local axes, reshape only the local tensor and reuse the existing bounded
local-gate implementation. For `r > 0`, compute the output block row owned by
the current rank, then stream partner shards in ascending mask order:

```python
out = apply_matrix_block(plan, rank_source=rank, shard=state.local_data)
for mask in plan.partner_masks:
    peer = rank ^ mask
    incoming = communicator.exchange(state.local_data, peer, plan.tag(mask))
    out = backend.add(
        out,
        apply_matrix_block(plan, rank_source=peer, shard=incoming),
    )
```

All accumulation on NPU must use `backend.add`, not raw complex addition.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_gate_planner.py tests/distributed/test_vector_kernel.py tests/distributed/test_vector_kernel_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/distributed/gates.py tests/distributed
git commit -m "feat(distributed): add sharded statevector gates"
```

---

### Task 5: Density Matrix and Local Kraus Noise

**Files:**
- Modify: `aicir/noise/base.py`
- Modify: `aicir/noise/channels.py`
- Create: `aicir/distributed/density.py`
- Create: `tests/noise/test_local_kraus.py`
- Create: `tests/distributed/test_density_kernel_multiprocess.py`
- Create: `tests/distributed/test_noise_multiprocess.py`

**Interfaces:**
- Produces: `NoiseChannel._local_kraus(n_qubits: int, backend) -> tuple[tuple[object, tuple[int, ...]], ...]`
- Consumes: `_VectorKernel` row action
- Produces: `_MatrixKernel.apply_unitary(...)`, `promote_vector(...)`, and `apply_noise_model(...)`

- [ ] **Step 1: Write failing compatibility and density tests**

```python
def test_local_kraus_matches_existing_embedded_channel():
    channel = AmplitudeDampingChannel(target_qubit=1, gamma=0.2)
    local = channel._local_kraus(3, backend)
    assert all(targets == (1,) for _, targets in local)
    assert all(tuple(matrix.shape) == (2, 2) for matrix, _ in local)
    assert len(channel.kraus_operators(3, backend)) == len(local)
```

Add Gloo comparisons for unitary density evolution, maximally mixed input,
amplitude damping, depolarizing noise, a two-qubit channel, and trace
preservation.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/noise/test_local_kraus.py tests/distributed/test_density_kernel_multiprocess.py tests/distributed/test_noise_multiprocess.py -q`

Expected: FAIL because the protected local Kraus hook and matrix kernel are absent.

- [ ] **Step 3: Add protected local Kraus representations**

The base default raises `NotImplementedError`. Each built-in channel returns
its existing finite local matrices paired with target qubits. Refactor
`kraus_operators()` to embed those same matrices, preserving returned values
and existing tests.

- [ ] **Step 4: Implement row-sharded density evolution**

Apply left action through the streaming row kernel with columns as batch,
then permute/reshape the local row block and apply `U†` to column axes locally.
For Kraus noise, accumulate each `K rho K†` local block with `backend.add`.

For vector promotion, gather only the complete statevector, construct
`local_rows @ full_vector†`, and include both buffers in capacity validation.

- [ ] **Step 5: Run focused and existing noise tests**

Run: `PYTHONPATH=. pytest tests/noise tests/distributed/test_density_kernel_multiprocess.py tests/distributed/test_noise_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aicir/noise aicir/distributed/density.py tests/noise tests/distributed
git commit -m "feat(distributed): add density matrices and Kraus noise"
```

---

### Task 6: Reductions, Observables, Sampling, and Result

**Files:**
- Create: `aicir/distributed/reducers.py`
- Modify: `aicir/distributed/result.py`
- Create: `tests/distributed/test_reducers_multiprocess.py`
- Create: `tests/distributed/test_result.py`

**Interfaces:**
- Produces: `_Reducer.probabilities(state)`, `expectations(state, observables)`, and `sample_z(...)`
- Produces: `DistResult.gather_probabilities(root=0)`

- [ ] **Step 1: Write failing result and reduction tests**

```python
def test_result_never_implicitly_gathers():
    result = DistResult(state, local_probabilities, {}, None, root=0)
    assert result.state is state
    communicator.gather.assert_not_called()


def test_non_root_counts_are_none():
    result = make_result(rank=1, world_size=2, counts=None)
    assert result.counts is None
    assert not result.is_root
```

Add multiprocess comparisons for vector/matrix probabilities, `PauliString`,
`Observable.pauli`, sums of Pauli terms, subset Z sampling, seeded
reproducibility, and one-shot collapse.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_result.py tests/distributed/test_reducers_multiprocess.py -q`

Expected: FAIL because reducers and result behavior are absent.

- [ ] **Step 3: Implement local probabilities and structured expectations**

All global norms and scalar expectations use `all_reduce(SUM)`. Reject an
unstructured full-system dense observable before allocating it. Preserve
logical qubit ordering through `_Layout`.

- [ ] **Step 4: Implement root-coordinated sampling**

Rank zero samples rank ownership from per-rank mass, broadcasts assignments,
and ranks sample local conditional indices with generators derived from
`(seed, rank)`. Gather only sampled indices. Permit `collapse=True` only for
one shot and normalize with a scalar all-reduce.

- [ ] **Step 5: Implement immutable result**

```python
@dataclass(frozen=True)
class DistResult:
    state: DistState | None
    local_probabilities: object | None
    expectations: Mapping[str, float]
    counts: Mapping[str, int] | None
    rank: int
    world_size: int

    @property
    def is_root(self):
        return self.rank == 0
```

`gather_probabilities()` performs the only explicit probability gather.

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_result.py tests/distributed/test_reducers_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aicir/distributed/reducers.py aicir/distributed/result.py tests/distributed
git commit -m "feat(distributed): add reductions sampling and results"
```

---

### Task 7: End-to-End Distributed Simulator

**Files:**
- Modify: `aicir/distributed/simulator.py`
- Create: `tests/distributed/test_simulator_validation.py`
- Create: `tests/distributed/test_simulator_multiprocess.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1–6
- Produces: the approved `DistSimulator.run(...) -> DistResult`
- Produces: `DistSimulator.from_env(...) -> DistSimulator`

- [ ] **Step 1: Write failing validation and integration tests**

Cover the exact public signature, root-owned initial inputs, already-sharded
inputs, zero initialization, unsupported instructions, `requires_grad`,
layout mismatch, invalid shots/collapse, circuit digest mismatch, and
`return_state`/`return_probabilities`.

```python
def test_rejects_multi_shot_collapse(simulator, circuit):
    with pytest.raises(ValueError, match="collapse=True"):
        simulator.run(circuit, shots=2, collapse=True)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_simulator_validation.py tests/distributed/test_simulator_multiprocess.py -q`

Expected: FAIL because the simulator is still a stub.

- [ ] **Step 3: Implement preflight and digest agreement**

Normalize circuit instructions, reject non-goals before allocation, compute
the deterministic layout, validate inputs, and compare a digest of circuit,
options, mode, and layout across ranks.

- [ ] **Step 4: Implement orchestration**

Run under `torch.no_grad()`. Initialize state, plan and apply instructions in
order, promote and apply matching deterministic noise, compute reductions,
sample terminal Z measurements, and construct `DistResult`. Never call the
existing `Measure.run()` distributed path.

- [ ] **Step 5: Run the complete distributed test directory**

Run: `PYTHONPATH=. pytest tests/distributed -q`

Expected: PASS.

- [ ] **Step 6: Run compatibility suites**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_backend.py tests/measure tests/noise -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aicir/distributed/simulator.py tests/distributed
git commit -m "feat(distributed): integrate distributed simulator"
```

---

### Task 8: Real NPU Probe, Documentation, and Full Verification

**Files:**
- Create: `scripts/npu/distributed_state_probe.py`
- Create: `docs/distributed.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `tests/distributed/test_probe_contract.py`

**Interfaces:**
- Consumes: public distributed API
- Produces: strict two-/four-NPU validation JSON and exit status

- [ ] **Step 1: Write failing probe contract test**

Test that the probe requires `fallback_to_cpu=False`, reports rank/device,
checks local and communicating gates, density/noise/reduction/sampling, and
fails when a local tensor is not on `npu:LOCAL_RANK`.

- [ ] **Step 2: Run test and verify failure**

Run: `PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q`

Expected: FAIL because the probe does not exist.

- [ ] **Step 3: Implement strict probe**

The script emits one root JSON object containing world size, devices,
statevector and density numerical errors, trace/norm errors, communication
path, local tensor sizes, and CPU-fallback status. Any failed invariant exits
nonzero.

- [ ] **Step 4: Write public documentation**

Document:

```bash
torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py
```

Include API examples, supported/rejected features, rank versus local rank,
logical versus storage axes, vector/matrix memory formulas, root-only gather
semantics, and the absence of an unverified speedup claim.

- [ ] **Step 5: Run local verification**

Run:

```bash
PYTHONPATH=. pytest tests/distributed tests/noise -q
PYTHONPATH=. pytest -q
git diff --check
```

Expected: all tests PASS and no whitespace errors.

- [ ] **Step 6: Run real Ascend verification**

Run on an Ascend host:

```bash
PYTHONPATH=. torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
PYTHONPATH=. torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py
```

Expected: both commands exit zero, report `fallback_to_cpu=false`, list only
the expected NPU devices, and meet `rtol=1e-5`, `atol=1e-6` state tolerances
and `atol=1e-5` probability/expectation/trace tolerances.

- [ ] **Step 7: Commit**

```bash
git add scripts/npu/distributed_state_probe.py docs/distributed.md README.md CHANGELOG.md tests/distributed
git commit -m "docs(distributed): add NPU validation and usage guide"
```

---

## Final Review Gate

- [ ] Map every design acceptance criterion to a passing test or strict probe field.
- [ ] Confirm no placeholder text remains in the plan or implementation.
- [ ] Confirm all public type names and method signatures match this plan.
- [ ] Confirm `git status --short` contains no unintended files.
- [ ] Report separately: local Gloo verification, compatibility regression,
  and real Ascend verification. Do not present an unrun real-NPU command as a
  verified result.

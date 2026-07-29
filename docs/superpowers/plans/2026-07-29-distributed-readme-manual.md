# Distributed API README Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `aicir/distributed/README.md` as a complete lifecycle-oriented functional manual for the public distributed API.

**Architecture:** Keep one canonical package-level manual and organize it by the order in which users launch, construct, run, and inspect a distributed simulation. Examples use only public imports and are checked against the current signatures and distributed tests.

**Tech Stack:** Markdown, Python examples, `torchrun`, PyTorch distributed, aicir public APIs, pytest.

## Global Constraints

- Modify documentation only; do not change distributed runtime behavior.
- Public distributed imports are exactly `DistNPUBackend`, `DistState`, `DistSimulator`, and `DistResult`.
- Do not import or recommend `_Layout`, `_ShardSpec`, `_Communicator`, or other internal symbols.
- State that `world_size=2^p`, `n_qubits >= p`, and the first release is forward-only `complex64`.
- Preserve the root `README.md` 7.2 link to `aicir/distributed/README.md`.
- Do not claim real-HCCL validation or speedup without a strict probe result from the target Ascend environment.

---

### Task 1: Rewrite the Package Manual by Runtime Lifecycle

**Files:**
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: `DistNPUBackend.from_env(...)`, `DistSimulator.from_env(...)`, `DistSimulator.run(...)`, `DistState` properties and explicit gather methods, and `DistResult` properties.
- Produces: the canonical complete user manual linked from root `README.md` section 7.2.

- [ ] **Step 1: Replace the quick-start structure with a full table of contents**

Use the following top-level structure:

```markdown
## 1. 功能模型与前提条件
## 2. 启动分布式进程
## 3. 构建分布式后端
## 4. 构建分布式量子态
## 5. 构建分布式量子线路
## 6. 构建并调用分布式模拟器
## 7. 状态向量模拟
## 8. 密度矩阵模拟
## 9. 确定性 Kraus 噪声
## 10. 观测量与期望值
## 11. 末端 Z 基采样与坍缩
## 12. 读取 DistResult 与显式聚合
## 13. 复用 DistState 继续演化
## 14. 自动与显式 layout
## 15. 存储、通信与内存公式
## 16. 支持边界和错误条件
## 17. 真机验证与故障定位
## 18. 公共 API 参考
```

- [ ] **Step 2: Document process launch and backend construction**

Include strict two- and four-NPU launch commands:

```bash
PYTHONPATH=. torchrun --nproc-per-node=2 your_program.py
PYTHONPATH=. torchrun --nproc-per-node=4 your_program.py
```

Explain `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, then show both supported construction forms:

```python
from aicir.distributed import DistNPUBackend, DistSimulator

simulator = DistSimulator.from_env(fallback_to_cpu=False)

backend = DistNPUBackend.from_env(fallback_to_cpu=False)
simulator = DistSimulator(backend)
```

- [ ] **Step 3: Document all supported state-construction paths**

Show implicit zero-state construction:

```python
result = simulator.run(Circuit(n_qubits=4))
state = result.state
```

Show a root-owned complete statevector:

```python
initial_state = (
    np.array([1, 0, 0, 0], dtype=np.complex64)
    if simulator.backend.rank == 0
    else None
)
result = simulator.run(circuit, initial_state=initial_state)
```

Show a root-owned density matrix:

```python
initial_density_matrix = (
    np.diag([0.25, 0.75]).astype(np.complex64)
    if simulator.backend.rank == 0
    else None
)
result = simulator.run(
    Circuit(n_qubits=1),
    initial_density_matrix=initial_density_matrix,
)
```

State explicitly that arbitrary user construction through
`DistState.from_local()` is not a public first-release workflow because it
requires internal shard metadata.

- [ ] **Step 4: Explain distributed circuit construction**

State that there is no `DistCircuit`: every rank constructs the same ordinary
`Circuit`, and distribution is selected by `DistSimulator`.

```python
from aicir import Circuit, cx, hadamard, rz

circuit = Circuit(
    hadamard(0),
    cx(target_qubit=1, control_qubits=(0,)),
    rz(0.3, 1),
    n_qubits=2,
)
```

Explain that a logical gate is local or communicating according to its
storage axes after layout, not according to a separate distributed gate API.

- [ ] **Step 5: Document simulator execution parameters**

Include the exact signature:

```python
DistSimulator.run(
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

Add a parameter table explaining accepted values, rank agreement, and
return-materialization effects.

- [ ] **Step 6: Add separate statevector, density, and noise sections**

For statevector simulation, explain contiguous amplitude shards and local
shape `(2^n / world_size, 1)`.

For density simulation, explain row shards and local shape
`(2^n / world_size, 2^n)`.

For noise, show gate-triggered deterministic evolution:

```python
from aicir import AmplitudeDampingChannel, NoiseModel

circuit.noise_model = NoiseModel().add_channel(
    AmplitudeDampingChannel(target_qubit=0, gamma=0.05)
)
result = simulator.run(circuit)
assert result.state.is_density
```

State that noise is applied after matching gates and promotes a statevector
to a density matrix.

- [ ] **Step 7: Document supported observable forms**

Show `PauliString`, `Hamiltonian`, and a local dense `Observable.matrix` with
explicit logical targets:

```python
observables = {
    "zz": PauliString("ZZ", n_qubits=2),
    "energy": Hamiltonian([("ZI", 0.5), ("XX", -0.25)]),
    "x0": Observable.matrix(
        np.array([[0, 1], [1, 0]], dtype=np.complex64),
        metadata={"qubits": [0]},
    ),
}
result = simulator.run(circuit, observables=observables)
```

State that expectation scalars are equal on all ranks and unstructured
full-system dense matrices are rejected.

- [ ] **Step 8: Document terminal sampling and collapse**

Cover all logical qubits, ordered subsets, deterministic seeds, root-only
counts, and the single-shot collapse restriction:

```python
result = simulator.run(
    circuit,
    shots=1024,
    measure_qubits=(1, 0),
    seed=7,
)

collapsed = simulator.run(
    circuit,
    shots=1,
    collapse=True,
    seed=7,
)
```

- [ ] **Step 9: Document result access and continuation**

Add a table for `state`, `local_probabilities`, `expectations`, `counts`,
`rank`, `world_size`, and `is_root`.

Show collective explicit gathers:

```python
full_state = result.state.to_numpy(root=0)
full_probabilities = result.gather_probabilities(root=0)
```

State that every rank must call these methods even though non-root ranks
receive `None`.

Show continuation with the same layout:

```python
layout = first.state.layout.logical_to_storage
second = simulator.run(
    next_circuit,
    initial_state=first.state,
    layout=layout,
)
```

- [ ] **Step 10: Document layout, memory, limits, and strict validation**

Explain logical-to-storage mapping, rank-prefix storage axes, and why rank
metadata consumes no logical qubit.

Retain these formulas:

```text
statevector per rank = 8 * 2^n / world_size bytes
density matrix per rank = 8 * 4^n / world_size bytes
```

Keep a supported/rejected feature table and the strict probe commands:

```bash
PYTHONPATH=. torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
PYTHONPATH=. torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py
```

- [ ] **Step 11: Add the four-type public API reference**

For each public type, list its public constructor or factory, properties, and
methods actually intended for users. Mark low-level `DistState.from_local()`
and `DistState.zero()` as implementation-facing in the first release rather
than recommended user entry points.

- [ ] **Step 12: Review the rewritten manual**

Run:

```bash
rg -n '快速使用|_Layout|_ShardSpec|_Communicator|TODO|TBD' aicir/distributed/README.md
```

Expected: no quick-start heading, no internal imports in examples, and no
placeholders. Internal names may appear only in a warning that they are not
public construction APIs.

- [ ] **Step 13: Commit the manual rewrite**

```bash
git add aicir/distributed/README.md
git commit -m "docs(distributed): expand complete API manual"
```

---

### Task 2: Validate Examples, Links, and Documentation Contract

**Files:**
- Modify if needed: `aicir/distributed/README.md`
- Verify: `README.md`
- Verify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the completed package manual from Task 1.
- Produces: evidence that examples and links match the current public API.

- [ ] **Step 1: Check public signatures and exports**

Run:

```bash
python - <<'PY'
import inspect
from aicir import distributed

assert distributed.__all__ == [
    "DistNPUBackend",
    "DistState",
    "DistSimulator",
    "DistResult",
]
print(inspect.signature(distributed.DistSimulator.run))
PY
```

Expected: the signature matches the parameter list documented in Task 1.

- [ ] **Step 2: Validate root links and stale paths**

Run:

```bash
rg -n 'aicir/distributed/README\\.md' README.md CHANGELOG.md
test ! -e docs/distributed.md
```

Expected: root README section 7.2 and CHANGELOG point to the package manual,
and no duplicate legacy manual exists.

- [ ] **Step 3: Run documentation-adjacent distributed tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/test_simulator_validation.py \
  tests/distributed/test_probe_contract.py -q
```

Expected: PASS.

- [ ] **Step 4: Check formatting and workspace scope**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intended documentation changes before
the final commit.

- [ ] **Step 5: Commit any validation corrections**

If validation required corrections:

```bash
git add aicir/distributed/README.md README.md CHANGELOG.md
git commit -m "docs(distributed): correct API manual examples"
```

If no correction was needed, do not create an empty commit.

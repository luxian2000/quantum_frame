# Distributed Simulator Future Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有非分布式 API 语义的前提下，逐项补齐分布式参数梯度、通信性能、多节点、恢复、确定性和基准能力，并用独立证据决定是否进入原生自动微分。

**Architecture:** 保持 `DistNPUBackend`、`DistSimulator` 和 `DistState` 的显式分布式边界。每个切片先冻结设计与失败测试，再实现内部接口并分别通过 Gloo 与严格 NPU 探针；原生自动微分仅设置可行性门槛，不在参数移位切片中放宽当前前向契约。

**Tech Stack:** Python 3.10、PyTorch、torch.distributed、torch_npu、HCCL、Gloo、NumPy、pytest、torchrun。

## Global Constraints

- 首期已在 2026-07-30 通过 world size 2 和 4 的九节严格 NPU 前向验收；该证据不覆盖梯度、性能、多节点、故障恢复或更大规模。
- `DistSimulator.run(...)` 当前仍必须拒绝任何 `requires_grad=True` 输入，并精确报错 `DistSimulator 首期仅支持前向模拟，不支持自动微分`。
- 当前 `torch_npu` 报告 HCCL 不支持原生 gather，并以 all-gather 实现；任何聚合优化必须分别测量 root 显存、全体 rank 显存和通信字节。
- world size 2 已覆盖分布式轴 `[0]`、每 rank 1 个 peer；world size 4 已覆盖轴 `[0, 1]`、每 rank 2 个 peer。更大 peer 拓扑必须由新探针证明，不能外推。
- 所有 NPU 验收均使用 `fallback_to_cpu=False`；NumPy 只能作为独立 oracle 或整理探针输入。
- 不假设 CANN 安装路径或版本；启动文档只使用平台实际提供的环境。
- 每个性能结论必须同时报告预热、重复次数、线路、qubit 数、world size、设备、峰值显存和原始样本，不用单次 wall-clock 结果下结论。
- 新增内部接口在对应设计文档批准前不得从 `aicir.distributed.__init__` 导出。

## Priority and Evidence

| 顺序 | 切片 | 本次证据给出的边界 |
| --- | --- | --- |
| P0 | 分布式参数移位 | 前向期望值已通过 2/4 NPU；梯度未验证，可先复用现有前向 API |
| P0 | 确定性分布式 RNG | sampling/collapse 正确性已过，但尚无跨 world size 可复现契约 |
| P1 | 通信重叠与 buffer 复用 | P2P peer/axis 路径已被调用；延迟、分配次数与重叠收益未测 |
| P1 | 性能与显存基准 | 正确性已过；没有单 NPU、复制模式、分片模式的对照数据 |
| P1 | 更大 world size 与状态压力 | 只有 world size 2/4 证据，不能推断 8 卡及更大状态 |
| P2 | 多节点 HCCL | 当前仅单节点；gather 由 all-gather 实现可能放大全网通信 |
| P2 | checkpoint、restart 与 collective failure | 当前任一 rank 失败依赖作业系统终止，没有恢复证据 |
| P3 | 原生自动微分 | 当前精确拒绝；必须先证明 NPU backward kernel 和 collective backward 安全 |

---

### Task 1: Freeze the Parameter-Shift Contract

**Files:**
- Create: `docs/superpowers/specs/2026-07-30-distributed-parameter-shift-design.md`
- Create: `tests/distributed/test_parameter_shift_contract.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: existing `DistSimulator.run(circuit, *, observables=..., layout=..., return_state=False, return_probabilities=False) -> DistResult`.
- Produces: an approved experimental contract for `scripts/npu/distributed_parameter_shift_probe.py`; no new public import.

- [ ] **Step 1: Write the design decision table**

Record these exact decisions in the design document:

```text
scope = expectation-value gradients for numeric RX/RY/RZ parameters
rule = 0.5 * (f(theta + pi/2) - f(theta - pi/2))
execution = every rank evaluates shifts in identical order
result owner = expectation scalar is identical on all ranks
initial modes = implicit zero, root statevector, root density matrix, DistState continuation
noise = excluded from the first gradient probe
shots = None
autograd = requires_grad inputs remain rejected
```

- [ ] **Step 2: Add a documentation contract test**

```python
from pathlib import Path


def test_parameter_shift_design_freezes_forward_contract():
    text = Path(
        "docs/superpowers/specs/"
        "2026-07-30-distributed-parameter-shift-design.md"
    ).read_text(encoding="utf-8")
    for phrase in (
        "0.5 * (f(theta + pi/2) - f(theta - pi/2))",
        "RX/RY/RZ",
        "requires_grad inputs remain rejected",
        "no new public import",
    ):
        assert phrase in text
```

- [ ] **Step 3: Run the contract test**

Run: `PYTHONPATH=. pytest tests/distributed/test_parameter_shift_contract.py -q`

Expected: PASS.

- [ ] **Step 4: Document the experimental boundary**

Add a “参数移位梯度” subsection to `aicir/distributed/README.md` stating that the probe uses repeated forward calls, does not enable PyTorch autograd, and does not imply native backward support.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-distributed-parameter-shift-design.md \
  tests/distributed/test_parameter_shift_contract.py \
  aicir/distributed/README.md
git commit -m "docs(distributed): freeze parameter shift contract"
```

### Task 2: Implement and Validate Distributed Parameter-Shift

**Files:**
- Create: `scripts/npu/distributed_parameter_shift_probe.py`
- Create: `tests/distributed/test_parameter_shift_multiprocess.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: the approved Task 1 contract and existing `DistSimulator.run`.
- Produces: probe-local `parameter_shift(simulator, circuit_factory, theta, observables, *, layout=None) -> dict[str, float]`; this function remains script-private.

- [ ] **Step 1: Write failing 2-rank Gloo tests**

Test RX, RY and RZ on both local and distributed layout axes. Compare the probe result with analytic derivatives using absolute tolerance `1e-5`; also assert the current `requires_grad=True` error remains exact.

Run: `PYTHONPATH=. pytest tests/distributed/test_parameter_shift_multiprocess.py -q`

Expected: FAIL because `distributed_parameter_shift_probe.py` does not exist.

- [ ] **Step 2: Implement the script-private evaluator**

Use the exact computation:

```python
def parameter_shift(
    simulator,
    circuit_factory,
    theta,
    observables,
    *,
    layout=None,
):
    values = []
    for shift in (np.pi / 2, -np.pi / 2):
        result = simulator.run(
            circuit_factory(float(theta + shift)),
            observables=observables,
            layout=layout,
            return_state=False,
            return_probabilities=False,
        )
        values.append(result.expectations)
    return {
        name: 0.5 * (values[0][name] - values[1][name])
        for name in observables
    }
```

The CLI must output one rank-0 JSON containing `passed`, `world_size`,
`fallback_to_cpu`, `max_gradient_error`, `local_axis_error`,
`distributed_axis_error`, and `autograd_rejection_exact`.

- [ ] **Step 3: Pass Gloo tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_parameter_shift_multiprocess.py tests/distributed/test_simulator_validation.py -q`

Expected: PASS.

- [ ] **Step 4: Run strict NPU acceptance**

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_parameter_shift_probe.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_parameter_shift_probe.py
```

Expected: both JSON reports have `passed=true`, `fallback_to_cpu=false`,
`max_gradient_error <= 1e-5`, and `autograd_rejection_exact=true`.

- [ ] **Step 5: Record evidence and commit**

Record both JSON summaries in the README without calling the result native automatic differentiation.

```bash
git add scripts/npu/distributed_parameter_shift_probe.py \
  tests/distributed/test_parameter_shift_multiprocess.py \
  aicir/distributed/README.md
git commit -m "test(distributed): validate parameter shift gradients"
```

### Task 3: Make Distributed Sampling Deterministic

**Files:**
- Create: `aicir/distributed/rng.py`
- Create: `tests/distributed/test_rng_multiprocess.py`
- Create: `scripts/npu/distributed_rng_probe.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: existing `DistSimulator.run(..., seed=..., shots=..., collapse=...)`.
- Produces: internal `_DistributedRNG(seed: int, operation_index: int)` with `uniform(shape, *, device, dtype) -> torch.Tensor`; public `run` signature remains unchanged.

- [ ] **Step 1: Write failing deterministic tests**

Assert:

```text
same seed + same circuit + same world size -> identical counts and collapse
different seed -> at least one differing 4096-shot count vector
same seed + world size 2/4 -> identical rank-0 counts
all ranks use the same operation-index stream
```

Run: `PYTHONPATH=. pytest tests/distributed/test_rng_multiprocess.py -q`

Expected: at least the cross-world-size test fails.

- [ ] **Step 2: Implement counter-derived random streams**

Derive each random stream from
`sha256(f"{seed}:{operation_index}:{sample_index}")`, broadcast only the
minimal seed/counter agreement data, and generate the rank-independent
measurement decision on rank 0. Do not use rank in the digest.

- [ ] **Step 3: Pass local multiprocess tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_rng_multiprocess.py tests/distributed/test_simulator_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 4: Validate on 2/4 NPU**

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_rng_probe.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_rng_probe.py
```

Expected: both reports contain the same SHA-256 digest for counts and
collapsed basis state, with `fallback_to_cpu=false`.

- [ ] **Step 5: Document and commit**

```bash
git add aicir/distributed/rng.py aicir/distributed/simulator.py \
  tests/distributed/test_rng_multiprocess.py \
  scripts/npu/distributed_rng_probe.py aicir/distributed/README.md
git commit -m "feat(distributed): make sampling deterministic"
```

### Task 4: Measure Then Optimize P2P Overlap and Buffer Reuse

**Files:**
- Create: `tests/distributed/test_communication_buffers.py`
- Create: `scripts/npu/distributed_communication_benchmark.py`
- Modify: `aicir/distributed/communication.py`
- Modify: `aicir/distributed/gates.py`
- Modify: `aicir/distributed/density.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: `_Communicator.exchange(tensor, *, peer, tag)` and `_GatePlan.partner_masks`.
- Produces: internal `_Communicator.exchange_into(tensor, receive, *, peer, tag) -> torch.Tensor` and `_ExchangeBufferPool.acquire(shape, dtype, device, peer) -> torch.Tensor`.
- Produces: benchmark CLI `distributed_communication_benchmark.py --mode {baseline,optimized} --n-qubits INT --depth INT --warmups INT --runs INT --output-json PATH`.
- Produces: rank-0 JSON with `mode`, `world_size`, `n_qubits`, `depth`, `warmup_runs`, `measured_runs`, `wall_time_samples_ms`, `peak_memory_bytes_per_rank`, `exchange_count_per_rank`, `allocated_receive_buffers_per_rank`, `numerical_digest`, and `fallback_to_cpu`.

- [ ] **Step 1: Add allocation and equivalence tests**

Verify `exchange_into` returns the supplied receive tensor, preserves complex64
real/imag paired tags on NPU transport, and reuses one buffer per
`(shape, dtype, device, peer)` key. Compare vector and density outputs with the
current implementation at `atol=1e-6`.

Run: `PYTHONPATH=. pytest tests/distributed/test_communication_buffers.py -q`

Expected: FAIL because the new internal methods are absent.

- [ ] **Step 2: Add a baseline benchmark before optimization**

The benchmark must emit JSON fields:

```text
mode, world_size, n_qubits, depth, warmup_runs, measured_runs,
wall_time_samples_ms, peak_memory_bytes_per_rank, exchange_count_per_rank,
allocated_receive_buffers_per_rank, fallback_to_cpu
```

Use 5 warmups and 30 measured runs for local-only, one distributed-axis gate,
and two-distributed-axis gate circuits. Its CLI is:

```text
--mode {baseline,optimized}
--n-qubits 20
--depth 50
--warmups 5
--runs 30
--output-json PATH
```

Before changing the kernels, run:

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_communication_benchmark.py \
  --mode baseline --n-qubits 20 --depth 50 --warmups 5 --runs 30 \
  --output-json /tmp/aicir-dist-comm-w2-baseline.json
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_communication_benchmark.py \
  --mode baseline --n-qubits 20 --depth 50 --warmups 5 --runs 30 \
  --output-json /tmp/aicir-dist-comm-w4-baseline.json
```

Expected: both files contain 30 timing samples for all three gate paths,
`fallback_to_cpu=false`, and the current allocation count.

- [ ] **Step 3: Implement buffer reuse without overlap**

Add `exchange_into` and `_ExchangeBufferPool`; first switch vector and density
kernels to reuse buffers while retaining the existing synchronous wait order.
Run:

`PYTHONPATH=. pytest tests/distributed/test_communication.py tests/distributed/test_communication_buffers.py tests/distributed/test_vector_kernel_multiprocess.py tests/distributed/test_density_kernel_multiprocess.py -q`

Expected: PASS with unchanged numerical results and lower recorded allocation count.

- [ ] **Step 4: Add measured overlap**

Split send/receive posting from completion inside `_Communicator`, post all
partner operations for one gate, compute the local-rank contribution, then
wait and accumulate peer contributions. Tests must inject delayed fake work
objects and prove local computation occurs before `wait()`.

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/test_communication.py \
  tests/distributed/test_communication_buffers.py \
  tests/distributed/test_vector_kernel_multiprocess.py \
  tests/distributed/test_density_kernel_multiprocess.py -q
```

Expected: PASS; the delayed fake-work trace records local computation before
the first `wait`, and vector/density errors remain at most `1e-6`.

- [ ] **Step 5: Capture optimized strict NPU samples**

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_communication_benchmark.py \
  --mode optimized --n-qubits 20 --depth 50 --warmups 5 --runs 30 \
  --output-json /tmp/aicir-dist-comm-w2-optimized.json
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_communication_benchmark.py \
  --mode optimized --n-qubits 20 --depth 50 --warmups 5 --runs 30 \
  --output-json /tmp/aicir-dist-comm-w4-optimized.json
```

Expected: both files contain 30 samples per gate path,
`fallback_to_cpu=false`, unchanged numerical digests, and an allocation count
not greater than the matching baseline.

- [ ] **Step 6: Compare baseline and optimized reports**

Report medians, p10/p90, peak memory, allocation counts and raw samples for
each matching world-size/gate-path pair. Accept the change only if numerical
tests pass and allocation count does not increase; do not require a fixed
speedup threshold.

- [ ] **Step 7: Commit**

```bash
git add aicir/distributed/communication.py aicir/distributed/gates.py \
  aicir/distributed/density.py \
  tests/distributed/test_communication_buffers.py \
  scripts/npu/distributed_communication_benchmark.py \
  aicir/distributed/README.md
git commit -m "perf(distributed): reuse and overlap P2P buffers"
```

### Task 5: Establish Single-NPU, Replicated, and Sharded Baselines

**Files:**
- Create: `scripts/npu/distributed_mode_benchmark.py`
- Create: `tests/distributed/test_mode_benchmark_contract.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: existing `NPUBackend`, `DistNPUBackend`, `DistSimulator`, and explicit root gather methods.
- Produces: a JSON-lines benchmark schema; no simulator API change.

- [ ] **Step 1: Freeze the benchmark matrix in a test**

Require vector and density runs for:

```text
single_npu: one process, one NPUBackend
replicated: N processes, each process owns a complete NPUBackend state
sharded: N processes, one DistSimulator state
world_sizes: single_npu=[1], replicated=[1,2,4], sharded=[1,2,4]
qubits: vector 16/20/24, density 8/10/12
depths: 10, 50
gather_modes: none, state, probabilities
warmups: 5
measured_runs: 30
```

The script must skip a case with `status="INSUFFICIENT_MEMORY"` after capturing
the exception type and device memory snapshot; it must not silently reduce
qubits.

Run: `PYTHONPATH=. pytest tests/distributed/test_mode_benchmark_contract.py -q`

Expected: FAIL because `distributed_mode_benchmark.py` does not exist.

- [ ] **Step 2: Implement the benchmark CLI and record schema**

The CLI must accept:

```text
--mode {single_npu,replicated,sharded}
--state-kind {vector,density}
--n-qubits INT
--depth INT
--gather-mode {none,state,probabilities}
--warmups 5
--runs 30
--seed 20260730
--output-json PATH
```

Every record must include hardware-visible device names, dtype, circuit digest,
seed, mode, world size, local/global state bytes, per-rank peak memory,
wall-time samples, gather mode and status. Implement
`validate_record(record: Mapping[str, object]) -> None` with these assertions:

```python
assert record["mode"] in {"single_npu", "replicated", "sharded"}
assert record["state_kind"] in {"vector", "density"}
assert record["gather_mode"] in {"none", "state", "probabilities"}
assert record["status"] in {"PASS", "INSUFFICIENT_MEMORY"}
assert record["warmup_runs"] == 5
assert record["measured_runs"] == 30
assert len(record["wall_time_samples_ms"]) == 30
assert len(record["peak_memory_bytes_per_rank"]) == record["world_size"]
assert record["local_state_bytes"] > 0
assert record["global_state_bytes"] > 0
assert record["seed"] == 20260730
```

For `INSUFFICIENT_MEMORY`, also require non-empty `exception_type` and
`device_memory_snapshot`; for `PASS`, require `fallback_to_cpu=false`.

Run: `PYTHONPATH=. pytest tests/distributed/test_mode_benchmark_contract.py -q`

Expected: PASS.

- [ ] **Step 3: Verify one CLI cell per mode**

```bash
PYTHONPATH=. python scripts/npu/distributed_mode_benchmark.py --help
PYTHONPATH=. pytest tests/distributed/test_mode_benchmark_contract.py -q
```

Expected: help lists all nine options and the contract tests pass.

- [ ] **Step 4: Execute each matrix cell independently**

Use one fresh `torchrun` job per matrix cell to avoid allocator history leaking
between modes. Run the complete frozen matrix:

```bash
bash -lc '
set -euo pipefail
for mode in single_npu replicated sharded; do
  for world_size in 1 2 4; do
    if [ "$mode" = single_npu ] && [ "$world_size" -ne 1 ]; then
      continue
    fi
    for state_spec in vector:16 vector:20 vector:24 density:8 density:10 density:12; do
      state_kind=${state_spec%%:*}
      n_qubits=${state_spec##*:}
      for depth in 10 50; do
        for gather_mode in none state probabilities; do
          PYTHONPATH=.:${PYTHONPATH:-} torchrun \
            --nproc-per-node="$world_size" \
            scripts/npu/distributed_mode_benchmark.py \
            --mode "$mode" --state-kind "$state_kind" \
            --n-qubits "$n_qubits" --depth "$depth" \
            --gather-mode "$gather_mode" \
            --warmups 5 --runs 30 --seed 20260730 \
            --output-json \
            "/tmp/aicir-mode-${mode}-${state_kind}-q${n_qubits}-d${depth}-g${gather_mode}-w${world_size}.json"
        done
      done
    done
  done
done
'
```

Expected: each output passes `validate_record`; retain raw JSON and summarize
only after all independent jobs finish.

- [ ] **Step 5: Publish bounded conclusions**

Document measured throughput, latency and peak-memory ratios for each completed
cell. Separate gather cost because the 2026-07-30 environment implements HCCL
gather via all-gather. Do not extrapolate beyond measured qubits/world sizes.

- [ ] **Step 6: Commit**

```bash
git add scripts/npu/distributed_mode_benchmark.py \
  tests/distributed/test_mode_benchmark_contract.py \
  aicir/distributed/README.md
git commit -m "bench(distributed): compare execution modes"
```

### Task 6: Stress Larger World Sizes and States

**Files:**
- Create: `scripts/npu/distributed_stress_probe.py`
- Create: `tests/distributed/test_stress_probe_contract.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: existing `DistSimulator.run` and `DistState.local_data`.
- Produces: sectioned stress JSON; no public API change.

- [ ] **Step 1: Define the exact stress grid**

```text
world_sizes = [2, 4, 8]
vector_qubits = [20, 24, 28, 30]
density_qubits = [8, 10, 12, 14]
depths = [10, 100]
layouts = [identity, reverse, auto]
gate_paths = [local_only, each_distributed_axis, all_distributed_axes]
```

The contract test must assert every grid cell is either `PASS`,
`INSUFFICIENT_MEMORY`, or `UNAVAILABLE_WORLD_SIZE`, with no missing cell.

- [ ] **Step 2: Implement per-rank invariants**

For each successful cell report state norm/trace, probability sum, local
element count, distributed axes, peer masks, peak memory and elapsed samples.
Use analytic GHZ/product-state references; never gather a state whose reported
global bytes exceed 25% of root device memory.

- [ ] **Step 3: Run Gloo shape tests**

Run: `PYTHONPATH=. pytest tests/distributed/test_stress_probe_contract.py tests/distributed/test_state_multiprocess.py -q`

Expected: PASS.

- [ ] **Step 4: Run available NPU grid**

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_stress_probe.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_stress_probe.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=8 \
  scripts/npu/distributed_stress_probe.py
```

Expected: every scheduled grid cell has `PASS`, `INSUFFICIENT_MEMORY`, or
`UNAVAILABLE_WORLD_SIZE`; successful cells have `fallback_to_cpu=false`.
An unavailable 8-NPU allocation is unavailable evidence, not a pass and not a
failure of 2/4 support.

- [ ] **Step 5: Document and commit**

```bash
git add scripts/npu/distributed_stress_probe.py \
  tests/distributed/test_stress_probe_contract.py \
  aicir/distributed/README.md
git commit -m "test(distributed): add state stress matrix"
```

### Task 7: Freeze and Validate Multi-Node HCCL Launch Contracts

**Files:**
- Create: `docs/superpowers/specs/2026-07-30-distributed-multinode-design.md`
- Create: `scripts/npu/distributed_multinode_probe.py`
- Create: `tests/distributed/test_multinode_contract.py`
- Modify: `aicir/distributed/backend.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: `DistNPUBackend.from_env(...)` and torchrun environment variables.
- Produces: internal `_validate_rank_topology(rank, local_rank, world_size, local_world_size, node_rank) -> None`; public constructor signature remains unchanged.

- [ ] **Step 1: Write topology rejection tests**

Cover duplicate global rank, out-of-range local rank, mismatched
`WORLD_SIZE`, inconsistent node count, and non-power-of-two global world size.
Every invalid case must fail before the first state collective with identical
error type/text on participating ranks.

Run: `PYTHONPATH=. pytest tests/distributed/test_multinode_contract.py -q`

Expected: FAIL because `_validate_rank_topology` and the probe are absent.

- [ ] **Step 2: Specify the launch contract**

The design must require `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, and `MASTER_PORT`; it must
describe global rank ownership and device binding without assuming a CANN path.

- [ ] **Step 3: Implement topology validation and probe**

The probe must report host identifier digest, global/local rank, node rank,
device, peer masks, cross-node peer count, distributed-axis P2P deltas,
gather/all-gather bytes and no-fallback status.

Run: `PYTHONPATH=. pytest tests/distributed/test_multinode_contract.py -q`

Expected: PASS; all invalid topologies fail before state collectives and valid
synthetic two-node topologies report unique global-rank ownership.

- [ ] **Step 4: Run a two-node launch**

On node rank 0:

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank=0 --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  scripts/npu/distributed_multinode_probe.py
```

On node rank 1, use the same address/port and `--node-rank=1`.
Expected: one rank-0 JSON with `world_size=8`, two host digests, cross-node P2P
evidence, `passed=true`, and `fallback_to_cpu=false`.

- [ ] **Step 5: Measure gather amplification**

Record bytes resident on every rank for explicit state/probability gather.
Because the observed torch_npu path maps gather to all-gather, distinguish API
semantics from physical memory/traffic in the report.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-distributed-multinode-design.md \
  scripts/npu/distributed_multinode_probe.py \
  tests/distributed/test_multinode_contract.py \
  aicir/distributed/backend.py aicir/distributed/README.md
git commit -m "feat(distributed): validate multi-node topology"
```

### Task 8: Add Checkpoint, Restart, and Collective Failure Protocols

**Files:**
- Create: `docs/superpowers/specs/2026-07-30-distributed-recovery-design.md`
- Create: `aicir/distributed/_checkpoint.py`
- Create: `aicir/distributed/_failure.py`
- Create: `tests/distributed/test_checkpoint_multiprocess.py`
- Create: `tests/distributed/test_collective_failure_multiprocess.py`
- Create: `scripts/npu/distributed_recovery_probe.py`
- Modify: `aicir/distributed/simulator.py`
- Modify: `aicir/distributed/README.md`

**Interfaces:**
- Consumes: `DistState`, `_ShardSpec`, `_Layout.digest()`, and communicator collectives.
- Produces: internal `_save_checkpoint(state, directory, *, step) -> pathlib.Path`, `_load_checkpoint(directory, *, backend, expected_layout) -> DistState`, and `_synchronize_failure(local_error, *, communicator, phase) -> None`.
- Produces: recovery CLI `distributed_recovery_probe.py --mode {save,resume} --checkpoint-dir PATH`.
- Produces: save JSON with `mode="save"`, manifest/shard digests, world size and `fallback_to_cpu`; resume JSON additionally contains `passed` and `uninterrupted_evolution_error`.

- [ ] **Step 1: Write failure-first tests**

Test vector and density checkpoints, non-identity layout, corrupt manifest,
missing rank shard, dtype mismatch, world-size mismatch, and one-rank injected
exception before P2P/all-reduce/gather. Each case must terminate all subprocesses
within 30 seconds and return one synchronized error summary.

- [ ] **Step 2: Freeze the checkpoint format**

The manifest must contain format version `1`, kind, dtype, n_qubits,
world_size, layout digest, global/local shapes, step, per-rank filenames,
byte sizes and SHA-256 digests. Write rank files to temporary names, fsync,
atomically rename them, barrier, then atomically publish the manifest.

- [ ] **Step 3: Implement exact-world-size restart**

Load only checkpoints whose world size and layout match the active backend.
Each rank validates its own digest, all ranks reduce validation status, and no
rank enters simulation until all shards pass.

- [ ] **Step 4: Implement collective failure synchronization**

Wrap simulation phases `preflight`, `initial_state`, `gate`, `noise`,
`reducer`, `sampling`, and `gather` with bounded UTF-8 failure payloads.
Use the first failing global rank as source and retain the existing 4096-byte
message bound.

- [ ] **Step 5: Validate process termination**

Run:

`PYTHONPATH=. pytest tests/distributed/test_checkpoint_multiprocess.py tests/distributed/test_collective_failure_multiprocess.py -q`

Expected: PASS; no child process remains after the 30-second deadline.

- [ ] **Step 6: Run the NPU recovery probe**

Run separate save and resume jobs so process-group destruction occurs between
commands:

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_recovery_probe.py \
  --mode save --checkpoint-dir /tmp/aicir-dist-recovery-w2
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_recovery_probe.py \
  --mode resume --checkpoint-dir /tmp/aicir-dist-recovery-w2
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_recovery_probe.py \
  --mode save --checkpoint-dir /tmp/aicir-dist-recovery-w4
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_recovery_probe.py \
  --mode resume --checkpoint-dir /tmp/aicir-dist-recovery-w4
```

Expected: both resume reports have `passed=true`, `fallback_to_cpu=false`,
matching manifest/shard digests, and uninterrupted-evolution error at most
`1e-6`.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-distributed-recovery-design.md \
  aicir/distributed/_checkpoint.py aicir/distributed/_failure.py \
  aicir/distributed/simulator.py \
  tests/distributed/test_checkpoint_multiprocess.py \
  tests/distributed/test_collective_failure_multiprocess.py \
  scripts/npu/distributed_recovery_probe.py aicir/distributed/README.md
git commit -m "feat(distributed): add checkpoint and failure protocols"
```

### Task 9: Gate Native Distributed Autograd

**Files:**
- Create: `docs/superpowers/specs/2026-07-30-distributed-native-autograd-gate.md`
- Create: `scripts/npu/distributed_backward_kernel_probe.py`
- Create: `tests/distributed/test_native_autograd_gate.py`
- Modify only after the gate passes: `docs/superpowers/plans/2026-07-30-distributed-native-autograd-implementation.md`

**Interfaces:**
- Consumes: current exact forward-only rejection, parameter-shift oracle from Task 2, and PyTorch/torch_npu backward behavior.
- Produces: a signed pass/fail evidence matrix and, only on pass, a separate implementation plan; it does not modify `DistSimulator.run`.
- Produces: probe CLI `distributed_backward_kernel_probe.py --iterations INT`; the gate commands use `--iterations 100`.
- Produces: rank-0 JSON containing per-primitive forward/backward evidence, per-rank retained-memory samples/growth, `memory_gate_passed`, `failed_primitives`, and `gate_status`.

- [ ] **Step 1: Keep the current rejection as a regression gate**

Run:

`PYTHONPATH=. pytest tests/distributed/test_simulator_validation.py tests/distributed/test_probe_contract.py -q`

Expected: all trainable inputs retain the exact forward-only error.

- [ ] **Step 2: Write backward-primitive gate tests**

Write cases for complex64 real/imag transport, local matrix application, reshape,
transpose, conjugation, complex construction, all-reduce, P2P send/receive,
and every density reducer operation under backward. Compare gradients with
float64 CPU finite differences and Task 2 parameter-shift at `atol=1e-4`.

Run: `PYTHONPATH=. pytest tests/distributed/test_native_autograd_gate.py -q`

Expected: FAIL because `distributed_backward_kernel_probe.py` and its evidence
schema are absent.

- [ ] **Step 3: Implement and validate the evidence schema**

Implement the Step 2 primitive operations outside `DistSimulator` and their
CPU finite-difference/parameter-shift comparisons.
The probe must emit per-primitive `forward_passed`, `backward_passed`,
`finite_gradients`, `gradient_digest`, `gradient_error`, and
`retained_memory_growth`; the top level must include `world_size`,
`fallback_to_cpu`, `gate_status`, and `failed_primitives`.

Run: `PYTHONPATH=. pytest tests/distributed/test_native_autograd_gate.py -q`

Expected: PASS for schema construction, exact blocked/pass decision logic, and
CPU reference calculations; this does not pass the NPU gate.

- [ ] **Step 4: Implement and test graph lifetime and memory**

Run 100 forward/backward iterations for vector and density cases. Synchronize
the allocator before sampling retained memory at iterations 20 and 100.
Compute
`retained_memory_growth = (bytes_at_100 - bytes_at_20) / bytes_at_20` and set
`memory_gate_passed=true` only when every rank reports growth at most `0.05`.
Include `iterations=100`, both retained-memory samples,
`retained_memory_growth`, and `memory_gate_passed` in the rank evidence.

Run: `PYTHONPATH=. pytest tests/distributed/test_native_autograd_gate.py -q`

Expected: PASS for the 100-iteration schedule, growth calculation, 5% boundary
cases and the rule that `memory_gate_passed=false` forces
`gate_status=BLOCKED`.

- [ ] **Step 5: Collect rank-synchronous NPU backward and memory evidence**

For world size 2 and 4, require every primitive to report
`forward_passed=true`, `backward_passed=true`, finite gradients,
rank-consistent SHA-256 gradient digests, no CPU fallback, and a passing
100-iteration memory gate. Any unsupported primitive or failed memory gate
keeps native autograd blocked.

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_backward_kernel_probe.py --iterations 100
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_backward_kernel_probe.py --iterations 100
```

Expected: each command emits one rank-0 JSON. `gate_status=PASS` is permitted
only when every primitive passes and `memory_gate_passed=true` on both world
sizes; otherwise the report is `gate_status=BLOCKED` with exact failing
primitives, errors, retained-memory samples and growth ratios.

- [ ] **Step 6: Decide the gate**

Write `gate_status: PASS` only if all primitive, 2/4 NPU, correctness and memory
conditions pass. Otherwise write `gate_status: BLOCKED` plus the exact failing
primitive and error. Parameter-shift remains the supported gradient path.

- [ ] **Step 7: Write the separate native-autograd plan only on PASS**

The generated plan must preserve the explicit distributed API, specify custom
autograd ownership for P2P/collectives, add gradcheck-style 2/4 rank tests, and
remove the forward-only rejection only for explicitly supported inputs. A
blocked gate must not create or modify that implementation plan.

- [ ] **Step 8: Commit evidence**

```bash
git add docs/superpowers/specs/2026-07-30-distributed-native-autograd-gate.md \
  scripts/npu/distributed_backward_kernel_probe.py \
  tests/distributed/test_native_autograd_gate.py
git commit -m "test(distributed): gate native autograd support"
```

### Task 10: Final Cross-Slice Verification

**Files:**
- Modify: `aicir/distributed/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: accepted outputs from Tasks 1–9.
- Produces: one evidence table distinguishing supported, experimental, measured, unavailable and blocked capabilities.

- [ ] **Step 1: Run the complete local regression**

```bash
PYTHONPATH=. pytest tests/distributed -q
PYTHONPATH=. pytest -q
git diff --check
```

Expected: all tests pass, platform-dependent tests skip with explicit reasons,
and the diff check emits no output.

- [ ] **Step 2: Re-run strict hardware probes**

Run the full API, parameter-shift, RNG, communication benchmark, mode
benchmark, stress, multi-node, recovery and backward-gate probes only on the
hardware configurations each probe declares. Preserve each rank-0 JSON and
command exit code.

- [ ] **Step 3: Audit claims**

Search:

```bash
rg -n "自动微分|autograd|加速|speedup|多节点|容错|CANN|gather" \
  aicir/distributed/README.md CHANGELOG.md \
  docs/superpowers/specs docs/superpowers/plans
```

Every claim must point to a completed probe. Native autograd remains
unsupported unless Task 9 records `gate_status: PASS` and the separate
implementation plan has subsequently been executed and verified.

- [ ] **Step 4: Commit final documentation**

```bash
git add aicir/distributed/README.md CHANGELOG.md
git commit -m "docs(distributed): summarize validated capabilities"
```

## Self-Review Checklist

- [ ] Every slice begins with a design or failing contract test before implementation.
- [ ] The roadmap does not claim native automatic differentiation is supported.
- [ ] The roadmap does not infer a CANN version or installation path.
- [ ] The roadmap records gather-to-all-gather as an observed warning, not a universal HCCL property.
- [ ] The roadmap makes no speedup, scale-out or fault-recovery claim without new measurements.
- [ ] World size 2/4 correctness evidence is separated from world size 8 and multi-node evidence.
- [ ] All new implementation interfaces remain internal until their design is approved.

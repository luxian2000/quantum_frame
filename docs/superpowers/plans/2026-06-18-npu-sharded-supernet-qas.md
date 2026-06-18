# NPU-sharded supernet QAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shard `supernet_qas`'s training, ranking, and finetune phases across multiple Ascend NPUs, with two training modes (`safe` = single-card-equivalent, `aggressive` = data-parallel), activating only on a distributed NPU run.

**Architecture:** A new pure-helper module `aicir/qas/primitives/sharding.py` provides the activation predicate and distributed collectives. `aicir/qas/algorithms/supernet.py` reads a `ShardContext` at the start of each phase and takes a sharded branch when active, otherwise runs today's exact code. `demos/BeH2/BeH2_npu.py` is switched from per-rank-seed best-of-N to same-seed in-search sharding with a `--mode` flag.

**Tech Stack:** Python, PyTorch (`torch.distributed`, gloo/hccl), aicir backends (`NPUBackend`, `GPUBackend`), pytest.

## Global Constraints

- Run everything from repo root with `PYTHONPATH=.` (no installed package).
- Comments/docstrings/READMEs are Chinese — match surrounding style.
- `torch` is optional in core but **required** for `aicir.qas`; tests that need it use `pytest.importorskip("torch")`.
- Conventions: state vector `(2^n, 1)`, complex dtype; NPU lacks complex kernels so **no complex-dtype collectives** — only real tensors and Python objects cross device boundaries.
- Activation predicate (verbatim): a run is sharded iff `isinstance(backend, NPUBackend)` **and** `torch.distributed` is initialized with `world_size > 1`. On a non-NPU machine `NPUBackend` falls back to CPU but is still an `NPUBackend` instance, so the sharded path is exercisable under a CPU gloo process group.
- Seed invariant: all ranks use the same `config.seed` ⇒ identical init weights + identical candidate sampling.
- Weight-sync invariant: all ranks hold identical shared weights at the start of every training step.
- Off the sharded path, every existing supernet test must stay green (no behavior change).
- Add a dated entry to `CHANGELOG.md` for the new interface.
- Two-word function names; never three-word `xxx_yyy_zzz`.

---

### Task 1: Sharding primitives module

**Files:**
- Create: `aicir/qas/primitives/sharding.py`
- Test: `tests/primitives/test_sharding.py`

**Interfaces:**
- Consumes: `aicir.backends.npu_backend.NPUBackend`; `torch.distributed`.
- Produces:
  - `@dataclass(frozen=True) class ShardContext: is_sharded: bool; rank: int; world_size: int`
  - `shard_context(backend) -> ShardContext`
  - `owned_indices(n: int, rank: int, world_size: int) -> list[int]`
  - `all_gather(obj) -> list[Any]` (rank order; `[obj]` when not sharded)
  - `all_reduce_mean(tensors: Sequence[torch.Tensor]) -> None` (in place; no-op when not sharded)
  - `broadcast_parameters(params: Mapping[Any, torch.Tensor], src: int = 0) -> None` (no-op when not sharded)

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# tests/primitives/test_sharding.py
import pytest

torch = pytest.importorskip("torch")

from aicir.qas.primitives.sharding import (
    ShardContext,
    owned_indices,
    shard_context,
    all_gather,
)
from aicir.backends.npu_backend import NPUBackend
from aicir.backends.gpu_backend import GPUBackend


@pytest.mark.parametrize("n,world", [(10, 4), (8, 2), (5, 3), (3, 4)])
def test_owned_indices_partition_covers_range_once(n, world):
    seen = []
    for rank in range(world):
        seen.extend(owned_indices(n, rank, world))
    assert sorted(seen) == list(range(n))


def test_owned_indices_is_strided():
    assert owned_indices(10, 1, 4) == [1, 5, 9]


def test_shard_context_off_distributed_is_inactive():
    # No process group initialized in this test process.
    ctx = shard_context(NPUBackend(device="npu:0"))
    assert ctx == ShardContext(is_sharded=False, rank=0, world_size=1)


def test_shard_context_non_npu_backend_is_inactive():
    ctx = shard_context(GPUBackend(device="cpu"))
    assert ctx.is_sharded is False


def test_all_gather_without_process_group_returns_singleton():
    assert all_gather({"x": 1}) == [{"x": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/primitives/test_sharding.py -q`
Expected: FAIL with `ModuleNotFoundError: aicir.qas.primitives.sharding`.

- [ ] **Step 3: Implement the module**

```python
# aicir/qas/primitives/sharding.py
"""多 NPU 任务并行的分片原语。

这些工具只在“分布式 NPU 运行”下生效：backend 是 NPUBackend 且 torch.distributed
已初始化、world_size > 1。其它情况下全部退化为本地无操作，调用方无需分支判断
（只读 shard_context 的结果即可）。这里只搬运实数张量和 Python 对象，不做任何
复数 dtype 的集合通信（Ascend NPU 没有复数算子）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ...backends.npu_backend import NPUBackend


@dataclass(frozen=True)
class ShardContext:
    is_sharded: bool
    rank: int
    world_size: int


def _dist_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def shard_context(backend: Any) -> ShardContext:
    """判定当前是否为分布式 NPU 运行，并返回 rank/world_size。"""
    if not isinstance(backend, NPUBackend) or not _dist_ready():
        return ShardContext(is_sharded=False, rank=0, world_size=1)
    world_size = dist.get_world_size()
    if world_size <= 1:
        return ShardContext(is_sharded=False, rank=0, world_size=1)
    return ShardContext(is_sharded=True, rank=dist.get_rank(), world_size=world_size)


def owned_indices(n: int, rank: int, world_size: int) -> list[int]:
    """跨 rank 的等距切分；各 rank 的并集恰好覆盖 range(n) 一次。"""
    return list(range(rank, n, world_size))


def all_gather(obj: Any) -> list[Any]:
    """按 rank 顺序收集每个 rank 的 Python 对象；未分布式时返回 [obj]。"""
    if not _dist_ready():
        return [obj]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def all_reduce_mean(tensors: Sequence[torch.Tensor]) -> None:
    """对一组实数张量做跨 rank 求和再除以 world_size（原地）。未分布式时不操作。"""
    if not _dist_ready():
        return
    world_size = dist.get_world_size()
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)


def broadcast_parameters(params: Mapping[Any, torch.Tensor], src: int = 0) -> None:
    """把 src rank 的实数参数张量广播到所有 rank（原地）。未分布式时不操作。"""
    if not _dist_ready():
        return
    for _, tensor in sorted(params.items(), key=lambda item: str(item[0])):
        dist.broadcast(tensor, src=src)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/primitives/test_sharding.py -q`
Expected: PASS (5 tests / parametrized).

- [ ] **Step 5: Commit**

```bash
git add aicir/qas/primitives/sharding.py tests/primitives/test_sharding.py
git commit -m "feat(qas): add NPU sharding primitives

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `shard_mode` config field + `mode` kwarg on `supernet_qas`

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py` (`SupernetConfig` ~line 137, `_validate_config` ~line 280, `supernet_qas` ~line 1455)
- Test: `tests/test_supernet_sharding.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SupernetConfig.shard_mode: str = "safe"`; `supernet_qas(..., mode: str = "safe")` forwarding to `shard_mode`. Accepted values `{"safe", "aggressive"}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_supernet_sharding.py
import pytest

pytest.importorskip("torch")

from aicir.qas.algorithms.supernet import SupernetConfig, Supernet, supernet_qas
from aicir.operators import Hamiltonian


def test_shard_mode_defaults_to_safe():
    assert SupernetConfig().shard_mode == "safe"


def test_shard_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        Supernet(SupernetConfig(shard_mode="turbo"))


def test_supernet_qas_forwards_mode():
    ham = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.2)])
    # Run a tiny search; just assert it accepts mode and returns a result.
    result = supernet_qas(
        ham, layers=1, supernet_num=1, supernet_steps=1,
        finetune_steps=1, ranking_num=1, seed=1, mode="aggressive",
    )
    assert result.best_circuit is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_supernet_sharding.py -q`
Expected: FAIL (`shard_mode` not a field / `mode` unexpected kwarg).

- [ ] **Step 3: Add the config field**

In `SupernetConfig` (after `noise_mode` / existing string fields), add:

```python
    shard_mode: str = "safe"
```

In `_validate_config` (alongside the existing `ranking_strategy`/`noise_mode` checks), add:

```python
        shard_mode = str(cfg.shard_mode).strip().lower()
        if shard_mode not in {"safe", "aggressive"}:
            raise ValueError("shard_mode must be 'safe' or 'aggressive'")
```

In `supernet_qas`, add `mode: str = "safe"` to the keyword-only params (next to `use_parameter_shift`) and, just before constructing `SupernetConfig`, inject it (config_overrides must not also carry it):

```python
    config_overrides.setdefault("shard_mode", mode)
```

Add a one-line Chinese docstring note for `mode` in the Args block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_supernet_sharding.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aicir/qas/algorithms/supernet.py tests/test_supernet_sharding.py
git commit -m "feat(qas): add shard_mode config and supernet_qas mode kwarg

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Shard the ranking phase

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py` (`rank_architectures` ~line 946)
- Test: `tests/test_supernet_sharding.py` (add a unit test on candidate ownership; full equivalence is covered by the gloo test in Task 8)

**Interfaces:**
- Consumes: `shard_context`, `owned_indices`, `all_gather` from `aicir.qas.primitives.sharding`.
- Produces: `rank_architectures` returns the identical merged+sorted `list[dict]` on every rank.

- [ ] **Step 1: Write a failing test (non-distributed must be unchanged)**

```python
# add to tests/test_supernet_sharding.py
from aicir.qas.algorithms.supernet import Supernet, SupernetConfig


def test_rank_architectures_single_process_unchanged():
    cfg = SupernetConfig(n_qubits=2, layers=1, supernet_num=1,
                         supernet_steps=1, ranking_num=4, finetune_steps=0,
                         two_qubit_pairs=((0, 1),), task="vqe", seed=3)
    trainer = Supernet(cfg)
    from aicir.operators import Hamiltonian
    ham = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.2)])
    records = trainer.rank_architectures("vqe", hamiltonian=ham, split="train")
    assert len(records) == 4
    assert [r["rank"] for r in records] == [1, 2, 3, 4]
    assert records == sorted(records, key=lambda r: r["score"])
```

- [ ] **Step 2: Run to verify it passes already (guard test), then add sharded branch**

Run: `PYTHONPATH=. pytest tests/test_supernet_sharding.py::test_rank_architectures_single_process_unchanged -q`
Expected: PASS (this guards against regressions; the change below must keep it green).

- [ ] **Step 3: Add the sharded branch**

At the top of `aicir/qas/algorithms/supernet.py` imports add:

```python
from ..primitives.sharding import (
    shard_context,
    owned_indices,
    all_gather,
    all_reduce_mean,
    broadcast_parameters,
)
```

Replace the candidate-evaluation loop in `rank_architectures` (the
`for architecture in candidates:` block that appends to `records`) with an
ownership-aware version:

```python
        ctx = shard_context(self.backend)
        owned = (
            set(owned_indices(len(candidates), ctx.rank, ctx.world_size))
            if ctx.is_sharded
            else set(range(len(candidates)))
        )

        local_records: list[dict[str, Any]] = []
        for index, architecture in enumerate(candidates):
            if index not in owned:
                continue
            selected_id, losses = self.select_supernet(
                architecture, objective, dataset, hamiltonian, split=split,
            )
            local_records.append(
                {
                    "candidate_index": index,
                    "architecture": architecture,
                    "architecture_indices": self.encode_architecture(architecture),
                    "selected_supernet_id": selected_id,
                    "score": losses[selected_id],
                    "candidate_losses": losses,
                    "cnot_count": self.cnot_count(architecture),
                    "two_qubit_count": self.two_qubit_count(architecture),
                }
            )

        if ctx.is_sharded:
            merged: list[dict[str, Any]] = []
            for part in all_gather(local_records):
                merged.extend(part)
            records = merged
        else:
            records = local_records
        records.sort(key=lambda item: (item["score"], item["candidate_index"]))
```

Keep the existing `for rank, record in enumerate(records, start=1): record["rank"] = rank` tail. (Sorting by `(score, candidate_index)` makes the order deterministic and tie-stable across ranks.)

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. pytest tests/test_supernet_sharding.py -q && PYTHONPATH=. pytest tests/test_vqa_qas.py -q`
Expected: PASS (single-process behavior preserved; `rank` values unchanged).

- [ ] **Step 5: Commit**

```bash
git add aicir/qas/algorithms/supernet.py tests/test_supernet_sharding.py
git commit -m "feat(qas): shard ranking candidate evaluation across NPUs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Shard training — `safe` mode

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py` (`optimize_supernet` ~line 879)

**Interfaces:**
- Consumes: `shard_context`, `owned_indices`, `all_gather`, `broadcast_parameters`.
- Produces: no signature change; `optimize_supernet` internally takes a sharded `safe` branch when `ctx.is_sharded and shard_mode == "safe"`.

- [ ] **Step 1: Add a sharded `select_supernet` helper**

Add a method that evaluates only owned supernet ids and all-gathers the full loss vector:

```python
    def _sharded_select(self, architecture, objective, dataset, hamiltonian, split, ctx):
        owned = set(owned_indices(self.config.supernet_num, ctx.rank, ctx.world_size))
        local = []
        with torch.no_grad():
            for supernet_id in range(self.config.supernet_num):
                if supernet_id not in owned:
                    continue
                loss, _, _, _ = self._loss(
                    architecture, supernet_id, objective, dataset, hamiltonian, split=split,
                )
                local.append((supernet_id, _float_value(loss)))
        losses = [math.inf] * self.config.supernet_num
        for part in all_gather(local):
            for supernet_id, value in part:
                losses[supernet_id] = value
        selected = min(range(len(losses)), key=losses.__getitem__)
        return selected, losses
```

- [ ] **Step 2: Branch inside `optimize_supernet`**

At the start of `optimize_supernet`, read the context:

```python
        ctx = shard_context(self.backend)
        safe_sharded = ctx.is_sharded and self.config.shard_mode == "safe"
```

Inside the step loop, replace the `selected_id, candidate_losses = self.select_supernet(...)` call with:

```python
            if safe_sharded:
                selected_id, candidate_losses = self._sharded_select(
                    architecture, objective, dataset, hamiltonian, "train", ctx
                )
            else:
                selected_id, candidate_losses = self.select_supernet(
                    architecture, objective, dataset, hamiltonian, split="train",
                )
```

After the existing `optimizer.step()` / parameter-shift update block, in safe-sharded
runs only rank 0 should have performed the grad step; gate it and broadcast:

```python
            if safe_sharded:
                # Only rank 0 owns the single grad step; broadcast to resync.
                broadcast_parameters(self.shared_parameters, src=0)
```

and wrap the grad-step block so non-zero ranks skip it under safe sharding:

```python
            if safe_sharded and ctx.rank != 0:
                grad_norm = 0.0
            elif self.config.use_parameter_shift:
                grad_norm = self._parameter_shift_update(active_tensors, optimizer, loss_closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                grad_norm = self._grad_norm(active_tensors)
                optimizer.step()
```

(The `loss, _, active_keys, active_tensors = self._loss(...)` call stays on all ranks so logging fields are populated; only the optimizer mutation is rank-0-only. Broadcast then makes all ranks identical.)

- [ ] **Step 3: Run single-process regression**

Run: `PYTHONPATH=. pytest tests/test_vqa_qas.py tests/test_supernet_sharding.py -q`
Expected: PASS (safe_sharded is False in-process, so the path is byte-identical to before).

- [ ] **Step 4: Commit**

```bash
git add aicir/qas/algorithms/supernet.py
git commit -m "feat(qas): safe-mode sharded supernet training (W-eval split + rank0 step + broadcast)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Shard training — `aggressive` mode

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py` (`optimize_supernet`)

**Interfaces:**
- Consumes: `all_reduce_mean`, `all_gather`.
- Produces: aggressive branch in `optimize_supernet`. Effective batch = `world_size` architectures/step.

- [ ] **Step 1: Add an aggressive step helper**

```python
    def _aggressive_step(self, objective, dataset, hamiltonian, ctx):
        # Sample world_size architectures identically on all ranks; this rank owns arch[rank].
        archs = [self.sample_architecture() for _ in range(ctx.world_size)]
        architecture = archs[ctx.rank]
        selected_id, candidate_losses = self.select_supernet(
            architecture, objective, dataset, hamiltonian, split="train",
        )
        loss, _, active_keys, active_tensors = self._loss(
            architecture, selected_id, objective, dataset, hamiltonian, split="train",
        )
        # Zero all shared grads, backprop this rank's loss.
        for param in self.shared_parameters.values():
            param.grad = None
        if loss.requires_grad:
            loss.backward()
        ordered_keys = sorted(self.shared_parameters.keys(), key=str)
        grads = [
            (self.shared_parameters[k].grad
             if self.shared_parameters[k].grad is not None
             else torch.zeros_like(self.shared_parameters[k]))
            for k in ordered_keys
        ]
        all_reduce_mean(grads)
        for key, grad in zip(ordered_keys, grads):
            self.shared_parameters[key].grad = grad
        # Step every optimizer whose supernet was selected by some rank.
        selected_ids = set()
        for ids in all_gather([selected_id]):
            selected_ids.update(ids)
        for supernet_id in sorted(selected_ids):
            self._optimizers[supernet_id].step()
        return architecture, selected_id, candidate_losses, _float_value(loss), active_keys
```

- [ ] **Step 2: Branch inside `optimize_supernet`**

Add near the top:

```python
        aggressive_sharded = ctx.is_sharded and self.config.shard_mode == "aggressive"
```

At the top of the step loop, when `aggressive_sharded`, take the dedicated path and `continue` the logging from its returns instead of the sequential body:

```python
            if aggressive_sharded:
                architecture, selected_id, candidate_losses, loss_float, active_keys = (
                    self._aggressive_step(objective, dataset, hamiltonian, ctx)
                )
                grad_norm = 0.0
                log.append({
                    "step": step, "architecture": architecture,
                    "architecture_indices": self.encode_architecture(architecture),
                    "selected_supernet_id": selected_id,
                    "candidate_losses": candidate_losses, "loss": loss_float,
                    "gradient_norm": grad_norm,
                    "active_parameter_count": len(active_keys),
                    "cnot_count": self.cnot_count(architecture),
                    "two_qubit_count": self.two_qubit_count(architecture),
                })
                continue
```

(Place this `if` immediately after `architecture = self.sample_architecture()` is *replaced*: in the aggressive branch the helper samples `world_size` archs itself, so guard the original `architecture = self.sample_architecture()` line to run only when not aggressive.)

- [ ] **Step 3: Run single-process regression**

Run: `PYTHONPATH=. pytest tests/test_vqa_qas.py tests/test_supernet_sharding.py -q`
Expected: PASS (aggressive_sharded False in-process).

- [ ] **Step 4: Commit**

```bash
git add aicir/qas/algorithms/supernet.py
git commit -m "feat(qas): aggressive-mode data-parallel supernet training (all_reduce grads)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Top-K parallel finetune in `train`

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py` (`train` ~line 1198-1244)

**Interfaces:**
- Consumes: `shard_context`, `all_gather`; `Circuit` (already imported).
- Produces: on a sharded run, `train` finetunes the top-`world_size` ranked architectures (one per rank), all-gathers, and returns a `SupernetResult` reflecting the global best on every rank. Single-card unchanged (top-1).

- [ ] **Step 1: Replace the finetune/selection block**

After `ranking_records = self.rank_architectures(...)`, replace the
`best_record = ranking_records[0]` … `finetune_architecture(...)` block with:

```python
        ctx = shard_context(self.backend)
        if ctx.is_sharded:
            cand_index = ctx.rank
        else:
            cand_index = 0

        if cand_index < len(ranking_records):
            local_record = ranking_records[cand_index]
            local_arch = local_record["architecture"]
            local_supernet_id = int(local_record["selected_supernet_id"])
            local_circuit, local_params, local_finetune_log, local_score = (
                self.finetune_architecture(
                    local_arch, local_supernet_id, objective,
                    dataset=dataset, hamiltonian=hamiltonian,
                )
            )
            local_payload = {
                "score": float(local_score),
                "supernet_id": local_supernet_id,
                "ranking_index": cand_index,
                "n_qubits": int(local_circuit.n_qubits),
                "gates": list(local_circuit.gates),
                "numeric_parameters": {str(k): float(v) for k, v in local_params.items()},
            }
        else:
            local_payload = None  # more ranks than candidates

        if ctx.is_sharded:
            payloads = [p for p in all_gather(local_payload) if p is not None]
            best_payload = min(payloads, key=lambda p: p["score"])
        else:
            best_payload = local_payload

        best_record = ranking_records[best_payload["ranking_index"]]
        best_architecture = best_record["architecture"]
        selected_supernet_id = int(best_payload["supernet_id"])
        best_circuit = Circuit(
            *best_payload["gates"], n_qubits=best_payload["n_qubits"], backend=self.backend,
        )
        finetune_parameters = best_payload["numeric_parameters"]
        # finetune_log reflects the local run; on the winning rank it is the real log,
        # on others it is the local rank's log (kept for diagnostics only).
        finetune_log = local_finetune_log if (best_payload["ranking_index"] == cand_index) else []
        finetune_score = best_payload["score"]
```

Then keep the existing `final_metrics = self._final_metrics(...)` call and the
rest of `train` unchanged. Note `_final_metrics` accepts
`Mapping[ParameterKey, torch.Tensor | float]`, and the reconstructed
`finetune_parameters` is `dict[str, float]` keyed by `str(key)`; verify
`_final_metrics` only reads values (energies) and does not require the original
`ParameterKey` objects. If it does index by `ParameterKey`, instead carry the
parameters as `numeric_parameters` already consumed by `build_circuit` inside
`finetune_architecture`, and pass the already-built `best_circuit` to
`_final_metrics` (which recomputes energy from the circuit, not the param keys).

- [ ] **Step 2: Verify `_final_metrics` parameter usage**

Run: `PYTHONPATH=. python -c "import inspect,aicir.qas.algorithms.supernet as s; print(inspect.getsource(s.Supernet._final_metrics))"`
Expected: confirm whether it indexes `finetune_parameters` by key or only computes from `circuit`. Adjust Step 1's parameter handling accordingly (use circuit-derived energy if keys are needed).

- [ ] **Step 3: Run single-process regression**

Run: `PYTHONPATH=. pytest tests/test_vqa_qas.py tests/test_supernet_sharding.py -q`
Expected: PASS (cand_index=0, single top-1 finetune; identical to before).

- [ ] **Step 4: Commit**

```bash
git add aicir/qas/algorithms/supernet.py
git commit -m "feat(qas): top-K parallel finetune across NPUs with global-best selection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `BeH2_npu.py` — `--mode` + same-seed sharding

**Files:**
- Modify: `demos/BeH2/BeH2_npu.py`

**Interfaces:**
- Consumes: `supernet_qas(..., mode=...)`.
- Produces: a `torchrun`-launched single in-search run; rank 0 writes `output.txt`, `BeH2_npu_cir.qasm`, `BeH2_npu_cir.py`.

- [ ] **Step 1: Switch from best-of-N to single sharded search**

Change the search so all ranks use the **same seed** and pass `mode`. Replace the
per-rank seed (`args.seed + rank`) with `args.seed`, add a `--mode` arg, and run
once (the trainer shards internally). Key edits:

```python
    parser.add_argument("--mode", choices=("safe", "aggressive"), default="safe",
                        help="训练分片模式：safe=与单卡数值等价；aggressive=数据并行(~world_size 倍，动态不同)。")
```

```python
    kwargs.update({
        "layers": args.layers,
        "supernet_num": args.supernet_num,
        "supernet_steps": args.supernet_steps,
        "ranking_num": args.ranking_num,
        "finetune_steps": args.finetune_steps,
        "seed": args.seed,        # 同一 seed，保证各 rank 权重/候选一致
        "device": device,
        "mode": args.mode,
    })
```

Since the search is now sharded internally, `result` is identical on all ranks
(global best). Rank 0 writes the report and circuit directly from `result`
(drop the per-rank energy gather; `result.final_metrics` already holds the
global-best energy). Update `_write_report` to summarize the single sharded run
(mode, world_size, fine-tuned vs baseline energy, CNOT/2q, wall time).

- [ ] **Step 2: CPU smoke (single process; sharded path inactive)**

Run: `PYTHONPATH=. python demos/BeH2/BeH2_npu.py --allow-cpu-fallback --layers 1 --supernet-num 1 --supernet-steps 1 --ranking-num 1 --finetune-steps 1 --mode safe`
Expected: completes on a machine with enough RAM, writing `output.txt` and circuit files. (On RAM-limited hosts the 16-qubit sim may OOM; verify the argument plumbing with `--help` if so: `PYTHONPATH=. python demos/BeH2/BeH2_npu.py --help` shows `--mode {safe,aggressive}`.)

- [ ] **Step 3: Commit**

```bash
git add demos/BeH2/BeH2_npu.py
git commit -m "feat(demo): BeH2_npu single sharded search with --mode safe/aggressive

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: CPU-gloo reproducibility test (safe == single-card; aggressive converges)

**Files:**
- Test: `tests/test_supernet_sharding_dist.py`

**Interfaces:**
- Consumes: `Supernet`, `SupernetConfig`, `supernet_qas`; `torch.distributed` (gloo); `torch.multiprocessing.spawn`.

- [ ] **Step 1: Write the distributed test**

```python
# tests/test_supernet_sharding_dist.py
import os
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from aicir.operators import Hamiltonian
from aicir.qas.algorithms.supernet import supernet_qas


HAM_TERMS = [("ZZ", -1.0), ("XI", 0.2), ("IX", 0.2)]
COMMON = dict(layers=1, supernet_num=2, supernet_steps=4,
              finetune_steps=3, ranking_num=4, seed=5, device="npu:0")


def _single_card_energy():
    ham = Hamiltonian(n_qubits=2, terms=HAM_TERMS)
    res = supernet_qas(ham, mode="safe", **COMMON)
    return float(res.final_metrics["fine_tuned_energy"])


def _worker(rank, world_size, mode, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ham = Hamiltonian(n_qubits=2, terms=HAM_TERMS)
        res = supernet_qas(ham, mode=mode, **COMMON)
        if rank == 0:
            return_dict["energy"] = float(res.final_metrics["fine_tuned_energy"])
    finally:
        dist.destroy_process_group()


def _run(mode):
    mgr = mp.Manager()
    return_dict = mgr.dict()
    mp.spawn(_worker, args=(2, mode, return_dict), nprocs=2, join=True)
    return return_dict["energy"]


def test_safe_mode_matches_single_card():
    single = _single_card_energy()
    sharded = _run("safe")
    assert sharded == pytest.approx(single, abs=1e-4)


def test_aggressive_mode_runs_and_is_finite():
    energy = _run("aggressive")
    assert energy == energy  # not NaN
    assert energy < 1.0      # below the trivially-bad regime for this tiny H
```

Note: `_worker` signature must match `mp.spawn` (it injects `rank` first); the
`args=(2, mode, return_dict)` tuple supplies `world_size, mode, return_dict`.

- [ ] **Step 2: Run the distributed test**

Run: `PYTHONPATH=. pytest tests/test_supernet_sharding_dist.py -q`
Expected: PASS. `safe` reproduces single-card energy within 1e-4; `aggressive` returns a finite energy.

- [ ] **Step 3: Run the full QAS suite for regressions**

Run: `PYTHONPATH=. pytest tests/test_vqa_qas.py tests/test_qas_runner.py tests/test_supernet_sharding.py tests/primitives/test_sharding.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_supernet_sharding_dist.py
git commit -m "test(qas): CPU-gloo reproducibility for safe/aggressive sharded supernet

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Documentation — `aicir/qas/README.md` + `CHANGELOG.md`

**Files:**
- Modify: `aicir/qas/README.md` (supernet section, e.g. after §3.7 multi-NPU sharding)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a README subsection (Chinese)**

Add a `### 3.8 supernet 单次搜索的多 NPU 分片（safe / aggressive）` subsection documenting:
- 触发条件：backend 为 `NPUBackend` 且 `torch.distributed` 已初始化、`world_size > 1`；否则与单卡完全一致。
- 三个阶段的分片方式：ranking（候选等距切分，结果与单卡一致）、training（`safe`=切分 W 次 select 评估+rank0 梯度步+广播，数值等价；`aggressive`=每步 `world_size` 个架构数据并行，约 `world_size` 倍提速、动态不同、可相应调小 `supernet_steps`）、finetune（top-`world_size` 并行微调取全局最优）。
- `seed` 必须各 rank 相同。
- 入口：`supernet_qas(..., device="npu:0", mode="safe"|"aggressive")`，以及 `torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py --mode safe`。

- [ ] **Step 2: Add a CHANGELOG entry**

Add under a new dated heading:

```markdown
## 2026-06-18

- `aicir.qas` supernet 支持单次搜索的多 NPU 分片：新增 `SupernetConfig.shard_mode`
  与 `supernet_qas(..., mode="safe"|"aggressive")`。仅在分布式 NPU 运行下生效，
  分片 training / ranking / finetune 三个阶段。`demos/BeH2/BeH2_npu.py` 改为
  同种子单次分片搜索并新增 `--mode`。
```

- [ ] **Step 3: Verify docs render / no broken references**

Run: `PYTHONPATH=. python -c "import aicir.qas; print('ok')"`
Expected: `ok` (imports still valid).

- [ ] **Step 4: Commit**

```bash
git add aicir/qas/README.md CHANGELOG.md
git commit -m "docs(qas): document multi-NPU sharded supernet search (safe/aggressive)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- Read `aicir/qas/README.md` §3 and the spec at
  `docs/superpowers/specs/2026-06-18-npu-sharded-supernet-qas-design.md` before
  starting.
- The single most important invariant is **off-the-sharded-path behavior must
  not change**. Every task includes a single-process regression run for that
  reason. If any existing test changes output, stop and reconcile.
- `select_supernet`, `_loss`, `build_circuit`, `finetune_architecture`,
  `_final_metrics` are the only trainer methods you touch indirectly — do not
  change their signatures.

# VQE Loop Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `aicir.qas.vqe_loop` read as a small P0/P1/fair-label framework with explicit ansatz-family boundaries and no misleading legacy closed-loop naming.

**Architecture:** Keep P0 bootstrap generation, P1 mutation/selection, and fair labeling as separate lanes joined only by benchmark-table rows. Shared helpers stay unprefixed (`benchmark_table.py`, `fair_vqe.py`, `training_free.py`) because they are used by both lanes. Optional experiments and diagnostics stay outside the core path.

**Tech Stack:** Python stdlib, `unittest`, existing aicir QAS ansatz genes, CSV benchmark table schema, COBYLA fair VQE protocol.

---

## Target File Structure

Core P0:

- `aicir/qas/vqe_loop/p0_bootstrap_fair.py`: P0 bootstrap + fair-label one-shot runner.
- `aicir/qas/vqe_loop/p0_chemistry_excitation.py`: chemistry excitation P0 candidate rows and reusable excitation pools.
- `aicir/qas/vqe_loop/p0_supernet_native.py`: native supernet P0 candidate rows and native E5 evaluator.

Core P1:

- `aicir/qas/vqe_loop/p1_round.py`: P1 round planner and output writer.
- `aicir/qas/vqe_loop/p1_evolution.py`: parent selection, mutation, crossover, operator growth, child rows.
- `aicir/qas/vqe_loop/p1_selection.py`: queue rows, fallback scoring, baselines.

Fair/shared:

- `aicir/qas/vqe_loop/fair_labeling.py`: queue rows to fair labels.
- `aicir/qas/vqe_loop/fair_vqe.py`: COBYLA VQE optimizer and energy evaluation.
- `default`: frozen fair-label protocol data.
- `aicir/qas/vqe_loop/benchmark_table.py`: benchmark schema, CSV/row parsing, built-in fair protocol defaults, and row-level P1 policies.
- `aicir/qas/vqe_loop/training_free.py`: shared zero-cost annotations for P0/P1.
- `aicir/qas/vqe_loop/sharding.py`: fair-label queue sharding.

Optional:

- `aicir/qas/vqe_loop/cheap_eval_experiment.py`: E1/E2/E5/fair diagnostic harness.
- `aicir/qas/vqe_loop/task_proxy.py`: B1 task-aware proxy.
- `aicir/qas/vqe_loop/graph_predictor.py`: B2 feature predictor.
- `aicir/qas/vqe_loop/p0_problem_aware.py`: diagnostic problem-aware supernet sampler.

---

### Task 1: Lock New Module Names With Import Tests

**Files:**
- Modify: `tests/test_vqe_loop_unified_interface.py`
- Modify: `tests/test_labeling_seed.py`
- Modify: `tests/test_chemistry_excitation_ansatz.py`
- Modify: `tests/test_supernet_native_e5.py`

- [ ] **Step 1: Write failing imports for the canonical names**

In `tests/test_vqe_loop_unified_interface.py`, ensure the public package exposes P0 names:

```python
from aicir.qas.vqe_loop import (
    P0BootstrapConfig,
    P0BootstrapResult,
    run_p0_bootstrap_fair,
)

assert P0BootstrapConfig is not None
assert P0BootstrapResult is not None
assert run_p0_bootstrap_fair is not None
```

In `tests/test_labeling_seed.py`, import fair labeling from the new module:

```python
from aicir.qas.vqe_loop.fair_labeling import _label_seed_for_row
```

In `tests/test_chemistry_excitation_ansatz.py`, import chemistry P0 helpers from:

```python
from aicir.qas.vqe_loop.p0_chemistry_excitation import (
    build_chemistry_excitation_rows,
    closed_shell_excitation_pools,
)
```

In `tests/test_supernet_native_e5.py`, import native supernet helpers from:

```python
from aicir.qas.vqe_loop.p0_supernet_native import build_native_supernet_e5_evaluator
```

- [ ] **Step 2: Run tests to verify any missing canonical name fails**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_vqe_loop_unified_interface tests.test_labeling_seed tests.test_chemistry_excitation_ansatz tests.test_supernet_native_e5
```

Expected before implementation: FAIL with `ModuleNotFoundError` or `ImportError` for missing canonical names.

- [ ] **Step 3: Implement minimal naming exports**

If a canonical file does not exist, move the old file to the new name:

```powershell
Move-Item -LiteralPath 'vqe_loop\labeling.py' -Destination 'vqe_loop\fair_labeling.py'
Move-Item -LiteralPath 'vqe_loop\vqe_qas_loop.py' -Destination 'vqe_loop\p0_bootstrap_fair.py'
Move-Item -LiteralPath 'vqe_loop\chemistry_excitation.py' -Destination 'vqe_loop\p0_chemistry_excitation.py'
Move-Item -LiteralPath 'vqe_loop\supernet_native.py' -Destination 'vqe_loop\p0_supernet_native.py'
```

Update `aicir/qas/vqe_loop/__init__.py` to export:

```python
from .p0_bootstrap_fair import (
    ClosedLoopConfig,
    ClosedLoopResolvedDefaults,
    ClosedLoopResult,
    P0BootstrapConfig,
    P0BootstrapResult,
    effective_supernet_bootstrap_count,
    resolve_closed_loop_defaults,
    run_p0_bootstrap_fair,
    run_vqe_qas_closed_loop,
    stamp_literal_hamiltonian_terms,
)
```

- [ ] **Step 4: Run tests to verify canonical names pass**

Run the same command from Step 2.

Expected: PASS.

---

### Task 2: Remove Misleading Legacy Imports

**Files:**
- Modify: all `vqe_loop/*.py`
- Modify: `tests/*.py`
- Modify: `demos/*.py`

- [ ] **Step 1: Scan for legacy module paths**

Run:

```powershell
rg -n "vqe_loop\.(labeling|vqe_qas_loop|chemistry_excitation|supernet_native)|from aicir\.qas\.vqe_loop\.(labeling|vqe_qas_loop|chemistry_excitation|supernet_native)" vqe_loop tests demos -S
```

Expected before cleanup: any remaining old imports are listed.

- [ ] **Step 2: Replace imports with canonical modules**

Use these replacements:

```text
aicir.qas.vqe_loop.fair_labeling -> aicir.qas.vqe_loop.fair_labeling
aicir.qas.vqe_loop.vqe_qas_loop -> aicir.qas.vqe_loop.p0_bootstrap_fair
aicir.qas.vqe_loop.chemistry_excitation -> aicir.qas.vqe_loop.p0_chemistry_excitation
aicir.qas.vqe_loop.supernet_native -> aicir.qas.vqe_loop.p0_supernet_native
```

Also update `python -m` strings:

```text
aicir.qas.vqe_loop.fair_labeling -> aicir.qas.vqe_loop.fair_labeling
```

- [ ] **Step 3: Re-scan for old imports**

Run the Step 1 command again.

Expected: no output.

- [ ] **Step 4: Compile affected files**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile vqe_loop\fair_labeling.py vqe_loop\p0_bootstrap_fair.py vqe_loop\p0_chemistry_excitation.py vqe_loop\p0_supernet_native.py vqe_loop\__init__.py vqe_loop\__main__.py vqe_loop\p1_round.py vqe_loop\p1_evolution.py vqe_loop\sharding.py demos\run_p1_round_demo.py demos\run_p0_diagnostic.py
```

Expected: exit code 0.

---

### Task 3: Make Fair Labeling Explicitly Shared

**Files:**
- Modify: `aicir/qas/vqe_loop/fair_labeling.py`
- Modify: `aicir/qas/vqe_loop/p1_round.py`
- Modify: `aicir/qas/vqe_loop/p0_bootstrap_fair.py`
- Modify: `aicir/qas/vqe_loop/sharding.py`
- Modify: `aicir/qas/vqe_loop/README.md`

- [ ] **Step 1: Update fair labeling docstring**

At the top of `fair_labeling.py`, keep this responsibility text:

```python
"""Run frozen fair-VQE labels for P0/P1 benchmark queues.

This runner intentionally does not rank architectures. It turns pending queue
rows into protocol-versioned labels, including literal-Hamiltonian support,
warm-start parameters, best traces, retry status, and backend/dtype metadata.
"""
```

- [ ] **Step 2: Update P1 note**

In `p1_round.py`, make the summary note say:

```python
"note": "P1 round planner writes equal-size fair-label queues; fair_labeling.py supplies the fair COBYLA labels."
```

- [ ] **Step 3: Update sharding module string**

In `sharding.py`, `_runner_command()` must call:

```python
"aicir.qas.vqe_loop.fair_labeling",
```

- [ ] **Step 4: Run fair labeling tests**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_labeling_seed tests.test_p1_round tests.test_p1_round_demo tests.test_p0_bootstrap_conversion
```

Expected: PASS.

---

### Task 4: Separate Core and Optional Diagnostics in README

**Files:**
- Modify: `aicir/qas/vqe_loop/README.md`

- [ ] **Step 1: Replace module responsibility section**

Use this structure:

```markdown
## Module Responsibilities

### P0

- `p0_bootstrap_fair.py`: P0 bootstrap queue writer plus fair-label one-call API.
- `p0_chemistry_excitation.py`: chemistry excitation P0 rows and mutation-space metadata.
- `p0_supernet_native.py`: native supernet P0 rows and E5 evaluator support.

### P1

- `p1_round.py`: one P1 queue plus equal-budget baselines.
- `p1_evolution.py`: parent selection, mutation, crossover, and child rows.
- `p1_selection.py`: fallback ranking, selector queues, baselines.
- `benchmark_table.py`: benchmark schema, CSV/row parsing, fair protocol defaults, and row-level P1 policies.
- `oracle.py`: gene-aware distance and fair-energy prediction/abstention.

### Fair/Shared

- `fair_labeling.py`: pending queue rows to fair VQE labels.
- `fair_vqe.py`: COBYLA VQE optimization.
- `benchmark_table.py` built-in fair protocol: frozen fair-label protocol.
- `benchmark_table.py`: benchmark schema, CSV/row parsing, built-in fair protocol defaults, and row-level P1 policies.
- `training_free.py`: P0/P1 zero-cost annotations.
- `sharding.py`: fair-label queue sharding.

### Optional

- `cheap_eval_experiment.py`: E1/E2/E5/fair diagnostic harness.
- `task_proxy.py`: B1 task proxy.
- `graph_predictor.py`: B2 feature predictor.
- `p0_problem_aware.py`: optional P0 diagnostic problem-aware supernet sampler.
```

- [ ] **Step 2: Run README scan**

Run:

```powershell
rg -n "labeling.py|vqe_qas_loop.py|chemistry_excitation.py|supernet_native.py" vqe_loop\README.md
```

Expected: no output.

---

### Task 5: Keep Backward Compatibility Deliberate and Small

**Files:**
- Modify: `aicir/qas/vqe_loop/p0_bootstrap_fair.py`
- Modify: `aicir/qas/vqe_loop/__init__.py`

- [ ] **Step 1: Keep compatibility aliases only at package/P0 boundary**

At the bottom of `p0_bootstrap_fair.py`, keep:

```python
# Compatibility aliases for older demos/tests that still use the historical
# closed-loop name. The implementation above is intentionally P0/fair only.
P0BootstrapConfig = ClosedLoopConfig
P0BootstrapResult = ClosedLoopResult
run_vqe_qas_closed_loop = run_p0_bootstrap_fair
```

- [ ] **Step 2: Do not add compatibility shim files**

Confirm these files do not exist:

```powershell
Test-Path vqe_loop\labeling.py
Test-Path vqe_loop\vqe_qas_loop.py
Test-Path vqe_loop\chemistry_excitation.py
Test-Path vqe_loop\supernet_native.py
```

Expected:

```text
False
False
False
False
```

- [ ] **Step 3: Run public interface smoke**

Run:

```powershell
$env:PYTHONPATH='C:\Users\lixin\Documents\GitHub\quantum_frame_test'
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "from aicir.qas.vqe_loop import P0BootstrapConfig, run_p0_bootstrap_fair, run_vqe_qas_closed_loop; assert run_p0_bootstrap_fair is run_vqe_qas_closed_loop; print(P0BootstrapConfig.__name__)"
```

Expected:

```text
ClosedLoopConfig
```

---

### Task 6: Final Verification

**Files:**
- No code changes.

- [ ] **Step 1: Compile affected modules**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile vqe_loop\fair_labeling.py vqe_loop\p0_bootstrap_fair.py vqe_loop\p0_chemistry_excitation.py vqe_loop\p0_supernet_native.py vqe_loop\p1_round.py vqe_loop\p1_evolution.py vqe_loop\p1_selection.py vqe_loop\benchmark_table.py vqe_loop\fair_vqe.py vqe_loop\benchmark_table.py vqe_loop\training_free.py vqe_loop\sharding.py
```

Expected: exit code 0.

- [ ] **Step 2: Run focused unittest suite**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_labeling_seed tests.test_vqe_loop_unified_interface tests.test_chemistry_excitation_ansatz tests.test_explicit_gate_ansatz tests.test_supernet_native_e5 tests.test_p1_policy tests.test_p1_evolution tests.test_p1_round tests.test_p1_round_demo tests.test_task_proxy tests.test_vqe_loop_shared_rows tests.test_cheap_eval_experiment tests.test_p0_bootstrap_conversion
```

Expected: PASS.

- [ ] **Step 3: Scan for legacy names**

Run:

```powershell
rg -n "aicir\.qas\.vqe_loop\.(labeling|vqe_qas_loop|chemistry_excitation|supernet_native)(\W|$)|from aicir\.qas\.vqe_loop\.(labeling|vqe_qas_loop|chemistry_excitation|supernet_native)\b|python -m aicir\.qas\.vqe_loop\.labeling\b|\blabeling\.py\b|\bvqe_qas_loop\.py\b|\bchemistry_excitation\.py\b|\bsupernet_native\.py\b|fair_fair_labeling" vqe_loop tests demos -S
```

Expected: no output.

---

## Self-Review

Spec coverage:

- P0/P1 file prefixes are explicit where the file belongs to one lane.
- `fair_labeling.py` is not prefixed because it is shared by P0 and P1.
- `fair_vqe.py`, `benchmark_table.py` built-in fair protocol, `benchmark_table.py` and `training_free.py` remain shared.
- Optional B1/B2/diagnostic modules are clearly separate from core P0/P1/fair.

Placeholder scan:

- No TBD/TODO placeholders.
- Every task has exact file paths and verification commands.

Type consistency:

- `P0BootstrapConfig` and `P0BootstrapResult` are aliases to the existing dataclasses for compatibility.
- `run_vqe_qas_closed_loop` remains an alias to `run_p0_bootstrap_fair`.



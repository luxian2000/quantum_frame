# VQE Loop Mainline Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the current `vqe_loop` mainline into clearer boundaries while keeping public P0/P1/fair behavior unchanged.

**Architecture:** Keep `p1_evolution.py` and `p1_round.py` as public entry points. Move operator-sequence growth logic into `p1_evolution.py` and P1 selection mechanics into `p1_selection.py`, with compatibility imports where current tests or demos rely on old module paths.

**Tech Stack:** Python standard library, existing `aicir.qas.vqe_loop` modules, `unittest`, `py_compile`.

---

### Task 1: Move Operator Growth Out of `p1_evolution.py`

**Files:**
- Create: `vqe_loop/p1_evolution.py`
- Modify: `vqe_loop/p1_evolution.py`
- Test: `tests/test_p1_p1_evolution.py`

- [ ] **Step 1: Write compatibility-focused failing test**

Add a test to `tests/test_p1_p1_evolution.py` that imports `_operator_growth_evaluator_from_row` from both modules and verifies they produce the same callable behavior:

```python
def test_operator_growth_reexport_matches_new_module(self):
    from aicir.qas.vqe_loop.p1_evolution import _operator_growth_evaluator_from_row as moved
    from aicir.qas.vqe_loop.p1_evolution import _operator_growth_evaluator_from_row as reexported

    row = {
        "hamiltonian_terms": "[[1.0, \"XI\"], [0.5, \"IZ\"]]",
    }
    self.assertIsNotNone(moved(row))
    self.assertIsNotNone(reexported(row))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_p1_evolution
```

Expected: fails because `aicir.qas.vqe_loop.p1_evolution` does not exist.

- [ ] **Step 3: Create `p1_evolution.py`**

Move these functions from `p1_evolution.py` into `p1_evolution.py`:

```text
_normalize_operator_pool
_default_operator_pool
_finite_difference_operator_growth_score
_operator_growth_evaluator_from_row
_select_adapt_growth_operator
```

The new module imports:

```python
import json
from typing import Any, Mapping, Sequence

from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene
from aicir.qas.vqe_loop.hamiltonian_rows import row_hamiltonian_terms
```

- [ ] **Step 4: Update `p1_evolution.py` imports**

In `p1_evolution.py`, import the moved helpers:

```python
from aicir.qas.vqe_loop.p1_evolution import (
    _default_operator_pool,
    _operator_growth_evaluator_from_row,
    _select_adapt_growth_operator,
)
```

Keep imports private because current tests use private helpers and this is a compatibility refactor.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_p1_evolution
```

Expected: OK.

---

### Task 2: Move P1 Selection Mechanics Out of `p1_round.py`

**Files:**
- Create: `vqe_loop/p1_selection.py`
- Modify: `vqe_loop/p1_round.py`
- Test: `tests/test_p1_round.py`

- [ ] **Step 1: Write failing import/behavior test**

Add a test to `tests/test_p1_round.py`:

```python
def test_p1_selection_module_ranks_rows_by_score(self):
    from aicir.qas.vqe_loop.p1_selection import rank_by_score

    rows = [
        {"architecture_id": "bad", "E2": "2.0"},
        {"architecture_id": "good", "E2": "1.0"},
        {"architecture_id": "missing", "E2": ""},
    ]
    ranked = rank_by_score(rows, "E2")
    self.assertEqual([row["architecture_id"] for row in ranked], ["good", "bad"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_p1_round
```

Expected: fails because `p1_selection.py` does not exist.

- [ ] **Step 3: Create `p1_selection.py`**

Move or wrap these responsibilities from `p1_round.py`:

```text
rank_by_score
rank_fallback_rows
baseline_selector_queue
```

The public names in the new module should omit leading underscores:

```python
rank_by_score
rank_fallback_rows
baseline_selector_queue
```

- [ ] **Step 4: Update `p1_round.py`**

Replace local helper calls with imports:

```python
from aicir.qas.vqe_loop.p1_selection import (
    baseline_selector_queue,
    rank_by_score,
    rank_fallback_rows,
)
```

Keep local aliases only if needed for compatibility:

```python
_rank_by_score = rank_by_score
_rank_fallback_rows = rank_fallback_rows
_baseline_selector_queue = baseline_selector_queue
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_p1_round tests.test_p1_round_demo tests.test_p1_policy
```

Expected: OK.

---

### Task 3: Document Mainline Boundaries

**Files:**
- Modify: `vqe_loop/README.md`
- Modify: `vqe_loop/vqe_qas_loop.py`

- [ ] **Step 1: Update README mainline section**

Add a short section to `vqe_loop/README.md`:

```markdown
## Mainline Module Boundaries

- `p1_evolution.py` is the public mutation entry point.
- `p1_evolution.py` owns operator-sequence A1/A2 growth helpers.
- `p1_round.py` is the public P1 orchestration entry point.
- `p1_selection.py` owns fallback ranking, baseline selector queues, and no-regret-lite selection helpers.
- `vqe_qas_loop.py` remains a compatibility runner for P0 bootstrap plus fair labels.
```

- [ ] **Step 2: Update `vqe_qas_loop.py` module docstring**

Ensure the docstring says it is a compatibility P0 bootstrap + fair-label runner, not a full multi-round closed-loop implementation.

- [ ] **Step 3: Run documentation-safe verification**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile vqe_loop\vqe_qas_loop.py
```

Expected: OK.

---

### Task 4: Final Verification

**Files:**
- No production changes unless tests expose a regression.

- [ ] **Step 1: Compile affected files**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile vqe_loop\p1_evolution.py vqe_loop\p1_selection.py vqe_loop\p1_evolution.py vqe_loop\p1_round.py vqe_loop\vqe_qas_loop.py
```

Expected: OK.

- [ ] **Step 2: Run mainline tests**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_p1_evolution tests.test_p1_policy tests.test_p1_round tests.test_p1_round_demo tests.test_task_proxy tests.test_vqe_loop_shared_rows
```

Expected: OK.

- [ ] **Step 3: Run P0/fair smoke tests**

Run:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_cheap_eval_experiment tests.test_labeling_seed tests.test_p0_bootstrap_conversion tests.test_vqe_loop_unified_interface
```

Expected: OK.


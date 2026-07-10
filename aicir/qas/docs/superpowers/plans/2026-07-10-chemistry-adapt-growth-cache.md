# Chemistry ADAPT Growth Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate repeated chemistry ADAPT proxy evaluations within one P1 parent expansion while preserving search and fair-label behavior.

**Architecture:** Add a parent-local score memoizer around the chemistry growth evaluator used by `generate_mutation_children()`. Add an evaluator-local energy cache to the shared finite-difference implementation so all candidates reuse the same parent energy.

**Tech Stack:** Python, `unittest`, existing QAS chemistry ansatz and fair VQE helpers.

---

### Task 1: Add regression coverage

**Files:**
- Modify: `tests/test_p1_evolution.py`

- [ ] Add `test_generate_mutation_children_caches_chemistry_growth_scores_per_parent`, generate multiple ADAPT children from one parent, and assert the injected scorer is called once per limited-pool candidate.
- [ ] Add `test_chemistry_growth_evaluator_reuses_parent_energy`, patch `evaluate_vqe_energy`, score two distinct candidates, and assert one base plus four trial evaluations.
- [ ] Run `python -m unittest tests.test_p1_evolution.P1VariationTests.test_generate_mutation_children_caches_chemistry_growth_scores_per_parent tests.test_p1_evolution.P1VariationTests.test_chemistry_growth_evaluator_reuses_parent_energy` and confirm both tests fail for excessive calls.

### Task 2: Implement bounded caches

**Files:**
- Modify: `vqe_loop/p1_evolution.py`

- [ ] Add a parent-local chemistry score cache in `generate_mutation_children()`, keyed by canonical gene payload and canonical excitation.
- [ ] Add an optional energy cache to `_finite_difference_chemistry_growth_score()` and use architecture identity plus parameter tuple as the key.
- [ ] Allocate that energy cache inside `_chemistry_growth_evaluator_from_row()` so it cannot leak across rows or Hamiltonians.
- [ ] Re-run the two focused tests and confirm they pass.

### Task 3: Verify and publish

**Files:**
- Verify: `vqe_loop/p1_evolution.py`
- Verify: `tests/test_p1_evolution.py`

- [ ] Run `python -m unittest discover -s tests -p 'test_p1_evolution.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_p1_round.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_p1_round_demo.py'`.
- [ ] Run `python -m py_compile vqe_loop/p1_evolution.py tests/test_p1_evolution.py`.
- [ ] Run `git diff --check` on the changed files.
- [ ] Commit only the scoped files and push branch `qas_1`.


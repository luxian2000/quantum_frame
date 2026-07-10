# Chemistry ADAPT Growth Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate repeated chemistry ADAPT proxy evaluations and support restarting formal P1 from completed P0 labels.

**Architecture:** Add a parent-local score memoizer around the chemistry growth evaluator used by `generate_mutation_children()`. Add an evaluator-local energy cache to the shared finite-difference implementation so all candidates reuse the same parent energy. Add an opt-in P1 resume branch to the CH4 formal runner while leaving its default P0 path unchanged.

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

### Task 3: Add a formal P1-only resume path

**Files:**
- Modify: `demos/run_ch4_18q_lineb_npu4.sh`
- Create: `tests/test_ch4_lineb_runner.py`

- [ ] Add a failing shell-level test that runs with `ROUNDS=0`, `SKIP_P0=1`, and a deliberately invalid `PYTHON`; require success and a copied `current_labeled_rows.csv`.
- [ ] Add `SKIP_P0` and `P1_BOOTSTRAP_LABELS_CSV` handling around the existing P0 block, validating the source before P1 starts.
- [ ] Run `python -m unittest discover -s tests -p 'test_ch4_lineb_runner.py'` and confirm it passes.

### Task 4: Verify and publish

**Files:**
- Verify: `vqe_loop/p1_evolution.py`
- Verify: `tests/test_p1_evolution.py`
- Verify: `demos/run_ch4_18q_lineb_npu4.sh`
- Verify: `tests/test_ch4_lineb_runner.py`

- [ ] Run `python -m unittest discover -s tests -p 'test_p1_evolution.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_p1_round.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_p1_round_demo.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_ch4_lineb_runner.py'`.
- [ ] Run `bash -n demos/run_ch4_18q_lineb_npu4.sh`.
- [ ] Run `python -m py_compile vqe_loop/p1_evolution.py tests/test_p1_evolution.py`.
- [ ] Run `git diff --check` on the changed files.
- [ ] Commit only the scoped files and push branch `qas_1`.

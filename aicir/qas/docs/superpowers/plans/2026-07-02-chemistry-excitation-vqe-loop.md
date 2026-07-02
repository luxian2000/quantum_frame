# Chemistry Excitation VQE Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone chemistry excitation ansatz family to the VQE loop so P0/P1 queues and fair-label row handling can use HF plus single/double excitation candidates.

**Architecture:** Keep chemistry excitation separate from `SupernetAnsatzGene`. Add a new ansatz gene and row builder that emit the existing benchmark-table schema, then wire the existing `labeling.py`, cheap E2 path, oracle distance, and P1 mutation planner to recognize the new family.

**Tech Stack:** Python dataclasses, existing `ArchitectureSpec.from_gates`, CSV benchmark-table rows, pytest/unittest.

---

### Task 1: Chemistry Gene And Pools

**Files:**
- Modify: `primitives/ansatz.py`
- Create: `vqe_loop/chemistry_excitation.py`
- Test: `tests/test_chemistry_excitation_ansatz.py`

- [ ] Write tests for closed-shell HF occupied qubits, excitation pool generation, gene round-trip, and architecture metadata.
- [ ] Run `python -m pytest -p no:cacheprovider tests/test_chemistry_excitation_ansatz.py -q` and confirm the new tests fail because imports are missing.
- [ ] Implement `closed_shell_excitation_pools`, `ChemistryExcitationAnsatzGene`, and `architecture_from_chemistry_excitation_gene`.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Labeling And Cheap Eval Recognition

**Files:**
- Modify: `vqe_loop/labeling.py`
- Modify: `vqe_loop/cheap_eval_experiment.py`
- Test: `tests/test_chemistry_excitation_ansatz.py`

- [ ] Add failing tests showing `labeling._architecture_from_row` and `cheap_eval_experiment._architecture_from_experiment_row` accept `kind="chemistry_excitation"`.
- [ ] Run the focused test and confirm it fails at row parsing.
- [ ] Import the new gene and builder in both modules and route by `kind`.
- [ ] Re-run the focused test.

### Task 3: P0 Row Generation

**Files:**
- Modify: `vqe_loop/chemistry_excitation.py`
- Test: `tests/test_chemistry_excitation_ansatz.py`

- [ ] Add failing tests for deterministic chemistry-supernet bootstrap rows with `family="chemistry_excitation"`, `screening_energy_is_final_label="false"`, and benchmark-table-compatible fields.
- [ ] Implement `build_chemistry_excitation_rows`.
- [ ] Re-run the focused test.

### Task 4: P1 Mutation And Oracle Distance

**Files:**
- Modify: `vqe_loop/p1_evolution.py`
- Modify: `vqe_loop/oracle.py`
- Test: `tests/test_p1_p1_evolution.py`
- Test: `tests/test_p1_oracle.py`

- [ ] Add failing tests for chemistry insert/delete/swap/change mutation rows and chemistry-vs-chemistry oracle distance.
- [ ] Extend `MutableGene`, mutation type lists, row metadata, and distance dispatch.
- [ ] Re-run the focused P1 tests.

### Task 5: Smoke Verification

**Files:**
- Test: focused tests above

- [ ] Run `python -m py_compile primitives/ansatz.py vqe_loop/chemistry_excitation.py vqe_loop/labeling.py vqe_loop/cheap_eval_experiment.py vqe_loop/p1_evolution.py vqe_loop/oracle.py`.
- [ ] Run the focused pytest set.
- [ ] If full COBYLA fair labeling fails because core excitation gates are unavailable, record the exact failure and defer core gate support to the next phase.

# QAS Growth Routes Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Line A and Line B execute the two-route architecture contract without stale selector reuse, repeated ADAPT scans, task leakage, or fair-budget drift.

**Architecture:** Keep the shared benchmark schema and fair VQE/COBYLA protocol unchanged. Clean architecture-derived fields at the mutation boundary, scope identities by task, share parent-local ADAPT caches, and make route defaults executable from one Python configuration module that is mirrored in the architecture document.

**Tech Stack:** Python, `unittest`, Bash launch wrappers, CSV benchmark rows.

---

### Task 1: Child Row Hygiene And Task Scope

**Files:**
- Modify: `vqe_loop/p1_evolution.py`
- Modify: `vqe_loop/benchmark_table.py`
- Modify: `vqe_loop/p1_round.py`
- Test: `tests/test_p1_evolution.py`
- Test: `tests/test_p1_round.py`

- [ ] Add failing tests proving mutated children do not inherit selector/proxy fields.
- [ ] Add failing tests proving parent selection and deduplication do not cross family or Hamiltonian task scope.
- [ ] Clear all architecture-derived measurements at child creation.
- [ ] Filter parents by the active route family and include task identity in deduplication/cache keys.
- [ ] Run the focused P1 evolution and round tests.

### Task 2: ADAPT Correctness And Reuse

**Files:**
- Modify: `library/ansatz.py`
- Modify: `vqe_loop/p1_evolution.py`
- Test: `tests/test_fair_vqe_pauli_terms.py`
- Test: `tests/test_p1_evolution.py`

- [ ] Add a failing fair-VQE test for X/Y operator-sequence gates.
- [ ] Add failing tests for operator score/energy reuse and exhausted pools.
- [ ] Normalize operator basis-change gate names for the shared simulator.
- [ ] Add parent-local operator score and energy caches matching chemistry behavior.
- [ ] Raise `MutationUnavailable` when no novel operator or excitation remains.
- [ ] Stop a round cleanly when no new child can be generated.

### Task 3: Fair Budget And Backend Propagation

**Files:**
- Modify: `demos/run_p1_round_demo.py`
- Modify: `demos/run_ch4_18q_lineb_npu4.sh`
- Modify: `vqe_loop/p1_evolution.py`
- Test: `tests/test_p1_round_demo.py`
- Test: `tests/test_p1_evolution.py`

- [ ] Add failing tests for zero-round budget exits, failed-call accounting, and duplicate downgraded baselines.
- [ ] Cap each planned queue by remaining fair budget and count attempted rows, including failures.
- [ ] Remove duplicate E5 queues when E5 deterministically downgrades to E2 for the active family.
- [ ] Thread backend/dtype into P1 E2 and ADAPT evaluators.
- [ ] Initialize early stopping from the P0/bootstrap best fair energy.

### Task 4: Executable Route Configuration

**Files:**
- Create: `vqe_loop/growth_routes.py`
- Modify: `demos/run_p1_round_demo.py`
- Modify: `demos/run_ch4_18q_lineb_npu4.sh`
- Modify: `docs/superpowers/specs/2026-07-08-qas-two-growth-routes-architecture.md`
- Test: `tests/test_p1_round_demo.py`

- [ ] Add failing parser tests for Line A and Line B defaults.
- [ ] Define typed defaults for route family, rounds, selector, mutation weights, depth, diversity, and stopping.
- [ ] Apply route defaults before explicit CLI overrides.
- [ ] Keep the shell wrapper responsible only for hardware and run-directory overrides.
- [ ] Update the architecture document to exactly match executable defaults.

### Task 5: Verification And Delivery

**Files:**
- Review all modified files.

- [ ] Run `py_compile` on every modified Python module.
- [ ] Run focused P1, fair-VQE, and demo unit tests.
- [ ] Run the broader QAS P0/P1 regression suite.
- [ ] Review `git diff --check` and confirm unrelated dirty files were untouched.
- [ ] Commit scoped changes and push `qas_1`.

# P1 Mutation Oracle Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the P1 foundation for mutation-driven QAS search with fair-call budget tracking, deduplication, oracle/L1 quota merge, and gene-aware oracle prediction.

**Architecture:** Keep P1 as small pure modules first, then wire orchestration later. `p1_policy.py` owns budget/dedup/quota/merge policy, `oracle.py` owns gene-aware architecture distance and trust-region prediction, and tests use `unittest` so they run in the current host environment without `pytest`.

**Tech Stack:** Python standard library, existing `SupernetAnsatzGene`, CSV-style row dictionaries, `unittest`.

---

### File Structure

- Create: `vqe_loop/p1_policy.py`
  - fair-call budget tracker
  - child deduplication against labeled/unlabeled/current-round rows
  - P1 selector resolution so `both` delegates to configured `cheap_eval_selector`
  - cold-start and adaptive quota policy
  - quota-based L0/L1/control merge without score calibration
  - optional `control_score_field` hook for diversity/control queues; default remains FIFO
- Create: `vqe_loop/oracle.py`
  - extract the duplicated inverse-distance kNN prediction currently embedded in `next_batch.py` and `calibration.py`
  - gene-aware distance for supernet-native ansatz genes, alongside the existing coarse trust-region geometry in `geometry.py`
  - exact-match and kNN trust-region prediction of `fair_best_energy`
  - abstain status when neighbors are insufficient or too far
  - keep E2/E5 proxy scores out of the oracle target
  - record `mutation_type` in prediction output for later per-mutation trust calibration
- Create: `tests/test_p1_policy.py`
  - regression tests for fair budget equality, dedup, selector resolution, quota, and merge
- Create: `tests/test_p1_oracle.py`
  - regression tests for gate/layer/depth distance ordering and trust-region prediction
- Create: `vqe_loop/p1_evolution.py`
  - pure parent selection from fair-labeled rows
  - optional diversity parent selection by gene-aware distance
  - supernet-native gate/connectivity/layer/depth mutations
  - layer crossover for compatible supernet-native parents
  - optional weighted mutation choice; default remains deterministic round-robin coverage
  - retry around unavailable/no-op mutations
  - child rows with `mutation_type` and `parent_architecture_id`
- Create: `tests/test_p1_p1_evolution.py`
  - regression tests for each mutation type and child row metadata
- Create: `vqe_loop/p1_round.py`
  - plan one P1 mutation/oracle/fallback queue without running fair VQE
  - write equal-size random/E2-only/E5-only baseline queues for cost-normalized labeling
  - preserve P1 metadata through the benchmark-table schema
- Create: `tests/test_p1_round.py`
  - regression tests for schema metadata, oracle/fallback merge, equal budget baselines, and output files

### Task 1: P1 Loop Pure Policy

- [ ] Write failing `unittest` tests in `tests/test_p1_policy.py`.
- [ ] Run `python tests/test_p1_policy.py` and confirm imports/functions are missing.
- [ ] Implement `vqe_loop/p1_policy.py` with minimal pure functions and dataclasses.
- [ ] Re-run `python tests/test_p1_policy.py` and confirm it passes.

### Task 2: Gene-Aware Oracle

- [ ] Write failing `unittest` tests in `tests/test_p1_oracle.py`.
- [ ] Run `python tests/test_p1_oracle.py` and confirm imports/functions are missing.
- [ ] Implement `vqe_loop/oracle.py` with gene-aware distance and kNN prediction.
- [ ] Update existing Stage-2 callers to reuse the public weighted kNN helper where safe.
- [ ] Re-run `python tests/test_p1_oracle.py` and confirm it passes.

### Task 3: Verification

- [ ] Run `python -m py_compile vqe_loop/p1_policy.py vqe_loop/oracle.py tests/test_p1_policy.py tests/test_p1_oracle.py`.
- [ ] Run `python tests/test_p1_policy.py`.
- [ ] Run `python tests/test_p1_oracle.py`.
- [ ] Report what is implemented and what remains for full `p1_policy.py` orchestration.

### Self-Review

- Spec coverage: The four requested details are covered by budget tracking, dedup, selector resolution, and quota policy. The larger mutation/fair-VQE runner is intentionally left for the next plan after these pure policies are stable.
- Placeholder scan: No undefined implementation placeholders remain in the tasks; each task names files and commands.
- Type consistency: Row-based APIs consistently use dictionaries keyed by `architecture_id`, `ansatz_gene`, `canonical_arch_hash`, and `fair_best_energy`.

# Cheap Evaluation Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic P0/P0.5 diagnostic tool that compares light VQE and native supernet screening as cheap architecture evaluators before adding problem-aware bias, mutation, surrogate models, or RL.

**Architecture:** Add a focused `vqe_loop.cheap_eval_diagnostics` module that works on existing `SupernetAnsatzGene` architecture rows and completed benchmark rows. The first implementation computes robust ranking diagnostics, top-K enrichment, cost curves for `N in {10,20,40,80}`, warm-start gain summaries from traces, and hit/exposure summaries when instrumentation is present; it avoids running heavy experiments in tests.

**Tech Stack:** Python standard library, `numpy`, existing `aicir.qas` VQE/supernet primitives, existing CSV benchmark-table contract.

---

### File Structure

- Create: `vqe_loop/cheap_eval_diagnostics.py`
  - Pure functions for Spearman/Kendall-style rank diagnostics, top-K recall/enrichment, cost curves, warm-start gain summaries, and decision status.
  - CSV loader and CLI entry point for analyzing rows already produced by P0 experiments.
- Modify: `vqe_loop/__init__.py`
  - Export the diagnostic summary helpers for tests and downstream scripts.
- Create: `tests/test_cheap_eval_diagnostics.py`
  - Unit tests with synthetic rows covering uniform-vs-biased separation, E1/E2/E5 proxy comparison, top-K enrichment, N-dependent costs, and fallback decisions.

### Task 1: Ranking And Top-K Metrics

**Files:**
- Create: `vqe_loop/cheap_eval_diagnostics.py`
- Test: `tests/test_cheap_eval_diagnostics.py`

- [ ] **Step 1: Write failing tests for rank metrics**

Add tests that build four rows with `fair_high`, `E1`, `E2`, and `E5` columns. Assert:
- Spearman is `1.0` for perfect ranking.
- Spearman is negative for reversed ranking.
- Top-2 recall and enrichment are computed against high-budget fair labels.

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: FAIL because `aicir.qas.vqe_loop.cheap_eval_diagnostics` does not exist.

- [ ] **Step 2: Implement rank and top-K helpers**

Implement:
- `_as_float(value) -> float | None`
- `_rank(values) -> list[float]`
- `spearman_correlation(rows, proxy_field, target_field) -> float | None`
- `kendall_pairwise_accuracy(rows, proxy_field, target_field) -> float | None`
- `top_k_enrichment(rows, proxy_field, target_field, k) -> dict[str, float]`

Lower energy is better for both proxy and target fields.

- [ ] **Step 3: Run tests**

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: PASS for Task 1 tests.

### Task 2: Cost Curves And Proxy Decision Summary

**Files:**
- Modify: `vqe_loop/cheap_eval_diagnostics.py`
- Modify: `tests/test_cheap_eval_diagnostics.py`

- [ ] **Step 1: Write failing tests for N-dependent cost**

Add tests for:
- `cost_curve(upfront_cost=100, per_arch_cost=5, n_values=[10, 20])` returns total and amortized costs.
- `proxy_quality_cost_frontier(...)` preserves quality metrics and cost metrics for `E1`, `E2`, and `E5`.

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: FAIL because the cost helpers do not exist.

- [ ] **Step 2: Implement cost helpers**

Implement:
- `cost_curve(upfront_cost, per_arch_cost, n_values=(10,20,40,80))`
- `proxy_quality_cost_frontier(rows, proxy_fields, target_field, cost_models, k)`

Each proxy summary includes `spearman`, `kendall_pairwise_accuracy`, `top_k_recall`, `top_k_enrichment`, `total_cost_N`, and `amortized_cost_N`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: PASS for Task 1 and Task 2 tests.

### Task 3: Warm-Start Gain And Hit/Exposure Stratification

**Files:**
- Modify: `vqe_loop/cheap_eval_diagnostics.py`
- Modify: `tests/test_cheap_eval_diagnostics.py`

- [ ] **Step 1: Write failing tests for warm-start and exposure summaries**

Add tests for:
- `warm_start_gain_summary(rows, warm_field, random_field)` returns mean, std, and count.
- `stratified_proxy_summary(rows, proxy_field, target_field, strata_field, bins)` reports separate Spearman/top-K summaries for low/high hit-rate groups.

Use synthetic rows with `hit_rate` and `exposure_count` fields.

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: FAIL because the helpers do not exist.

- [ ] **Step 2: Implement warm-start and stratification helpers**

Implement:
- `warm_start_gain_summary(rows, warm_field="fair_warm", random_field="fair_random")`
- `stratified_proxy_summary(rows, proxy_field, target_field, strata_field, threshold, k)`

For energies, gain is `random_energy - warm_energy`; positive means warm-start improved the energy.

- [ ] **Step 3: Run tests**

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: PASS for Task 1-3 tests.

### Task 4: P0/P0.5 Decision Gate And CLI

**Files:**
- Modify: `vqe_loop/cheap_eval_diagnostics.py`
- Modify: `vqe_loop/__init__.py`
- Modify: `tests/test_cheap_eval_diagnostics.py`

- [ ] **Step 1: Write failing tests for decision status**

Add tests for:
- A proxy with `spearman >= 0.6`, enrichment above random, and acceptable cost is marked `pass`.
- A proxy with low Spearman and low enrichment is marked `repair`.
- A case after `max_repair_rounds=3` is marked `fallback`.

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: FAIL because decision gate helpers do not exist.

- [ ] **Step 2: Implement decision helpers and CLI**

Implement:
- `decide_proxy_status(summary, spearman_threshold=0.6, min_enrichment=0.0, repair_round=0, max_repair_rounds=3)`
- `summarize_proxy_diagnostics(rows, proxy_fields, target_field, cost_models, k=3, n_values=(10,20,40,80))`
- CLI: `python -m aicir.qas.vqe_loop.cheap_eval_diagnostics --input rows.csv --target fair_high --proxies E1,E2,E5 --output summary.json`

The CLI reads CSV rows and writes JSON. It does not run VQE or supernet training.

- [ ] **Step 3: Export module helpers**

Update `vqe_loop/__init__.py` to export `summarize_proxy_diagnostics`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py -q`
Expected: PASS.

### Task 5: Verification

**Files:**
- No additional file changes expected.

- [ ] **Step 1: Compile changed Python files**

Run: `python -m py_compile vqe_loop/cheap_eval_diagnostics.py tests/test_cheap_eval_diagnostics.py`
Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest -p no:cacheprovider tests/test_cheap_eval_diagnostics.py tests/test_supernet_native_expansion.py -q`
Expected: exit code 0.

- [ ] **Step 3: Report implemented protocol**

Summarize that the tool implements the first executable layer of the agreed protocol:
- multi-proxy E1/E2/E5 analysis,
- top-K enrichment,
- cost(N),
- warm-start gain,
- hit/exposure stratification,
- explicit repair/fallback gate.

---

### Self-Review

- Spec coverage: The plan implements the agreed P0/P0.5 diagnostic analysis layer, not the heavy experiment runner. It deliberately leaves actual H2/LiH/Ising runs to a follow-up once the analyzer is tested.
- Placeholder scan: No placeholder tasks remain; every task names files, functions, commands, and expected outcomes.
- Type consistency: Rows are `Mapping[str, Any]`, proxy/target fields are strings, summaries are JSON-serializable dictionaries.

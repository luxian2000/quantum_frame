# VQE Loop Mainline Architecture Design

## Goal

Refactor `vqe_loop` around the current P0/P1/fair-label mainline without creating a large directory tree or changing behavior.

The mainline remains:

```text
P0 bootstrap -> P1 mutation/oracle/selector -> fair label -> benchmark summary
```

The primary metric remains `fair_best_energy`. Cheap evaluators, task proxies, graph proxies, zero-cost scores, and native supernet E5 are selector signals only.

## Scope

This design uses a small-file approach. It does not introduce a broad `schema/`, `selection/`, `variation/`, or `runner/` package tree yet.

The first refactor creates at most two new production files:

- `vqe_loop/p1_evolution.py`
- `vqe_loop/p1_selection.py`

Existing public entry points stay stable:

- `vqe_loop.p1_evolution.generate_mutation_children`
- `vqe_loop.p1_evolution.mutate_gene`
- `vqe_loop.p1_round.plan_p1_round`
- `vqe_loop.p1_round.write_p1_round_outputs`
- `vqe_loop.vqe_qas_loop.run_vqe_qas_closed_loop`

## Module Boundaries

### Existing Benchmark Table Helper

enchmark_table.py remains the shared row/schema helper. It owns benchmark fields, CSV IO, row-object parsing, built-in fair protocol defaults, and row-level P1 policies. It is not candidate generation or fair VQE execution logic.

### P1 Evolution

`p1_evolution.py` is the public P1 evolution entry point. It keeps parent selection, supernet mutation, chemistry excitation mutation, operator-sequence mutation, optional A1/A2-style operator growth, crossover, and mutation result to queue-row conversion in one file for now.

The operator-growth helpers stay inside `p1_evolution.py` because they are an optional operator-sequence mutation strategy rather than a separate mainline stage.

### P1 Selection

`p1_round.py` remains the public P1 orchestration entry point. It should coordinate child generation, oracle prediction, fallback selection, baseline queue creation, summaries, and output writing.

Selection mechanics should move out because they are not orchestration:

- fallback ranking
- no-regret-lite merge policy
- baseline selector queue construction

New file:

- `p1_selection.py`: score ranking, fallback topK selection helpers, no-regret-lite queue merge helpers, and baseline selector queue creation.

### Stable Files

Do not split these in this phase:

- `fair_labeling.py`
- `fair_vqe.py`
- `benchmark_table.py`\r\n- `p0_chemistry_excitation.py`
- `p0_supernet_native.py`
- `task_proxy.py`
- `graph_predictor.py`
- `training_free.py`
- `p0_bootstrap_fair.py`
- `geometry.py`

`vqe_qas_loop.py` should be documented as the compatibility P0 bootstrap + fair runner. `geometry.py` legacy cleanup is deferred until the mainline split is stable.

## Compatibility

Tests and demos should import P1 evolution behavior from `p1_evolution.py`; no legacy mutation/growth compatibility shim is kept in this cleanup pass.

No benchmark schema changes are allowed in this phase.

## Verification

Run focused checks after each split:

```text
py_compile affected files
tests.test_p1_evolution
tests.test_p1_policy
tests.test_p1_round
tests.test_p1_round_demo
tests.test_task_proxy
tests.test_vqe_loop_shared_rows
```

If P0/fair code is touched, also run:

```text
tests.test_cheap_eval_experiment
tests.test_labeling_seed
tests.test_p0_bootstrap_conversion
tests.test_vqe_loop_unified_interface
```


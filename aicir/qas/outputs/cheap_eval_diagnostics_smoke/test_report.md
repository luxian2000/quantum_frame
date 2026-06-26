# Cheap Evaluation Diagnostics Test Report

Date: 2026-06-26

## Scope

This report verifies the executable P0/P0.5 diagnostic layer and the experiment-output framework for comparing cheap architecture evaluators before mutation, surrogate modeling, or RL.

Covered capabilities:

- Spearman and pairwise ordering diagnostics
- architecture-id based top-K overlap, directional recall, and enrichment
- N-dependent cost curves
- warm-start gain with variance and zero-mean CV handling
- hit-rate/exposure stratification through CLI
- temporal/order-window diagnostics for supernet warmup confounds
- pass/repair/fallback gate
- CLI JSON summary generation
- E1-E5 experiment manifest, row schema, and injectable runner contract
- real light-VQE evaluator registry for `E1`, `E2`, and `fair_high`
- shared-seed proxy comparison, row-level architecture/problem cache, and warm-start parameter passthrough

## Files Under Test

- `vqe_loop/cheap_eval_diagnostics.py`
- `vqe_loop/cheap_eval_experiment.py`
- `tests/test_cheap_eval_diagnostics.py`
- `tests/test_cheap_eval_experiment.py`
- `vqe_loop/__init__.py`

## Environment Note

`pytest` is not installed in the available bundled Python environment, so tests were executed by directly importing the test modules and calling each `test_*` function. This covers the same assertions but does not use pytest's runner/reporting.

## Verification Commands

### Direct Test Function Run

Command:

```powershell
@'
import pathlib
import sys
import tempfile

repo = pathlib.Path(r'C:\Users\lixin\Documents\GitHub\quantum_frame_test')
sys.path.insert(0, str(repo))

modules = [
    'aicir.qas.tests.test_cheap_eval_diagnostics',
    'aicir.qas.tests.test_cheap_eval_experiment',
]
passed = []
for module_name in modules:
    module = __import__(module_name, fromlist=['*'])
    for name in sorted(dir(module)):
        if not name.startswith('test_'):
            continue
        fn = getattr(module, name)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(pathlib.Path(tmp))
            except TypeError:
                fn()
        passed.append(f'{module_name}.{name}')
print(f'passed={len(passed)}')
for name in passed:
    print(name)
'@ | & 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -
```

Result:

```text
passed=28
aicir.qas.tests.test_cheap_eval_diagnostics.test_cli_accepts_cost_models_file
aicir.qas.tests.test_cheap_eval_diagnostics.test_cli_writes_json_summary
aicir.qas.tests.test_cheap_eval_diagnostics.test_cost_curve_is_n_dependent
aicir.qas.tests.test_cheap_eval_diagnostics.test_decision_gate_passes_repairs_and_falls_back
aicir.qas.tests.test_cheap_eval_diagnostics.test_quality_cost_frontier_keeps_cost_and_quality_together
aicir.qas.tests.test_cheap_eval_diagnostics.test_rank_metrics_detect_good_and_reversed_proxies
aicir.qas.tests.test_cheap_eval_diagnostics.test_stratified_proxy_summary_reports_high_and_low_hit_groups
aicir.qas.tests.test_cheap_eval_diagnostics.test_summarize_proxy_diagnostics_includes_temporal_analysis
aicir.qas.tests.test_cheap_eval_diagnostics.test_summarize_proxy_diagnostics_marks_each_proxy
aicir.qas.tests.test_cheap_eval_diagnostics.test_temporal_analysis_reports_order_window_correlations
aicir.qas.tests.test_cheap_eval_diagnostics.test_top_k_enrichment_distinguishes_missing_proxy_recall_directions
aicir.qas.tests.test_cheap_eval_diagnostics.test_top_k_enrichment_measures_screening_usefulness
aicir.qas.tests.test_cheap_eval_diagnostics.test_top_k_enrichment_uses_architecture_identity_not_target_value
aicir.qas.tests.test_cheap_eval_diagnostics.test_warm_start_gain_cv_is_infinite_for_zero_mean_nonzero_variance
aicir.qas.tests.test_cheap_eval_diagnostics.test_warm_start_gain_summary_uses_random_minus_warm
aicir.qas.tests.test_cheap_eval_experiment.test_experiment_cli_writes_manifest_and_empty_csv
aicir.qas.tests.test_cheap_eval_experiment.test_experiment_schema_includes_required_proxy_fields
aicir.qas.tests.test_cheap_eval_experiment.test_light_vqe_registry_can_build_problem_from_literal_row_terms
aicir.qas.tests.test_cheap_eval_experiment.test_light_vqe_registry_passes_initial_parameters_for_warm_start
aicir.qas.tests.test_cheap_eval_experiment.test_light_vqe_registry_reuses_cached_architecture_and_problem
aicir.qas.tests.test_cheap_eval_experiment.test_light_vqe_registry_runs_e1_e2_and_fair_with_configured_budgets
aicir.qas.tests.test_cheap_eval_experiment.test_light_vqe_registry_uses_same_proxy_seed_set_for_e1_and_e2
aicir.qas.tests.test_cheap_eval_experiment.test_run_experiment_evaluator_results_do_not_overwrite_sampler_identity
aicir.qas.tests.test_cheap_eval_experiment.test_run_experiment_preserves_rows_when_evaluator_or_fair_runner_fails
aicir.qas.tests.test_cheap_eval_experiment.test_run_experiment_requires_configured_proxy_evaluators
aicir.qas.tests.test_cheap_eval_experiment.test_run_experiment_writes_rows_from_injected_evaluators
aicir.qas.tests.test_cheap_eval_experiment.test_write_empty_experiment_csv_creates_header
aicir.qas.tests.test_cheap_eval_experiment.test_write_experiment_manifest_records_e1_e5_protocol
```

### Syntax Compile

Command:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile vqe_loop\cheap_eval_diagnostics.py vqe_loop\cheap_eval_experiment.py tests\test_cheap_eval_diagnostics.py tests\test_cheap_eval_experiment.py vqe_loop\__init__.py
```

Result: passed with exit code 0.

## CLI Smoke Test

Input:

- `outputs/cheap_eval_diagnostics_smoke/smoke_rows.csv`
- `outputs/cheap_eval_diagnostics_smoke/cost_models.json`

Command:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m aicir.qas.vqe_loop.cheap_eval_diagnostics `
  --input aicir\qas\outputs\cheap_eval_diagnostics_smoke\smoke_rows.csv `
  --target fair_high `
  --proxies E1,E2,E5 `
  --output aicir\qas\outputs\cheap_eval_diagnostics_smoke\summary.json `
  --cost-models-file aicir\qas\outputs\cheap_eval_diagnostics_smoke\cost_models.json `
  --warm-fields fair_warm,fair_random `
  --strata-field hit_rate `
  --strata-threshold 0.5 `
  --temporal-order-field evaluation_order_index `
  --temporal-window-size 2 `
  --top-k 2 `
  --n-values 10,20,40,80
```

Result: passed with exit code 0.

Output:

- `outputs/cheap_eval_diagnostics_smoke/summary.json`

## Smoke Summary

The synthetic smoke data intentionally makes `E1` a bad proxy and `E2`/`E5` good proxies.

| Proxy | Spearman | Pairwise Accuracy | Fair Coverage (`top_k_recall`) | Proxy Purity (`fair_top_recall`) | Enrichment | Status |
|---|---:|---:|---:|---:|---:|---|
| E1 | -1.000 | 0.000 | 0.000 | 0.000 | -1.000 | repair |
| E2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | pass |
| E5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | pass |

Top-K overlap now uses `architecture_id` rather than target energy values, so duplicate fair energies cannot create false matches. The analyzer also emits `intersection_count`, `proxy_top_recall`, and `fair_top_recall`.

Warm-start summary:

```text
count=4
mean_gain=0.25
gain_variance=0.0225
std_gain=0.15
gain_cv_abs=0.60
```

The zero-mean nonzero-variance edge case is covered by tests and returns `gain_cv_abs=inf`.

Hit-rate stratification is present in `summary.json` under `strata.proxies`.

Temporal/order-window analysis is present in `summary.json` under `temporal.proxies`; the smoke run uses `evaluation_order_index` with two-row windows.

N-dependent cost from the smoke cost model:

| Proxy | N=10 amortized | N=20 amortized | N=40 amortized | N=80 amortized |
|---|---:|---:|---:|---:|
| E1 | 1.00 | 1.00 | 1.00 | 1.00 |
| E2 | 5.00 | 5.00 | 5.00 | 5.00 |
| E5 | 12.00 | 7.00 | 4.50 | 3.25 |

This confirms the intended behavior: the supernet-like proxy can be expensive at small `N` due to upfront cost, then cheaper amortized at larger `N`.

## Experiment Framework Smoke Test

Command:

```powershell
& 'C:\Users\lixin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m aicir.qas.vqe_loop.cheap_eval_experiment `
  --manifest aicir\qas\outputs\cheap_eval_diagnostics_smoke\experiment_manifest.json `
  --output-csv aicir\qas\outputs\cheap_eval_diagnostics_smoke\experiment_rows_template.csv `
  --benchmarks h2_4q,lih_12q,ising8_lattice `
  --depths 3,4 `
  --n-architectures 40 `
  --sampling-mode uniform
```

Result: passed with exit code 0.

Outputs:

- `outputs/cheap_eval_diagnostics_smoke/experiment_manifest.json`
- `outputs/cheap_eval_diagnostics_smoke/experiment_rows_template.csv`

The manifest standardizes:

- benchmark set: `h2_4q`, `lih_12q`, `ising8_lattice`
- fixed depths: `3`, `4`
- proxy fields: `E1`, `E2`, `E3`, `E4`, `E5`
- target field: `fair_high`

The row template includes:

- `E1`, `E2`, `E3`, `E4`, `E5`
- `E5_mean`, `E5_min`, `E5_std`
- `fair_high`, `fair_warm`, `fair_random`
- `hit_rate`, `exposure_count`, `evaluation_order_index`
- per-proxy walltime fields

`run_experiment()` is now an executable injectable runner. Its callable contract is:

- `architecture_sampler(config)` generates architecture rows
- `evaluator_registry` supplies E1-E5 evaluator callables
- `fair_vqe_runner` supplies high-budget fair labels

The runner writes rows using `EXPERIMENT_FIELDS`, fills `evaluation_order_index`,
records per-proxy walltime when the evaluator did not provide it, enforces that
all configured proxy evaluators exist before writing partial output, and caps
execution at `config.n_architectures`.

Runtime safety behavior is covered by tests:

- evaluator and fair-runner exceptions are recorded in `error_log` and do not
  stop later rows from being written
- each completed row is flushed to disk
- evaluator result dictionaries cannot overwrite existing non-empty sampler
  identity fields such as `architecture_id`, `depth`, or `hit_rate`
- `error_log` is part of `EXPERIMENT_FIELDS`

`build_light_vqe_evaluator_registry()` now provides a first real evaluator
bundle for the P0 experiment:

- `E1`: random-init light VQE with configurable low budget
- `E2`: random-init light VQE with configurable higher budget
- `fair_high`: high-budget fair VQE runner

The registry accepts an explicit `VQEProblem` or builds one from row-local
`hamiltonian_terms`. Tests inject a fake optimizer, so this verifies wiring,
budget/seed routing, architecture reconstruction, and row output without
running expensive COBYLA work.

Seed handling is intentionally paired: `E1` and `E2` use the same
`proxy_seed_offsets` for each architecture, so their comparison isolates budget
rather than initialization luck. Multiple proxy seeds are supported; the scalar
proxy field records the best energy across that shared seed set. `fair_high`
uses a separate `fair_seed_offsets` set.

The light-VQE path also caches parsed architecture/problem objects on the row
under private keys ignored by CSV writing, avoiding repeated JSON/gene parsing
across `E1`, `E2`, and `fair_high`. `initial_parameters` is passed through to
`optimize_vqe_energy`, so the same helper can later support warm-started
`E3`/`E4` evaluators.

## Interpretation

The analyzer now covers the P0 CLI gaps and the two review bugs:

- warm-start gain is included in JSON summaries when `--warm-fields` is provided
- hit-rate/exposure stratification is included when `--strata-field` and `--strata-threshold` are provided
- temporal/order-window diagnostics are included when `--temporal-order-field` is provided
- top-K overlap uses architecture identity, not fair-energy values
- zero-mean warm-start gain with nonzero variance is no longer mislabeled as stable

The experiment framework defines the schema and callable boundary needed to generate real E1-E5 rows. It can now execute injected evaluators and write completed diagnostic rows. The first real callable bundle covers `E1`, `E2`, and `fair_high`; `E3`, `E4`, and `E5` still require the supernet warm-start/native-screening state to be wired separately.

## Limitations

- This report does not run the expensive H2/LiH/Ising P0 benchmark.
- It validates the analyzer, CLI, experiment-output schema, and light-VQE evaluator wiring, not the actual quality of `E1`-`E5` on physical Hamiltonians.
- Cost model/walltime unification remains a later integration step.
- Supernet-backed `E3`, `E4`, and `E5` evaluators are not implemented in this report.
- Pytest runner output is unavailable until `pytest` is installed in the active Python environment.

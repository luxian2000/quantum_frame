# QAS Two Growth Routes Architecture

This is the human-readable contract for the two supported iterative QAS/VQE growth routes:

- Line A: Pauli operator-sequence search.
- Line B: chemistry-excitation search.

The executable defaults live in `vqe_loop/growth_routes.py`. This document mirrors those values. A route-specific CLI default is activated only when `--growth-route` is explicitly supplied; explicit CLI arguments override the route default.

Cheap evaluators are selectors only. Final labels always use the shared fair VQE/COBYLA path, and the primary metric is always `fair_best_energy`.

## Executable Route Configuration

```yaml
qas_growth_search:
  invariants:
    final_metric: fair_best_energy
    final_label_path: fair_labeling.py -> fair_vqe.py -> COBYLA
    cheap_metrics_are_selectors_only: true
    benchmark_schema: BENCHMARK_TABLE_FIELDS
    identity_scope:
      - Hamiltonian task
      - protocol_version
      - ansatz_family
      - canonical_architecture

  routes:
    line_a_operator_sequence:
      ansatz_family: operator_sequence

      p0:
        initializer: external_operator_fair_labels
        required_input: completed operator_sequence rows with fair_best_energy
        supported_exporter: demos/export_adaptvqe_explicit_seed.py
        e5_to_operator_adapter_implemented: false

      operator_pool:
        source: explicit_cli_or_hamiltonian_terms
        hamiltonian_order: descending_absolute_coefficient
        coefficient_abs_floor: 0.0
        include_identity: false
        max_pool_size: 128
        skip_existing_operators: true

      p1:
        parent_selection:
          metric: fair_best_energy
          parent_count: 4
          diversity_count: 0

        generation:
          children_per_parent: 8
          min_layers: 0
          max_layers: null
          genetic_weight: 0.5
          adapt_growth_weight: 0.5
          mutation_weights_within_genetic:
            operator_insert: 0.4
            operator_delete: 0.2
            operator_swap: 0.2
            operator_big_mutation: 0.2
          adapt_mutation: operator_adapt_growth
          parent_local_score_cache: true
          parent_energy_cache: true

        selection:
          selector: e5
          operator_e5_behavior: downgrade_to_e2
          baseline_selectors: [E2]
          fair_top_k: 4
          e2_max_evals: 250

        iteration:
          rounds: 5
          early_stop_epsilon: 1.0e-4
          early_stop_patience: 2
          max_total_fair_calls: 100
          stop_on_no_new_child: true

    line_b_chemistry_excitation:
      ansatz_family: chemistry_excitation

      p0:
        initializer: hf_empty_then_excitation_population
        first_candidate:
          hf_preparation: true
          excitations: []
          layers: 0
        remaining_candidates:
          pool: active_space_closed_shell
          include_spin_preserving_singles: true
          include_paired_doubles: true

      p1:
        parent_selection:
          metric: fair_best_energy
          parent_count: 8
          diversity_count: 0

        generation:
          children_per_parent: 16
          min_layers: 0
          max_layers: 32
          genetic_weight: 0.3
          adapt_growth_weight: 0.7
          mutation_weights_within_genetic:
            chemistry_insert: 0.4
            chemistry_delete: 0.15
            chemistry_swap: 0.15
            chemistry_change: 0.3
          adapt_mutation: chemistry_adapt_growth
          adapt_append_k: 4
          adapt_pool_limit: 24
          append_semantics: batch_top_k_from_one_parent_scan
          preserve_single_and_double_candidates_under_limit: true
          skip_existing_excitations: true
          parent_local_score_cache: true
          parent_energy_cache: true

        selection:
          selector: e2
          baseline_selectors: [E2]
          fair_top_k: 8
          e2_max_evals: 250
          task_proxy_enabled: false
          graph_proxy_enabled: false

        iteration:
          rounds: 6
          early_stop_epsilon: 1.0e-4
          early_stop_patience: 3
          max_total_fair_calls: 100
          stop_on_no_new_child: true
```

## Route Data Flow

```text
route-specific P0 fair labels
  -> filter one ansatz family and one Hamiltonian task
  -> select parents by fair_best_energy
  -> generate route-specific children
  -> clear inherited fair/proxy/selector measurements
  -> deduplicate by task + protocol + family + canonical architecture
  -> oracle or configured cheap selector
  -> shared fair-label queue
  -> fair_labeling.py / COBYLA
  -> append completed labels to the unified benchmark table
  -> repeat until a configured stop condition
```

Failed fair-label attempts consume fair budget. The final round is reduced to the remaining budget instead of submitting a full batch beyond `max_total_fair_calls`.

## Line A Details

Line A searches `OperatorSequenceAnsatzGene` rows. Each layer is one parameterized Pauli evolution. X/Y basis changes use canonical `hadamard` and `s_gate` instructions so the same exact-statevector fair path can evaluate them.

The current P0 boundary is explicit: Line A starts from an external CSV containing completed, task-matched `operator_sequence` fair labels. `demos/export_adaptvqe_explicit_seed.py` can create such a seed row from an external ADAPT result. The previously proposed E5-supernet-to-operator adapter is not implemented and is therefore not presented as an active route.

When no explicit pool is supplied, Line A takes nonidentity Hamiltonian Pauli terms, sorts them by absolute coefficient, and keeps at most 128. The ADAPT scorer evaluates the parent energy once and each candidate at plus/minus finite difference once. Repeated children from the same parent reuse those scores.

E5 is not a native operator-sequence evaluator. Selecting E5 for Line A deterministically downgrades the child prescreen to shared low-budget E2. The baseline list contains only E2, preventing duplicate E2/E5 fair-label queues.

## Line B Details

Line B searches `ChemistryExcitationAnsatzGene` rows. P0 always places the HF-empty architecture first, followed by bounded single/double-excitation candidates.

`adapt_append_k=4` is deliberately a batch top-k operation: the candidate pool is scored once against the same parent and the best four novel excitations are appended together. It is not four sequential ADAPT steps with gradient recomputation after each append. The pool limit keeps both single and double excitation classes.

Task and graph proxies are currently disabled for chemistry rows because their chemistry-specific feature encoders are not implemented. Attempting to select them is rejected instead of silently producing all-zero ties.

## CH4 18-Qubit Runner

`demos/run_ch4_18q_lineb_npu4.sh` imports the Line B defaults from `growth_routes.py`; environment variables remain explicit overrides.

```yaml
ch4_18q_runner:
  active_electrons: 10
  active_spatial_orbitals: 9
  p0_candidates: 64
  p0_max_excitations: 6
  p1_route: line_b_chemistry_excitation
  backend: npu
  dtype: complex64
  num_shards: 4
  fair_label_n_seeds: 3
  fair_label_max_evals: 1000
  success_delta_ref: 0.02
```

`BACKEND` and `DTYPE` are passed to P1 E2, ADAPT growth, and final fair labeling. For resumed runs, `current_labeled_rows.csv` is reused and P0 is not rerun.

## Stop Conditions

Both routes stop when any of these conditions is met:

```text
configured round count reached
remaining fair-call budget is zero
fair_best_energy improvement stays below epsilon for patience rounds
no novel, nonduplicate child can be generated
no candidate survives the configured prescreen
```

Early stopping is initialized from the P0/bootstrap global best `fair_best_energy`, not from the first P1 result.

## CH4 Energy Scope

The 12-qubit active-space CH4 run and original 18-qubit CH4 Hamiltonian are different tasks and must not be compared by absolute energy.

- 12q active-space reference: approximately `-17.316821650897 Ha`.
- 18q electronic reference: `-53.276955083160 Ha`.
- 18q total energy including nuclear repulsion: approximately `-39.805693225904 Ha`.

# QAS Two Growth Routes Architecture

This document is the editable architecture and configuration surface for two iterative QAS/VQE growth routes:

- Line A: operator-sequence search with genetic mutation plus ADAPT growth.
- Line B: chemistry-excitation search with genetic mutation plus ADAPT growth.

Cheap evaluators are selectors only. Final comparison uses shared fair COBYLA labels and the primary metric remains `fair_best_energy`.

## Editable Architecture Configuration

The YAML block is organized by execution architecture. First choose `global.active_route`, then configure that route's P0, P1 generation, selector, iteration, and stop conditions. Each selectable method has `enabled`, `weight`, and `parameters` so the document can become a machine-readable algorithm config.

```yaml
qas_growth_search:
  global:
    active_route: line_a_operator_sequence
    route_options:
      - line_a_operator_sequence
      - line_b_chemistry_excitation
    enable_fair_label: true
    fair_metric: fair_best_energy
    seed: 42

  routes:
    line_a_operator_sequence:
      ansatz_family: operator_sequence

      p0:
        enabled: true
        purpose: initial_population
        initial_state_method: e5_seeded_operator_pool
        initial_state_method_options:
          empty:
            enabled: false
            weight: 0.0
            parameters:
              initial_operators: 0
          random_pool:
            enabled: false
            weight: 0.0
            parameters:
              initial_min_operators: 1
              initial_max_operators: 3
              pool_source: operator_pool
          from_operator_p0_labels:
            enabled: false
            weight: 0.0
            parameters:
              parent_metric: fair_best_energy
              parent_count: 8
          e5_seeded_operator_pool:
            enabled: true
            weight: 1.0
            parameters:
              supernet_candidate_count: 32
              e5_seed_count: 8
              selector: E5
              adapter: hamiltonian_guided_pauli_projection
              initial_min_operators: 1
              initial_max_operators: 3

        operator_pool:
          method: hamiltonian_guided
          method_options:
            explicit:
              enabled: false
              weight: 0.0
              parameters:
                operators: []
            default_pauli_local:
              enabled: true
              weight: 0.2
              parameters:
                include_single_qubit: true
                include_two_qubit_same_axis: true
            hamiltonian_guided:
              enabled: true
              weight: 0.6
              parameters:
                coefficient_abs_floor: 0.0
                include_hamiltonian_terms: true
                include_commutator_neighbors: true
            qiskit_like_excitation_mapped:
              enabled: true
              weight: 0.1
              parameters:
                mapping: jordan_wigner
                include_singles: true
                include_doubles: true
            pennylane_like_operator_pool:
              enabled: true
              weight: 0.1
              parameters:
                include_spin_adapted_like_terms: true
                mapping: jordan_wigner
          max_pool_size: 128

      p1:
        enabled: true
        parent_selection:
          metric: fair_best_energy
          parent_count: 4
          diversity_count: 0

        generation:
          mode: mixed
          method_options:
            genetic_mutation:
              enabled: true
              weight: 0.5
              mutation_weights:
                operator_insert: 0.4
                operator_delete: 0.2
                operator_swap: 0.2
                operator_big_mutation: 0.2
              parameters:
                min_layers: 0
                max_layers: null
            adapt_growth:
              enabled: true
              weight: 0.5
              mutation_type: operator_adapt_growth
              parameters:
                pool_source: p0.operator_pool
                skip_existing_operators: true
                gradient_proxy_weight: 1.0
                energy_drop_weight: 1.0
          children_per_parent: 8

        selection:
          selector: e5
          selector_options:
            e2:
              enabled: true
              weight: 1.0
              parameters:
                max_evals: 250
            e5:
              enabled: true
              weight: 1.0
              parameters:
                usage: initial_state_only_or_downgrade
                downgrade_for_operator_rows: e2
            task_proxy:
              enabled: false
              weight: 0.0
              parameters:
                require_family_support: true
            graph_predictor:
              enabled: false
              weight: 0.0
              parameters:
                require_family_support: true
            ensemble:
              enabled: false
              weight: 0.0
              parameters:
                aggregate_available_signals_only: true
          fair_top_k: 4

        iteration:
          rounds: 5
          stop_conditions:
            early_stop_epsilon: 1.0e-4
            early_stop_patience: 2
            max_total_fair_calls: 100
            stop_on_no_new_child: true

    line_b_chemistry_excitation:
      ansatz_family: chemistry_excitation

      p0:
        enabled: true
        purpose: initial_population
        initial_state_method: hf_empty
        initial_state_method_options:
          hf_empty:
            enabled: true
            weight: 1.0
            parameters:
              initial_excitations: 0
          random_excitation:
            enabled: false
            weight: 0.0
            parameters:
              initial_min_excitations: 1
              initial_max_excitations: 2
              pool_source: excitation_pool
          from_chemistry_p0_labels:
            enabled: false
            weight: 0.0
            parameters:
              parent_metric: fair_best_energy
              parent_count: 8

        excitation_pool:
          method: active_space_closed_shell
          method_options:
            active_space_closed_shell:
              enabled: true
              weight: 1.0
              parameters:
                active_electrons: null
                active_spatial_orbitals: null
                include_single_excitation: true
                include_double_excitation: true

      p1:
        enabled: true
        parent_selection:
          metric: fair_best_energy
          parent_count: 8
          diversity_count: 0

        generation:
          mode: mixed
          method_options:
            genetic_mutation:
              enabled: true
              weight: 0.3
              mutation_weights:
                chemistry_insert: 0.4
                chemistry_delete: 0.15
                chemistry_swap: 0.15
                chemistry_change: 0.3
              parameters:
                pool_source: p0.excitation_pool
                min_layers: 0
                max_layers: null
            adapt_growth:
              enabled: true
              weight: 0.7
              mutation_type: chemistry_adapt_growth
              parameters:
                pool_source: p0.excitation_pool
                skip_existing_excitations: true
                gradient_proxy_weight: 1.0
                energy_drop_weight: 1.0
          children_per_parent: 16

        selection:
          selector: e2
          selector_options:
            e2:
              enabled: true
              weight: 1.0
              parameters:
                max_evals: 250
            e5:
              enabled: false
              weight: 0.0
              parameters:
                downgrade_for_chemistry_rows: e2
            task_proxy:
              enabled: false
              weight: 0.0
              parameters:
                require_chemistry_aware_features: true
            graph_predictor:
              enabled: false
              weight: 0.0
              parameters:
                require_family_aware_features: true
            ensemble:
              enabled: false
              weight: 0.0
              parameters:
                aggregate_available_signals_only: true
          fair_top_k: 8

        iteration:
          rounds: 10
          stop_conditions:
            early_stop_epsilon: 1.0e-4
            early_stop_patience: 3
            max_total_fair_calls: 100
            stop_on_no_new_child: true
```

The intended execution order is:

```text
global.active_route
  -> routes[active_route].p0
       -> choose initial-state method
       -> build route-specific pool
       -> produce initial candidates or parents
       -> optional initial selector/fair labels
  -> routes[active_route].p1
       -> select parents by fair_best_energy
       -> generate children by weighted methods
       -> run configured selector as cheap prescreen
       -> label selected rows through fair_labeling.py
       -> repeat until iteration stop condition
```

## Line A: OperatorSequence Route

Line A searches `OperatorSequenceAnsatzGene` architectures. Its native search object is a sequence of Pauli evolutions, not a supernet mask.

### Initial State

Line A can include an E5-based initial state, but E5 should be treated as an initializer or selector over native supernet candidates, not as a final evaluator for operator-sequence rows.

The recommended E5 initial-state path is:

```text
native supernet candidates
  -> E5 screening
  -> select top E5 candidates
  -> derive Hamiltonian-guided Pauli operator seeds
  -> create OperatorSequenceAnsatzGene seed population
```

This needs an explicit adapter because native supernet architectures and Pauli operator sequences are different ansatz families. The adapter should be named and documented, for example `hamiltonian_guided_pauli_projection`, so that the semantic conversion is visible.

### Generator

The generator first samples between genetic mutation and ADAPT growth using the weights under `p1.generation.method_options`. If genetic is selected, it samples one of `operator_insert`, `operator_delete`, `operator_swap`, or `operator_big_mutation`. If ADAPT growth is selected, it runs `operator_adapt_growth`.

### ADAPT Pool

The preferred pool should not be random. It should be Hamiltonian-guided, similar in spirit to Qiskit or PennyLane ADAPT pools:

- Parse Hamiltonian Pauli terms from the task.
- Keep non-identity Pauli supports with coefficients above a configured floor.
- Add symmetry/commutator-inspired nearby Pauli terms when available.
- Optionally cap the pool with `max_pool_size`.
- Fall back to local one- and two-body Pauli terms only when Hamiltonian terms are unavailable.

Explicit user-provided pools remain useful for smoke tests and ablations.

### Selector And E5

E5 can be used before Line A to produce better seeds, but final `OperatorSequenceAnsatzGene` rows are not native-supernet rows. Therefore:

```text
E5 for Line A initial state: allowed
E5 for final operator-sequence row scoring: unsupported, downgrade/guard
```

The final comparison still goes through fair labels.

## Line B: ChemistryExcitation Route

Line B searches `ChemistryExcitationAnsatzGene` architectures. Its pool is always based on active-space chemistry excitations:

```text
single_excitation
double_excitation
```

Both chemistry genetic mutation and chemistry ADAPT growth use this same pool.

### Initial State

The default initial state is `hf_empty`: Hartree-Fock preparation with zero excitations. Other allowed modes are random excitation seeds and existing chemistry P0 labels.

### Generator

The generator first samples between chemistry genetic mutation and chemistry ADAPT growth using the weights under `p1.generation.method_options`. If genetic is selected, it samples one of `chemistry_insert`, `chemistry_delete`, `chemistry_swap`, or `chemistry_change`. If ADAPT growth is selected, it runs `chemistry_adapt_growth`.

### ADAPT Pool

The chemistry ADAPT pool is not Pauli-string based. It is generated from:

```text
closed_shell_excitation_pools(active_electrons, active_spatial_orbitals)
```

The growth step scans all configured single and double excitations, scores each candidate by gradient proxy and/or energy drop, skips existing excitations when configured, and appends the best chemistry excitation.

## Iteration And Stop Conditions

Both lines share the same stopping rules, configured per route under `p1.iteration`:

```text
stop if rounds >= configured rounds
stop if total fair calls >= max_total_fair_calls
stop if best fair_best_energy improves by less than early_stop_epsilon for early_stop_patience rounds
stop if no new non-duplicate child can be generated
```

Only `fair_best_energy` can drive convergence and final comparison. Proxy scores may influence selection, but they are not final winner metrics.

## CH4 Energy Scope

The 12-qubit active-space CH4 run and the original 18-qubit CH4 demo are different Hamiltonians. Do not compare their absolute energies directly.

- 12q active-space CH4 reference: about `-17.316821650897 Ha`.
- Original 18q CH4 electronic energy: about `-53.276955083160 Ha`.
- Original 18q CH4 total with nuclear repulsion: about `-39.805693225904 Ha`.

# Chemistry ADAPT Growth Cache Design

## Goal

Remove deterministic repeated energy evaluations from P1 chemistry ADAPT growth without changing child generation, selector ranking, or fair-label semantics.

## Context

For each parent, `generate_mutation_children()` may choose `chemistry_adapt_growth` several times. Every choice scans the same limited excitation pool. With a pool limit of 24, each scan currently performs 72 energy evaluations: the parent at its prefix parameters and each trial excitation at `+epsilon` and `-epsilon`. The parent evaluation is repeated for every candidate, and the complete scan is repeated for every ADAPT child generated from the same parent.

## Design

Use two in-memory caches whose lifetime is limited to one parent during one `generate_mutation_children()` call.

1. Wrap the parent's chemistry growth evaluator with a score cache keyed by the canonical parent gene and canonical excitation. Repeated ADAPT mutations for the same parent return the previously computed score.
2. Let `_chemistry_growth_evaluator_from_row()` keep an energy cache keyed by architecture identity and parameter tuple. During the first pool scan, the common parent energy is evaluated once while each distinct trial still evaluates both finite-difference points.

The caches are not persisted and are not shared across parents, rounds, Hamiltonians, or processes. Exceptions retain the existing behavior: a failed finite-difference score becomes positive infinity.

## Behavioral Guarantees

- Candidate ordering and the finite-difference score formula are unchanged.
- Mutation weights, generated child rows, and deterministic seeds are unchanged.
- Proxy scores remain selector-only values.
- Final comparison continues through shared fair labeling and `fair_best_energy`.

## Verification

- A regression test proves repeated ADAPT children score each candidate only once per parent.
- A regression test proves two distinct candidates share one parent/base energy evaluation.
- Existing P1 evolution, P1 round, and demo tests remain green.


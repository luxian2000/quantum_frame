# QAS Project Instructions

This repository should use the Superpowers workflow for all coding-agent work.

## Superpowers

- At the start of each task, check whether a Superpowers skill applies.
- For design or feature work, use `brainstorming` before implementation.
- For bug fixes and refactors, use `systematic-debugging` or `test-driven-development` when applicable.
- Before claiming work is complete, use `verification-before-completion` and report the actual verification commands.
- Prefer small, reviewed changes over broad rewrites. Keep implementation aligned with the existing QAS architecture.

## QAS Constraints

- The primary comparison metric is `fair_best_energy`.
- Cheap evaluators are selectors only. Do not use proxy energy as the final winner metric.
- Fair labels must go through the shared `labeling.py` + COBYLA protocol.
- P0/P1 outputs should continue to feed the unified benchmark table schema.
- Chemistry excitation, native supernet, task proxy, graph proxy, and zero-cost paths should remain ansatz-family aware.

## Cleanup Rules

- Delete unused legacy code only after checking references with `rg`.
- Prefer shared row/schema helpers over duplicated CSV or JSON parsing.
- Do not collapse ansatz-family-specific logic into generic helpers when the semantics differ.
- When touching P0/P1/fair paths, run focused tests for the affected path before reporting completion.

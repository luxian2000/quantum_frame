# Supernet: use aicir gates instead of local gate definitions

**Date:** 2026-06-29
**Status:** Approved design, ready for implementation plan
**Scope:** `aicir/qas/algorithms/supernet.py` (single-file internal refactor)

## Goal

Remove supernet's locally-defined gate layer so the supernet QAS module
**defines no gates of its own**. Gate construction must come from the
`aicir.core` gate factories, and gate metadata (parameter counts, name
validity) must come from the `aicir.gates` registry.

Today [`supernet.py`](../../../aicir/qas/algorithms/supernet.py) declares its
own `GateSpec` dataclass and two module-level builder dicts
(`_SINGLE_QUBIT_GATES`, `_TWO_QUBIT_GATES`) that duplicate gate metadata
(arity, parameter count) and wrap the aicir factory functions. This is the
"local gate definition" we are eliminating.

## Hard invariant

This is a **behavior-preserving internal refactor**. The refactored code MUST
emit byte-for-byte identical circuits:

- same gate dicts (same `type` strings, same qubit fields, same params),
- same number and identity of trainable parameters,
- same RNG consumption during architecture sampling,
- existing `tests/qas/` suite stays green with no test changes (other than
  added characterization tests).

It is plumbing, not numerics. The gate matrices already live entirely in
`aicir.core.gates.gate_to_matrix` / `apply_gate_to_state`; supernet's local
`GateSpec` never defined gate semantics, only wrapped aicir factories.

## Background facts (verified)

- A gate in aicir is a plain dict, e.g. `{"type": "rx", "params": [theta],
  "target_qubit": 0}`. Factory functions in `aicir/core/circuit.py` (`rx`,
  `ry`, `rz`, `hadamard`, `cx`, `rzz`) are thin constructors returning such
  dicts.
- The name to matrix logic is a hardcoded `if/elif gate_type == ...` dispatch
  inside `gate_to_matrix` / `apply_gate_to_state`, after normalizing the name
  via `aicir.gates.canonical_gate_name`. (Making this data-driven is NEXT.md
  §7's deferred `matrix`-field migration — explicitly OUT OF SCOPE here.)
- `aicir.gates` registry holds **metadata only** (`name`, `num_qubits`,
  `num_params`, `aliases`, `controlled`, `qasm_name`, `symbol`, `generator`,
  `decomposition`). No builder, no matrix.
- Registry metadata for supernet's gates:
  - `rx`/`ry`/`rz` → `num_qubits=1, num_params=1, generator=X/Y/Z`
  - `rzz` → `num_qubits=2, num_params=1, generator=ZZ`
  - `hadamard` → `num_qubits=1, num_params=0`
  - `cx` → `num_qubits=None, num_params=0, controlled=True`
  - `identity` → `num_qubits=None, num_params=0`
  - The short tokens `"i"` and `"h"` are **not** registry names/aliases.
- supernet's current builders already emit pure aicir gate dicts; `"i"`
  emits nothing (a search-space "skip this slot" sentinel), `"h"` builds
  `hadamard(q)`.

## Design

### Search tokens (unchanged public surface)

Keep the searchable token alphabets exactly as today so config/back-compat is
preserved:

- single-qubit: `("i", "h", "rx", "ry", "rz")`
- two-qubit: `("cx", "rzz")`

These are **search-space tokens** (the `SupernetConfig.single_qubit_gates` /
`two_qubit_gates` public surface), not gate definitions.

### Token → aicir mapping

Add a token-to-canonical map:

```python
_TOKEN_CANONICAL = {
    "h": "hadamard",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "cx": "cx", "rzz": "rzz",
}
```

`"i"` is intentionally absent: it represents *no gate* (emit nothing), a
search "skip" sentinel — not a locally-defined identity gate.

### Construction

Replace the `GateSpec.builder` lambdas with dispatch to the **aicir.core
factory functions** supernet already imports (`hadamard`, `rx`, `ry`, `rz`,
`cx`, `rzz`):

- single-qubit token (non-`i`): call the matching factory with the active
  parameter (if any) and the target qubit; `"i"` → emit nothing.
- two-qubit token: `cx` → `cx(target_qubit=t, control_qubits=[c])`; `rzz` →
  `rzz(theta, qubit_1=c, qubit_2=t)`.

The emitted dicts are identical to today's output.

### Parameter counts from the registry

The number of trainable angles per gate is sourced from
`get_gate_spec(canonical).num_params` (registry = single source of truth);
`"i"` contributes 0. This removes the duplicated `n_params` metadata that the
local `GateSpec` carried.

### Validation

`_normalize_single_gate` / `_normalize_two_qubit_gate` change from "is this
key in my local builder dict?" to:

- token is in the searchable alphabet, **and**
- for non-`"i"` tokens, `get_gate_spec(_TOKEN_CANONICAL[token]) is not None`
  (the registry confirms it is a real aicir gate).

Error behavior for unknown tokens is preserved (`ValueError`).

### Accepted local residue (honest scope note)

A small amount of **construction wiring** remains local: which factory and
calling convention applies per arity class — single-target vs. control+target
(`cx`) vs. symmetric pair (`rzz`) — because the registry's
`num_qubits=None`/`controlled` fields cannot fully drive dict assembly. This
is dispatch to aicir's own constructors, **not** a gate definition: no
matrices, no arity/param metadata are hardcoded locally. "No local gates" is
met in both letter and spirit — the eliminated thing is the duplicate
metadata + semantics wrapper, which is fully gone.

## Affected call sites within supernet.py

The local `GateSpec` / builder dicts are consumed in:

- `_ensure_architecture_params` — reads `spec.n_params` to lazily create
  shared parameters.
- `build_circuit` — reads `spec.n_params` and calls `spec.builder(...)`.

Both are rewritten to use the registry param-count and the factory dispatch.
No change to parameter-key structure, weight-sharing, sharding, or autograd
paths.

## Testing

1. Establish a green baseline: run `tests/qas/test_supernet*.py` before any
   change.
2. Add a **characterization test**: for a fixed `Architecture` and seed,
   assert the emitted gate-dict sequence (types, qubit fields, param
   positions) is unchanged, and that every emitted `type` string is an aicir
   canonical gate name.
3. Refactor; confirm the full `tests/qas/` suite stays green.

No behavior changes, so no new behavioral tests are required beyond
characterization.

## Out of scope (decoupled follow-on work)

- NEXT.md §7 `GateSpec.matrix` migration (high blast radius; deferred).
- Adding a particle-conserving Givens / excitation gate.
- HF-reference initial state for the supernet VQE path.
- Opening the search pool to arbitrary registered gates / user-defined pools.

These were explicitly deferred during design to keep this change low-risk.

## References

- `aicir/qas/algorithms/supernet.py` — target file
- `aicir/gates/spec.py`, `aicir/gates/registry.py` — metadata source
- `aicir/core/circuit.py` — factory functions
- `aicir/core/gates.py` — `gate_to_matrix` / `apply_gate_to_state`
- `NEXT.md` §1.3 (QAS Target/GateSpec integration), §7 (GateSpec registry)

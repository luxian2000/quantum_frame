# Supernet: use aicir gates instead of local gate definitions

**Date:** 2026-06-29
**Status:** Approved design, ready for implementation plan
**Scope:** `aicir/qas/algorithms/supernet.py` (internal refactor) plus a small
additive metadata field in `aicir/gates` (`GateSpec.num_controls`) so the
controlled-gate convention lives as registry data.

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

To let supernet source the controlled-gate convention from data (rather than
a local "cx = 1 control" assumption), the design also adds one small piece of
metadata — `GateSpec.num_controls` — to the shared `aicir.gates` registry.

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

- Factory functions in `aicir/core/circuit.py` (`rx`, `ry`, `rz`, `hadamard`,
  `cx`, `rzz`) return typed `Operation`s; `Circuit` normalizes them to its
  internal gate-dict form (e.g. `{"type": "rx", "params": [theta],
  "target_qubit": 0}`). supernet consumes the gate-dict form.
- The name to matrix logic is a hardcoded `if/elif gate_type == ...` dispatch
  inside `gate_to_matrix` / `apply_gate_to_state`, after normalizing the name
  via `aicir.gates.canonical_gate_name`. (Making this data-driven is NEXT.md
  §7's deferred `matrix`-field migration — explicitly OUT OF SCOPE here.)
- `aicir.gates` registry holds **metadata only** (`name`, `num_qubits`,
  `num_params`, `aliases`, `controlled`, `qasm_name`, `symbol`, `generator`,
  `decomposition`). No builder, no matrix, and **no control count**.
- The aicir factories return typed `Operation`s with **uniform fields**
  (`qubits`, `controls`, `params`) — there is no `target_qubit`-vs-`qubit_1`
  field-name divergence to worry about. Construction is fully covered by
  deciding, for the qubits supernet supplies, which are targets (`qubits=`)
  and which are controls (`controls=`), plus how many params.
- Registry metadata for supernet's gates:
  - `rx`/`ry`/`rz` → `num_qubits=1, num_params=1, controlled=False, generator=X/Y/Z`
  - `rzz` → `num_qubits=2, num_params=1, controlled=False, generator=ZZ`
  - `hadamard` → `num_qubits=1, num_params=0, controlled=False`
  - `cx` → `num_qubits=None, num_params=0, controlled=True`
  - `identity` → `num_qubits=None, num_params=0`
  - The short tokens `"i"` and `"h"` are **not** registry names/aliases.
- Why `cx` has `num_qubits=None`: its target count is genuinely variable —
  aicir's `cx` accepts a single target *or* multiple targets (multi-target CX
  ≡ several single-target CXs sharing one control). The other controlled
  gates (`cy`, `cz`, `crx`/`cry`/`crz`, `toffoli`) fix `num_qubits=1`.
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

### Registry change: encode the control count as data (item 2)

Add a new optional field to `aicir.gates.GateSpec`:

```python
num_controls: int = 0   # number of control qubits the gate carries
```

Populate it for the controlled gates in the registry:

| gate | num_controls |
|---|---|
| `cx` | 1 |
| `cy`, `cz` | 1 |
| `crx`, `cry`, `crz` | 1 |
| `toffoli` | 2 |
| everything else | 0 (default) |

This is **additive and non-breaking**: the new field defaults to `0`, all
existing positional `GateSpec(...)` registrations and existing consumers are
unaffected (the existing `controlled` boolean is retained; it now equals
`num_controls > 0`). It is metadata in the same spirit as `num_qubits` /
`generator` that already landed under NEXT.md §7 — **not** the deferred
`matrix` migration.

With `num_controls` present, the control/target split of a gate is now
**registry data**, not a hardcoded supernet constant.

### Construction

Replace the `GateSpec.builder` lambdas with construction driven by registry
metadata, dispatched to the **aicir.core factory functions** supernet already
imports (`hadamard`, `rx`, `ry`, `rz`, `cx`, `rzz`):

- For a token, look up its canonical name and read `num_controls` /
  `num_params` from the registry.
- Split the qubits supernet supplies for the slot using `num_controls`
  (supernet's two-qubit pair is ordered `(control, target)`, so the first
  `num_controls` qubits are controls, the rest targets; single-qubit slots
  have `num_controls=0`).
- Call the matching aicir factory with that split and the active parameters;
  `"i"` → emit nothing.

The factory is still used for actual construction (it encodes defaults such
as `control_states` all-ones), which keeps output **byte-identical** to today
while the *qubit-role split* now comes from registry data rather than a
supernet-local "cx = 1 control" assumption. The emitted dicts are identical
to today's output.

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

In the *current* registry there are **two** distinct reasons it cannot fully
drive `cx` construction:

1. **Variable target count** — `cx` has `num_qubits=None` because its arity is
   genuinely unfixed (single- or multi-target). So the target count is not
   readable from the registry.
2. **Control split not encoded** — `controlled` is only a boolean; it says the
   gate *has* controls, not *how many*, nor how to split a supplied qubit
   tuple into controls vs. targets. This "1 control + 1 target" convention
   lived only in the `cx` factory and `gate_to_matrix`, not as data.

Item 2 of this design **closes reason 2** by adding `num_controls` to the
registry. After it, the control/target split is registry-driven. Reason 1
remains (cx's arity is legitimately variable), but it is harmless here:
supernet uses the single-target form, so "targets = the slot's qubits minus
the `num_controls` controls" resolves it without any cx-specific constant.

What is left **local to supernet** is therefore only:

- the **search-space tokens** `i` (no-op skip) and `h` (→ `hadamard`), which
  are search alphabet, not gate definitions; and
- **dispatch to aicir's own factory functions** for construction (because the
  registry exposes no constructor — adding one is the deferred §7 work).

No gate matrices, arities, parameter counts, or control conventions are
defined locally anymore — those are all sourced from `aicir.gates` /
`aicir.core`. "No local gates" is met in letter and spirit.

## Affected call sites

In `aicir/gates/` (item 2):

- `spec.py` — add the `num_controls` field to `GateSpec`.
- `registry.py` — set `num_controls` on `cx`/`cy`/`cz`/`crx`/`cry`/`crz`/
  `toffoli`.

In `aicir/qas/algorithms/supernet.py`, the local `GateSpec` / builder dicts
are consumed in:

- `_ensure_architecture_params` — reads `spec.n_params` to lazily create
  shared parameters.
- `build_circuit` — reads `spec.n_params` and calls `spec.builder(...)`.

Both are rewritten to use the registry param-count / `num_controls` split and
the factory dispatch. No change to parameter-key structure, weight-sharing,
sharding, or autograd paths.

## Testing

1. Establish a green baseline: run `tests/qas/test_supernet*.py` before any
   change.
2. Add a registry test (`tests/gates/`): `num_controls` defaults to `0`,
   equals the expected count for each controlled gate, and stays consistent
   with `controlled` (`controlled == num_controls > 0`).
3. Add a **characterization test** (`tests/qas/`): for a fixed `Architecture`
   and seed, assert the emitted gate-dict sequence (types, qubit fields, param
   positions, `control_states`) is byte-identical to the pre-refactor output,
   and that every emitted `type` string is an aicir canonical gate name.
4. Refactor; confirm the full `tests/qas/` and `tests/gates/` suites stay
   green.

No behavior changes, so no new behavioral tests are required beyond the
registry-field and characterization tests.

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

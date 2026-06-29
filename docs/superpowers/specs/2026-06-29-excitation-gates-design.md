# Particle-conserving excitation gates (single + double) — Design

**Date:** 2026-06-29
**Status:** Approved design, ready for implementation plan
**Scope:** add two general, reusable parametric gates to the gate subsystem, a
correct four-term parameter-shift rule for them, and an H2 VQE validation.

## Goal

Add native, particle-number-conserving **single-excitation** and
**double-excitation** gates to `aicir` as first-class gate-subsystem citizens
(beside `rxx`/`rzz`), plus the correct four-term parameter-shift rule they
require, and validate that they create genuine electron correlation by reaching
chemical accuracy on H2 via direct VQE.

These gates are **general** (not chemistry-tagged): they are reusable by any
caller. This spec deliberately does **not** touch supernet, does not add a
chemistry HF helper, and does not wire the gates into any search pool — those
are separate, decoupled follow-ons.

## Motivation (why these gates)

A circuit of only single-excitation (orbital-rotation) gates maps one Slater
determinant to another single determinant (Thouless' theorem) — it cannot
create a superposition of determinants, i.e. no electron correlation. Genuine
correlation (e.g. H2's `HF + double excitation`) requires a **double-excitation**
gate. Both gates conserve particle number (commute with `N = Σ(I−Z_i)/2`), so
they keep a state in its electron-number sector — unlike `rx`/`ry`/`h`/`cx`.

## Background facts (verified)

- Gates are dicts; factories in `aicir/core/circuit.py` build them; matrices are
  produced by a hardcoded name dispatch in `aicir/core/gates.py`
  (`gate_to_matrix` / `apply_gate_to_state`). NEXT.md §7's `GateSpec.matrix`
  migration is deferred and OUT OF SCOPE.
- `rxx`/`rzz` are the template: numpy matrix builder (`_rxx`), graph-preserving
  torch kernel (`_rxx_backend`, which builds each grad-bearing complex cell as a
  fresh tensor so Ascend NPU's real-only autograd add works), factory in
  `circuit.py`, `GateSpec` in `registry.py`, top-level re-export in `__init__.py`.
- `_apply_local_matrix_to_state(state, matrix, axes, n_qubits, backend)`
  (`aicir/core/gates.py:678`) applies a `2^k×2^k` matrix on an arbitrary qubit
  subset `axes` (handles non-adjacent qubits and the NPU flat path). This is what
  makes a native **4-qubit** gate feasible; the generic `unitary` gate type only
  places on qubits `0..k-1` (`gates.py:956`) and is therefore unusable here.
- `aicir.gates` registry already drives qml self-inspection: `generator` +
  `parametric_pauli_gates()` decide which gates are standard 2-term-PSR
  differentiable. A gate with `generator=None` is correctly NOT treated as a
  standard-Pauli rotation.
- `aicir.chemistry` ships an `h2_jw` preset (4-qubit Jordan-Wigner H2) used by
  the validation.

## Design

### Unit 1 — `single_excitation` gate (alias `givens`)

- 2-qubit, 1 parameter. Real Givens rotation on the `{|01⟩,|10⟩}` subspace,
  identity on `|00⟩`/`|11⟩`:

  ```
  [[1,        0,         0,        0],
   [0,  cos(θ/2), -sin(θ/2),       0],
   [0,  sin(θ/2),  cos(θ/2),       0],
   [0,        0,         0,        1]]
  ```

- Particle-conserving (block-diagonal by Hamming weight).
- `GateSpec("single_excitation", num_qubits=2, num_params=1, aliases=("givens",), qasm_name=None, generator=None, shift_rule="four_term")`. `qasm_name=None`: OpenQASM has no standard excitation gate; QASM export is out of scope (a caller that needs it can decompose first).

### Unit 2 — `double_excitation` gate

- 4-qubit, 1 parameter. 16×16 identity except the `{|0011⟩,|1100⟩}` 2×2 block,
  which rotates by `cos(θ/2)` on the diagonal and `∓sin(θ/2)` off-diagonal
  (same Givens form, sign convention fixed and verified in tests). All other
  basis states are fixed.
- Particle-conserving.
- Applied on the 4 chosen qubits via `_apply_local_matrix_to_state`.
- `GateSpec("double_excitation", num_qubits=4, num_params=1, generator=None, shift_rule="four_term")`.

### Touch-points for each gate (mirror `rxx`)

- `aicir/core/gates.py`: numpy matrix builder + torch backend kernel
  (graph-preserving per the `_rxx_backend` fresh-cell rule) + entries in the
  apply-to-state dispatch and the global `gate_to_matrix` dispatch (and the
  remaining two backend dispatch branches that currently special-case
  `{"rzz","rxx"}`).
- `aicir/core/circuit.py`: factory functions
  `single_excitation(theta, qubit_1, qubit_2)` and
  `double_excitation(theta, q1, q2, q3, q4)`.
- `aicir/gates/registry.py`: the two `GateSpec` registrations.
- `aicir/__init__.py`: top-level re-export of both factories.

### Unit 3 — four-term parameter-shift rule

The excitation gates have generators with an equidistant three-eigenvalue
spectrum `{−1, 0, +1}` (two distinct gaps), so the standard two-term `±π/2`
rule is **incorrect** for them; they need the general four-term rule with shifts
`{±π/2, ±3π/2}` and two coefficient pairs.

- Add a reusable four-term shift function to `aicir/qml/deriv.py` (e.g.
  `psr4(objective, params, ...)`), keeping `qml.deriv` the single source of
  truth for shift rules (consistent with how `psr` is shared by
  VQE/SSVQE/VQD/QAS).
- Add per-gate shift-rule **metadata**: a `GateSpec.shift_rule: str | None = None`
  field (values `"two_term"` / `"four_term"`), set to `"four_term"` on both
  excitation gates, exposed by a registry helper (e.g.
  `gate_shift_rule(name)` / `four_term_shift_gates()`) analogous to
  `parametric_pauli_gates()`. This lets a later auto-dispatch select the right
  rule per parameter.
- The exact four-term coefficients come from the general parameter-shift rule
  for a two-gap `{−1,0,1}` spectrum; correctness is **verified against autograd**
  (the autograd/backprop gradient is the ground-truth oracle) in tests, so no
  possibly-wrong constant is taken on faith.

**Out of scope for Unit 3 (explicit):** rewiring VQE/SSVQE/VQD/QAS gradient
dispatch to auto-select the four-term rule per parameter for circuits containing
excitation gates. The standalone rule, the metadata, and the tests are
delivered; end-to-end auto-dispatch is a follow-on.

### Unit 4 — H2 correlation validation

- A `tests/` test running VQE on the 4-qubit JW H2 Hamiltonian
  (`aicir.chemistry` `h2_jw`):
  - prepare the HF determinant with inline `X` gates (test-local, **not** a
    reusable chemistry helper),
  - apply the `double_excitation` gate over the four qubits,
  - optimize the single angle on the autograd (`GPUBackend`/torch) backend via
    `BasicVQE` or `qfun`,
  - assert the energy is within **chemical accuracy (1.6e-3 Ha)** of exact
    dense diagonalization of the same Hamiltonian.
- This proves the gates create real correlation (a single-excitation-only
  circuit cannot pass this).

## Testing

1. **Gate matrices** (`tests/gates/`): assert `single_excitation` /
   `double_excitation` numpy matrices equal the closed-form Givens blocks; assert
   particle-number conservation (gate commutes with `N` — maps each basis state
   to same-Hamming-weight states); assert unitarity.
2. **Autograd kernels** (`tests/gates/` or `tests/backends/`): on the torch
   backend, gradient of `⟨H⟩` through each gate matches finite difference;
   confirm the torch kernel preserves the autograd graph.
3. **Registry** (`tests/gates/`): both gates registered with correct
   `num_qubits`/`num_params`/`shift_rule`; `givens` alias resolves to
   `single_excitation`; `generator is None`.
4. **Four-term rule** (`tests/qml/`): `psr4` gradient on a circuit containing a
   single-excitation gate and one containing a double-excitation gate matches the
   autograd gradient to tolerance; assert the standard two-term `psr` does **not**
   match (guards against silently using the wrong rule).
5. **Validation** (`tests/`): the H2 VQE reaches within 1.6e-3 Ha of exact
   diagonalization.

## Out of scope (decoupled follow-ons)

- Supernet `initial_state` hook and chemistry HF-occupation helper.
- Wiring `single_excitation` into the supernet search pool.
- Auto-dispatch of the four-term rule across VQE/SSVQE/VQD/QAS.
- NEXT.md §7 `GateSpec.matrix` migration.
- BeH2/CH4 chemical-accuracy tuning.

## References

- `aicir/core/gates.py` — `_rxx`/`_rxx_backend`, `_apply_local_matrix_to_state`, dispatch
- `aicir/core/circuit.py` — `rxx`/`rzz` factories
- `aicir/gates/registry.py`, `aicir/gates/spec.py` — GateSpec + registry helpers
- `aicir/qml/deriv.py` — `psr` (parameter-shift single source of truth)
- `aicir/chemistry/` — `h2_jw` preset
- Wierichs et al., "General parameter-shift rules for quantum gradients" (four-term rule for two-gap spectra)

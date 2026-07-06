# Move ansatz templates to top level (aicir.ansatze) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `aicir/vqc/ansatz/` to `aicir/ansatze/` (clean rename, no back-compat shim), since ansatz has zero dependency on `vqc` and is already consumed by `qas`/`optimization/qubo` and (via decoupled data) `chemistry`.

**Architecture:** Pure package relocation. `git mv` the directory; update every importer (`vqc/__init__.py`, tests, demos, READMEs, CLAUDE.md) from `aicir.vqc.ansatz` to `aicir.ansatze`; move the ansatz-specific test files from `tests/vqc/` to `tests/ansatze/`; add one new CHANGELOG entry (do not edit historical entries); fix `CONTENTS.md`'s tree listing.

**Tech Stack:** Python. No new dependencies.

## Global Constraints

- Run from repo root with `PYTHONPATH=.`; tests `PYTHONPATH=. pytest`.
- Clean break, no backward-compat alias for `aicir.vqc.ansatz` (matches CLAUDE.md convention: "old long aliases are intentionally not kept").
- Do NOT edit historical `CHANGELOG.md` entries (lines documenting past `aicir.vqc.ansatz.*` shipping dates stay as literal history); add exactly one new dated entry for this rename.
- Do NOT edit `docs/superpowers/specs/2026-07-02-uccsd-chemistry-pipeline-design.md` or `docs/superpowers/plans/2026-07-02-uccsd-chemistry-pipeline.md` — historical design record, not live docs.
- Use `git mv` for renames (preserve history).
- Comments/docstrings/READMEs stay in Chinese, matching surrounding style.
- End each commit message with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 1: Move package, fix all code + doc importers, verify

**Files:**
- `git mv aicir/vqc/ansatz aicir/ansatze` (moves `__init__.py`, `hea.py`, `hea_ti.py`, `uccsd.py`, `_excitation.py`)
- Modify: `aicir/vqc/__init__.py` (remove the `ansatz` re-export block — it no longer lives under `vqc`)
- Modify: `aicir/vqc/README.md` (5 import lines: `from aicir.vqc.ansatz import ...` → `from aicir.ansatze import ...`)
- Modify: `aicir/chemistry/README.md` (3 references: 2 import lines + 1 prose mention)
- Modify: `CLAUDE.md` (ansatz bullet under "Subsystems" — currently describes ansatz as living in `aicir/vqc/`; update path)
- Modify: `demos/vqe_h2_demo.py`, `demos/vqe_circuit_figure_demo.py`, `demos/qas_evaluator_demo.py` (import lines + the two `tags=[...]`/docstring mentions of `"vqc.ansatz"` in `qas_evaluator_demo.py`, cosmetic but should match)
- Modify: `tests/vqc/test_hea_ansatz.py`, `tests/vqc/test_hea_ti_ansatz.py`, `tests/vqc/test_uccsd_ansatz.py`, `tests/vqc/test_uccsd_vqe_integration.py`, `tests/vqc/test_excitation_circuits.py` (import lines only — these files themselves move to `tests/ansatze/` in Task 2, edit content now while path is still `tests/vqc/`)

**Interfaces:**
- Produces: `aicir.ansatze` importable at `from aicir.ansatze import hea, hea_ti, uccsd, hea_parameter_count, hea_ti_parameter_count, uccsd_parameter_count, entangling_edges, hardware_efficient_ansatz, power_law_couplings` (whatever `aicir/vqc/ansatz/__init__.py` currently exports, unchanged contents — only the parent path changes).
- Consumes: nothing new.

- [ ] **Step 1: Confirm current export surface before moving (baseline)**

Run: `PYTHONPATH=. python -c "from aicir.vqc import ansatz; print(sorted(ansatz.__all__))"`
Record the printed list — Task 1 Step 6 must reproduce the identical list from the new path.

- [ ] **Step 2: Move the package**

```bash
git mv aicir/vqc/ansatz aicir/ansatze
```

- [ ] **Step 3: Remove the ansatz re-export from `aicir/vqc/__init__.py`**

Read the file first. It currently contains (near the top, after the `QAOA` import block):

```python
try:
    from . import ansatz
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.append("ansatz")
```

Delete this whole block. `vqc` no longer re-exports `ansatz` — callers import `aicir.ansatze` directly.

- [ ] **Step 4: Fix importer paths — code files**

In each of these files, replace `aicir.vqc.ansatz` → `aicir.ansatze` and `from aicir.vqc.ansatz import` → `from aicir.ansatze import` (exact string substitution, no logic change):

- `demos/vqe_h2_demo.py` (line ~28: `from aicir.vqc.ansatz import hea`)
- `demos/vqe_circuit_figure_demo.py` (line ~33: `from aicir.vqc.ansatz import hea`)
- `demos/qas_evaluator_demo.py` (line ~34: `from aicir.vqc.ansatz import hea, hea_parameter_count, hea_ti, hea_ti_parameter_count`; also update the two docstring/tag mentions of the string `"vqc.ansatz"` at lines ~120, ~128, ~132, ~136 to `"aicir.ansatze"` / `"ansatz"` respectively — read the file to match exact current wording, keep Chinese prose intact)
- `tests/vqc/test_hea_ansatz.py` (line ~5)
- `tests/vqc/test_hea_ti_ansatz.py` (line ~5, multi-line import)
- `tests/vqc/test_uccsd_ansatz.py` (line ~5)
- `tests/vqc/test_uccsd_vqe_integration.py` (line ~25)
- `tests/vqc/test_excitation_circuits.py` (line ~6: `from aicir.vqc.ansatz._excitation import ...`)

- [ ] **Step 5: Fix importer paths — docs**

- `aicir/vqc/README.md`: 5 occurrences of `from aicir.vqc.ansatz import ...` (lines ~189, ~271, ~288, ~306, ~328) → `from aicir.ansatze import ...`.
- `aicir/chemistry/README.md`: 3 occurrences (line ~6 prose: `aicir.vqc.ansatz.uccsd` → `aicir.ansatze.uccsd`; line ~80: `from aicir.vqc.ansatz import uccsd` → `from aicir.ansatze import uccsd`; line ~119: `from aicir.vqc.ansatz import hea` → `from aicir.ansatze import hea`).
- `CLAUDE.md`: find the `aicir/vqc/` subsystem bullet (mentions "ansatz templates in `ansatz/`"). Read the current exact wording and update it to describe `aicir/ansatze/` as its own top-level subsystem entry (not nested under the `aicir/vqc/` bullet), matching the style of neighboring subsystem bullets (e.g. `aicir/optimizer/`, `aicir/encoder/`). Keep the same content (hea/hea_ti/uccsd description, fSWAP note, decoupled-from-chemistry note) — only the location/heading changes.

- [ ] **Step 6: Verify new import path works and matches baseline**

Run: `PYTHONPATH=. python -c "from aicir import ansatz; print(sorted(ansatz.__all__))"`
Expected: identical list to Step 1's baseline.

Run: `PYTHONPATH=. python -c "import aicir.vqc; assert not hasattr(aicir.vqc, 'ansatz'); print('ok: vqc no longer exposes ansatz')"`
Expected: prints `ok: vqc no longer exposes ansatz` (no `AttributeError` suppressed — the assert must pass).

- [ ] **Step 7: Run affected test files at their current (pre-move) location to confirm imports resolve**

Run: `PYTHONPATH=. pytest tests/vqc/test_hea_ansatz.py tests/vqc/test_hea_ti_ansatz.py tests/vqc/test_uccsd_ansatz.py tests/vqc/test_uccsd_vqe_integration.py tests/vqc/test_excitation_circuits.py -v`
Expected: all PASS (same pass count as before the move — this only changed import paths, not logic).

- [ ] **Step 8: Run full suite to catch any missed importer**

Run: `PYTHONPATH=. pytest -q`
Expected: same pass/skip counts as pre-move baseline (1284 passed, 2 skipped) — any failure here means a missed `aicir.vqc.ansatz` reference; grep for it: `grep -rn "vqc\.ansatz\|vqc import ansatz" aicir/ tests/ demos/` and fix.

- [ ] **Step 9: Commit**

```bash
git add -A aicir/ansatze aicir/vqc tests/vqc demos CLAUDE.md
git commit -m "refactor(ansatz): move aicir.vqc.ansatz to aicir.ansatze (top level)

ansatz has zero dependency on vqc and is already consumed by qas,
optimization/qubo, and chemistry (via decoupled data) — nesting it under
vqc implied it was vqc-specific. Clean rename, no back-compat alias.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Move ansatz test files to `tests/ansatze/`, update CHANGELOG + CONTENTS.md

**Files:**
- `git mv tests/vqc/test_hea_ansatz.py tests/ansatze/test_hea_ansatz.py`
- `git mv tests/vqc/test_hea_ti_ansatz.py tests/ansatze/test_hea_ti_ansatz.py`
- `git mv tests/vqc/test_uccsd_ansatz.py tests/ansatze/test_uccsd_ansatz.py`
- `git mv tests/vqc/test_uccsd_vqe_integration.py tests/ansatze/test_uccsd_vqe_integration.py`
- `git mv tests/vqc/test_excitation_circuits.py tests/ansatze/test_excitation_circuits.py`
- Modify: `CHANGELOG.md` (append one new entry; do not touch existing entries)
- Modify: `CONTENTS.md` (fix the tree: `ansatz/` moves out from under `vqc/` to be its own top-level entry, in both the `aicir/` tree at line ~43-44 and the `tests/` tree at line ~88)

**Interfaces:**
- Consumes: Task 1's completed rename (must run after Task 1 — these test files' import lines were already fixed in Task 1 Step 4, this task only moves the files).

- [ ] **Step 1: Move the test files**

```bash
mkdir -p tests/ansatz
git mv tests/vqc/test_hea_ansatz.py tests/ansatze/test_hea_ansatz.py
git mv tests/vqc/test_hea_ti_ansatz.py tests/ansatze/test_hea_ti_ansatz.py
git mv tests/vqc/test_uccsd_ansatz.py tests/ansatze/test_uccsd_ansatz.py
git mv tests/vqc/test_uccsd_vqe_integration.py tests/ansatze/test_uccsd_vqe_integration.py
git mv tests/vqc/test_excitation_circuits.py tests/ansatze/test_excitation_circuits.py
```

- [ ] **Step 2: Run the moved tests from their new location**

Run: `PYTHONPATH=. pytest tests/ansatze/ -v`
Expected: all PASS (same tests, same count, just relocated — no import path changes needed here since Task 1 already fixed the `from aicir.ansatze import ...` lines).

- [ ] **Step 3: Confirm nothing left in `tests/vqc/` needed the move**

Run: `PYTHONPATH=. pytest tests/vqc/ -v`
Expected: PASS — only VQE-orchestration tests remain (`test_parameter_shift_uses_qml.py`, `test_qaoa_qfun.py`, `test_vqe_orchestration.py`, `test_vqe_qfun.py`, `test_vqe_target.py`), none of which import `ansatz`.

- [ ] **Step 4: Read `CHANGELOG.md`'s current top (most recent dated section) to match format**

Read the file's first ~40 lines to see the exact `## YYYY-MM-DD` / `### Added` / `### Changed` heading style used for the most recent entries.

- [ ] **Step 5: Add one new CHANGELOG entry — today's date, do not touch anything else**

Add a new dated section at the top of the file (above the existing most-recent section), following the exact heading format found in Step 4. Content:

```markdown
## 2026-07-03

### Changed

- **Breaking:** `aicir.vqc.ansatz` moved to `aicir.ansatze` (top-level package). `hea`/`hea_parameter_count`/`hea_ti`/`hea_ti_parameter_count`/`uccsd`/`uccsd_parameter_count`/`entangling_edges`/`hardware_efficient_ansatz`/`power_law_couplings` now import from `aicir.ansatze`, not `aicir.vqc.ansatz`. `aicir.vqc` no longer re-exports `ansatz`. Reason: ansatz has no dependency on `vqc` and is already consumed by `aicir.qas`, `aicir.optimization.qubo`, and `aicir.chemistry` (via decoupled data) — nesting it under `vqc` implied a coupling that didn't exist. No backward-compatible alias (see CLAUDE.md's "old long aliases are intentionally not kept" convention).
```

Do not reformat, reorder, or edit any existing section of the file.

- [ ] **Step 6: Fix `CONTENTS.md`'s tree listing**

Read the file around line 43-44 (the `aicir/` tree, showing `vqc/` with `ansatz/` nested under it) and around line 88 (the `tests/` tree). Update both:
- In the `aicir/` tree: `ansatz/` becomes a sibling top-level entry under `aicir/`, not nested under `vqc/`.
- In the `tests/` tree: add a `tests/ansatze/` entry (sibling of `tests/vqc/`), removing any ansatz-specific test files that were previously listed under `tests/vqc/` (if `CONTENTS.md` lists individual test files — read it first to see the actual granularity before editing; if it only lists directory names, just ensure `ansatz/` appears as its own top-level directory in both trees).

Preserve the file's existing tree-drawing characters/indentation style exactly (`|--`, `` `-- ``, etc. — read a few surrounding lines to match).

- [ ] **Step 7: Run full suite one more time**

Run: `PYTHONPATH=. pytest -q`
Expected: 1284 passed, 2 skipped (same as pre-move baseline — this task only moved files and touched docs, no logic).

- [ ] **Step 8: Commit**

```bash
git add -A tests/ansatz tests/vqc CHANGELOG.md CONTENTS.md
git commit -m "refactor(ansatz): move ansatz tests to tests/ansatze/; update CHANGELOG + CONTENTS.md

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** package move (Task 1) + vqc/__init__.py decoupling (Task 1) + all code importers: demos, tests, aicir/vqc/README.md, aicir/chemistry/README.md, CLAUDE.md (Task 1) + test-file relocation (Task 2) + CHANGELOG (Task 2, additive only) + CONTENTS.md (Task 2). All identified referencing files from the pre-plan grep are covered. Historical docs (CHANGELOG old entries, the 2026-07-02 spec/plan docs) explicitly excluded per Global Constraints — matches the "don't rewrite history" requirement from the conversation.

**Placeholder scan:** every step has a concrete command or exact-text instruction; Step 5/Task 1 and Step 6/Task 2 ask the implementer to read exact current text before editing (line numbers are "~approximate, read first" since I haven't re-verified every line number against the live file) rather than assuming exact line content — this is deliberate given doc prose can drift, not a placeholder.

**Type consistency:** no function signatures change in this plan — pure path/location rename. `aicir.ansatze`'s exported names are asserted identical pre/post move (Task 1 Step 1 vs Step 6).

# Supernet: use aicir gates instead of local gate definitions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove supernet's locally-defined `GateSpec` + builder dicts so gate construction comes from `aicir.core` factories and gate metadata from the `aicir.gates` registry, adding one small registry field (`num_controls`) so the control/target split is registry-driven.

**Architecture:** Two changes. (1) Add an additive `GateSpec.num_controls` metadata field to the shared `aicir.gates` registry and populate it for controlled gates. (2) Refactor `aicir/qas/algorithms/supernet.py` to delete its local gate layer and instead validate tokens against the registry, read parameter counts and control counts from the registry, and construct gates by dispatching to the aicir factory functions it already imports. The refactor is behavior-preserving and guarded by a characterization test.

**Tech Stack:** Python, PyTorch (supernet path is torch-gated), pytest. Run from repo root with `PYTHONPATH=.`.

## Global Constraints

- **Behavior-preserving:** the refactored supernet MUST emit byte-identical gate dicts (same `type` strings, qubit fields, `control_states`, parameter placement) and consume RNG identically. The existing supernet test suite stays green with no edits.
- **No local gate definitions:** supernet must not declare gate matrices, arities, parameter counts, or control conventions. Those come from `aicir.gates` / `aicir.core`. Search-space tokens (`i`, `h`) and dispatch-to-factory wiring are allowed (they define no gates).
- **`num_controls` is additive/non-breaking:** new field defaults to `0`; the existing `controlled: bool` is retained and must satisfy `controlled == (num_controls > 0)`.
- **Out of scope:** NEXT.md §7 `GateSpec.matrix` migration; Givens/excitation gates; HF-reference initial state; opening the pool to arbitrary gates.
- **Comments/docstrings in Chinese** to match surrounding code.
- **Run tests from repo root** with `PYTHONPATH=. pytest ...`.

Spec: `docs/superpowers/specs/2026-06-29-supernet-use-aicir-gates-design.md`

---

## File Structure

- **Modify** `aicir/gates/spec.py` — add `num_controls: int = 0` field + validation.
- **Modify** `aicir/gates/registry.py` — set `num_controls` on `cx`/`cy`/`cz`/`crx`/`cry`/`crz`/`toffoli`.
- **Create** `tests/gates/test_gatespec_num_controls.py` — registry-field test.
- **Modify** `aicir/qas/algorithms/supernet.py` — remove local `GateSpec` + builder dicts; add token tables + factory-dispatch helpers; rewrite validators and the two consumer call sites.
- **Create** `tests/test_supernet_gate_pool.py` — characterization test locking the emitted gate dicts.

---

## Task 1: Add `num_controls` registry metadata

**Files:**
- Modify: `aicir/gates/spec.py` (the `GateSpec` dataclass + `__post_init__`)
- Modify: `aicir/gates/registry.py:121-127` (controlled-gate registrations)
- Test: `tests/gates/test_gatespec_num_controls.py`

**Interfaces:**
- Produces: `GateSpec.num_controls: int` (default `0`). For every registered gate, `controlled == (num_controls > 0)`. Concretely `cx/cy/cz/crx/cry/crz` → `1`, `toffoli` → `2`, all others → `0`.

- [ ] **Step 1: Write the failing test**

Create `tests/gates/test_gatespec_num_controls.py`:

```python
"""GateSpec.num_controls：受控门的控制位数量作为注册表数据。"""

from aicir.gates import get_gate_spec, registered_gate_names


def test_num_controls_defaults_to_zero_for_plain_gates():
    assert get_gate_spec("rx").num_controls == 0
    assert get_gate_spec("ry").num_controls == 0
    assert get_gate_spec("rz").num_controls == 0
    assert get_gate_spec("hadamard").num_controls == 0
    assert get_gate_spec("rzz").num_controls == 0
    assert get_gate_spec("swap").num_controls == 0


def test_num_controls_set_for_controlled_gates():
    assert get_gate_spec("cx").num_controls == 1
    assert get_gate_spec("cy").num_controls == 1
    assert get_gate_spec("cz").num_controls == 1
    assert get_gate_spec("crx").num_controls == 1
    assert get_gate_spec("cry").num_controls == 1
    assert get_gate_spec("crz").num_controls == 1
    assert get_gate_spec("toffoli").num_controls == 2


def test_num_controls_consistent_with_controlled_flag():
    for name in registered_gate_names():
        spec = get_gate_spec(name)
        assert spec.controlled == (spec.num_controls > 0), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/gates/test_gatespec_num_controls.py -q`
Expected: FAIL — `AttributeError: 'GateSpec' object has no attribute 'num_controls'`.

- [ ] **Step 3: Add the field to `GateSpec`**

In `aicir/gates/spec.py`, add the field immediately after `controlled: bool = False`:

```python
    controlled: bool = False
    num_controls: int = 0
```

Add a short doc line in the class docstring's attribute list (after the `controlled` line):

```python
    - ``num_controls``：控制位数量；``controlled == (num_controls > 0)``。
```

In `__post_init__`, after the existing `num_params` check, add:

```python
        if int(self.num_controls) < 0:
            raise ValueError("num_controls must be non-negative")
```

- [ ] **Step 4: Populate the registry**

In `aicir/gates/registry.py`, replace the controlled-gate block (currently lines 121-127) with the same registrations plus `num_controls=`:

```python
    # num_qubits=None：cx/cnot 支持单目标或多目标（多目标等价于多个单目标 CX）。
    GateSpec("cx", None, 0, aliases=("cnot",), controlled=True, num_controls=1, qasm_name="cx", symbol="X"),
    GateSpec("cy", 1, 0, controlled=True, num_controls=1, qasm_name="cy", symbol="Y", decomposition=decompose_cy),
    GateSpec("cz", 1, 0, controlled=True, num_controls=1, qasm_name="cz", symbol="Z", decomposition=decompose_cz),
    GateSpec("crx", 1, 1, controlled=True, num_controls=1, qasm_name="crx", symbol="Rx", generator="X"),
    GateSpec("cry", 1, 1, controlled=True, num_controls=1, qasm_name="cry", symbol="Ry", generator="Y"),
    GateSpec("crz", 1, 1, controlled=True, num_controls=1, qasm_name="crz", symbol="Rz", generator="Z"),
    GateSpec("toffoli", 1, 0, aliases=("ccnot",), controlled=True, num_controls=2, qasm_name="ccx", symbol="X"),
```

- [ ] **Step 5: Run the new test + the full gates suite**

Run: `PYTHONPATH=. pytest tests/gates/ -q`
Expected: PASS (new file passes; no regressions in existing gates tests).

- [ ] **Step 6: Commit**

```bash
git add aicir/gates/spec.py aicir/gates/registry.py tests/gates/test_gatespec_num_controls.py
git commit -m "feat(gates): add GateSpec.num_controls registry metadata"
```

---

## Task 2: Refactor supernet to use aicir gates

**Files:**
- Modify: `aicir/qas/algorithms/supernet.py`
  - imports (~line 46): add `from ...gates import get_gate_spec`
  - replace the local `GateSpec` dataclass + `_SINGLE_QUBIT_GATES` + `_TWO_QUBIT_GATES` + the two `_normalize_*` validators (lines 71-129)
  - `_ensure_architecture_params` (lines 355-381): use `_token_n_params`
  - `build_circuit` (lines 518-574): use `_token_n_params` + the build helpers
- Test: `tests/test_supernet_gate_pool.py`

**Interfaces:**
- Consumes: `get_gate_spec(name).num_params`, `get_gate_spec(name).num_controls` (from Task 1), and the aicir factories `hadamard`, `rx`, `ry`, `rz`, `cx`, `rzz` (already imported in supernet).
- Produces (module-level, used only inside supernet.py):
  - `_SINGLE_QUBIT_TOKENS = ("i","h","rx","ry","rz")`, `_TWO_QUBIT_TOKENS = ("cx","rzz")`
  - `_TOKEN_CANONICAL: dict[str,str]`
  - `_token_n_params(token: str) -> int`
  - `_build_single_gate(token: str, params, qubit: int) -> dict | None`
  - `_build_two_qubit_gate(token: str, params, control: int, target: int) -> dict`
  - `_normalize_single_gate(name) -> str`, `_normalize_two_qubit_gate(name) -> str` (signatures unchanged)

- [ ] **Step 1: Write the characterization test (locks current behavior)**

This test passes against the CURRENT code — it is a golden lock for the refactor, not a red-green test. Create `tests/test_supernet_gate_pool.py`:

```python
"""Characterization: supernet build_circuit emits aicir-canonical gate dicts.

Locks the gate-dict output so the "use aicir gates" refactor stays
byte-identical (types, qubit fields, control_states, parameter placement).
"""

import pytest

pytest.importorskip("torch")

from aicir.gates import canonical_gate_name, registered_gate_names
from aicir.qas.algorithms.supernet import (
    Architecture,
    LayerArchitecture,
    Supernet,
    SupernetConfig,
)


def _qas():
    return Supernet(
        SupernetConfig(
            n_qubits=3,
            layers=1,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (1, 2)),
            seed=0,
        )
    )


def test_mixed_architecture_emits_expected_gate_dicts():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("h", "rx", "rz"), ("cx", "rzz")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    gates = circuit.gates

    assert [g["type"] for g in gates] == ["hadamard", "rx", "rz", "cx", "rzz"]

    assert gates[0] == {"type": "hadamard", "target_qubit": 0}
    assert gates[1]["type"] == "rx" and gates[1]["target_qubit"] == 1 and "parameter" in gates[1]
    assert gates[2]["type"] == "rz" and gates[2]["target_qubit"] == 2 and "parameter" in gates[2]
    # cx on pair (0, 1): target = pair[1], control = pair[0]
    assert gates[3] == {
        "type": "cx",
        "target_qubit": 1,
        "control_qubits": [0],
        "control_states": [1],
    }
    # rzz on pair (1, 2): qubit_1 = pair[0], qubit_2 = pair[1]
    assert gates[4]["type"] == "rzz" and gates[4]["qubit_1"] == 1 and gates[4]["qubit_2"] == 2
    assert "parameter" in gates[4]


def test_every_emitted_type_is_a_registered_aicir_gate():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("h", "rx", "ry"), ("cx", "rzz")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    valid = set(registered_gate_names())
    for gate in circuit.gates:
        assert canonical_gate_name(gate["type"]) in valid


def test_identity_token_emits_no_gate():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("i", "i", "i"), ("none", "none")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    assert circuit.gates == []
```

- [ ] **Step 2: Run the characterization test against current code**

Run: `PYTHONPATH=. pytest tests/test_supernet_gate_pool.py -q`
Expected: PASS (locks the pre-refactor behavior).

- [ ] **Step 3: Add the registry import**

In `aicir/qas/algorithms/supernet.py`, after the line
`from ...core.gates import apply_gate_to_state, gate_to_matrix`, add:

```python
from ...gates import get_gate_spec
```

- [ ] **Step 4: Replace the local gate layer + validators (lines 71-129)**

Delete the `@dataclass(frozen=True) class GateSpec`, `_SINGLE_QUBIT_GATES`, `_TWO_QUBIT_GATES`, and the two `_normalize_*` functions (lines 71-129). Replace that whole block with:

```python
# 可搜索的单/双比特门 token：搜索空间字母表（SupernetConfig 公开面），非门定义。
_SINGLE_QUBIT_TOKENS = ("i", "h", "rx", "ry", "rz")
_TWO_QUBIT_TOKENS = ("cx", "rzz")

# 搜索 token → aicir 规范门名。"i" 表示"该槽不放门"，故不在表内。
_TOKEN_CANONICAL: dict[str, str] = {
    "h": "hadamard",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cx": "cx",
    "rzz": "rzz",
}

# 单比特旋转 token → aicir.core 工厂；门语义全部来自 aicir，本模块不定义门。
_SINGLE_ROTATION_FACTORY = {"rx": rx, "ry": ry, "rz": rz}


def _token_n_params(token: str) -> int:
    """该 token 的可训练角度数；取自 aicir.gates 注册表（"i" 无门、0 个）。"""
    if token == "i":
        return 0
    return int(get_gate_spec(_TOKEN_CANONICAL[token]).num_params)


def _build_single_gate(token: str, params: Sequence[Any], qubit: int) -> dict[str, Any] | None:
    """构造单比特门（经 aicir.core 工厂）；"i" 不放门，返回 None。"""
    if token == "i":
        return None
    canonical = _TOKEN_CANONICAL[token]
    if canonical == "hadamard":
        return hadamard(int(qubit))
    return _SINGLE_ROTATION_FACTORY[canonical](params[0], target_qubit=int(qubit))


def _build_two_qubit_gate(
    token: str, params: Sequence[Any], control: int, target: int
) -> dict[str, Any]:
    """构造双比特门（经 aicir.core 工厂）；控制/目标拆分由注册表 num_controls 驱动。"""
    canonical = _TOKEN_CANONICAL[token]
    n_controls = int(get_gate_spec(canonical).num_controls)
    ordered = (int(control), int(target))
    controls = ordered[:n_controls]
    targets = ordered[n_controls:]
    if canonical == "cx":
        return cx(target_qubit=targets[0], control_qubits=list(controls))
    return rzz(params[0], qubit_1=targets[0], qubit_2=targets[1])


def _normalize_single_gate(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _SINGLE_QUBIT_TOKENS:
        raise ValueError(
            f"supernet single-qubit gates are {tuple(_SINGLE_QUBIT_TOKENS)}; got {name!r}"
        )
    if gate != "i" and get_gate_spec(_TOKEN_CANONICAL[gate]) is None:
        raise ValueError(
            f"supernet single-qubit token {name!r} maps to an unregistered aicir gate"
        )
    return gate


def _normalize_two_qubit_gate(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _TWO_QUBIT_TOKENS:
        raise ValueError(
            f"supernet two-qubit gates are {tuple(_TWO_QUBIT_TOKENS)}; got {name!r}"
        )
    if get_gate_spec(_TOKEN_CANONICAL[gate]) is None:
        raise ValueError(
            f"supernet two-qubit token {name!r} maps to an unregistered aicir gate"
        )
    return gate
```

(`_normalize_two_qubit_choice`, which calls `_normalize_two_qubit_gate`, is unchanged and stays directly below.)

- [ ] **Step 5: Update `_ensure_architecture_params`**

In `_ensure_architecture_params`, replace the single-qubit inner loop:

```python
                for qubit_id, gate_type in enumerate(single_layout):
                    spec = _SINGLE_QUBIT_GATES[gate_type]
                    for param_index in range(spec.n_params):
```

with:

```python
                for qubit_id, gate_type in enumerate(single_layout):
                    for param_index in range(_token_n_params(gate_type)):
```

and replace the two-qubit inner loop:

```python
                    spec = _TWO_QUBIT_GATES[choice]
                    for param_index in range(spec.n_params):
```

with:

```python
                    for param_index in range(_token_n_params(choice)):
```

- [ ] **Step 6: Update `build_circuit` (single-qubit section)**

Replace:

```python
            for qubit_id, gate_type in enumerate(single_layout):
                spec = _SINGLE_QUBIT_GATES[gate_type]
                params: list[torch.Tensor | float] = []
                for param_index in range(spec.n_params):
```

with:

```python
            for qubit_id, gate_type in enumerate(single_layout):
                params: list[torch.Tensor | float] = []
                for param_index in range(_token_n_params(gate_type)):
```

and replace:

```python
                gate = spec.builder(params, (qubit_id,))
                if gate is not None:
                    gates.append(gate)
```

with:

```python
                gate = _build_single_gate(gate_type, params, qubit_id)
                if gate is not None:
                    gates.append(gate)
```

- [ ] **Step 7: Update `build_circuit` (two-qubit section)**

Replace:

```python
                spec = _TWO_QUBIT_GATES[choice]
                control, target = self.config.two_qubit_pairs[pair_index]
                params = []
                for param_index in range(spec.n_params):
```

with:

```python
                control, target = self.config.two_qubit_pairs[pair_index]
                params = []
                for param_index in range(_token_n_params(choice)):
```

and replace:

```python
                gate = spec.builder(params, (int(control), int(target)))
                if gate is not None:
                    gates.append(gate)
```

with:

```python
                gate = _build_two_qubit_gate(choice, params, control, target)
                if gate is not None:
                    gates.append(gate)
```

- [ ] **Step 8: Verify no stale references remain**

Run: `PYTHONPATH=. python -c "import re,sys; s=open('aicir/qas/algorithms/supernet.py').read(); bad=[t for t in ('_SINGLE_QUBIT_GATES','_TWO_QUBIT_GATES','spec.builder','spec.n_params') if t in s]; print('STALE:',bad); sys.exit(1 if bad else 0)"`
Expected: `STALE: []` and exit 0.

- [ ] **Step 9: Run the characterization test + full supernet/qas suite**

Run: `PYTHONPATH=. pytest tests/test_supernet_gate_pool.py tests/test_vqa_qas.py tests/test_supernet_lazy_layouts.py tests/test_supernet_sharding.py tests/test_qas_runner.py tests/test_vqa_qas.py -q`
Expected: PASS (characterization unchanged; existing behavior preserved).

- [ ] **Step 10: Run the broader suite touching QAS/gates**

Run: `PYTHONPATH=. pytest tests/gates/ tests/test_supernet_gate_pool.py tests/test_vqa_qas.py tests/test_supernet_lazy_layouts.py tests/test_qas_runner.py tests/test_maxcut_demo.py -q`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add aicir/qas/algorithms/supernet.py tests/test_supernet_gate_pool.py
git commit -m "refactor(qas): supernet uses aicir gates, drops local GateSpec"
```

---

## Self-Review notes

- **Spec coverage:** Goal (remove local gates) → Task 2; hard invariant (byte-identical) → Task 2 characterization test (Steps 1-2, 9); item 1 (two-part residue) → documented in spec, no code; item 2 (`num_controls` as registry data) → Task 1; validation change → Task 2 Step 4; affected call sites → Task 2 Steps 5-7; out-of-scope items excluded.
- **Placeholder scan:** none — all steps carry concrete code/commands.
- **Type consistency:** `_token_n_params` / `_build_single_gate` / `_build_two_qubit_gate` names used identically across Steps 4-7; `num_controls` field name consistent between Task 1 and Task 2 Step 4.

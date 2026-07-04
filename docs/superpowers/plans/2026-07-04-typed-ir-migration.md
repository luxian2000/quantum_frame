# Typed `Circuit.gates` API Migration

Date: 2026-07-04
Branch: `feature/typed-circuit-gates-api`

## Goal

Make `Circuit.gates` return typed instruction objects:

- `Operation`
- `Measurement`
- `ControlFlow`

Keep legacy dict output available through explicit compatibility APIs:

- `Circuit.legacy_gates`
- `Circuit.to_gate_dicts()`

## API Change

Before:

```python
cir.gates == [{"type": "rx", "target_qubit": 0, "parameter": 0.1}]
cir.gates[0]["parameter"] = 0.2
```

After:

```python
cir.gates[0].name == "rx"
cir.gates[0]["type"] == "rx"        # read-compatible
cir.gates[0]["parameter"] = 0.2     # TypeError: typed gates are immutable

gates = cir.to_gate_dicts()
gates[0]["parameter"] = 0.2         # OK: detached legacy dict
```

## Migration Rules

- Read-only legacy access can continue temporarily: `gate["type"]`, `gate.get("parameter")`.
- New typed code should prefer attributes: `gate.name`, `gate.qubits`, `gate.params`, `gate.controls`.
- Any code that mutates gate dicts must first call `circuit.to_gate_dicts()` or `circuit.legacy_gates`.
- Any code that serializes gates to JSON/QASM/third-party interop should use `to_gate_dicts()`.
- Tests that assert exact old dict output should assert `circuit.legacy_gates`.

## Implementation Done

- Added typed API tests covering construction, append/extend, parameter binding, and JSON round trip.
- Changed `Circuit.__init__`, `append`, and `extend` to normalize inputs with `as_instruction`.
- Changed `Circuit.operations` to expose the same typed instruction list as a tuple.
- Added `Circuit.legacy_gates` and `Circuit.to_gate_dicts()` as detached dict output.
- Updated control-flow construction to store legacy dict body snapshots.
- Updated QAS/VQE parameter binding and supernet payloads to use legacy dict snapshots at mutation/serialization boundaries.
- Migrated runtime internals from legacy dict reads to typed instruction accessors:
  - `aicir.core.circuit`
  - `aicir.core.gates`
  - `aicir.core.batch`
  - `aicir.measure.trajectory`
  - `aicir.simulator.network`
  - `aicir.qml.deriv`
  - `aicir.metrics`
  - `aicir.visual.plot`
  - MoG-VQE and VQE-QAS preparation/fair binding
  - main transpile passes outside explicit legacy rewrite boundaries
- Added `tests/circuit/test_typed_internal_gate_access.py` as an AST guard: non-boundary production code must not read gate business fields through `gate["..."]` or `gate.get(...)`.
- Updated README and CHANGELOG with breaking-change migration notes.

## Remaining Dict Boundaries

These dict uses are intentional compatibility or payload boundaries:

- IR conversion and compatibility: `aicir.ir.*` `to_dict/from_dict`, `instruction_to_gate_dict`.
- Core IO and third-party interop: JSON/QASM/Qiskit/PennyLane/WuYue/DAG adapters.
- Legacy transpile rewrite APIs: local rewrite and DAG rewrite paths that explicitly accept gate-dict lists.
- CRLQAS action-space/search DTOs, where candidate payloads are mutable config records rather than `Circuit.gates` internals.

## Verification

Run before merge:

```bash
pytest tests/circuit/test_circuit_typed_gates_api.py \
  tests/circuit/test_typed_ir_internal_migration.py \
  tests/circuit/test_operation_ir.py \
  tests/circuit/test_typed_factories.py \
  tests/circuit/test_parameterized_circuit.py \
  tests/core/test_control_flow_build.py \
  tests/core/test_control_flow_json.py \
  tests/circuit/io/test_json_qasm_io.py -q

pytest tests/circuit tests/core tests/measure tests/transpile tests/visual tests/primitives -q

pytest -q
```

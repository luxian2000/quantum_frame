"""Shared accessors for consuming circuits through typed IR.

The public ``Circuit.gates`` surface remains a list of gate dictionaries for
compatibility. Internal consumers should use these helpers so they can accept
``CircuitIR`` and typed instructions without re-implementing dict field rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .circuit_ir import CircuitIR, CircuitInstruction
from .control_flow import ControlFlow
from .measurement import Measurement
from .operation import Operation


def as_instruction(value: CircuitInstruction | Mapping[str, Any]) -> CircuitInstruction:
    """Normalize a typed instruction or old gate mapping to typed IR."""

    if isinstance(value, (Operation, Measurement, ControlFlow)):
        return value
    if isinstance(value, Mapping):
        t = str(value.get("type", "")).lower()
        if t in {"if", "while"}:
            return ControlFlow.from_dict(value)
        if t in {"measure", "measurement", "reset"}:
            return Measurement.from_dict(value)
        return Operation.from_dict(value)
    raise TypeError("instruction must be Operation, Measurement, ControlFlow, or a gate mapping")


def instruction_name(instruction: CircuitInstruction | Mapping[str, Any]) -> str:
    """Return the operation or measurement type name."""

    inst = as_instruction(instruction)
    if isinstance(inst, ControlFlow):
        return inst.name
    if isinstance(inst, Measurement):
        return inst.measurement_type
    return inst.name


def instruction_params(instruction: CircuitInstruction | Mapping[str, Any]) -> tuple[Any, ...]:
    """Return instruction parameters as a tuple."""

    inst = as_instruction(instruction)
    if isinstance(inst, (Measurement, ControlFlow)):
        return ()
    return inst.params


def instruction_parameter(instruction: CircuitInstruction | Mapping[str, Any], default: Any = None) -> Any:
    """Return the old gate-dict ``parameter`` value shape for compatibility."""

    params = instruction_params(instruction)
    if not params:
        return default
    if len(params) == 1:
        return params[0]
    return list(params)


def instruction_qubits(instruction: CircuitInstruction | Mapping[str, Any]) -> tuple[int, ...]:
    """Return the explicit target/readout qubits carried by an instruction."""

    inst = as_instruction(instruction)
    if isinstance(inst, ControlFlow):
        return ()
    return inst.qubits


def instruction_controls(instruction: CircuitInstruction | Mapping[str, Any]) -> tuple[int, ...]:
    """Return control qubits for operations; measurements have no controls."""

    inst = as_instruction(instruction)
    if isinstance(inst, (Measurement, ControlFlow)):
        return ()
    return inst.controls


def instruction_control_states(instruction: CircuitInstruction | Mapping[str, Any]) -> tuple[int, ...]:
    """Return control states, defaulting omitted controls to state 1."""

    inst = as_instruction(instruction)
    if isinstance(inst, (Measurement, ControlFlow)):
        return ()
    if inst.control_states:
        return inst.control_states
    return tuple(1 for _ in inst.controls)


def instruction_to_gate_dict(instruction: CircuitInstruction | Mapping[str, Any]) -> dict[str, Any]:
    """Convert an instruction to the legacy gate dictionary surface."""

    return as_instruction(instruction).to_dict()


def circuit_instructions(circuit: Any) -> tuple[CircuitInstruction, ...]:
    """Return typed instructions from CircuitIR, Circuit, or circuit-like objects."""

    if isinstance(circuit, CircuitIR):
        return tuple(circuit.operations)

    operations = getattr(circuit, "operations", None)
    if operations is not None:
        return tuple(as_instruction(operation) for operation in operations)

    ir = getattr(circuit, "ir", None)
    if isinstance(ir, CircuitIR):
        return tuple(ir.operations)

    gates = getattr(circuit, "gates", None)
    if gates is not None:
        return tuple(as_instruction(gate) for gate in gates)

    raise TypeError("circuit must provide CircuitIR operations or a gates sequence")


def has_circuit_instructions(circuit: Any) -> bool:
    """Return whether a circuit-like object exposes a consumable instruction list."""

    if isinstance(circuit, CircuitIR):
        return True
    return any(hasattr(circuit, attr) for attr in ("operations", "ir", "gates"))


def circuit_gate_dicts(circuit: Any) -> list[dict[str, Any]]:
    """Return legacy gate dictionaries generated from typed instructions."""

    return [instruction_to_gate_dict(instruction) for instruction in circuit_instructions(circuit)]


def circuit_instruction_count(circuit: Any) -> int:
    """Return the number of typed circuit instructions."""

    return len(circuit_instructions(circuit))

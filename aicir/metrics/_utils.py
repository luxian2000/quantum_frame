"""Shared helpers for circuit-level algorithm metrics."""

from __future__ import annotations

from typing import List, Tuple

from ..core.circuit import Circuit
from ..ir import (
    circuit_instruction_count,
    circuit_instructions,
    instruction_controls,
    instruction_name,
    instruction_qubits,
)


def gate_type(gate) -> str:
    return instruction_name(gate).lower()


def is_two_qubit_gate(gate) -> bool:
    gate_name = gate_type(gate)
    return bool(
        instruction_controls(gate)
        or gate_name in {
            "cx",
            "cnot",
            "cy",
            "cz",
            "crx",
            "cry",
            "crz",
            "swap",
            "rzz",
            "rxx",
            "zz",
            "toffoli",
            "ccnot",
        }
    )


def count_gate_families(circuit: Circuit) -> Tuple[int, int]:
    single_qubit_ops = 0
    two_qubit_ops = 0
    for gate in circuit_instructions(circuit):
        if is_two_qubit_gate(gate):
            two_qubit_ops += 1
        else:
            single_qubit_ops += 1
    return single_qubit_ops, two_qubit_ops


def count_two_qubit_gates(circuit: Circuit) -> int:
    return sum(1 for gate in circuit_instructions(circuit) if is_two_qubit_gate(gate))


def gate_qubits(gate, n_qubits: int) -> List[int]:
    """Return explicit qubits touched by a gate, falling back to all qubits."""
    qubits = [*instruction_qubits(gate), *instruction_controls(gate)]

    if not qubits:
        return list(range(int(n_qubits)))

    return list(dict.fromkeys(qubits))


def depth_proxy(circuit: Circuit) -> float:
    """Simple layer-like circuit depth proxy without backend scheduling."""
    if circuit_instruction_count(circuit) == 0:
        return 0.0

    qubit_layers = [0] * int(circuit.n_qubits)
    max_layer = 0
    for gate in circuit_instructions(circuit):
        involved_qubits = gate_qubits(gate, int(circuit.n_qubits))
        layer = max((qubit_layers[qubit] for qubit in involved_qubits), default=0) + 1
        for qubit in involved_qubits:
            qubit_layers[qubit] = layer
        max_layer = max(max_layer, layer)
    return float(max_layer)


__all__ = [
    "count_gate_families",
    "count_two_qubit_gates",
    "depth_proxy",
    "gate_qubits",
    "gate_type",
    "is_two_qubit_gate",
]

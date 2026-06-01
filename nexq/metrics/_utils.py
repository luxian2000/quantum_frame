"""Shared helpers for circuit-level algorithm metrics."""

from __future__ import annotations

from typing import Tuple

from ..core.circuit import Circuit


def gate_type(gate: dict) -> str:
    return str(gate.get("type", "")).lower()


def is_two_qubit_gate(gate: dict) -> bool:
    gate_name = gate_type(gate)
    return bool(
        gate.get("control_qubits")
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
            "zz",
            "toffoli",
            "ccnot",
        }
    )


def count_gate_families(circuit: Circuit) -> Tuple[int, int]:
    single_qubit_ops = 0
    two_qubit_ops = 0
    for gate in circuit.gates:
        if is_two_qubit_gate(gate):
            two_qubit_ops += 1
        else:
            single_qubit_ops += 1
    return single_qubit_ops, two_qubit_ops


def count_two_qubit_gates(circuit: Circuit) -> int:
    return sum(1 for gate in circuit.gates if is_two_qubit_gate(gate))


def depth_proxy(circuit: Circuit) -> float:
    """Simple layer-like circuit depth proxy without backend scheduling."""
    if not circuit.gates:
        return 0.0

    qubit_layers = [0] * int(circuit.n_qubits)
    max_layer = 0
    for gate in circuit.gates:
        qubits = []
        target = gate.get("target_qubit")
        if target is not None:
            qubits.append(int(target))
        controls = gate.get("control_qubits") or []
        qubits.extend(int(qubit) for qubit in controls)
        gate_qubits = qubits or list(range(int(circuit.n_qubits)))
        layer = max((qubit_layers[qubit] for qubit in gate_qubits), default=0) + 1
        for qubit in gate_qubits:
            qubit_layers[qubit] = layer
        max_layer = max(max_layer, layer)
    return float(max_layer)


__all__ = [
    "count_gate_families",
    "count_two_qubit_gates",
    "depth_proxy",
    "gate_type",
    "is_two_qubit_gate",
]

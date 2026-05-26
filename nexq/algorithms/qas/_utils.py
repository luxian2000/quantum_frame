"""Shared low-level helpers for QAS modules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from ...channel.backends.base import Backend
from ...core.circuit import Circuit


TWO_QUBIT_GATE_TYPES = {
    "cx",
    "cnot",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "swap",
    "toffoli",
    "ccnot",
    "rzz",
}


class RankedScore(Protocol):
    weighted_score: float
    rank: Optional[int]


def get_default_backend() -> Backend:
    from ...channel.backends.numpy_backend import NumpyBackend

    return NumpyBackend()


def ensure_backend(backend: Optional[Backend]) -> Backend:
    return backend if backend is not None else get_default_backend()


def gate_type(gate: Dict[str, Any]) -> str:
    return str(gate.get("type", ""))


def is_two_qubit_gate(gate: Dict[str, Any]) -> bool:
    gate_name = gate_type(gate)
    return gate_name in TWO_QUBIT_GATE_TYPES or bool(gate.get("control_qubits"))


def count_gate_families(circuit: Circuit) -> Tuple[int, int]:
    single_qubit = 0
    two_qubit = 0
    for gate in circuit.gates:
        if is_two_qubit_gate(gate):
            two_qubit += 1
        else:
            single_qubit += 1
    return single_qubit, two_qubit


def count_two_qubit_gates(circuit: Circuit) -> int:
    return sum(1 for gate in circuit.gates if is_two_qubit_gate(gate))


def count_parameters(circuit: Circuit) -> int:
    total = 0
    for gate in circuit.gates:
        if "parameter" not in gate:
            continue
        parameter = gate.get("parameter")
        if parameter is None:
            continue
        array = np.asarray(parameter)
        total += int(array.size) if array.shape else 1
    return total


def depth_proxy(circuit: Circuit) -> float:
    return len(circuit.gates) / max(1, int(circuit.n_qubits))


def clipped_score(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def assign_ranks(scores: Iterable[RankedScore]) -> List[RankedScore]:
    ranked = sorted(scores, key=lambda item: item.weighted_score, reverse=True)
    for rank, score in enumerate(ranked, start=1):
        score.rank = rank
    return ranked

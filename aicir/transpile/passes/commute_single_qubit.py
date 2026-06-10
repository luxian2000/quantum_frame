"""Conservative local commutation pass for single-qubit gates."""

from __future__ import annotations

from ...core.circuit import Circuit
from ...ir import circuit_gate_dicts
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates, commute_single_qubit_gates


class CommuteSingleQubitPass(TransformationPass):
    """Look back through known-safe commuting gates to cancel or merge operations."""

    def __init__(self, *, max_reorder_hops: int = 8) -> None:
        self.max_reorder_hops = int(max_reorder_hops)
        if self.max_reorder_hops < 0:
            raise ValueError("max_reorder_hops must be non-negative")

    def run(self, circuit: Circuit) -> Circuit:
        return circuit_from_gates(
            circuit,
            commute_single_qubit_gates(circuit_gate_dicts(circuit), max_reorder_hops=self.max_reorder_hops),
        )

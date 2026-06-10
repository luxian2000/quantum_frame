"""Inverse-gate cancellation pass."""

from __future__ import annotations

from ...core.circuit import Circuit
from ..base import TransformationPass
from ._local_rewrite import cancel_inverse_gates, circuit_from_gates


class CancelInversePass(TransformationPass):
    """Cancel adjacent inverse gate pairs."""

    def run(self, circuit: Circuit) -> Circuit:
        return circuit_from_gates(circuit, cancel_inverse_gates(circuit.gates))

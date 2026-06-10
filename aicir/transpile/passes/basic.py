"""Basic validation and canonicalization passes."""

from __future__ import annotations

from ...core.circuit import Circuit
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates


class ValidatePass(TransformationPass):
    """Validate that the circuit can be represented by the current ``Circuit`` surface."""

    def run(self, circuit: Circuit) -> Circuit:
        return circuit_from_gates(circuit, circuit.gates)


class CanonicalizePass(TransformationPass):
    """Return a copy normalized through the current ``Circuit`` constructor."""

    def run(self, circuit: Circuit) -> Circuit:
        return circuit_from_gates(circuit, circuit.gates)

"""Single-qubit rotation merge pass."""

from __future__ import annotations

from ...core.circuit import Circuit
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates, merge_adjacent_rotations


class MergeRotationsPass(TransformationPass):
    """Merge adjacent ``rx``/``ry``/``rz`` gates on the same qubit."""

    def run(self, circuit: Circuit) -> Circuit:
        return circuit_from_gates(circuit, merge_adjacent_rotations(circuit.gates))

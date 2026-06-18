"""Pass manager for ordered circuit transformations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.circuit import Circuit
from ..ir import circuit_gate_dicts, has_circuit_instructions
from .base import TransformationPass


def _pass_from_name(name: str) -> TransformationPass:
    from .passes import (
        CancelInversePass,
        CanonicalizePass,
        CommuteSingleQubitPass,
        DecomposePass,
        LayoutPass,
        MergeRotationsPass,
        ValidatePass,
    )

    key = str(name).strip().lower()
    mapping = {
        "validate": ValidatePass,
        "canonicalize": CanonicalizePass,
        "cancel_inverse": CancelInversePass,
        "cancel": CancelInversePass,
        "merge_rotations": MergeRotationsPass,
        "merge_rotation": MergeRotationsPass,
        "commute_single_qubit": CommuteSingleQubitPass,
        "commute": CommuteSingleQubitPass,
        "decompose": DecomposePass,
        "layout": LayoutPass,
    }
    try:
        return mapping[key]()
    except KeyError as exc:
        raise ValueError(f"Unknown transpile pass: {name}") from exc


def _coerce_pass(item: str | TransformationPass) -> TransformationPass:
    if isinstance(item, str):
        return _pass_from_name(item)
    if isinstance(item, TransformationPass):
        return item
    if hasattr(item, "run"):
        return item
    raise TypeError("passes must be pass names or TransformationPass objects")


class PassManager:
    """Run circuit transformation passes in sequence."""

    def __init__(
        self,
        passes: Iterable[str | TransformationPass],
        *,
        fixed_point: bool = False,
        max_rounds: int = 64,
    ) -> None:
        self.passes = tuple(_coerce_pass(item) for item in passes)
        self.fixed_point = bool(fixed_point)
        self.max_rounds = int(max_rounds)
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

    def run(self, circuit: Circuit) -> Circuit:
        if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
            raise TypeError("PassManager.run expects a Circuit or CircuitIR-like object")
        if not isinstance(circuit, Circuit):
            circuit = Circuit(*circuit_gate_dicts(circuit), n_qubits=int(circuit.n_qubits))

        current = circuit
        rounds = self.max_rounds if self.fixed_point else 1
        for _ in range(rounds):
            before = circuit_gate_dicts(current)
            for item in self.passes:
                current = item.run(current)
            if not self.fixed_point or circuit_gate_dicts(current) == before:
                return current
        return current


def default_optimization_pipeline(*, max_rounds: int = 64, max_reorder_hops: int = 8) -> PassManager:
    """Return the default local circuit-optimization pipeline."""

    from .passes import CancelInversePass, CommuteSingleQubitPass, MergeRotationsPass

    return PassManager(
        [
            CancelInversePass(),
            MergeRotationsPass(),
            CommuteSingleQubitPass(max_reorder_hops=max_reorder_hops),
        ],
        fixed_point=True,
        max_rounds=max_rounds,
    )

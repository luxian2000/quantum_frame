"""Noise model composition utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set


@dataclass
class NoiseRule:
    channel: object
    after_gates: Optional[Set[str]] = None
    exclude_gate_qubits: bool = False


@dataclass
class NoiseModel:
    """Noise model composed of gate-triggered channel rules."""

    rules: List[NoiseRule] = field(default_factory=list)

    def add_channel(
        self,
        channel,
        after_gates: Optional[Sequence[str]] = None,
        *,
        exclude_gate_qubits: bool = False,
    ) -> "NoiseModel":
        gate_set = None if after_gates is None else {str(g) for g in after_gates}
        self.rules.append(
            NoiseRule(
                channel=channel,
                after_gates=gate_set,
                exclude_gate_qubits=bool(exclude_gate_qubits),
            )
        )
        return self

    def _match_rule(self, rule: NoiseRule, gate_type: Optional[str]) -> bool:
        if rule.after_gates is None:
            return True
        if gate_type is None:
            return False
        return gate_type in rule.after_gates

    def _gate_qubits(self, gate: Optional[dict]) -> Set[int]:
        if not gate:
            return set()
        qubits: Set[int] = set()
        target = gate.get("target_qubit")
        if target is not None:
            qubits.add(int(target))
        for key in ("control_qubits", "qubits", "targets"):
            values = gate.get(key)
            if values is None:
                continue
            if isinstance(values, (list, tuple, set)):
                qubits.update(int(q) for q in values)
            else:
                qubits.add(int(values))
        return qubits

    def _should_apply_to_gate(self, rule: NoiseRule, gate: Optional[dict]) -> bool:
        if not rule.exclude_gate_qubits:
            return True
        target_qubit = getattr(rule.channel, "target_qubit", None)
        if target_qubit is None:
            return True
        return int(target_qubit) not in self._gate_qubits(gate)

    def apply(self, rho, n_qubits: int, backend, gate_type: Optional[str] = None, gate: Optional[dict] = None):
        """Apply all matching noise rules to a density matrix."""
        if gate_type is None and gate is not None:
            gate_type = gate.get("type")
        out = rho
        for rule in self.rules:
            if not self._match_rule(rule, gate_type):
                continue
            if not self._should_apply_to_gate(rule, gate):
                continue
            kraus = rule.channel.kraus_operators(n_qubits, backend)
            acc = backend.zeros(out.shape)
            for k in kraus:
                acc = acc + backend.matmul(backend.matmul(k, out), backend.dagger(k))
            out = acc
        return out

    def __len__(self) -> int:
        return len(self.rules)

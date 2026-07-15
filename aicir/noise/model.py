"""Noise model composition utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

from ..ir import instruction_controls, instruction_name, instruction_qubits


@dataclass
class NoiseRule:
    channel: object
    after_gates: Optional[Set[str]] = None
    exclude_gate_qubits: bool = False


@dataclass
class NoiseModel:
    """Noise model composed of gate-triggered channel rules.

    全系统嵌入的 Kraus 算符按 (规则下标, n_qubits, backend) 缓存——
    多 shot / 多门重复 apply 时只构建一次。信道对象加入规则后
    不应再就地修改参数（否则缓存不失效）。
    """

    rules: List[NoiseRule] = field(default_factory=list)
    _kraus_cache: dict = field(default_factory=dict, repr=False, compare=False)

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

    def _gate_qubits(self, gate: Optional[Any]) -> Set[int]:
        if not gate:
            return set()
        return {int(q) for q in (*instruction_qubits(gate), *instruction_controls(gate))}

    def _should_apply_to_gate(self, rule: NoiseRule, gate: Optional[Any]) -> bool:
        if not rule.exclude_gate_qubits:
            return True
        target_qubit = getattr(rule.channel, "target_qubit", None)
        if target_qubit is None:
            return True
        return int(target_qubit) not in self._gate_qubits(gate)

    def apply(self, rho, n_qubits: int, backend, gate_type: Optional[str] = None, gate: Optional[Any] = None):
        """Apply all matching noise rules to a density matrix."""
        if gate_type is None and gate is not None:
            gate_type = instruction_name(gate)
        out = rho
        for rule_idx, rule in enumerate(self.rules):
            if not self._match_rule(rule, gate_type):
                continue
            if not self._should_apply_to_gate(rule, gate):
                continue
            cache_key = (rule_idx, int(n_qubits), id(backend))
            pairs = self._kraus_cache.get(cache_key)
            if pairs is None:
                kraus = rule.channel.kraus_operators(n_qubits, backend)
                pairs = [(k, backend.dagger(k)) for k in kraus]
                self._kraus_cache[cache_key] = pairs
            # 累加走 backend.add：NPU complex64 无 aclnnAdd，裸 `+` 真机报错
            acc = backend.zeros(out.shape)
            for k, k_dag in pairs:
                acc = backend.add(acc, backend.matmul(backend.matmul(k, out), k_dag))
            out = acc
        return out

    def __len__(self) -> int:
        return len(self.rules)

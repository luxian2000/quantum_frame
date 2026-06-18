"""DecomposePass：把高级门分解到目标门集（NEXT.md 第 2 节）。

本片提供一组经数值验证的标准分解规则，把双比特高级门
（``swap``/``cz``/``cy``）改写为 ``cx`` + 单比特门形式：

- ``swap(a, b)``  -> ``cx(a,[b]) · cx(b,[a]) · cx(a,[b])``
- ``cz(t,[c])``   -> ``h(t) · cx(t,[c]) · h(t)``
- ``cy(t,[c])``   -> ``rz(-pi/2,t) · cx(t,[c]) · rz(pi/2,t)``

分解只针对“不在目标门集内且存在规则”的门；已在门集内的门原样保留。
规则展开产生的单比特门为 ``hadamard``/``rz``，本片不再把任意单比特门
进一步做 Euler 基底翻译（留待后续）。受控形式仅支持单控制位。
"""

from __future__ import annotations

import math

from ...core.circuit import Circuit
from ...gates import canonical_gate_name
from ...ir import circuit_gate_dicts, instruction_controls, instruction_qubits
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates

__all__ = ["DecomposePass"]


def _single_control(gate: dict) -> int | None:
    controls = instruction_controls(gate)
    states = gate.get("control_states")
    if len(controls) != 1:
        return None
    if states is not None and tuple(int(s) for s in states) != (1,):
        return None
    return int(controls[0])


def _decompose_swap(gate: dict) -> list[dict] | None:
    qubits = instruction_qubits(gate)
    if len(qubits) != 2 or instruction_controls(gate):
        return None
    a, b = int(qubits[0]), int(qubits[1])
    cab = {"type": "cx", "target_qubit": a, "control_qubits": [b], "control_states": [1]}
    cba = {"type": "cx", "target_qubit": b, "control_qubits": [a], "control_states": [1]}
    return [dict(cab), dict(cba), dict(cab)]


def _decompose_cz(gate: dict) -> list[dict] | None:
    qubits = instruction_qubits(gate)
    control = _single_control(gate)
    if len(qubits) != 1 or control is None:
        return None
    t = int(qubits[0])
    return [
        {"type": "hadamard", "target_qubit": t},
        {"type": "cx", "target_qubit": t, "control_qubits": [control], "control_states": [1]},
        {"type": "hadamard", "target_qubit": t},
    ]


def _decompose_cy(gate: dict) -> list[dict] | None:
    qubits = instruction_qubits(gate)
    control = _single_control(gate)
    if len(qubits) != 1 or control is None:
        return None
    t = int(qubits[0])
    return [
        {"type": "rz", "target_qubit": t, "parameter": -math.pi / 2.0},
        {"type": "cx", "target_qubit": t, "control_qubits": [control], "control_states": [1]},
        {"type": "rz", "target_qubit": t, "parameter": math.pi / 2.0},
    ]


_RULES = {
    "swap": _decompose_swap,
    "cz": _decompose_cz,
    "cy": _decompose_cy,
}


class DecomposePass(TransformationPass):
    """把高级门分解到目标门集。

    参数：

    - ``basis_gates``：目标原生门集（门名，支持别名）。门集内的门保留不动。
    - ``target``：可传入 ``Target``，从中取 ``basis_gates``（与显式
      ``basis_gates`` 二选一；都未给时默认目标门集为 ``("cx",)``）。
    - ``skip_unsupported``：``True`` 时，对不在门集内且无分解规则的门保持原样；
      ``False``（默认）时遇到这类双比特门抛 ``ValueError``。
    """

    def __init__(self, basis_gates=None, *, target=None, skip_unsupported: bool = False) -> None:
        if basis_gates is None and target is not None:
            basis_gates = getattr(target, "basis_gates", None)
        if not basis_gates:
            basis_gates = ("cx",)
        self.basis_gates = frozenset(canonical_gate_name(name) for name in basis_gates)
        self.skip_unsupported = bool(skip_unsupported)

    def run(self, circuit: Circuit) -> Circuit:
        out: list[dict] = []
        for gate in circuit_gate_dicts(circuit):
            name = canonical_gate_name(gate["type"])
            if name in self.basis_gates:
                out.append(gate)
                continue
            rule = _RULES.get(name)
            replacement = rule(gate) if rule is not None else None
            if replacement is not None:
                out.extend(replacement)
                continue
            if self.skip_unsupported:
                out.append(gate)
                continue
            qubits = instruction_qubits(gate)
            if len(qubits) + len(instruction_controls(gate)) >= 2:
                raise ValueError(
                    f"DecomposePass: gate '{name}' is not in basis {sorted(self.basis_gates)} "
                    "and has no decomposition rule"
                )
            out.append(gate)
        return circuit_from_gates(circuit, out)

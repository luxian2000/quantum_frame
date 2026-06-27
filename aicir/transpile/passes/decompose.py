"""DecomposePass：把高级门分解到目标门集（NEXT.md 第 2 节）。

分解规则由 ``GateSpec.decomposition`` 驱动（NEXT.md §7），内置一组经数值
验证的标准规则，把双比特高级门（``swap``/``cz``/``cy``）改写为 ``cx`` +
单比特门形式：

- ``swap(a, b)``  -> ``cx(a,[b]) · cx(b,[a]) · cx(a,[b])``
- ``cz(t,[c])``   -> ``h(t) · cx(t,[c]) · h(t)``
- ``cy(t,[c])``   -> ``rz(-pi/2,t) · cx(t,[c]) · rz(pi/2,t)``

分解只针对“不在目标门集内且存在规则”的门；已在门集内的门原样保留。
规则展开产生的单比特门为 ``hadamard``/``rz``，本片不再把任意单比特门
进一步做 Euler 基底翻译（留待后续）。受控形式仅支持单控制位。注册自定义门
时携带 ``decomposition`` 即可被本 pass 自动识别。
"""

from __future__ import annotations

from ...core.circuit import Circuit
from ...gates import canonical_gate_name, gate_decomposition
from ...ir import circuit_gate_dicts, instruction_controls, instruction_qubits
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates

__all__ = ["DecomposePass"]


def _apply_rule(rule, gate: dict) -> list[dict] | None:
    """把门规范化为 ``(qubits, controls, control_states, params)`` 后调用分解规则。"""

    qubits = tuple(instruction_qubits(gate))
    controls = tuple(instruction_controls(gate))
    states = gate.get("control_states")
    states = tuple(int(s) for s in states) if states is not None else None
    params = gate.get("parameter")
    return rule(qubits, controls, states, params)


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
            rule = gate_decomposition(name)
            replacement = _apply_rule(rule, gate) if rule is not None else None
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

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

import math

import numpy as np

from ...core.circuit import Circuit
from ...gates import canonical_gate_name, gate_decomposition, gate_matrix
from ...ir import (
    circuit_instructions,
    instruction_control_states,
    instruction_controls,
    instruction_name,
    instruction_parameter,
    instruction_qubits,
    instruction_to_gate_dict,
)
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates

__all__ = ["DecomposePass"]

_ZYZ_BASIS = frozenset({"rz", "ry"})


def _zyz_angles(u: np.ndarray) -> tuple[float, float, float]:
    """把 2x2 幺正分解为 ``Rz(beta)·Ry(gamma)·Rz(delta)`` 的角度（忽略全局相位）。"""
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    v = u / np.sqrt(det + 0j)  # 归一到 SU(2)
    c, s = abs(v[0, 0]), abs(v[1, 0])
    gamma = 2.0 * math.atan2(s, c)
    if s < 1e-12:  # 对角：仅 beta+delta 可定
        return float(2.0 * np.angle(v[1, 1])), float(gamma), 0.0
    if c < 1e-12:  # 反对角：仅 beta-delta 可定
        ang = float(np.angle(v[1, 0]))
        return ang, float(gamma), -ang
    a, b = float(np.angle(v[1, 1])), float(np.angle(v[1, 0]))
    return a + b, float(gamma), a - b


def _zyz_decomposition(name: str, instruction, qubit: int) -> list[dict] | None:
    """单比特门 -> ``rz·ry·rz`` 门字典序列（基底等价至全局相位）；不可分解返回 ``None``。"""
    local = gate_matrix(name, instruction_parameter(instruction, ()), None)
    if local is None:
        return None
    u = np.asarray(local, dtype=complex)
    if u.shape != (2, 2):
        return None
    beta, gamma, delta = _zyz_angles(u)
    return [
        {"type": "rz", "target_qubit": qubit, "parameter": delta},
        {"type": "ry", "target_qubit": qubit, "parameter": gamma},
        {"type": "rz", "target_qubit": qubit, "parameter": beta},
    ]


def _apply_rule(rule, instruction) -> list[dict] | None:
    """把门规范化为 ``(qubits, controls, control_states, params)`` 后调用分解规则。"""

    qubits = tuple(instruction_qubits(instruction))
    controls = tuple(instruction_controls(instruction))
    states = tuple(int(s) for s in instruction_control_states(instruction)) if controls else None
    params = instruction_parameter(instruction)
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
        for instruction in circuit_instructions(circuit):
            name = canonical_gate_name(instruction_name(instruction))
            if name in self.basis_gates:
                out.append(instruction_to_gate_dict(instruction))
                continue
            rule = gate_decomposition(name)
            replacement = _apply_rule(rule, instruction) if rule is not None else None
            if replacement is not None:
                out.extend(replacement)
                continue
            # 单比特门：基底含 rz/ry 时经 ZYZ 翻译到目标基底（等价至全局相位）。
            qubits = instruction_qubits(instruction)
            if (
                len(qubits) == 1
                and not instruction_controls(instruction)
                and _ZYZ_BASIS <= self.basis_gates
            ):
                zyz = _zyz_decomposition(name, instruction, int(qubits[0]))
                if zyz is not None:
                    out.extend(zyz)
                    continue
            if self.skip_unsupported:
                out.append(instruction_to_gate_dict(instruction))
                continue
            qubits = instruction_qubits(instruction)
            if len(qubits) + len(instruction_controls(instruction)) >= 2:
                raise ValueError(
                    f"DecomposePass: gate '{name}' is not in basis {sorted(self.basis_gates)} "
                    "and has no decomposition rule"
                )
            out.append(instruction_to_gate_dict(instruction))
        return circuit_from_gates(circuit, out)

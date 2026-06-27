"""门分解规则（NEXT.md §7：``GateSpec.decomposition`` 字段驱动）。

每条规则签名 ``(qubits, controls, control_states, params) -> list[dict] | None``，
只构造**纯门字典**（不依赖 ``aicir.ir``/``aicir.transpile``，避免与门注册表的
循环导入）。``transpile`` 的 ``DecomposePass`` 先把门规范化为
``(qubits, controls, control_states, params)`` 再调用本规则。

返回 ``None`` 表示当前入参形态不适用（例如受控形式带多个控制位），调用方
应保留原门或按目标门集策略处理。受控规则仅支持单控制位、控制态为 1。
"""

from __future__ import annotations

import math

__all__ = ["decompose_swap", "decompose_cz", "decompose_cy"]


def _single_control(controls, control_states) -> int | None:
    """受控门归一：恰有一个控制位且控制态为 1 时返回控制位，否则 ``None``。"""

    if len(controls) != 1:
        return None
    if control_states is not None and tuple(int(s) for s in control_states) != (1,):
        return None
    return int(controls[0])


def decompose_swap(qubits, controls, control_states, params) -> list[dict] | None:
    """``swap(a, b)`` -> ``cx(a,[b]) · cx(b,[a]) · cx(a,[b])``。"""

    if len(qubits) != 2 or controls:
        return None
    a, b = int(qubits[0]), int(qubits[1])
    cab = {"type": "cx", "target_qubit": a, "control_qubits": [b], "control_states": [1]}
    cba = {"type": "cx", "target_qubit": b, "control_qubits": [a], "control_states": [1]}
    return [dict(cab), dict(cba), dict(cab)]


def decompose_cz(qubits, controls, control_states, params) -> list[dict] | None:
    """``cz(t,[c])`` -> ``h(t) · cx(t,[c]) · h(t)``。"""

    control = _single_control(controls, control_states)
    if len(qubits) != 1 or control is None:
        return None
    t = int(qubits[0])
    return [
        {"type": "hadamard", "target_qubit": t},
        {"type": "cx", "target_qubit": t, "control_qubits": [control], "control_states": [1]},
        {"type": "hadamard", "target_qubit": t},
    ]


def decompose_cy(qubits, controls, control_states, params) -> list[dict] | None:
    """``cy(t,[c])`` -> ``rz(-pi/2,t) · cx(t,[c]) · rz(pi/2,t)``。"""

    control = _single_control(controls, control_states)
    if len(qubits) != 1 or control is None:
        return None
    t = int(qubits[0])
    return [
        {"type": "rz", "target_qubit": t, "parameter": -math.pi / 2.0},
        {"type": "cx", "target_qubit": t, "control_qubits": [control], "control_states": [1]},
        {"type": "rz", "target_qubit": t, "parameter": math.pi / 2.0},
    ]

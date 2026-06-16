"""
aicir/core/io/wuyue_io.py

WuYue 互操作桥接层。

本模块把 WuYue 作为可选依赖：导入 aicir 不要求安装 wuyue，
只有调用转换函数时才尝试导入 WuYue SDK 对象。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ...gates import canonical_gate_name
from ...ir import circuit_gate_dicts, has_circuit_instructions
from ..circuit import Circuit


def _require_wuyue():
    try:
        from wuyue.circuit.circuit import QuantumCircuit
        from wuyue.element import gate as gates
        from wuyue.register.classicalregister import ClassicalRegister
        from wuyue.register.quantumregister import QuantumRegister
    except ImportError as exc:  # pragma: no cover - 本地测试环境安装了 wuyue。
        raise ImportError(
            "WuYue 互操作需要安装可选依赖 wuyue；请先安装 wuyue 后再调用本函数。"
        ) from exc
    return SimpleNamespace(
        QuantumCircuit=QuantumCircuit,
        QuantumRegister=QuantumRegister,
        ClassicalRegister=ClassicalRegister,
        gates=gates,
    )


def _float_param(value: Any) -> float:
    """把 WuYue/aicir 参数值转换为当前互操作层支持的数值角度。"""

    try:
        return float(value)
    except TypeError as exc:
        raise TypeError("WuYue 互操作当前仅支持已绑定的数值参数") from exc


def _control_data(gate: dict[str, Any]) -> tuple[list[int], list[int]]:
    controls = [int(qubit) for qubit in gate.get("control_qubits", []) or []]
    raw_states = gate.get("control_states")
    if raw_states is None:
        return controls, [1] * len(controls)
    states = [int(state) for state in raw_states]
    if len(states) != len(controls):
        raise ValueError("control_states 的长度必须与 control_qubits 一致")
    if any(state not in (0, 1) for state in states):
        raise ValueError("WuYue 互操作仅支持二值控制态 0/1")
    return controls, states


def _measurement_width(gate: dict[str, Any], n_qubits: int) -> int:
    qubits = [int(qubit) for qubit in gate.get("qubits", []) or []]
    return len(qubits) if qubits else int(n_qubits)


def _bit_list(register: Any, indices: list[int]):
    return register[indices]


def _append_single(qc: Any, wuyue: Any, qreg: Any, gate_cls: Any, qubit: int) -> None:
    qc.add(gate_cls, _bit_list(qreg, [int(qubit)]))


def _with_zero_control_wrappers(
    qc: Any,
    wuyue: Any,
    qreg: Any,
    controls: list[int],
    states: list[int],
    emit,
) -> None:
    """用前后 X 门表达 |0> 控制态，避免依赖 WuYue 私有控制态形态。"""

    zero_controls = [control for control, state in zip(controls, states) if state == 0]
    for control in zero_controls:
        _append_single(qc, wuyue, qreg, wuyue.gates.X, control)
    emit()
    for control in reversed(zero_controls):
        _append_single(qc, wuyue, qreg, wuyue.gates.X, control)


def circuit_to_wuyue(circuit: Any):
    """将 aicir Circuit/CircuitIR 转换为 WuYue ``QuantumCircuit``。

    当前支持 WuYue 原生门集中的基础单比特门、参数旋转门、u2/u3、
    cx/cz、swap、IsingZZ、toffoli 和线路内测量标记。
    """

    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("circuit 需要具备 n_qubits 和 typed IR operations 或 gates 序列")

    wuyue = _require_wuyue()
    gates = circuit_gate_dicts(circuit)
    n_qubits = int(circuit.n_qubits)
    n_clbits = sum(
        _measurement_width(gate, n_qubits)
        for gate in gates
        if canonical_gate_name(str(gate["type"])) == "measure"
    )
    qreg = wuyue.QuantumRegister(n_qubits)
    creg = wuyue.ClassicalRegister(max(1, n_clbits))
    qc = wuyue.QuantumCircuit(qreg, creg)
    next_clbit = 0

    for gate in gates:
        gtype = canonical_gate_name(str(gate["type"]))

        if gtype == "measure":
            qubits = [int(qubit) for qubit in gate.get("qubits", []) or []]
            if not qubits:
                qubits = list(range(n_qubits))
            width = len(qubits)
            qc.add(
                wuyue.gates.MEASURE,
                _bit_list(qreg, qubits),
                cbit=_bit_list(creg, list(range(next_clbit, next_clbit + width))),
            )
            next_clbit += width
            continue

        if gtype == "identity":
            for qubit in range(n_qubits):
                _append_single(qc, wuyue, qreg, wuyue.gates.I, qubit)
            continue

        if gtype in {"pauli_x", "pauli_y", "pauli_z", "hadamard", "s_gate", "t_gate"}:
            q = int(gate["target_qubit"])
            gate_cls = {
                "pauli_x": wuyue.gates.X,
                "pauli_y": wuyue.gates.Y,
                "pauli_z": wuyue.gates.Z,
                "hadamard": wuyue.gates.H,
                "s_gate": wuyue.gates.S,
                "t_gate": wuyue.gates.T,
            }[gtype]
            _append_single(qc, wuyue, qreg, gate_cls, q)
            continue

        if gtype in {"rx", "ry", "rz"}:
            q = int(gate["target_qubit"])
            theta = _float_param(gate["parameter"])
            gate_cls = {"rx": wuyue.gates.RX, "ry": wuyue.gates.RY, "rz": wuyue.gates.RZ}[gtype]
            qc.add(gate_cls, _bit_list(qreg, [q]), paras=[theta])
            continue

        if gtype == "u2":
            q = int(gate["target_qubit"])
            phi, lam = gate["parameter"]
            qc.add(
                wuyue.gates.U2,
                _bit_list(qreg, [q]),
                paras=[_float_param(phi), _float_param(lam)],
            )
            continue

        if gtype == "u3":
            q = int(gate["target_qubit"])
            theta, phi, lam = gate["parameter"]
            qc.add(
                wuyue.gates.U3,
                _bit_list(qreg, [q]),
                paras=[_float_param(theta), _float_param(phi), _float_param(lam)],
            )
            continue

        if gtype == "swap":
            qc.add(
                wuyue.gates.SWAP,
                _bit_list(qreg, [int(gate["qubit_1"]), int(gate["qubit_2"])]),
            )
            continue

        if gtype == "rzz":
            qc.add(
                wuyue.gates.IsingZZ,
                _bit_list(qreg, [int(gate["qubit_1"]), int(gate["qubit_2"])]),
                paras=[_float_param(gate["parameter"])],
            )
            continue

        if gtype in {"cx", "cz", "toffoli"}:
            controls, states = _control_data(gate)
            target = int(gate["target_qubit"])
            if gtype == "toffoli":
                if len(controls) != 2:
                    raise ValueError("WuYue 互操作仅支持双控制 toffoli")

                def emit_toffoli():
                    qc.add(
                        wuyue.gates.TOFFOLI,
                        _bit_list(qreg, [target]),
                        control=_bit_list(qreg, controls),
                    )

                _with_zero_control_wrappers(qc, wuyue, qreg, controls, states, emit_toffoli)
                continue

            if len(controls) != 1:
                raise ValueError(f"WuYue 互操作仅支持单控制门，当前: {gtype} controls={controls}")

            gate_cls = {"cx": wuyue.gates.CX, "cz": wuyue.gates.CZ}[gtype]

            def emit_controlled():
                qc.add(gate_cls, _bit_list(qreg, [target]), control=_bit_list(qreg, controls))

            _with_zero_control_wrappers(qc, wuyue, qreg, controls, states, emit_controlled)
            continue

        raise ValueError(f"WuYue 互操作暂不支持门类型: {gtype}")

    return qc


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in values]


def _wuyue_gate_sequence(wuyue_circuit: Any) -> list[Any]:
    if not hasattr(wuyue_circuit, "apply_circuit") or not hasattr(wuyue_circuit, "gates"):
        raise TypeError("wuyue_circuit 需要是 WuYue QuantumCircuit 或兼容对象")
    wuyue_circuit.apply_circuit()
    return list(wuyue_circuit.gates)


def circuit_from_wuyue(wuyue_circuit: Any) -> Circuit:
    """将 WuYue ``QuantumCircuit`` 转换为 aicir ``Circuit``。"""

    _require_wuyue()
    gates: list[dict[str, Any]] = []

    for op in _wuyue_gate_sequence(wuyue_circuit):
        name = str(getattr(op, "name", "")).upper()
        targets = _int_list(getattr(op, "target", []))
        controls = _int_list(getattr(op, "ctrl", []))
        params = [_float_param(param) for param in getattr(op, "paras", []) or []]

        if name in {"BARRIER", "RESET", "I"}:
            continue
        if name == "MEASURE":
            if len(targets) != 1:
                raise ValueError("WuYue MEASURE 指令应作用在单个 qubit")
            gates.append({"type": "measure", "qubits": [targets[0]]})
            continue

        if name in {"X", "Y", "Z", "H", "S", "T"}:
            gate_name = {
                "X": "pauli_x",
                "Y": "pauli_y",
                "Z": "pauli_z",
                "H": "hadamard",
                "S": "s_gate",
                "T": "t_gate",
            }[name]
            gates.append({"type": gate_name, "target_qubit": targets[0]})
            continue

        if name in {"RX", "RY", "RZ"}:
            gates.append({"type": name.lower(), "target_qubit": targets[0], "parameter": params[0]})
            continue

        if name == "U2":
            gates.append({"type": "u2", "target_qubit": targets[0], "parameter": [params[0], params[1]]})
            continue

        if name == "U3":
            gates.append({"type": "u3", "target_qubit": targets[0], "parameter": [params[0], params[1], params[2]]})
            continue

        if name == "SWAP":
            gates.append({"type": "swap", "qubit_1": targets[0], "qubit_2": targets[1]})
            continue

        if name in {"IZZ", "ISINGZZ"}:
            gates.append({"type": "rzz", "qubit_1": targets[0], "qubit_2": targets[1], "parameter": params[0]})
            continue

        if name in {"CX", "CNOT", "CZ"}:
            if len(controls) != 1:
                raise ValueError(f"WuYue {name} 指令应包含一个控制位")
            gate_name = "cx" if name in {"CX", "CNOT"} else "cz"
            gates.append(
                {
                    "type": gate_name,
                    "target_qubit": targets[0],
                    "control_qubits": controls,
                    "control_states": [1],
                }
            )
            continue

        if name in {"TOFFOLI", "CCX"}:
            if len(controls) != 2:
                raise ValueError(f"WuYue {name} 指令应包含两个控制位")
            gates.append({"type": "toffoli", "target_qubit": targets[0], "control_qubits": controls})
            continue

        raise ValueError(f"WuYue 互操作暂不支持指令: {name}")

    n_qubits = int(getattr(wuyue_circuit, "qubits", 0))
    if n_qubits <= 0 and hasattr(wuyue_circuit, "qreg"):
        n_qubits = len(wuyue_circuit.qreg)
    return Circuit(*gates, n_qubits=n_qubits)


to_wuyue = circuit_to_wuyue
from_wuyue = circuit_from_wuyue


__all__ = ["circuit_to_wuyue", "circuit_from_wuyue", "to_wuyue", "from_wuyue"]

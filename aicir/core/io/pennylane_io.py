"""
aicir/core/io/pennylane_io.py

PennyLane 互操作桥接层。

本模块把 PennyLane 作为可选依赖：导入 aicir 不要求安装 pennylane，
只有调用转换函数时才尝试导入 PennyLane 对象。
"""

from __future__ import annotations

from typing import Any

from ...gates import canonical_gate_name
from ...ir import circuit_gate_dicts, has_circuit_instructions
from ..circuit import Circuit


def _require_pennylane():
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - 本地测试环境安装了 pennylane。
        raise ImportError(
            "PennyLane 互操作需要安装可选依赖 pennylane；请先安装 pennylane 后再调用本函数。"
        ) from exc
    return qml


def _float_param(value: Any) -> float:
    """把 PennyLane/aicir 参数值转换为当前互操作层支持的数值角度。"""

    try:
        return float(value)
    except TypeError as exc:
        raise TypeError("PennyLane 互操作当前仅支持已绑定的数值参数") from exc


def _int_wires(wires: Any) -> list[int]:
    """将 PennyLane wires 转换为 aicir 使用的非负整数 qubit 下标。"""

    try:
        values = wires.tolist()
    except AttributeError:
        values = list(wires)
    out: list[int] = []
    for wire in values:
        if isinstance(wire, bool):
            raise TypeError("PennyLane wire 必须是非负整数")
        try:
            value = int(wire)
        except (TypeError, ValueError) as exc:
            raise TypeError("PennyLane wire 必须是非负整数") from exc
        if value < 0 or value != wire:
            raise TypeError("PennyLane wire 必须是非负整数")
        out.append(value)
    return out


def _control_data(gate: dict[str, Any]) -> tuple[list[int], list[int]]:
    controls = [int(qubit) for qubit in gate.get("control_qubits", []) or []]
    raw_states = gate.get("control_states")
    if raw_states is None:
        return controls, [1] * len(controls)
    states = [int(state) for state in raw_states]
    if len(states) != len(controls):
        raise ValueError("control_states 的长度必须与 control_qubits 一致")
    if any(state not in (0, 1) for state in states):
        raise ValueError("PennyLane 互操作仅支持二值控制态 0/1")
    return controls, states


def _with_zero_control_wrappers(qml: Any, ops: list[Any], controls: list[int], states: list[int], emit) -> None:
    """用前后 PauliX 表达 |0> 控制态，避免依赖 PennyLane 私有 controlled 形态。"""

    zero_controls = [control for control, state in zip(controls, states) if state == 0]
    for control in zero_controls:
        ops.append(qml.PauliX(wires=control))
    emit()
    for control in reversed(zero_controls):
        ops.append(qml.PauliX(wires=control))


def circuit_to_pennylane(circuit: Any):
    """将 aicir Circuit/CircuitIR 转换为 PennyLane ``QuantumScript``。

    当前互操作面聚焦幺正门序列；aicir 的线路内 ``measure`` 标记不具备
    PennyLane 普通 Operation 语义，因此暂不在本函数中转换。
    """

    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("circuit 需要具备 n_qubits 和 typed IR operations 或 gates 序列")

    qml = _require_pennylane()
    ops: list[Any] = []
    n_qubits = int(circuit.n_qubits)

    for gate in circuit_gate_dicts(circuit):
        gtype = canonical_gate_name(str(gate["type"]))

        if gtype == "measure":
            raise ValueError("PennyLane 互操作当前不支持线路内 measure 标记")

        if gtype == "identity":
            ops.append(qml.Identity(wires=list(range(n_qubits))))
            continue

        if gtype in {"pauli_x", "pauli_y", "pauli_z", "hadamard", "s_gate", "t_gate"}:
            q = int(gate["target_qubit"])
            op_cls = {
                "pauli_x": qml.PauliX,
                "pauli_y": qml.PauliY,
                "pauli_z": qml.PauliZ,
                "hadamard": qml.Hadamard,
                "s_gate": qml.S,
                "t_gate": qml.T,
            }[gtype]
            ops.append(op_cls(wires=q))
            continue

        if gtype in {"rx", "ry", "rz"}:
            q = int(gate["target_qubit"])
            theta = _float_param(gate["parameter"])
            ops.append({"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}[gtype](theta, wires=q))
            continue

        if gtype == "u2":
            q = int(gate["target_qubit"])
            phi, lam = gate["parameter"]
            ops.append(qml.U2(_float_param(phi), _float_param(lam), wires=q))
            continue

        if gtype == "u3":
            q = int(gate["target_qubit"])
            theta, phi, lam = gate["parameter"]
            ops.append(qml.U3(_float_param(theta), _float_param(phi), _float_param(lam), wires=q))
            continue

        if gtype == "swap":
            ops.append(qml.SWAP(wires=[int(gate["qubit_1"]), int(gate["qubit_2"])]))
            continue

        if gtype == "rzz":
            ops.append(qml.IsingZZ(_float_param(gate["parameter"]), wires=[int(gate["qubit_1"]), int(gate["qubit_2"])]))
            continue

        if gtype == "rxx":
            ops.append(qml.IsingXX(_float_param(gate["parameter"]), wires=[int(gate["qubit_1"]), int(gate["qubit_2"])]))
            continue

        if gtype in {"cx", "cy", "cz", "crx", "cry", "crz", "toffoli"}:
            controls, states = _control_data(gate)
            target = int(gate["target_qubit"])
            if gtype == "toffoli":
                if len(controls) != 2:
                    raise ValueError("PennyLane 互操作仅支持双控制 toffoli")

                def emit_toffoli():
                    ops.append(qml.Toffoli(wires=[controls[0], controls[1], target]))

                _with_zero_control_wrappers(qml, ops, controls, states, emit_toffoli)
                continue

            if len(controls) != 1:
                raise ValueError(f"PennyLane 互操作仅支持单控制门，当前: {gtype} controls={controls}")

            control = controls[0]
            if gtype in {"cx", "cy", "cz"}:
                op_cls = {"cx": qml.CNOT, "cy": qml.CY, "cz": qml.CZ}[gtype]

                def emit_controlled():
                    ops.append(op_cls(wires=[control, target]))

            else:
                theta = _float_param(gate["parameter"])
                op_cls = {"crx": qml.CRX, "cry": qml.CRY, "crz": qml.CRZ}[gtype]

                def emit_controlled():
                    ops.append(op_cls(theta, wires=[control, target]))

            _with_zero_control_wrappers(qml, ops, controls, states, emit_controlled)
            continue

        raise ValueError(f"PennyLane 互操作暂不支持门类型: {gtype}")

    return qml.tape.QuantumScript(ops, [])


def _operations_from_pennylane(pennylane_circuit: Any) -> list[Any]:
    operations = getattr(pennylane_circuit, "operations", None)
    if operations is not None:
        return list(operations)
    tape = getattr(pennylane_circuit, "tape", None) or getattr(pennylane_circuit, "qtape", None)
    if tape is not None and getattr(tape, "operations", None) is not None:
        return list(tape.operations)
    raise TypeError("pennylane_circuit 需要提供 operations 序列或 tape.operations")


def circuit_from_pennylane(pennylane_circuit: Any) -> Circuit:
    """将 PennyLane ``QuantumScript``/tape-like 对象转换为 aicir ``Circuit``。"""

    _require_pennylane()
    gates: list[dict[str, Any]] = []
    max_wire = -1

    for op in _operations_from_pennylane(pennylane_circuit):
        name = str(getattr(op, "name", "")).lower()
        wires = _int_wires(op.wires)
        params = [_float_param(param) for param in getattr(op, "parameters", [])]
        if wires:
            max_wire = max(max_wire, max(wires))

        if name == "barrier":
            continue
        if name == "identity":
            continue

        if name in {"paulix", "pauliy", "pauliz", "hadamard", "s", "t"}:
            gate_name = {
                "paulix": "pauli_x",
                "pauliy": "pauli_y",
                "pauliz": "pauli_z",
                "hadamard": "hadamard",
                "s": "s_gate",
                "t": "t_gate",
            }[name]
            gates.append({"type": gate_name, "target_qubit": wires[0]})
            continue

        if name in {"rx", "ry", "rz"}:
            gates.append({"type": name, "target_qubit": wires[0], "parameter": params[0]})
            continue

        if name == "u2":
            gates.append({"type": "u2", "target_qubit": wires[0], "parameter": [params[0], params[1]]})
            continue

        if name == "u3":
            gates.append({"type": "u3", "target_qubit": wires[0], "parameter": [params[0], params[1], params[2]]})
            continue

        if name == "swap":
            gates.append({"type": "swap", "qubit_1": wires[0], "qubit_2": wires[1]})
            continue

        if name == "isingzz":
            gates.append({"type": "rzz", "qubit_1": wires[0], "qubit_2": wires[1], "parameter": params[0]})
            continue

        if name == "isingxx":
            gates.append({"type": "rxx", "qubit_1": wires[0], "qubit_2": wires[1], "parameter": params[0]})
            continue

        if name in {"cnot", "cy", "cz"}:
            gate_name = "cx" if name == "cnot" else name
            gates.append(
                {
                    "type": gate_name,
                    "target_qubit": wires[1],
                    "control_qubits": [wires[0]],
                    "control_states": [1],
                }
            )
            continue

        if name in {"crx", "cry", "crz"}:
            gates.append(
                {
                    "type": name,
                    "target_qubit": wires[1],
                    "control_qubits": [wires[0]],
                    "control_states": [1],
                    "parameter": params[0],
                }
            )
            continue

        if name == "toffoli":
            gates.append({"type": "toffoli", "target_qubit": wires[2], "control_qubits": [wires[0], wires[1]]})
            continue

        raise ValueError(f"PennyLane 互操作暂不支持指令: {name}")

    return Circuit(*gates, n_qubits=max_wire + 1 if max_wire >= 0 else 0)


to_pennylane = circuit_to_pennylane
from_pennylane = circuit_from_pennylane


__all__ = ["circuit_to_pennylane", "circuit_from_pennylane", "to_pennylane", "from_pennylane"]

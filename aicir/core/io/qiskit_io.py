"""
aicir/core/io/qiskit_io.py

Qiskit 互操作桥接层。

本模块把 Qiskit 作为可选依赖：导入 aicir 不要求安装 qiskit，
只有调用转换函数时才尝试导入 Qiskit 对象。
"""

from __future__ import annotations

from typing import Any

from ...gates import canonical_gate_name
from ...ir import circuit_gate_dicts, has_circuit_instructions
from ..circuit import Circuit


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:  # pragma: no cover - 本地测试环境安装了 qiskit。
        raise ImportError(
            "Qiskit 互操作需要安装可选依赖 qiskit；请先安装 qiskit 后再调用本函数。"
        ) from exc
    return QuantumCircuit


def _qiskit_qubit_index(qc: Any, qubit: Any) -> int:
    """从 Qiskit qubit 对象取得线路内的整数下标。"""

    try:
        return int(qc.find_bit(qubit).index)
    except AttributeError as exc:
        raise TypeError("qiskit_circuit 需要提供 find_bit(qubit).index") from exc


def _float_param(value: Any) -> float:
    """把 Qiskit/aicir 参数值转换为当前互操作层支持的数值角度。"""

    try:
        return float(value)
    except TypeError as exc:
        raise TypeError("Qiskit 互操作当前仅支持已绑定的数值参数") from exc


def _control_data(gate: dict[str, Any]) -> tuple[list[int], list[int]]:
    controls = [int(qubit) for qubit in gate.get("control_qubits", []) or []]
    raw_states = gate.get("control_states")
    if raw_states is None:
        return controls, [1] * len(controls)
    states = [int(state) for state in raw_states]
    if len(states) != len(controls):
        raise ValueError("control_states 的长度必须与 control_qubits 一致")
    if any(state not in (0, 1) for state in states):
        raise ValueError("Qiskit 互操作仅支持二值控制态 0/1")
    return controls, states


def _with_zero_control_wrappers(qc: Any, controls: list[int], states: list[int], emit) -> None:
    """用前后 X 门表达 |0> 控制态，避免依赖 Qiskit 私有 controlled-gate 形态。"""

    zero_controls = [control for control, state in zip(controls, states) if state == 0]
    for control in zero_controls:
        qc.x(control)
    emit()
    for control in reversed(zero_controls):
        qc.x(control)


def _measurement_width(gate: dict[str, Any], n_qubits: int) -> int:
    qubits = [int(qubit) for qubit in gate.get("qubits", []) or []]
    return len(qubits) if qubits else int(n_qubits)


def _append_measurement(qc: Any, qubits: list[int], next_clbit: int) -> int:
    if not qubits:
        qubits = list(range(qc.num_qubits))
    for offset, qubit in enumerate(qubits):
        qc.measure(qubit, next_clbit + offset)
    return next_clbit + len(qubits)


def circuit_to_qiskit(circuit: Any):
    """将 aicir Circuit/CircuitIR 转换为 Qiskit ``QuantumCircuit``。

    支持门集与当前 QASM/GateSpec 第一批互操作面保持一致：基础单比特门、
    参数旋转、受控门、swap、rzz/rxx、u2/u3、toffoli 和线路内测量标记。
    """

    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("circuit 需要具备 n_qubits 和 typed IR operations 或 gates 序列")

    QuantumCircuit = _require_qiskit()
    gates = circuit_gate_dicts(circuit)
    n_qubits = int(circuit.n_qubits)
    n_clbits = sum(
        _measurement_width(gate, n_qubits)
        for gate in gates
        if canonical_gate_name(str(gate["type"])) == "measure"
    )
    qc = QuantumCircuit(n_qubits, n_clbits)
    next_clbit = 0

    for gate in gates:
        gtype = canonical_gate_name(str(gate["type"]))

        if gtype == "measure":
            qubits = [int(qubit) for qubit in gate.get("qubits", []) or []]
            next_clbit = _append_measurement(qc, qubits, next_clbit)
            continue

        if gtype in {"pauli_x", "pauli_y", "pauli_z", "hadamard", "s_gate", "t_gate"}:
            q = int(gate["target_qubit"])
            {
                "pauli_x": qc.x,
                "pauli_y": qc.y,
                "pauli_z": qc.z,
                "hadamard": qc.h,
                "s_gate": qc.s,
                "t_gate": qc.t,
            }[gtype](q)
            continue

        if gtype in {"rx", "ry", "rz"}:
            q = int(gate["target_qubit"])
            theta = _float_param(gate["parameter"])
            {"rx": qc.rx, "ry": qc.ry, "rz": qc.rz}[gtype](theta, q)
            continue

        if gtype == "u2":
            q = int(gate["target_qubit"])
            phi, lam = gate["parameter"]
            qc.u(1.5707963267948966, _float_param(phi), _float_param(lam), q)
            continue

        if gtype == "u3":
            q = int(gate["target_qubit"])
            theta, phi, lam = gate["parameter"]
            qc.u(_float_param(theta), _float_param(phi), _float_param(lam), q)
            continue

        if gtype == "swap":
            qc.swap(int(gate["qubit_1"]), int(gate["qubit_2"]))
            continue

        if gtype in {"rzz", "rxx"}:
            q1 = int(gate["qubit_1"])
            q2 = int(gate["qubit_2"])
            theta = _float_param(gate["parameter"])
            {"rzz": qc.rzz, "rxx": qc.rxx}[gtype](theta, q1, q2)
            continue

        if gtype in {"cx", "cy", "cz", "crx", "cry", "crz", "toffoli"}:
            controls, states = _control_data(gate)
            target = int(gate["target_qubit"])
            if gtype == "toffoli":
                if len(controls) != 2:
                    raise ValueError("Qiskit 互操作仅支持双控制 toffoli")

                def emit_toffoli():
                    qc.ccx(controls[0], controls[1], target)

                _with_zero_control_wrappers(qc, controls, states, emit_toffoli)
                continue

            if len(controls) != 1:
                raise ValueError(f"Qiskit 互操作仅支持单控制门，当前: {gtype} controls={controls}")

            control = controls[0]
            if gtype in {"cx", "cy", "cz"}:
                method = {"cx": qc.cx, "cy": qc.cy, "cz": qc.cz}[gtype]

                def emit_controlled():
                    method(control, target)

            else:
                theta = _float_param(gate["parameter"])
                method = {"crx": qc.crx, "cry": qc.cry, "crz": qc.crz}[gtype]

                def emit_controlled():
                    method(theta, control, target)

            _with_zero_control_wrappers(qc, controls, states, emit_controlled)
            continue

        raise ValueError(f"Qiskit 互操作暂不支持门类型: {gtype}")

    return qc


def circuit_from_qiskit(qiskit_circuit: Any) -> Circuit:
    """将 Qiskit ``QuantumCircuit`` 转换为 aicir ``Circuit``。"""

    if not hasattr(qiskit_circuit, "data") or not hasattr(qiskit_circuit, "num_qubits"):
        raise TypeError("qiskit_circuit 需要是 Qiskit QuantumCircuit 或兼容对象")

    gates: list[dict[str, Any]] = []
    for item in qiskit_circuit.data:
        name = str(item.operation.name).lower()
        qubits = [_qiskit_qubit_index(qiskit_circuit, qubit) for qubit in item.qubits]
        params = [_float_param(param) for param in item.operation.params]

        if name == "barrier":
            continue
        if name == "measure":
            if len(qubits) != 1:
                raise ValueError("Qiskit measure 指令应作用在单个 qubit")
            gates.append({"type": "measure", "qubits": [qubits[0]]})
            continue
        if name in {"id", "delay"}:
            continue

        if name in {"x", "y", "z", "h", "s", "t"}:
            gate_name = {
                "x": "pauli_x",
                "y": "pauli_y",
                "z": "pauli_z",
                "h": "hadamard",
                "s": "s_gate",
                "t": "t_gate",
            }[name]
            gates.append({"type": gate_name, "target_qubit": qubits[0]})
            continue

        if name in {"rx", "ry", "rz", "p", "u1"}:
            gate_name = "rz" if name in {"p", "u1"} else name
            gates.append({"type": gate_name, "target_qubit": qubits[0], "parameter": params[0]})
            continue

        if name == "u2":
            gates.append({"type": "u2", "target_qubit": qubits[0], "parameter": [params[0], params[1]]})
            continue

        if name in {"u", "u3"}:
            gates.append(
                {"type": "u3", "target_qubit": qubits[0], "parameter": [params[0], params[1], params[2]]}
            )
            continue

        if name == "swap":
            gates.append({"type": "swap", "qubit_1": qubits[0], "qubit_2": qubits[1]})
            continue

        if name in {"rzz", "rxx"}:
            gates.append({"type": name, "qubit_1": qubits[0], "qubit_2": qubits[1], "parameter": params[0]})
            continue

        if name in {"cx", "cy", "cz"}:
            gates.append(
                {
                    "type": name,
                    "target_qubit": qubits[1],
                    "control_qubits": [qubits[0]],
                    "control_states": [1],
                }
            )
            continue

        if name in {"crx", "cry", "crz"}:
            gates.append(
                {
                    "type": name,
                    "target_qubit": qubits[1],
                    "control_qubits": [qubits[0]],
                    "control_states": [1],
                    "parameter": params[0],
                }
            )
            continue

        if name == "ccx":
            gates.append({"type": "toffoli", "target_qubit": qubits[2], "control_qubits": [qubits[0], qubits[1]]})
            continue

        raise ValueError(f"Qiskit 互操作暂不支持指令: {name}")

    return Circuit(*gates, n_qubits=int(qiskit_circuit.num_qubits))


to_qiskit = circuit_to_qiskit
from_qiskit = circuit_from_qiskit


__all__ = ["circuit_to_qiskit", "circuit_from_qiskit", "to_qiskit", "from_qiskit"]

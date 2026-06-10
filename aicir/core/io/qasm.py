"""
aicir/core/io/qasm.py

OpenQASM 2.0 / 3.0 导入导出（子集支持）。

支持门：
- 单比特: x, y, z, h, s, t, rx, ry, rz, p/u1, u2, u3/u
- 双比特: cx, cy, cz, swap, crx, cry, crz, rzz, rxx
- 三比特: ccx

当前不处理：if、reset、opaque、自定义 gate、cp/cu 家族。
导入时对 measure/barrier 语句采用跳过策略。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List

from ...ir import circuit_gate_dicts, has_circuit_instructions
from ..circuit import Circuit

_QASM2_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
_QASM3_HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'

_SINGLE_NO_PARAM_EXPORT = {
    "pauli_x": "x",
    "X": "x",
    "pauli_y": "y",
    "Y": "y",
    "pauli_z": "z",
    "Z": "z",
    "hadamard": "h",
    "H": "h",
    "s_gate": "s",
    "S": "s",
    "t_gate": "t",
    "T": "t",
}

_PARAM_EXPORT = {
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "u2": "u2",
}

_DOUBLE_EXPORT = {
    "cnot": "cx",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
    "rzz": "rzz",
    "rxx": "rxx",
}

_THREE_EXPORT = {
    "toffoli": "ccx",
    "ccnot": "ccx",
}

_IMPORT_SINGLE = {
    "x": "pauli_x",
    "y": "pauli_y",
    "z": "pauli_z",
    "h": "hadamard",
    "s": "s_gate",
    "t": "t_gate",
}

_IMPORT_PARAM = {"rx", "ry", "rz"}
_IMPORT_DOUBLE = {"cx", "cy", "cz", "swap"}


def _normalize_gate_type_for_export(gtype: str) -> str:
    """规范化门类型标签，便于统一导出映射。"""
    if gtype in {"u", "U"}:
        return "u3"
    if gtype == "u1":
        return "rz"
    if gtype == "p":
        return "rz"
    return gtype


def _parse_qubit_ref(token: str, reg_info: Dict[str, Dict[str, int]]) -> int:
    """解析量子位引用（支持 q[1] 或单比特寄存器名 q）。"""
    tok = token.strip()
    m = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", tok)
    if m:
        name, idx_s = m.group(1), m.group(2)
        if name not in reg_info:
            raise ValueError(f"未知量子寄存器: {name}")
        idx = int(idx_s)
        size = reg_info[name]["size"]
        if idx < 0 or idx >= size:
            raise ValueError(f"量子位索引越界: {name}[{idx}] (size={size})")
        return reg_info[name]["offset"] + idx

    # QASM 3 标量寄存器：qubit q;
    if tok in reg_info and reg_info[tok]["size"] == 1:
        return reg_info[tok]["offset"]

    raise ValueError(f"无法解析量子位引用: {token}")


def _split_operands(operand_text: str) -> List[str]:
    """按逗号切分操作数列表。"""
    return [x.strip() for x in operand_text.split(",") if x.strip()]


def _format_angle(theta) -> str:
    """将角度格式化为 QASM 可读表达式。"""
    value = float(theta)
    # 对常见 pi 比例做友好输出
    ratio = value / math.pi if math.pi != 0 else 0.0
    for n, d in [(0, 1), (1, 4), (1, 2), (3, 4), (1, 1), (3, 2), (2, 1)]:
        target = n / d
        if abs(ratio - target) < 1e-10:
            if n == 0:
                return "0"
            if d == 1:
                return "pi" if n == 1 else f"{n}*pi"
            return f"pi/{d}" if n == 1 else f"{n}*pi/{d}"
    return f"{value:.15g}"


def _parse_qasm_angle(expr: str) -> float:
    """解析 QASM 角度表达式，支持 pi 及其基本四则组合。"""
    safe = expr.strip().replace("^", "**")
    # 仅允许数字、pi、空白和 + - * / . ( )
    if not re.fullmatch(r"[0-9piPIeE+\-*/().\s]+", safe):
        raise ValueError(f"不支持的角度表达式: {expr}")
    safe = safe.replace("PI", "pi").replace("Pi", "pi")
    try:
        value = eval(safe, {"__builtins__": {}}, {"pi": math.pi})
    except Exception as exc:
        raise ValueError(f"角度表达式解析失败: {expr}") from exc
    return float(value)


def _normalized_control_data(gate: Dict[str, object]) -> tuple[List[int], List[int]]:
    """返回规范化后的控制位与控制态列表。"""
    controls = [int(q) for q in gate.get("control_qubits", []) or []]
    raw_states = gate.get("control_states")
    if raw_states is None:
        return controls, [1] * len(controls)

    control_states = [int(state) for state in raw_states]
    if len(control_states) != len(controls):
        raise ValueError("control_states 的长度必须与 control_qubits 一致")
    if any(state not in (0, 1) for state in control_states):
        raise ValueError("QASM 导出仅支持二值控制态 0/1")
    return controls, control_states


def _control_state_wrapper_lines(controls: List[int], control_states: List[int]) -> tuple[List[str], List[str]]:
    """将 |0> 控制态转换为前后包裹的 X 门。"""
    zero_controls = [ctrl for ctrl, state in zip(controls, control_states) if state == 0]
    pre = [f"x q[{ctrl}];" for ctrl in zero_controls]
    post = [f"x q[{ctrl}];" for ctrl in reversed(zero_controls)]
    return pre, post


def _qasm3_required_ancilla_count(circuit: Circuit) -> int:
    """计算 QASM 3.0 导出多控 crx/cry/crz 所需的最大辅助比特数。"""
    max_ancillas = 0
    for gate in circuit_gate_dicts(circuit):
        gtype = _normalize_gate_type_for_export(gate["type"])
        if gtype not in {"crx", "cry", "crz"}:
            continue
        controls, _ = _normalized_control_data(gate)
        if len(controls) > 1:
            max_ancillas = max(max_ancillas, len(controls) - 1)
    return max_ancillas


def _append_qasm3_multi_control_rotation(
    lines: List[str], gate_name: str, controls: List[int], target: int, theta: str
) -> None:
    """将多控 crx/cry/crz 分解为 ccx + 单控旋转门 + 反计算序列。"""
    if len(controls) < 2:
        raise ValueError(f"多控 {gate_name} 分解至少需要两个控制位")

    lines.append(f"ccx q[{controls[0]}],q[{controls[1]}],anc[0];")
    for control_index in range(2, len(controls)):
        prev_anc = f"anc[{control_index - 2}]"
        next_anc = f"anc[{control_index - 1}]"
        lines.append(f"ccx q[{controls[control_index]}],{prev_anc},{next_anc};")

    aggregate_control = f"anc[{len(controls) - 2}]"
    lines.append(f"{gate_name}({theta}) {aggregate_control},q[{target}];")

    for control_index in range(len(controls) - 1, 1, -1):
        prev_anc = f"anc[{control_index - 2}]"
        next_anc = f"anc[{control_index - 1}]"
        lines.append(f"ccx q[{controls[control_index]}],{prev_anc},{next_anc};")
    lines.append(f"ccx q[{controls[0]}],q[{controls[1]}],anc[0];")


def circuit_to_qasm(circuit: Circuit, version: str = "2.0") -> str:
    """将 Circuit 导出为 OpenQASM 字符串，支持 2.0 和 3.0。"""
    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("circuit 需要具备 n_qubits 和 typed IR operations 或 gates 序列")

    version_norm = str(version).strip()
    if version_norm not in {"2.0", "3.0"}:
        raise ValueError("version 仅支持 '2.0' 或 '3.0'")

    lines: List[str] = []
    if version_norm == "2.0":
        lines.append(_QASM2_HEADER.rstrip("\n"))
        lines.append(f"qreg q[{int(circuit.n_qubits)}];")
    else:
        lines.append(_QASM3_HEADER.rstrip("\n"))
        lines.append(f"qubit[{int(circuit.n_qubits)}] q;")
        ancilla_count = _qasm3_required_ancilla_count(circuit)
        if ancilla_count > 0:
            lines.append(f"qubit[{ancilla_count}] anc;")

    for gate in circuit_gate_dicts(circuit):
        gtype = _normalize_gate_type_for_export(gate["type"])
        controls, control_states = _normalized_control_data(gate)
        pre_lines, post_lines = _control_state_wrapper_lines(controls, control_states)
        lines.extend(pre_lines)

        if gtype in _SINGLE_NO_PARAM_EXPORT:
            qasm_gate = _SINGLE_NO_PARAM_EXPORT[gtype]
            q = int(gate["target_qubit"])
            lines.append(f"{qasm_gate} q[{q}];")
        elif gtype in _PARAM_EXPORT:
            qasm_gate = _PARAM_EXPORT[gtype]
            q = int(gate["target_qubit"])
            if qasm_gate == "u2":
                phi = _format_angle(gate["parameter"][0])
                lam = _format_angle(gate["parameter"][1])
                lines.append(f"u2({phi},{lam}) q[{q}];")
            else:
                theta = _format_angle(gate["parameter"])
                lines.append(f"{qasm_gate}({theta}) q[{q}];")
        elif gtype == "u3":
            q = int(gate["target_qubit"])
            theta = _format_angle(gate["parameter"][0])
            phi = _format_angle(gate["parameter"][1])
            lam = _format_angle(gate["parameter"][2])
            gate_name = "u3" if version_norm == "2.0" else "u"
            lines.append(f"{gate_name}({theta},{phi},{lam}) q[{q}];")
        elif gtype in _DOUBLE_EXPORT:
            qasm_gate = _DOUBLE_EXPORT[gtype]
            if qasm_gate == "swap":
                q1 = int(gate["qubit_1"])
                q2 = int(gate["qubit_2"])
                lines.append(f"swap q[{q1}],q[{q2}];")
            elif qasm_gate in {"rzz", "rxx"}:
                q1 = int(gate["qubit_1"])
                q2 = int(gate["qubit_2"])
                theta = _format_angle(gate["parameter"])
                lines.append(f"{qasm_gate}({theta}) q[{q1}],q[{q2}];")
            elif qasm_gate in {"crx", "cry", "crz"}:
                t = int(gate["target_qubit"])
                theta = _format_angle(gate["parameter"])
                if version_norm == "3.0" and len(controls) > 1:
                    _append_qasm3_multi_control_rotation(lines, qasm_gate, controls, t, theta)
                else:
                    if len(controls) != 1:
                        raise ValueError(f"QASM 导出仅支持单控制门，当前: {gtype} controls={controls}")
                    c = int(controls[0])
                    lines.append(f"{qasm_gate}({theta}) q[{c}],q[{t}];")
            else:
                if len(controls) != 1:
                    raise ValueError(f"QASM 导出仅支持单控制门，当前: {gtype} controls={controls}")
                c = int(controls[0])
                t = int(gate["target_qubit"])
                lines.append(f"{qasm_gate} q[{c}],q[{t}];")
        elif gtype in _THREE_EXPORT:
            qasm_gate = _THREE_EXPORT[gtype]
            if len(controls) != 2:
                raise ValueError(f"QASM 导出仅支持双控制 Toffoli，当前 controls={controls}")
            c1, c2 = int(controls[0]), int(controls[1])
            t = int(gate["target_qubit"])
            lines.append(f"{qasm_gate} q[{c1}],q[{c2}],q[{t}];")
        else:
            raise ValueError(f"QASM 导出暂不支持门类型: {gtype}")

        lines.extend(post_lines)

    return "\n".join(lines) + "\n"


def circuit_to_qasm3(circuit: Circuit) -> str:
    """将 Circuit 导出为 OpenQASM 3.0 字符串。"""
    return circuit_to_qasm(circuit, version="3.0")


def _strip_comments_and_blank(qasm_text: str) -> List[str]:
    rows = []
    for raw in qasm_text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if line:
            rows.append(line)
    return rows


def _detect_qasm_version(first_line: str) -> str:
    m = re.match(r"^openqasm\s+([0-9]+(?:\.[0-9]+)?)\s*;?$", first_line.strip(), re.IGNORECASE)
    if not m:
        raise ValueError("无法识别 OpenQASM 版本头")
    version = m.group(1)
    if version.startswith("2"):
        return "2.0"
    if version.startswith("3"):
        return "3.0"
    raise ValueError(f"不支持的 OpenQASM 版本: {version}")


def circuit_from_qasm(qasm_text: str) -> Circuit:
    """从 OpenQASM 2.0/3.0 字符串解析 Circuit（子集）。"""
    rows = _strip_comments_and_blank(qasm_text)
    if not rows:
        raise ValueError("QASM 内容为空")

    qasm_version = _detect_qasm_version(rows[0])

    n_qubits = None
    reg_info: Dict[str, Dict[str, int]] = {}
    next_offset = 0
    gates: List[Dict[str, object]] = []

    qreg_re = re.compile(r"^qreg\s+([A-Za-z_]\w*)\[(\d+)\];$", re.IGNORECASE)
    qubit_re = re.compile(r"^qubit\[(\d+)\]\s+([A-Za-z_]\w*)\s*;$", re.IGNORECASE)
    qubit_scalar_re = re.compile(r"^qubit\s+([A-Za-z_]\w*)\s*;$", re.IGNORECASE)
    gate3_re = re.compile(r"^(ccx)\s+(.+);$", re.IGNORECASE)
    gate2_re = re.compile(r"^(cx|cy|cz|swap|crx|cry|crz|rzz|rxx)\s+(.+);$", re.IGNORECASE)
    gate1_re = re.compile(r"^(x|y|z|h|s|t)\s+(.+);$", re.IGNORECASE)
    gatep1_re = re.compile(r"^(rx|ry|rz|p|u1)\(([^)]+)\)\s+(.+);$", re.IGNORECASE)
    gateu2_re = re.compile(r"^(u2)\(([^,]+),([^\)]+)\)\s+(.+);$", re.IGNORECASE)
    gateu3_re = re.compile(r"^(u3|u)\(([^,]+),([^,]+),([^\)]+)\)\s+(.+);$", re.IGNORECASE)

    for line in rows[1:]:
        low = line.lower()
        if low.startswith("include "):
            continue
        if low.startswith("creg "):
            continue
        if re.match(r"^bit(\[\d+\])?\s+", low):
            continue
        if low.startswith("measure ") or low.startswith("barrier "):
            continue
        if "= measure" in low:
            continue

        m = qreg_re.match(line)
        if m:
            rname = m.group(1)
            rsize = int(m.group(2))
            reg_info[rname] = {"offset": next_offset, "size": rsize}
            next_offset += rsize
            continue

        m = qubit_re.match(line)
        if m:
            rsize = int(m.group(1))
            rname = m.group(2)
            reg_info[rname] = {"offset": next_offset, "size": rsize}
            next_offset += rsize
            continue

        m = qubit_scalar_re.match(line)
        if m:
            rname = m.group(1)
            reg_info[rname] = {"offset": next_offset, "size": 1}
            next_offset += 1
            continue

        m = gate1_re.match(line)
        if m:
            name = m.group(1).lower()
            ops = _split_operands(m.group(2))
            if len(ops) != 1:
                raise ValueError(f"单比特门参数个数错误: {line}")
            q = _parse_qubit_ref(ops[0], reg_info)
            gates.append({"type": _IMPORT_SINGLE[name], "target_qubit": q})
            continue

        m = gatep1_re.match(line)
        if m:
            name = m.group(1).lower()
            theta = _parse_qasm_angle(m.group(2))
            ops = _split_operands(m.group(3))
            if len(ops) != 1:
                raise ValueError(f"单比特参数门操作数错误: {line}")
            q = _parse_qubit_ref(ops[0], reg_info)
            # p/u1 统一映射为 rz
            gate_name = "rz" if name in {"p", "u1"} else name
            gates.append({"type": gate_name, "target_qubit": q, "parameter": theta})
            continue

        m = gateu2_re.match(line)
        if m:
            phi = _parse_qasm_angle(m.group(2))
            lam = _parse_qasm_angle(m.group(3))
            ops = _split_operands(m.group(4))
            if len(ops) != 1:
                raise ValueError(f"u2 操作数错误: {line}")
            q = _parse_qubit_ref(ops[0], reg_info)
            gates.append({"type": "u2", "target_qubit": q, "parameter": [phi, lam]})
            continue

        m = gateu3_re.match(line)
        if m:
            theta = _parse_qasm_angle(m.group(2))
            phi = _parse_qasm_angle(m.group(3))
            lam = _parse_qasm_angle(m.group(4))
            ops = _split_operands(m.group(5))
            if len(ops) != 1:
                raise ValueError(f"u/u3 操作数错误: {line}")
            q = _parse_qubit_ref(ops[0], reg_info)
            gates.append({"type": "u3", "target_qubit": q, "parameter": [theta, phi, lam]})
            continue

        m = gate2_re.match(line)
        if m:
            name = m.group(1).lower()
            ops = _split_operands(m.group(2))
            if len(ops) != 2:
                raise ValueError(f"双比特门操作数错误: {line}")
            q1 = _parse_qubit_ref(ops[0], reg_info)
            q2 = _parse_qubit_ref(ops[1], reg_info)
            if name == "swap":
                gates.append({"type": "swap", "qubit_1": q1, "qubit_2": q2})
            elif name in {"crx", "cry", "crz", "rzz", "rxx"}:
                raise ValueError(
                    f"无法从无参数语句解析 {name}，请检查语法。"
                )
            else:
                gates.append(
                    {
                        "type": name,
                        "target_qubit": q2,
                        "control_qubits": [q1],
                        "control_states": [1],
                    }
                )
            continue

        # 双比特参数门（crx/cry/crz/rzz/rxx）
        m = re.match(r"^(crx|cry|crz|rzz|rxx)\(([^)]+)\)\s+(.+);$", line, re.IGNORECASE)
        if m:
            name = m.group(1).lower()
            theta = _parse_qasm_angle(m.group(2))
            ops = _split_operands(m.group(3))
            if len(ops) != 2:
                raise ValueError(f"{name} 操作数错误: {line}")
            q1 = _parse_qubit_ref(ops[0], reg_info)
            q2 = _parse_qubit_ref(ops[1], reg_info)
            if name in {"rzz", "rxx"}:
                gates.append({"type": name, "qubit_1": q1, "qubit_2": q2, "parameter": theta})
                continue
            gates.append(
                {
                    "type": name,
                    "target_qubit": q2,
                    "control_qubits": [q1],
                    "control_states": [1],
                    "parameter": theta,
                }
            )
            continue

        m = gate3_re.match(line)
        if m:
            ops = _split_operands(m.group(2))
            if len(ops) != 3:
                raise ValueError(f"ccx 操作数错误: {line}")
            c1 = _parse_qubit_ref(ops[0], reg_info)
            c2 = _parse_qubit_ref(ops[1], reg_info)
            t = _parse_qubit_ref(ops[2], reg_info)
            gates.append(
                {
                    "type": "toffoli",
                    "target_qubit": t,
                    "control_qubits": [c1, c2],
                }
            )
            continue

        raise ValueError(f"无法解析或暂不支持的 QASM 语句: {line}")

    if reg_info:
        n_qubits = sum(v["size"] for v in reg_info.values())

    if n_qubits is None:
        # 从门索引推断
        max_idx = -1
        for g in gates:
            if "target_qubit" in g:
                max_idx = max(max_idx, int(g["target_qubit"]))
            if "control_qubits" in g:
                max_idx = max(max_idx, *(int(x) for x in g["control_qubits"]))
            if "qubit_1" in g:
                max_idx = max(max_idx, int(g["qubit_1"]))
            if "qubit_2" in g:
                max_idx = max(max_idx, int(g["qubit_2"]))
        if max_idx < 0:
            if qasm_version == "3.0":
                raise ValueError("QASM 3.0 未定义 qubit 寄存器且无法从门推断量子比特数")
            raise ValueError("QASM 2.0 未定义 qreg 且无法从门推断量子比特数")
        n_qubits = max_idx + 1

    return Circuit(*gates, n_qubits=n_qubits)


def save_circuit_qasm(circuit: Circuit, file_path: str | Path, version: str = "2.0") -> None:
    """将 Circuit 导出为 OpenQASM 文件，支持 2.0 和 3.0。"""
    path = Path(file_path)
    path.write_text(circuit_to_qasm(circuit, version=version), encoding="utf-8")


def save_circuit_qasm3(circuit: Circuit, file_path: str | Path) -> None:
    """将 Circuit 导出为 OpenQASM 3.0 文件。"""
    save_circuit_qasm(circuit, file_path, version="3.0")


def load_circuit_qasm(file_path: str | Path) -> Circuit:
    """从 OpenQASM 2.0/3.0 文件加载 Circuit。"""
    path = Path(file_path)
    return circuit_from_qasm(path.read_text(encoding="utf-8"))

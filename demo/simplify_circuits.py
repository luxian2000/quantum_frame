#!/usr/bin/env python3
"""
对保存的电路执行简单的局部优化：
- 取消连续相同的自反门（X,X 或 H,H 等）
- 合并连续的同轴旋转（rx/ry/rz 合并参数）
- 取消连续相同的 CNOT（同控制、同目标）

目标：减少随机生成电路的冗余门，尽量不改变电路的全局保真度。

注意：这是局部、保守的优化，不保证能得到最小电路。
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexq.circuit.io.json_io import load_circuit_json, save_circuit_json
from nexq.channel.backends.numpy_backend import NumpyBackend


# 可取消的自反门
SELF_INV = {"pauli_x", "pauli_y", "pauli_z", "hadamard"}
# 可合并的旋转
ROTATIONS = {"rx", "ry", "rz"}
# CNOT 等名字
CNOT_TYPES = {"cnot", "cx"}


def is_same_single_target(g1, g2):
    return ("target_qubit" in g1 and "target_qubit" in g2 and
            g1["target_qubit"] == g2["target_qubit"] and
            ("control_qubits" not in g1 and "control_qubits" not in g2))


def simplify_gates(gates):
    out = []

    for g in gates:
        if not out:
            out.append(g)
            continue

        last = out[-1]
        # 1) 合并 rotation
        if last["type"] in ROTATIONS and g["type"] == last["type"] and is_same_single_target(last, g):
            # 合并参数
            p1 = last.get("parameter", 0.0)
            p2 = g.get("parameter", 0.0)
            newp = float(p1) + float(p2)
            # 规范到 (-pi, pi]
            newp = (newp + math.pi) % (2 * math.pi) - math.pi
            if abs(newp) < 1e-12:
                out.pop()
            else:
                last["parameter"] = newp
            continue

        # 2) 取消自反门对
        if last["type"] == g["type"] and g["type"] in SELF_INV and is_same_single_target(last, g):
            # remove last (两门抵消)
            out.pop()
            continue

        # 3) 取消连续相同 CNOT
        if last.get("type") in CNOT_TYPES and g.get("type") in CNOT_TYPES:
            # 比较 target 和 control
            l_ctrl = last.get("control_qubits", [])
            g_ctrl = g.get("control_qubits", [])
            if l_ctrl == g_ctrl and last.get("target_qubit") == g.get("target_qubit"):
                out.pop()
                continue

        # 4) 合并 RZ 与 RZ 的小优化（也包含在 ROTATIONS，因为 rz 在集合中）
        # 此处已覆盖

        # 默认：保留
        out.append(g)

    return out


def fidelity_of_circuit(circ):
    # circ 是 Circuit 对象
    backend = NumpyBackend()
    U = circ.unitary(backend=backend)
    if hasattr(U, 'numpy'):
        U = U.numpy()
    else:
        U = np.asarray(U)
    n = circ.n_qubits
    init = np.zeros(1 << n, dtype=complex)
    init[0] = 1.0
    state = U @ init
    return state


def process(path_json: Path, out_json: Path):
    c = load_circuit_json(path_json)
    before = list(c.gates)
    print(f"Processing {path_json.name}: n_qubits={c.n_qubits}, gates_before={len(before)}")

    new_gates = simplify_gates(before)
    # 如果长度变化，写回
    if len(new_gates) < len(before):
        new_c = type(c)(*new_gates, n_qubits=c.n_qubits)
        save_circuit_json(new_c, out_json)
        print(f"  simplified -> gates_after={len(new_gates)} (saved to {out_json.name})")
    else:
        # 未变化，直接复制原文件到输出
        save_circuit_json(c, out_json)
        print(f"  no simplification applied (gates_after={len(new_gates)}). Copied to {out_json.name}")

    # 输出保真度比较
    U0 = c.unitary(backend=NumpyBackend())
    if hasattr(U0, 'numpy'):
        U0 = U0.numpy()
    else:
        U0 = np.asarray(U0)
    n = c.n_qubits
    init = np.zeros(1 << n, dtype=complex); init[0] = 1.0
    state0 = U0 @ init

    c1 = load_circuit_json(out_json)
    U1 = c1.unitary(backend=NumpyBackend())
    if hasattr(U1, 'numpy'):
        U1 = U1.numpy()
    else:
        U1 = np.asarray(U1)
    state1 = U1 @ init

    # 对目标保真度无法知道，这里只比较两电路是否相等（幺正相等 up to numerical error)
    diff = np.linalg.norm(U0 - U1)
    print(f"  unitary_diff_norm = {diff:.3e}")


if __name__ == '__main__':
    base = Path(__file__).parent
    inputs = ['canonical_ghz.json', 'best_random_ghz.json']
    for name in inputs:
        p = base / name
        outp = base / (Path(name).stem + '_simplified.json')
        if p.exists():
            process(p, outp)
        else:
            print(f"Not found: {p}")
"""
简单电路简化器：合并相邻同轴旋转与消除自反门

用法：
    python simplify_circuits.py

会处理 demo 中的 canonical_ghz.json 与 best_random_ghz.json，
并生成 simplified_*.json 与 simplified_*.qasm
"""
import sys
from pathlib import Path
import math
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexq.circuit.io.json_io import load_circuit_json, save_circuit_json
from nexq.circuit.io.qasm import save_circuit_qasm
from nexq.circuit.model import Circuit


# 可反门（自身的逆）集合
SELF_INVERSE = {"pauli_x", "pauli_y", "pauli_z", "hadamard", "cx", "cnot", "swap"}

# 旋转门集合
ROTATIONS = {"rx", "ry", "rz"}


def norm_type(gtype: str) -> str:
    if gtype == "cnot":
        return "cx"
    return gtype


def same_targets(g1: dict, g2: dict) -> bool:
    # 比较 target_qubit
    if g1.get("target_qubit") != g2.get("target_qubit"):
        return False
    # 比较控制比特（如果有）
    c1 = g1.get("control_qubits") or []
    c2 = g2.get("control_qubits") or []
    return list(c1) == list(c2)


def angle_normalize(theta: float) -> float:
    # 归一化到 (-pi, pi]
    return ((theta + math.pi) % (2 * math.pi)) - math.pi


def simplify_gates(gates: list) -> list:
    stack = []
    for gate in gates:
        g = dict(gate)  # shallow copy
        g['type'] = norm_type(g['type'])
        if not stack:
            stack.append(g)
            continue
        prev = stack[-1]
        # unify prev type
        prev_type = norm_type(prev['type'])
        cur_type = g['type']

        # Case 1: same non-parameter gate and same targets -> self-inverse cancellation
        if prev_type == cur_type and cur_type in SELF_INVERSE and same_targets(prev, g):
            stack.pop()
            continue

        # Case 2: rotations on same axis and same target -> merge parameters
        if prev_type == cur_type and cur_type in ROTATIONS and same_targets(prev, g):
            p = float(prev.get('parameter', 0.0))
            q = float(g.get('parameter', 0.0))
            s = angle_normalize(p + q)
            # if effectively zero, remove prev
            if abs(s) < 1e-12:
                stack.pop()
            else:
                prev['parameter'] = float(s)
            continue

        # Case 3: RZ + RZ might be represented as 'rz'
        if prev_type in ROTATIONS and cur_type in ROTATIONS and prev_type == cur_type and same_targets(prev, g):
            # handled above
            pass

        # Default: push
        stack.append(g)
    return stack


def process_file(json_path: Path):
    circ = load_circuit_json(json_path)
    g_before = list(circ.gates)
    simplified = simplify_gates(g_before)

    circ_s = Circuit(*simplified, n_qubits=circ.n_qubits)
    out_json = json_path.parent / (json_path.stem + '.simplified.json')
    out_qasm = json_path.parent / (json_path.stem + '.simplified.qasm')

    save_circuit_json(circ_s, out_json)
    try:
        save_circuit_qasm(circ_s, out_qasm, version='2.0')
    except Exception:
        # 如果 QASM 导出失败则跳过
        pass

    return len(g_before), len(simplified), str(out_json), str(out_qasm)


if __name__ == '__main__':
    base = Path(__file__).parent
    files = ['canonical_ghz.json', 'best_random_ghz.json']
    for name in files:
        p = base / name
        if not p.exists():
            print(f"Not found: {p}")
            continue
        b, a, jpath, qpath = process_file(p)
        print(f"{name}: {b} -> {a} gates after simplification")
        print(f"  simplified json: {jpath}")
        print(f"  simplified qasm: {qpath}")

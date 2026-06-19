"""多格式本地线路化简前端（dict / qasm / dag）。

线路结构优化的统一入口在 :func:`aicir.transpile.optimize`；本模块在其
基础上提供：

- :func:`optimize_circuit`：Circuit 专用包装，等价于 ``optimize``；
- :func:`optimize_basic`：根据输入类型分派到 dict/qasm/dag 路径。

dict/dag 路径复用 :func:`._local_rewrite.optimize_gates`（本地化简规则的
单一来源），qasm 路径是独立的正则文本改写。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.circuit import Circuit
from ..ir import (
    has_circuit_instructions,
    instruction_controls,
    instruction_qubits,
)
from .passes._local_rewrite import _gate_family_from_name, optimize_gates
from .passmanager import optimize


def optimize_circuit(
    circuit: Circuit,
    *,
    max_rounds: int = 64,
    max_reorder_hops: int = 8,
) -> Circuit:
    """返回 :class:`~aicir.core.circuit.Circuit` 的优化副本。

    Circuit 专用入口，等价于 :func:`aicir.transpile.optimize`，保留
    ``n_qubits`` 与绑定的 backend。
    """

    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("optimize_circuit 输入必须是 Circuit 或 CircuitIR-like 对象")
    return optimize(circuit, max_rounds=max_rounds, max_reorder_hops=max_reorder_hops)


def _gate_qubits(gate: dict[str, Any]) -> list[int]:
    return sorted(set(int(q) for q in (*instruction_qubits(gate), *instruction_controls(gate))))


@dataclass
class _QasmStmt:
    raw: str
    kind: str
    family: str | None = None
    q0: str | None = None
    q1: str | None = None
    theta: float | None = None


_RE_UNARY = re.compile(r"^(x|y|z|h|s|sdg)\s+([^;]+);$", re.IGNORECASE)
_RE_CNOT = re.compile(r"^(cx|cnot)\s+([^,;]+)\s*,\s*([^;]+);$", re.IGNORECASE)
_RE_ROT = re.compile(r"^(rx|ry|rz)\s*\(([^\)]*)\)\s+([^;]+);$", re.IGNORECASE)


def _parse_qasm_angle(expr: str) -> float:
    text = expr.strip().lower()
    if not text:
        raise ValueError("空角度表达式")
    if not re.fullmatch(r"[0-9eE+\-*/().\spi]+", text):
        raise ValueError(f"不支持的角度表达式: {expr}")
    val = eval(text, {"__builtins__": {}}, {"pi": np.pi})
    return float(val)


def _format_qasm_angle(theta: float) -> str:
    if np.isclose(theta, np.pi, atol=1e-15):
        return "pi"
    if np.isclose(theta, -np.pi, atol=1e-15):
        return "-pi"
    return format(float(theta), ".15g")


def _parse_qasm_stmt(line: str) -> _QasmStmt:
    # Keep comments/blanks/unknown statements unchanged as barriers.
    code = line.split("//", 1)[0].strip()
    if not code:
        return _QasmStmt(raw=line, kind="barrier")

    m = _RE_UNARY.match(code)
    if m:
        fam = _gate_family_from_name(m.group(1))
        return _QasmStmt(raw=line, kind="gate", family=fam, q0=m.group(2).strip())

    m = _RE_CNOT.match(code)
    if m:
        fam = _gate_family_from_name(m.group(1))
        return _QasmStmt(raw=line, kind="gate", family=fam, q0=m.group(2).strip(), q1=m.group(3).strip())

    m = _RE_ROT.match(code)
    if m:
        fam = _gate_family_from_name(m.group(1))
        try:
            theta = _parse_qasm_angle(m.group(2))
        except Exception:
            return _QasmStmt(raw=line, kind="barrier")
        return _QasmStmt(raw=line, kind="gate", family=fam, q0=m.group(3).strip(), theta=theta)

    return _QasmStmt(raw=line, kind="barrier")


def _cancel_qasm_pair(prev: _QasmStmt, curr: _QasmStmt) -> bool:
    if prev.kind != "gate" or curr.kind != "gate" or prev.family is None or curr.family is None:
        return False

    if prev.family == curr.family and prev.family in {"x", "y", "z", "h"}:
        return prev.q0 == curr.q0

    if prev.family == curr.family == "cnot":
        return prev.q0 == curr.q0 and prev.q1 == curr.q1

    if {prev.family, curr.family} == {"s", "sdg"}:
        return prev.q0 == curr.q0

    return False


def _is_single_qubit_qasm_gate(stmt: _QasmStmt) -> bool:
    return stmt.kind == "gate" and stmt.q0 is not None and stmt.q1 is None


def _is_cnot_qasm_gate(stmt: _QasmStmt) -> bool:
    return stmt.kind == "gate" and stmt.family == "cnot" and stmt.q0 is not None and stmt.q1 is not None


def _single_qubit_commutes_with_cnot(single: _QasmStmt, cnot: _QasmStmt) -> bool:
    if not _is_single_qubit_qasm_gate(single) or not _is_cnot_qasm_gate(cnot):
        return False

    assert single.q0 is not None and cnot.q0 is not None and cnot.q1 is not None
    q = single.q0
    ctrl = cnot.q0
    targ = cnot.q1
    fam = single.family

    # Known safe commuting subsets for CNOT:
    # - On control qubit: Z-family phase ops commute.
    # - On target qubit: X-family ops commute.
    if q == ctrl and fam in {"z", "s", "sdg", "rz"}:
        return True
    if q == targ and fam in {"x", "rx"}:
        return True
    return False


def _optimize_qasm_text(qasm_text: str) -> str:
    rows = qasm_text.splitlines()
    kept: list[_QasmStmt] = []
    max_reorder_hops = 8
    for row in rows:
        stmt = _parse_qasm_stmt(row)

        # Merge adjacent rx/ry/rz on the same qubit.
        if (
            kept
            and kept[-1].kind == "gate"
            and stmt.kind == "gate"
            and kept[-1].family in {"rx", "ry", "rz"}
            and kept[-1].family == stmt.family
            and kept[-1].q0 == stmt.q0
            and kept[-1].theta is not None
            and stmt.theta is not None
        ):
            merged = kept[-1].theta + stmt.theta
            if np.isclose(merged, 0.0, atol=1e-15):
                kept.pop()
            else:
                gate_name = kept[-1].family
                q = kept[-1].q0
                assert gate_name is not None and q is not None
                kept[-1] = _QasmStmt(
                    raw=f"{gate_name}({_format_qasm_angle(merged)}) {q};",
                    kind="gate",
                    family=gate_name,
                    q0=q,
                    theta=merged,
                )
            continue

        if kept and _cancel_qasm_pair(kept[-1], stmt):
            kept.pop()
            continue

        # Safe limited reordering (no actual text swap):
        # For a single-qubit gate on q, look back through at most N trailing
        # single-qubit gates on *other* qubits; if we can cancel/merge with
        # a same-qubit predecessor, do it directly.
        if _is_single_qubit_qasm_gate(stmt):
            consumed = False
            hops = 0
            idx = len(kept) - 1
            while idx >= 0 and hops < max_reorder_hops:
                prev = kept[idx]

                # Barriers stop lookback.
                if prev.kind != "gate":
                    break

                # Known commuting two-qubit pattern: single-qubit gate commuting with CNOT.
                if _is_cnot_qasm_gate(prev):
                    if _single_qubit_commutes_with_cnot(stmt, prev):
                        hops += 1
                        idx -= 1
                        continue
                    break

                # Unknown/multi-qubit gates stop lookback.
                if not _is_single_qubit_qasm_gate(prev):
                    break

                assert stmt.q0 is not None and prev.q0 is not None
                if prev.q0 != stmt.q0:
                    hops += 1
                    idx -= 1
                    continue

                # Found a same-qubit predecessor.
                if _cancel_qasm_pair(prev, stmt):
                    del kept[idx]
                    consumed = True
                    break

                if (
                    prev.family in {"rx", "ry", "rz"}
                    and prev.family == stmt.family
                    and prev.theta is not None
                    and stmt.theta is not None
                ):
                    merged = prev.theta + stmt.theta
                    if np.isclose(merged, 0.0, atol=1e-15):
                        del kept[idx]
                    else:
                        gate_name = prev.family
                        q = prev.q0
                        assert gate_name is not None and q is not None
                        kept[idx] = _QasmStmt(
                            raw=f"{gate_name}({_format_qasm_angle(merged)}) {q};",
                            kind="gate",
                            family=gate_name,
                            q0=q,
                            theta=merged,
                        )
                    consumed = True
                    break

                # Same qubit but not combinable: stop.
                break
            if not consumed:
                kept.append(stmt)
            continue

        kept.append(stmt)

    optimized = "\n".join(s.raw for s in kept)
    if qasm_text.endswith("\n") and not optimized.endswith("\n"):
        optimized += "\n"
    return optimized


def _optimize_qasm_text_fixed_point(qasm_text: str, *, max_rounds: int = 64) -> str:
    current = qasm_text
    for _ in range(max_rounds):
        nxt = _optimize_qasm_text(current)
        if nxt == current:
            return nxt
        current = nxt
    return current


def _topological_order_from_adj(adj: np.ndarray) -> list[int]:
    n = int(adj.shape[0])
    indeg = adj.sum(axis=0).astype(int)
    ready = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while ready:
        ready.sort()
        node = ready.pop(0)
        order.append(node)
        children = np.where(adj[node] > 0)[0]
        for ch in children:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(int(ch))
    if len(order) != n:
        raise ValueError("DAG 邻接矩阵含环或非法")
    return order


def _dag_nodes_from_arrays(
    X: np.ndarray,
    A: np.ndarray,
    type_onehot: np.ndarray,
    gate_types: list[str],
) -> list[dict[str, Any]]:
    f_type = len(gate_types)
    order = _topological_order_from_adj(np.asarray(A, dtype=np.float32))
    gates: list[dict[str, Any]] = []
    for idx in order:
        if idx == 0 or idx == X.shape[0] - 1:
            continue
        row_t = type_onehot[idx]
        if float(np.sum(row_t)) <= 0:
            continue
        type_idx = int(np.argmax(row_t))
        gtype = gate_types[type_idx]
        qubits = [int(q) for q in np.where(X[idx, f_type:] > 0.5)[0].tolist()]
        fam = _gate_family_from_name(gtype)
        if fam in {"x", "y", "z", "h", "s", "sdg"} and qubits:
            gates.append({"type": gtype, "target_qubit": qubits[0]})
        elif fam == "cnot" and len(qubits) >= 2:
            # Note: circuit_to_dag stores only qubit set, not control/target order.
            # Use sorted order for deterministic round-trip in optimizer.
            q_sorted = sorted(qubits)
            gates.append(
                {
                    "type": gtype,
                    "control_qubits": [q_sorted[0]],
                    "control_states": [1],
                    "target_qubit": q_sorted[1],
                }
            )
        else:
            # Unsupported/unknown gate types are kept as barriers by encoding them as-is.
            if qubits:
                gates.append({"type": gtype, "target_qubit": qubits[0]})
    return gates


def _dag_arrays_from_gates(gates: list[dict[str, Any]], gate_types: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not gate_types:
        raise ValueError("DAG gate_types 不能为空")
    f_type = len(gate_types)
    gate_to_idx = {g: i for i, g in enumerate(gate_types)}

    max_q = -1
    for g in gates:
        qubits = _gate_qubits(g)
        if qubits:
            max_q = max(max_q, max(qubits))
    n_qubits = max_q + 1 if max_q >= 0 else 1

    total = len(gates) + 2
    X = np.zeros((total, f_type + n_qubits), dtype=np.float32)
    type_onehot = np.zeros((total, f_type), dtype=np.float32)

    last_on_qubit = [0] * n_qubits
    edges: set[tuple[int, int]] = set()

    for idx, g in enumerate(gates, start=1):
        gtype = g.get("type")
        if gtype in gate_to_idx:
            t_idx = gate_to_idx[gtype]
            type_onehot[idx, t_idx] = 1.0
            X[idx, t_idx] = 1.0

        qubits = _gate_qubits(g)

        for q in qubits:
            X[idx, f_type + q] = 1.0
            edges.add((last_on_qubit[q], idx))
            last_on_qubit[q] = idx

    end_idx = total - 1
    for q in range(n_qubits):
        edges.add((last_on_qubit[q], end_idx))

    A = np.zeros((total, total), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
    return X, A, type_onehot


def optimize_basic(
    obj: Any,
    *,
    input_type: str | None = None,
    dag_gate_types: list[str] | None = None,
) -> Any:
    """Optimize simple adjacent gate-cancellation rules.

    Supported inputs:
    - dict order: Circuit | list[gate_dict] | {"gates": ..., "n_qubits": ...}
    - openqasm text: str
    - dag: tuple/list (X, A, type_onehot) with dag_gate_types,
      or dict {"X":..., "A":..., "type_onehot":..., "gate_types":...}

    Output type matches input type.
    """
    kind = input_type
    if kind is None:
        if isinstance(obj, str):
            kind = "qasm"
        elif (
            isinstance(obj, Circuit)
            or (hasattr(obj, "n_qubits") and has_circuit_instructions(obj))
            or isinstance(obj, list)
            or (isinstance(obj, dict) and "gates" in obj)
        ):
            kind = "dict"
        else:
            kind = "dag"

    if kind == "dict":
        if isinstance(obj, Circuit) or (hasattr(obj, "n_qubits") and has_circuit_instructions(obj)):
            return optimize_circuit(obj)
        if isinstance(obj, list):
            return optimize_gates(obj)
        if isinstance(obj, dict) and "gates" in obj:
            out = copy.deepcopy(obj)
            out["gates"] = optimize_gates(obj["gates"])
            return out
        raise TypeError("dict 输入需为 Circuit、list[dict] 或含 gates 的 dict")

    if kind == "qasm":
        if not isinstance(obj, str):
            raise TypeError("qasm 输入必须是字符串")
        return _optimize_qasm_text_fixed_point(obj)

    if kind == "dag":
        if isinstance(obj, dict):
            if not all(k in obj for k in ("X", "A", "type_onehot")):
                raise TypeError("dag dict 输入必须包含 X/A/type_onehot")
            gate_types = obj.get("gate_types") or dag_gate_types
            if gate_types is None:
                raise ValueError("dag 输入需要 gate_types（可放在 obj['gate_types'] 或 dag_gate_types）")
            X = np.asarray(obj["X"], dtype=np.float32)
            A = np.asarray(obj["A"], dtype=np.float32)
            T = np.asarray(obj["type_onehot"], dtype=np.float32)
            gates = _dag_nodes_from_arrays(X, A, T, list(gate_types))
            gates_opt = optimize_gates(gates)
            X2, A2, T2 = _dag_arrays_from_gates(gates_opt, list(gate_types))
            out = copy.deepcopy(obj)
            out["X"], out["A"], out["type_onehot"] = X2, A2, T2
            out["gate_types"] = list(gate_types)
            return out

        if isinstance(obj, (tuple, list)) and len(obj) == 3:
            if dag_gate_types is None:
                raise ValueError("dag tuple/list 输入需要提供 dag_gate_types")
            X = np.asarray(obj[0], dtype=np.float32)
            A = np.asarray(obj[1], dtype=np.float32)
            T = np.asarray(obj[2], dtype=np.float32)
            gates = _dag_nodes_from_arrays(X, A, T, list(dag_gate_types))
            gates_opt = optimize_gates(gates)
            X2, A2, T2 = _dag_arrays_from_gates(gates_opt, list(dag_gate_types))
            return (X2, A2, T2) if isinstance(obj, tuple) else [X2, A2, T2]

        raise TypeError("dag 输入需为 (X,A,type_onehot) 或含 X/A/type_onehot 的 dict")

    raise ValueError(f"不支持的 input_type: {kind}")

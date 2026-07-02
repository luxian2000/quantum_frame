"""Local circuit rewrite helpers used by transpiler passes."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np

from ...core.circuit import Circuit
from ...ir import instruction_to_gate_dict

_DEFAULT_MAX_ROUNDS = 64
_DEFAULT_MAX_REORDER_HOPS = 8


def _gate_family_from_name(name: str) -> str | None:
    low = str(name).strip().lower()
    if low in {"pauli_x", "x"}:
        return "x"
    if low in {"pauli_y", "y"}:
        return "y"
    if low in {"pauli_z", "z"}:
        return "z"
    if low in {"hadamard", "h"}:
        return "h"
    if low in {"cx", "cnot"}:
        return "cnot"
    if low in {"s", "s_gate"}:
        return "s"
    if low in {"sdg", "sdag", "s_dagger", "sdagger"}:
        return "sdg"
    if low in {"rx"}:
        return "rx"
    if low in {"ry"}:
        return "ry"
    if low in {"rz"}:
        return "rz"
    return None


def _normalize_control_states(gate: dict[str, Any]) -> tuple[int, ...]:
    controls = list(gate.get("control_qubits", []) or [])
    raw = gate.get("control_states")
    if raw is None:
        return tuple(1 for _ in controls)
    return tuple(int(x) for x in raw)


def _is_single_qubit_gate(gate: dict[str, Any]) -> bool:
    fam = _gate_family_from_name(gate.get("type", ""))
    if fam not in {"x", "y", "z", "h", "s", "sdg", "rx", "ry", "rz"}:
        return False
    if gate.get("target_qubit") is None:
        return False
    if gate.get("control_qubits") or gate.get("qubit_1") is not None or gate.get("qubit_2") is not None:
        return False
    if gate.get("qubits") is not None or gate.get("targets") is not None:
        return False
    return True


def _cnot_control_target(gate: dict[str, Any]) -> tuple[int, int] | None:
    fam = _gate_family_from_name(gate.get("type", ""))
    if fam != "cnot" or gate.get("target_qubit") is None:
        return None
    controls = tuple(int(x) for x in (gate.get("control_qubits", []) or []))
    if len(controls) != 1:
        return None
    return controls[0], int(gate["target_qubit"])


def _is_cnot_gate(gate: dict[str, Any]) -> bool:
    return _cnot_control_target(gate) is not None


def _single_qubit_commutes_with_cnot_gate(single: dict[str, Any], cnot: dict[str, Any]) -> bool:
    if not _is_single_qubit_gate(single):
        return False
    pair = _cnot_control_target(cnot)
    if pair is None:
        return False

    ctrl, targ = pair
    q = int(single["target_qubit"])
    fam = _gate_family_from_name(single.get("type", ""))
    if q == ctrl and fam in {"z", "s", "sdg", "rz"}:
        return True
    if q == targ and fam in {"x", "rx"}:
        return True
    return False


def _cancel_gate_pair(prev: dict[str, Any], curr: dict[str, Any]) -> bool:
    pf = _gate_family_from_name(prev.get("type", ""))
    cf = _gate_family_from_name(curr.get("type", ""))

    if pf is None or cf is None:
        return False

    if pf == cf and pf in {"x", "y", "z", "h"}:
        return int(prev.get("target_qubit", -1)) == int(curr.get("target_qubit", -1))

    if pf == cf == "cnot":
        p_controls = tuple(int(x) for x in (prev.get("control_qubits", []) or []))
        c_controls = tuple(int(x) for x in (curr.get("control_qubits", []) or []))
        if len(p_controls) != 1 or len(c_controls) != 1:
            return False
        return (
            p_controls == c_controls
            and int(prev.get("target_qubit", -1)) == int(curr.get("target_qubit", -1))
            and _normalize_control_states(prev) == _normalize_control_states(curr)
        )

    if {pf, cf} == {"s", "sdg"}:
        return int(prev.get("target_qubit", -1)) == int(curr.get("target_qubit", -1))

    return False


def _is_close_to_zero(value: Any) -> bool:
    try:
        return bool(np.isclose(float(value), 0.0, atol=1e-15))
    except (TypeError, ValueError):
        return False


def _excitation_key(gate: dict[str, Any]) -> tuple[str, tuple[int, ...]] | None:
    """返回 excitation 门的 (类型, 操作数) 键；非 excitation 返回 ``None``。

    ``single_excitation``/``double_excitation`` 是固定生成元的旋转门，角度可加
    （``G(θ1)·G(θ2)=G(θ1+θ2)``），故同类型、同操作数（同顺序）的相邻门可合并。
    """
    gtype = gate.get("type")
    if gtype == "single_excitation":
        a, b = gate.get("qubit_1"), gate.get("qubit_2")
        if a is None or b is None:
            return None
        return ("single_excitation", (int(a), int(b)))
    if gtype == "double_excitation":
        qubits = gate.get("qubits")
        if qubits is None:
            return None
        return ("double_excitation", tuple(int(q) for q in qubits))
    return None


def _try_merge_rotation_at(gates: list[dict[str, Any]], index: int, gate: dict[str, Any]) -> bool:
    prev = gates[index]
    gf = _gate_family_from_name(gate.get("type", ""))
    pf = _gate_family_from_name(prev.get("type", ""))
    if (
        gf in {"rx", "ry", "rz"}
        and pf == gf
        and int(prev.get("target_qubit", -1)) == int(gate.get("target_qubit", -1))
        and not (prev.get("control_qubits") or [])
        and not (gate.get("control_qubits") or [])
    ):
        try:
            merged_param = prev.get("parameter", 0.0) + gate.get("parameter", 0.0)
        except TypeError:
            return False
        if _is_close_to_zero(merged_param):
            del gates[index]
        else:
            prev["parameter"] = merged_param
        return True

    exc_key = _excitation_key(gate)
    if exc_key is not None and exc_key == _excitation_key(prev):
        try:
            merged_param = prev.get("parameter", 0.0) + gate.get("parameter", 0.0)
        except TypeError:
            return False
        if _is_close_to_zero(merged_param):
            del gates[index]
        else:
            prev["parameter"] = merged_param
        return True
    return False


def _try_consume_against_gate_at(gates: list[dict[str, Any]], index: int, gate: dict[str, Any]) -> bool:
    if _cancel_gate_pair(gates[index], gate):
        del gates[index]
        return True
    return _try_merge_rotation_at(gates, index, gate)


def _consume_single_qubit_gate_by_lookback(
    gates: list[dict[str, Any]], gate: dict[str, Any], *, max_reorder_hops: int
) -> bool:
    hops = 0
    idx = len(gates) - 1
    while idx >= 0 and hops < max_reorder_hops:
        prev = gates[idx]

        if _is_cnot_gate(prev):
            if _single_qubit_commutes_with_cnot_gate(gate, prev):
                hops += 1
                idx -= 1
                continue
            break

        if not _is_single_qubit_gate(prev):
            break

        if int(prev["target_qubit"]) != int(gate["target_qubit"]):
            hops += 1
            idx -= 1
            continue

        return _try_consume_against_gate_at(gates, idx, gate)

    return False


def _copy_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return dict(instruction_to_gate_dict(gate))


_QUBIT_INT_FIELDS = ("target_qubit", "qubit_1", "qubit_2")
_QUBIT_LIST_FIELDS = ("qubits", "targets", "control_qubits")


def remap_gate(gate: dict[str, Any], mapping) -> dict[str, Any]:
    """返回 ``gate`` 的副本，把其中所有比特下标按 ``mapping`` 重映射。

    ``mapping`` 可为 ``dict`` 或可调用对象（``old -> new``）。覆盖
    ``target_qubit``/``qubit_1``/``qubit_2`` 单值字段与 ``qubits``/
    ``targets``/``control_qubits`` 列表字段；其余字段原样保留。
    """

    fn = mapping.__getitem__ if hasattr(mapping, "__getitem__") else mapping
    out = _copy_gate(gate)
    for key in _QUBIT_INT_FIELDS:
        if out.get(key) is not None:
            out[key] = int(fn(int(out[key])))
    for key in _QUBIT_LIST_FIELDS:
        value = out.get(key)
        if value is not None:
            out[key] = [int(fn(int(q))) for q in value]
    return out


def circuit_from_gates(circuit: Circuit, gates: Iterable[dict[str, Any]]) -> Circuit:
    return Circuit(
        *[_copy_gate(gate) for gate in gates],
        n_qubits=circuit.n_qubits,
        backend=getattr(circuit, "backend", None),
    )


def cancel_inverse_gates(gates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in gates:
        gate = _copy_gate(raw)
        if out and _cancel_gate_pair(out[-1], gate):
            out.pop()
        else:
            out.append(gate)
    return out


def merge_adjacent_rotations(gates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in gates:
        gate = _copy_gate(raw)
        if out and _try_merge_rotation_at(out, len(out) - 1, gate):
            continue
        out.append(gate)
    return out


def commute_single_qubit_gates(
    gates: Iterable[dict[str, Any]], *, max_reorder_hops: int = 8
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in gates:
        gate = _copy_gate(raw)
        if _is_single_qubit_gate(gate) and _consume_single_qubit_gate_by_lookback(
            out, gate, max_reorder_hops=max_reorder_hops
        ):
            continue
        out.append(gate)
    return out


def _optimize_gate_dict_list(
    gates: Iterable[dict[str, Any]], *, max_reorder_hops: int = _DEFAULT_MAX_REORDER_HOPS
) -> list[dict[str, Any]]:
    """单趟交错执行 commute/cancel/merge 规则。"""

    out: list[dict[str, Any]] = []
    for g in gates:
        gate = copy.deepcopy(g)
        if _is_single_qubit_gate(gate) and _consume_single_qubit_gate_by_lookback(
            out, gate, max_reorder_hops=max_reorder_hops
        ):
            continue
        if out and _try_consume_against_gate_at(out, len(out) - 1, gate):
            continue
        out.append(gate)
    return out


def optimize_gates(
    gates: Iterable[dict[str, Any]],
    *,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    max_reorder_hops: int = _DEFAULT_MAX_REORDER_HOPS,
) -> list[dict[str, Any]]:
    """对门字典列表反复应用本地化简规则，直到不动点。

    本地化简规则的单一来源；``optimize_basic`` 的 dict/dag 路径与
    transpile 的 pass 流水线都从这里取规则。
    """

    current = [instruction_to_gate_dict(gate) for gate in copy.deepcopy(list(gates))]
    for _ in range(max_rounds):
        nxt = _optimize_gate_dict_list(current, max_reorder_hops=max_reorder_hops)
        if nxt == current:
            return nxt
        current = nxt
    return current

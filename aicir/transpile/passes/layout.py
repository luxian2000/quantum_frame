"""LayoutPass：选择 logical-to-physical 比特映射（NEXT.md 第 2 节）。

支持三种模式：

- **显式**：给定 ``logical -> physical`` 映射（``dict`` 或序列）。
- **平凡**：``initial_layout=None``，恒等不改动。
- **自动**：``initial_layout="auto"``，按线路双比特门交互频率贪心地把高频交互
  的逻辑比特放到耦合图相邻的物理比特上，减少后续 ``RoutingPass`` 的 SWAP。

任何模式都只重标号、不插门，故线路在比特置换意义下与原线路等价。
自动布局为贪心启发式（非全局最优）；按噪声择优留待后续。
"""

from __future__ import annotations

from ...core.circuit import Circuit
from ...ir import circuit_gate_dicts, instruction_controls, instruction_qubits
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates, remap_gate

__all__ = ["LayoutPass"]


def _interaction_pairs(circuit: Circuit) -> list[tuple[int, int]]:
    """按交互频率降序返回线路中的双比特逻辑对。"""
    weights: dict[frozenset, int] = {}
    for gate in circuit_gate_dicts(circuit):
        qubits = tuple(dict.fromkeys((*instruction_qubits(gate), *instruction_controls(gate))))
        if len(qubits) == 2:
            key = frozenset(int(q) for q in qubits)
            weights[key] = weights.get(key, 0) + 1
    ordered = sorted(weights, key=lambda k: weights[k], reverse=True)
    return [tuple(sorted(pair)) for pair in ordered]


def _auto_layout(circuit: Circuit, target) -> dict[int, int]:
    """贪心地把高频交互逻辑对放到相邻物理比特上。"""
    n_logical = int(circuit.n_qubits)
    free = set(range(int(target.n_qubits)))
    mapping: dict[int, int] = {}

    def place(logical: int, physical: int) -> None:
        mapping[logical] = physical
        free.discard(physical)

    for a, b in _interaction_pairs(circuit):
        if a in mapping and b in mapping:
            continue
        if a not in mapping and b not in mapping:
            edge = next(
                ((x, y) for x, y in target.coupling_map if x in free and y in free),
                None,
            )
            if edge is not None:
                place(a, edge[0])
                place(b, edge[1])
            else:
                place(a, min(free))
                place(b, min(free))
        else:
            placed, unplaced = (a, b) if a in mapping else (b, a)
            nbrs = [p for p in target.neighbors(mapping[placed]) if p in free]
            place(unplaced, nbrs[0] if nbrs else min(free))

    for logical in range(n_logical):
        if logical not in mapping:
            place(logical, min(free))
    return mapping


class LayoutPass(TransformationPass):
    """把逻辑比特重新标号到物理比特。

    参数：

    - ``initial_layout``：逻辑->物理映射。可为 ``dict``（``logical -> physical``）
      或序列（下标为逻辑位、值为物理位）。``None`` 表示平凡布局（恒等，不改动）。
    - ``target``：可选 ``Target``；给出时输出线路的 ``n_qubits`` 取
      ``target.n_qubits``，并校验物理位落在 ``[0, target.n_qubits)``。
      未给 ``target`` 时，输出 ``n_qubits`` 自动取「逻辑位数」与「最大物理位+1」
      的较大值——即映射到较大物理下标会相应加宽线路。

    映射必须是单射（不同逻辑位不能映射到同一物理位）。
    """

    def __init__(self, initial_layout=None, *, target=None) -> None:
        self.initial_layout = initial_layout
        self.target = target
        # run() 后置：最近一次实际使用的 logical->physical 映射，供 TranspileResult 读取。
        self.last_layout: dict[int, int] | None = None

    def _build_map(self, n_logical: int, circuit: Circuit) -> dict[int, int]:
        layout = self.initial_layout
        if layout is None:
            return {q: q for q in range(n_logical)}
        if isinstance(layout, str):
            if layout != "auto":
                raise ValueError(f"LayoutPass: unknown layout mode {layout!r}")
            if self.target is None:
                raise ValueError("LayoutPass: 'auto' layout requires a target")
            if self.target.fully_connected:
                return {q: q for q in range(n_logical)}
            return _auto_layout(circuit, self.target)
        if isinstance(layout, dict):
            mapping = {int(k): int(v) for k, v in layout.items()}
        else:
            mapping = {logical: int(physical) for logical, physical in enumerate(layout)}

        for q in range(n_logical):
            if q not in mapping:
                raise ValueError(f"LayoutPass: no physical qubit assigned to logical qubit {q}")
        physical = list(mapping.values())
        if len(set(physical)) != len(physical):
            raise ValueError("LayoutPass: initial_layout must be injective")
        if self.target is not None:
            for p in physical:
                if p < 0 or p >= int(self.target.n_qubits):
                    raise ValueError(
                        f"LayoutPass: physical qubit {p} out of range for "
                        f"{self.target.n_qubits}-qubit target"
                    )
        return mapping

    def run(self, circuit: Circuit) -> Circuit:
        n_logical = int(circuit.n_qubits)
        mapping = self._build_map(n_logical, circuit)
        self.last_layout = dict(mapping)
        gates = [remap_gate(gate, mapping) for gate in circuit_gate_dicts(circuit)]

        if self.target is not None:
            n_physical = int(self.target.n_qubits)
        else:
            n_physical = max([n_logical, *(p + 1 for p in mapping.values())])
        return Circuit(*gates, n_qubits=n_physical, backend=getattr(circuit, "backend", None))

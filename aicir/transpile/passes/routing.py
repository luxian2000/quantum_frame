"""RoutingPass：插入 SWAP 以满足硬件连接拓扑（NEXT.md 第 2 节）。

本片实现一个“基础但正确”的路由器：对每个作用在非相邻物理比特上的
双比特门，沿耦合图最短路径插入 SWAP 把两比特移到相邻位置，施加该门后
再按相反顺序插回 SWAP 复位。因此整条线路在比特顺序意义上与原线路
**完全等价**（无需跟踪置换），代价是 SWAP 数量非最优。基于置换跟踪的
最优路由留待后续。

约束：仅支持单比特门与“恰好 2 个不同比特”的双比特门；更高阶门
（如 toffoli）抛 ``NotImplementedError``。
"""

from __future__ import annotations

from collections import deque

from ...core.circuit import Circuit
from ...ir import circuit_gate_dicts, instruction_controls, instruction_qubits
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates, remap_gate

__all__ = ["RoutingPass"]


def _shortest_path(target, a: int, b: int) -> list[int]:
    """耦合图上 ``a`` 到 ``b`` 的最短路径（含端点）；不连通时抛错。"""

    if a == b:
        return [a]
    prev = {a: a}
    queue = deque([a])
    while queue:
        node = queue.popleft()
        for nxt in target.neighbors(node):
            if nxt in prev:
                continue
            prev[nxt] = node
            if nxt == b:
                path = [b]
                while path[-1] != a:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    raise ValueError(f"RoutingPass: qubits {a} and {b} are not connected on the coupling map")


def _swap_gate(x: int, y: int) -> dict:
    return {"type": "swap", "qubit_1": int(x), "qubit_2": int(y)}


class RoutingPass(TransformationPass):
    """沿耦合图插入 SWAP，使每个双比特门作用在相邻物理比特上。

    参数：

    - ``target``：提供 ``coupling_map`` 的 ``Target``。全连接（``coupling_map``
      为 ``None``）时本 pass 为恒等。

    假设线路比特已是物理比特（通常先经 ``LayoutPass``）。
    """

    def __init__(self, target) -> None:
        if target is None:
            raise ValueError("RoutingPass requires a Target with a coupling map")
        self.target = target

    def run(self, circuit: Circuit) -> Circuit:
        n_physical = int(self.target.n_qubits)
        if int(circuit.n_qubits) > n_physical:
            raise ValueError(
                f"RoutingPass: circuit needs {circuit.n_qubits} qubits but target has {n_physical}"
            )
        if self.target.fully_connected:
            return circuit_from_gates(circuit, circuit_gate_dicts(circuit))

        out: list[dict] = []
        for gate in circuit_gate_dicts(circuit):
            qubits = tuple(dict.fromkeys((*instruction_qubits(gate), *instruction_controls(gate))))
            if len(qubits) <= 1:
                out.append(gate)
                continue
            if len(qubits) > 2:
                raise NotImplementedError(
                    f"RoutingPass: gate '{gate['type']}' acts on {len(qubits)} qubits; "
                    "basic routing supports at most 2-qubit gates"
                )
            p0, p1 = int(qubits[0]), int(qubits[1])
            if self.target.coupled(p0, p1):
                out.append(gate)
                continue

            path = _shortest_path(self.target, p0, p1)
            swaps = [(path[i], path[i + 1]) for i in range(len(path) - 2)]
            for x, y in swaps:
                out.append(_swap_gate(x, y))
            moved = path[-2]
            out.append(remap_gate(gate, lambda q, _p0=p0, _m=moved: _m if q == _p0 else q))
            for x, y in reversed(swaps):
                out.append(_swap_gate(x, y))

        return Circuit(*out, n_qubits=n_physical, backend=getattr(circuit, "backend", None))

"""RoutingPass：插入 SWAP 以满足硬件连接拓扑（NEXT.md 第 2 节）。

本片实现**置换跟踪**路由器：对作用在非相邻物理比特上的双比特门，沿耦合图
最短路径插入 SWAP 把其中一个比特移到与另一比特相邻的位置，施加该门后
**不再复位**——而是把由此产生的比特置换向前携带，后续门按当前位置重新映射。
因此整条线路与原线路等价**至最终比特置换**（``final_layout``），SWAP 数量
比“插入-复位”方案大致减半，且置换可跨门复用（相邻化的比特对在后续门上
无需再插 SWAP）。

约束：仅支持单比特门与“恰好 2 个不同比特”的双比特门；更高阶门
（如 toffoli）抛 ``NotImplementedError``。

最终置换通过 ``final_layout``（``logical -> physical wire``，覆盖全部物理线，
未移动者为恒等）暴露；``last_layout`` 在置换非恒等时镜像它，供
``PassManager.run_with_result`` 记入 ``TranspileResult.layout``（恒等时为 ``None``）。
"""

from __future__ import annotations

from collections import deque

from ...core.circuit import Circuit
from ...ir import circuit_gate_dicts, instruction_controls, instruction_name, instruction_qubits
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
    """沿耦合图插入 SWAP，使每个双比特门作用在相邻物理比特上，并向前携带置换。

    参数：

    - ``target``：提供 ``coupling_map`` 的 ``Target``。全连接（``coupling_map``
      为 ``None``）时本 pass 为恒等。

    假设线路比特已是物理比特（通常先经 ``LayoutPass``）。

    运行后属性：

    - ``final_layout``：``logical -> physical wire`` 映射，覆盖全部物理线
      （未移动者恒等）。
    - ``last_layout``：``final_layout`` 的镜像；恒等时为 ``None``。
    """

    def __init__(self, target) -> None:
        if target is None:
            raise ValueError("RoutingPass requires a Target with a coupling map")
        self.target = target
        # run() 后置：最终比特置换，供直接消费者与 TranspileResult 读取。
        self.final_layout: dict[int, int] | None = None
        self.last_layout: dict[int, int] | None = None

    def _identity_layout(self, n_physical: int) -> dict[int, int]:
        return {q: q for q in range(n_physical)}

    def _finalize(self, loc: dict[int, int], n_physical: int) -> None:
        self.final_layout = dict(loc)
        identity = self._identity_layout(n_physical)
        self.last_layout = None if loc == identity else dict(loc)

    def run(self, circuit: Circuit) -> Circuit:
        n_physical = int(self.target.n_qubits)
        if int(circuit.n_qubits) > n_physical:
            raise ValueError(
                f"RoutingPass: circuit needs {circuit.n_qubits} qubits but target has {n_physical}"
            )
        if self.target.fully_connected:
            self._finalize(self._identity_layout(n_physical), n_physical)
            return circuit_from_gates(circuit, circuit_gate_dicts(circuit))

        # loc: 逻辑比特 -> 当前所在物理线；inv: 物理线 -> 当前承载的逻辑比特。
        loc = self._identity_layout(n_physical)
        inv = self._identity_layout(n_physical)

        def do_swap(out: list[dict], w1: int, w2: int) -> None:
            out.append(_swap_gate(w1, w2))
            l1, l2 = inv[w1], inv[w2]
            loc[l1], loc[l2] = w2, w1
            inv[w1], inv[w2] = l2, l1

        out: list[dict] = []
        for gate in circuit_gate_dicts(circuit):
            qubits = tuple(dict.fromkeys((*instruction_qubits(gate), *instruction_controls(gate))))
            if len(qubits) <= 1:
                # 单比特门也要跟随当前置换重映射到逻辑比特所在的物理线。
                out.append(remap_gate(gate, loc))
                continue
            if len(qubits) > 2:
                raise NotImplementedError(
                    f"RoutingPass: gate '{instruction_name(gate)}' acts on {len(qubits)} qubits; "
                    "basic routing supports at most 2-qubit gates"
                )
            q0, q1 = int(qubits[0]), int(qubits[1])
            a, b = loc[q0], loc[q1]
            if not self.target.coupled(a, b):
                # 沿最短路径把 q0 逐跳移到与 b 相邻的位置（停在 path[-2]），不复位。
                path = _shortest_path(self.target, a, b)
                for i in range(len(path) - 2):
                    do_swap(out, path[i], path[i + 1])
            out.append(remap_gate(gate, loc))

        self._finalize(loc, n_physical)
        return Circuit(*out, n_qubits=n_physical, backend=getattr(circuit, "backend", None))

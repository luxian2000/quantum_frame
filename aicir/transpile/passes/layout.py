"""LayoutPass：选择 logical-to-physical 比特映射（NEXT.md 第 2 节）。

本片实现“显式 / 平凡”布局：按给定的逻辑->物理映射重新标号线路比特，
不插入任何门，因此线路在比特置换意义下与原线路等价。自动布局
（按拓扑/噪声择优）留待后续。
"""

from __future__ import annotations

from ...core.circuit import Circuit
from ...ir import circuit_gate_dicts
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates, remap_gate

__all__ = ["LayoutPass"]


class LayoutPass(TransformationPass):
    """把逻辑比特重新标号到物理比特。

    参数：

    - ``initial_layout``：逻辑->物理映射。可为 ``dict``（``logical -> physical``）
      或序列（下标为逻辑位、值为物理位）。``None`` 表示平凡布局（恒等，不改动）。
    - ``target``：可选 ``Target``；给出时输出线路的 ``n_qubits`` 取
      ``target.n_qubits``，并校验物理位落在 ``[0, target.n_qubits)``。

    映射必须是单射（不同逻辑位不能映射到同一物理位）。
    """

    def __init__(self, initial_layout=None, *, target=None) -> None:
        self.initial_layout = initial_layout
        self.target = target

    def _build_map(self, n_logical: int) -> dict[int, int]:
        layout = self.initial_layout
        if layout is None:
            return {q: q for q in range(n_logical)}
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
        mapping = self._build_map(n_logical)
        gates = [remap_gate(gate, mapping) for gate in circuit_gate_dicts(circuit)]

        if self.target is not None:
            n_physical = int(self.target.n_qubits)
        else:
            n_physical = max([n_logical, *(p + 1 for p in mapping.values())])
        return Circuit(*gates, n_qubits=n_physical, backend=getattr(circuit, "backend", None))

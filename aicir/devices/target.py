"""Target：硬件能力描述（NEXT.md 第 3 节）。

``Target`` 统一描述一个后端/设备支持的门集、连接拓扑和执行能力，
作为 ``transpile``（分解、布局、路由）、``measure``、``vqc``、``qas``、
``metrics`` 的共同输入。本片先落地数据模型与连接拓扑查询，供
``DecomposePass``/``LayoutPass``/``RoutingPass`` 消费。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..gates import canonical_gate_name

__all__ = ["Target"]


def _normalize_pairs(coupling: Iterable[Iterable[int]] | None) -> tuple[tuple[int, int], ...] | None:
    if coupling is None:
        return None
    pairs: list[tuple[int, int]] = []
    for edge in coupling:
        a, b = (int(x) for x in edge)
        if a == b:
            raise ValueError(f"coupling edge connects a qubit to itself: {edge}")
        pairs.append((a, b))
    return tuple(pairs)


@dataclass(frozen=True)
class Target:
    """设备能力描述。

    属性：

    - ``n_qubits``：物理比特数。
    - ``basis_gates``：原生门集（规范门名）；空元组表示不限制门集。
    - ``coupling_map``：双比特门可作用的物理比特对；``None`` 表示全连接。
      连接关系按无向图处理（``(a, b)`` 等价 ``(b, a)``）。
    - ``supports_shots`` / ``supports_statevector`` /
      ``supports_density_matrix`` / ``supports_autodiff``：执行能力标志，
      供 primitives 与梯度方法选择执行路径。
    """

    n_qubits: int
    basis_gates: tuple[str, ...] = ()
    coupling_map: tuple[tuple[int, int], ...] | None = None
    supports_shots: bool = True
    supports_statevector: bool = True
    supports_density_matrix: bool = False
    supports_autodiff: bool = False
    _adjacency: dict[int, frozenset[int]] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.n_qubits) <= 0:
            raise ValueError("n_qubits must be positive")
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        object.__setattr__(
            self,
            "basis_gates",
            tuple(canonical_gate_name(name) for name in self.basis_gates),
        )
        pairs = _normalize_pairs(self.coupling_map)
        object.__setattr__(self, "coupling_map", pairs)

        adjacency: dict[int, set[int]] = {}
        if pairs is not None:
            for a, b in pairs:
                for q in (a, b):
                    if q < 0 or q >= self.n_qubits:
                        raise ValueError(
                            f"coupling qubit {q} out of range for {self.n_qubits}-qubit target"
                        )
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
        object.__setattr__(
            self, "_adjacency", {q: frozenset(ns) for q, ns in adjacency.items()}
        )

    # -- 门集查询 --------------------------------------------------------
    def supports(self, gate: str) -> bool:
        """门是否属于原生门集（按规范名比较）；门集为空时一律返回 ``True``。"""

        if not self.basis_gates:
            return True
        return canonical_gate_name(gate) in self.basis_gates

    # -- 拓扑查询 --------------------------------------------------------
    @property
    def fully_connected(self) -> bool:
        return self.coupling_map is None

    def coupled(self, a: int, b: int) -> bool:
        """物理比特 ``a`` 与 ``b`` 是否直接相连（无向）。全连接时恒为 ``True``。"""

        if self.coupling_map is None:
            return a != b
        return b in self._adjacency.get(int(a), frozenset())

    def neighbors(self, qubit: int) -> tuple[int, ...]:
        """返回与 ``qubit`` 直接相连的物理比特（升序）。"""

        if self.coupling_map is None:
            return tuple(q for q in range(self.n_qubits) if q != qubit)
        return tuple(sorted(self._adjacency.get(int(qubit), frozenset())))

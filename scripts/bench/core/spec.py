"""框架无关的线路规格。

跨框架基准的第一条有效性规则：**各框架必须跑同一条线路**。因此这里不直接构造
任何框架的线路对象，而是产出一份声明式的 ``CircuitSpec``——一个 ``(gate, qubits,
params)`` 三元组序列——再由各适配器翻译成自己的线路类型。

这样做的好处是可复现性可以被单测钉死：同 family 同 seed 必然给出逐字节相同的
``operations``，与任何框架的随机数状态无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["CircuitSpec", "Operation", "build_spec", "available_families"]

#: 规格里允许出现的门。刻意保持最小集——每个门都必须在所有对照框架里有
#: 语义完全一致的对应物，否则比较就不成立。
SUPPORTED_GATES = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz", "cp", "swap")


@dataclass(frozen=True)
class Operation:
    """一条门指令：门名 + 作用比特 + 参数。"""

    gate: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self):
        if self.gate not in SUPPORTED_GATES:
            raise ValueError(f"不支持的门 {self.gate!r}；可用：{SUPPORTED_GATES}")


@dataclass(frozen=True)
class CircuitSpec:
    """一条基准线路的完整声明。

    ``operations`` 是唯一的真源；``family``/``seed`` 等字段只用于清单记录与复现。
    """

    family: str
    n_qubits: int
    operations: tuple[Operation, ...]
    depth: int | None = None
    seed: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_gates(self) -> int:
        return len(self.operations)

    @property
    def n_two_qubit_gates(self) -> int:
        return sum(1 for op in self.operations if len(op.qubits) == 2)

    def label(self) -> str:
        parts = [self.family, f"n{self.n_qubits}"]
        if self.depth is not None:
            parts.append(f"d{self.depth}")
        if self.seed is not None:
            parts.append(f"s{self.seed}")
        return "-".join(parts)


# ────────────────────────────── 线路族 ──────────────────────────────


def _ghz(n_qubits: int, **_) -> list[Operation]:
    """GHZ 制备：最浅的纠缠基准，主要压门派发开销而非收缩。"""

    ops = [Operation("h", (0,))]
    ops += [Operation("cx", (q, q + 1)) for q in range(n_qubits - 1)]
    return ops


def _qft(n_qubits: int, **_) -> list[Operation]:
    """标准 QFT（不含末端比特反转）：稠密受控相位，双比特门数为 O(n²)。"""

    ops: list[Operation] = []
    for target in range(n_qubits):
        ops.append(Operation("h", (target,)))
        for control in range(target + 1, n_qubits):
            angle = math.pi / (2 ** (control - target))
            ops.append(Operation("cp", (control, target), (angle,)))
    return ops


def _random(n_qubits: int, depth: int = 4, seed: int | None = None, **_) -> list[Operation]:
    """随机线路：每层单比特随机旋转 + 交错的最近邻纠缠层。

    近似 supremacy 类基准的结构，但只用各框架语义一致的门。
    """

    rng = np.random.default_rng(seed)
    single = ("rx", "ry", "rz")
    ops: list[Operation] = []
    for layer in range(depth):
        for qubit in range(n_qubits):
            gate = single[int(rng.integers(len(single)))]
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            ops.append(Operation(gate, (qubit,), (angle,)))
        # 偶数层接 (0,1),(2,3)…；奇数层接 (1,2),(3,4)… —— 标准砖墙纠缠。
        for qubit in range(layer % 2, n_qubits - 1, 2):
            ops.append(Operation("cx", (qubit, qubit + 1)))
    return ops


def _layered_ansatz(n_qubits: int, depth: int = 2, seed: int | None = None, **_) -> list[Operation]:
    """硬件高效 ansatz：变分工作负载的代表，参数全部显式给定。"""

    rng = np.random.default_rng(seed)
    ops: list[Operation] = []
    for _ in range(depth):
        for qubit in range(n_qubits):
            ops.append(Operation("ry", (qubit,), (float(rng.uniform(0.0, 2.0 * math.pi)),)))
            ops.append(Operation("rz", (qubit,), (float(rng.uniform(0.0, 2.0 * math.pi)),)))
        for qubit in range(n_qubits - 1):
            ops.append(Operation("cx", (qubit, qubit + 1)))
    return ops


_FAMILIES = {
    "ghz": _ghz,
    "qft": _qft,
    "random": _random,
    "layered_ansatz": _layered_ansatz,
}


def available_families() -> tuple[str, ...]:
    """返回全部可用线路族名。"""

    return tuple(sorted(_FAMILIES))


def build_spec(family: str, *, n_qubits: int, depth: int | None = None, seed: int | None = None) -> CircuitSpec:
    """构造一条基准线路规格。

    参数:
        family:   线路族名，见 ``available_families()``。
        n_qubits: 比特数。
        depth:    层数（``ghz``/``qft`` 忽略该参数）。
        seed:     随机种子；同 seed 必然复现同一条线路。
    """

    if family not in _FAMILIES:
        raise KeyError(f"未知线路族 {family!r}；可用：{available_families()}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits 必须为正，收到 {n_qubits}")

    kwargs = {}
    if depth is not None:
        kwargs["depth"] = depth
    if seed is not None:
        kwargs["seed"] = seed

    operations = tuple(_FAMILIES[family](n_qubits, **kwargs))
    return CircuitSpec(
        family=family,
        n_qubits=n_qubits,
        operations=operations,
        depth=depth,
        seed=seed,
    )


def gate_counts(spec: CircuitSpec) -> dict[str, int]:
    """按门名统计数量，供清单记录工作量。"""

    counts: dict[str, int] = {}
    for op in spec.operations:
        counts[op.gate] = counts.get(op.gate, 0) + 1
    return counts

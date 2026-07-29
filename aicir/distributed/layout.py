"""Distribution metadata and logical-to-storage qubit layouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from ..ir import circuit_instructions, instruction_controls, instruction_qubits


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class _Layout:
    """Immutable mapping from logical qubits to storage tensor axes."""

    logical_to_storage: tuple[int, ...]
    distributed_axes: int

    @classmethod
    def explicit(
        cls,
        mapping,
        *,
        n_qubits: int,
        distributed_axes: int,
    ) -> "_Layout":
        n_qubits = int(n_qubits)
        distributed_axes = int(distributed_axes)
        values = tuple(int(value) for value in mapping)
        if sorted(values) != list(range(n_qubits)):
            raise ValueError("layout 必须是 range(n_qubits) 的完整双射")
        if distributed_axes < 0 or distributed_axes > n_qubits:
            raise ValueError(
                f"distributed_axes={distributed_axes} 必须位于 [0, {n_qubits}]"
            )
        return cls(values, distributed_axes)

    @classmethod
    def auto(
        cls,
        circuit,
        *,
        n_qubits: int,
        distributed_axes: int,
    ) -> "_Layout":
        n_qubits = int(n_qubits)
        distributed_axes = int(distributed_axes)
        if distributed_axes < 0 or distributed_axes > n_qubits:
            raise ValueError(
                f"distributed_axes={distributed_axes} 必须位于 [0, {n_qubits}]"
            )

        instruction_qubit_sets = []
        for instruction in circuit_instructions(circuit):
            qubits = {
                int(qubit)
                for qubit in (
                    *instruction_qubits(instruction),
                    *instruction_controls(instruction),
                )
            }
            instruction_qubit_sets.append(qubits)

        selected: set[int] = set()
        while len(selected) < distributed_axes:
            candidates = []
            for qubit in range(n_qubits):
                if qubit in selected:
                    continue
                candidate = selected | {qubit}
                score = sum(
                    (1 << len(targets & candidate)) - 1
                    for targets in instruction_qubit_sets
                )
                candidates.append((score, qubit))
            selected.add(min(candidates)[1])

        distributed = sorted(selected)
        local = [qubit for qubit in range(n_qubits) if qubit not in selected]
        logical_to_storage = [0] * n_qubits
        for storage, logical in enumerate((*distributed, *local)):
            logical_to_storage[logical] = storage
        return cls(tuple(logical_to_storage), distributed_axes)

    @property
    def n_qubits(self) -> int:
        return len(self.logical_to_storage)

    @property
    def storage_to_logical(self) -> tuple[int, ...]:
        inverse = [0] * self.n_qubits
        for logical, storage in enumerate(self.logical_to_storage):
            inverse[storage] = logical
        return tuple(inverse)

    @property
    def distributed_logical_qubits(self) -> tuple[int, ...]:
        return tuple(
            logical
            for logical, storage in enumerate(self.logical_to_storage)
            if storage < self.distributed_axes
        )

    def digest(self) -> str:
        payload = (self.logical_to_storage, self.distributed_axes)
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ShardSpec:
    """Global and rank-local shapes for one distributed state."""

    n_qubits: int
    world_size: int
    rank: int
    kind: str
    layout: _Layout
    global_shape: tuple[int, int]
    local_shape: tuple[int, int]
    global_start: int
    global_stop: int

    @classmethod
    def build(
        cls,
        n_qubits: int,
        world_size: int,
        rank: int,
        kind: str,
        layout: _Layout,
    ) -> "_ShardSpec":
        n_qubits = int(n_qubits)
        world_size = int(world_size)
        rank = int(rank)
        if not _is_power_of_two(world_size):
            raise ValueError("world_size 必须是 2 的幂")
        distributed_axes = int(math.log2(world_size))
        if n_qubits < distributed_axes:
            raise ValueError(
                f"n_qubits={n_qubits} 小于分布式存储轴数 {distributed_axes}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} 必须位于 [0, {world_size})")
        if kind not in {"vector", "matrix"}:
            raise ValueError("kind 必须是 'vector' 或 'matrix'")
        if layout.n_qubits != n_qubits:
            raise ValueError("layout 的 n_qubits 与状态不一致")
        if layout.distributed_axes != distributed_axes:
            raise ValueError(
                "layout.distributed_axes 必须等于 log2(world_size)"
            )

        dimension = 1 << n_qubits
        local_rows = dimension // world_size
        global_shape = (
            (dimension, 1) if kind == "vector" else (dimension, dimension)
        )
        local_shape = (
            (local_rows, 1)
            if kind == "vector"
            else (local_rows, dimension)
        )
        global_start = rank * local_rows
        return cls(
            n_qubits=n_qubits,
            world_size=world_size,
            rank=rank,
            kind=kind,
            layout=layout,
            global_shape=global_shape,
            local_shape=local_shape,
            global_start=global_start,
            global_stop=global_start + local_rows,
        )

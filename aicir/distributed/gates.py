"""Gate planning and streaming sharded-statevector kernels."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

from ..core.gates import (
    _apply_local_matrix_to_state,
    _cast_local_matrix,
    _gate_local_matrix,
)
from ..gates import canonical_gate_name
from ..ir import as_instruction, instruction_name
from ._contracts import reject_requires_grad
from .state import DistState


@dataclass(frozen=True)
class _GatePlan:
    instruction_index: int
    local_matrix: object
    logical_axes: tuple[int, ...]
    storage_axes: tuple[int, ...]
    distributed_storage_axes: tuple[int, ...]
    local_storage_axes: tuple[int, ...]
    partner_masks: tuple[int, ...]
    distributed_axes: int

    def partner_for(self, *, rank: int, mask: int) -> int:
        if int(mask) not in self.partner_masks:
            raise ValueError(f"mask={mask} 不属于该门的 partner_masks")
        return int(rank) ^ int(mask)

    def tag(self, mask: int) -> int:
        return int(self.instruction_index) * 4096 + int(mask)


class _GatePlanner:
    """Resolve a typed instruction into storage axes and rank partners."""

    def __init__(self, backend, layout, n_qubits: int):
        self._backend = backend
        self._layout = layout
        self._n_qubits = int(n_qubits)
        if layout.n_qubits != self._n_qubits:
            raise ValueError("layout 与 n_qubits 不一致")

    def plan(self, gate, instruction_index: int) -> _GatePlan:
        instruction = as_instruction(gate)
        gate_type = canonical_gate_name(instruction_name(instruction))
        local, logical_axes, cache_key = _gate_local_matrix(
            instruction,
            gate_type,
            self._backend,
        )
        if local is None or logical_axes is None:
            raise ValueError(
                f"指令 {gate_type!r} 没有可用于分布式执行的局部门矩阵"
            )

        matrix = _cast_local_matrix(
            self._backend,
            local,
            cache_key=cache_key,
        )
        return self.plan_matrix(
            matrix,
            logical_axes,
            instruction_index=instruction_index,
        )

    def plan_matrix(
        self,
        local_matrix,
        logical_axes,
        *,
        instruction_index: int,
    ) -> _GatePlan:
        logical_axes = tuple(int(axis) for axis in logical_axes)
        if len(set(logical_axes)) != len(logical_axes):
            raise ValueError("局部矩阵的逻辑量子比特不能重复")
        if any(
            logical < 0 or logical >= self._n_qubits
            for logical in logical_axes
        ):
            raise ValueError("局部矩阵的逻辑量子比特超出范围")
        storage_axes = tuple(
            self._layout.logical_to_storage[logical]
            for logical in logical_axes
        )
        distributed_storage_axes = tuple(
            sorted(
                storage
                for storage in storage_axes
                if storage < self._layout.distributed_axes
            )
        )
        local_storage_axes = tuple(
            storage
            for storage in storage_axes
            if storage >= self._layout.distributed_axes
        )

        rank_bit_masks = tuple(
            1 << (self._layout.distributed_axes - 1 - storage)
            for storage in distributed_storage_axes
        )
        partner_masks = []
        for count in range(1, len(rank_bit_masks) + 1):
            for combination in itertools.combinations(rank_bit_masks, count):
                mask = 0
                for bit in combination:
                    mask ^= bit
                partner_masks.append(mask)

        matrix = self._backend.cast_local_matrix(local_matrix)
        dimension = 1 << len(logical_axes)
        if tuple(int(axis) for axis in matrix.shape) != (
            dimension,
            dimension,
        ):
            raise ValueError(
                f"局部门矩阵形状必须是 ({dimension}, {dimension})"
            )
        reject_requires_grad(matrix)
        return _GatePlan(
            instruction_index=int(instruction_index),
            local_matrix=matrix,
            logical_axes=logical_axes,
            storage_axes=storage_axes,
            distributed_storage_axes=distributed_storage_axes,
            local_storage_axes=local_storage_axes,
            partner_masks=tuple(sorted(partner_masks)),
            distributed_axes=self._layout.distributed_axes,
        )


def _rank_axis_bit(rank: int, storage_axis: int, distributed_axes: int) -> int:
    shift = int(distributed_axes) - 1 - int(storage_axis)
    return (int(rank) >> shift) & 1


def _matrix_block(
    matrix,
    storage_axes,
    *,
    output_rank: int,
    input_rank: int,
    distributed_axes: int,
):
    """Fix distributed output/input bits and retain local target axes."""

    storage_axes = tuple(int(axis) for axis in storage_axes)
    arity = len(storage_axes)
    shaped = matrix.reshape((2,) * (2 * arity))
    selectors = [slice(None)] * (2 * arity)
    local_count = 0
    for position, storage_axis in enumerate(storage_axes):
        if storage_axis < distributed_axes:
            selectors[position] = _rank_axis_bit(
                output_rank,
                storage_axis,
                distributed_axes,
            )
            selectors[arity + position] = _rank_axis_bit(
                input_rank,
                storage_axis,
                distributed_axes,
            )
        else:
            local_count += 1
    block = shaped[tuple(selectors)]
    local_dimension = 1 << local_count
    return block.reshape(local_dimension, local_dimension)


class _VectorKernel:
    """Apply local-matrix blocks to one statevector shard."""

    def __init__(self, backend):
        self._backend = backend

    def _apply_source(self, source, plan: _GatePlan, source_rank: int):
        block = _matrix_block(
            plan.local_matrix,
            plan.storage_axes,
            output_rank=self._backend.rank,
            input_rank=source_rank,
            distributed_axes=plan.distributed_axes,
        )
        local_axes = tuple(
            storage - plan.distributed_axes
            for storage in plan.storage_axes
            if storage >= plan.distributed_axes
        )
        if not local_axes:
            return self._backend.mul(source, block.reshape(()))
        local_n_qubits = int(source.shape[0]).bit_length() - 1
        return _apply_local_matrix_to_state(
            source,
            block,
            local_axes,
            local_n_qubits,
            self._backend,
        )

    def apply(self, state: DistState, plan: _GatePlan) -> DistState:
        if state.kind != "vector":
            raise TypeError("_VectorKernel 仅接受 vector DistState")

        output = self._apply_source(
            state.local_data,
            plan,
            self._backend.rank,
        )
        for mask in plan.partner_masks:
            peer = plan.partner_for(rank=self._backend.rank, mask=mask)
            incoming = self._backend.communicator.exchange(
                state.local_data,
                peer=peer,
                tag=plan.tag(mask),
            )
            contribution = self._apply_source(incoming, plan, peer)
            output = self._backend.add(output, contribution)

        return DistState.from_local(
            output,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

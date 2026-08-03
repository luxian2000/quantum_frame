"""Row-sharded density-matrix evolution."""

from __future__ import annotations

import torch

from .gates import _GatePlanner, _matrix_block
from .layout import _ShardSpec
from .state import DistState


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for index, axis in enumerate(permutation):
        inverse[int(axis)] = index
    return tuple(inverse)


def _contiguous(value):
    return value.contiguous() if isinstance(value, torch.Tensor) else value


def _apply_matrix_to_rows(
    rows,
    local_matrix,
    axes,
    n_row_qubits: int,
    backend,
):
    """Apply a local matrix along row axes and preserve trailing batches."""

    axes = tuple(int(axis) for axis in axes)
    if not axes:
        return backend.mul(rows, local_matrix.reshape(()))
    target_set = set(axes)
    group_shape = []
    target_dimensions = {}
    gap_dimensions = []
    pending_gap = 1

    def flush_gap():
        nonlocal pending_gap
        if pending_gap > 1:
            gap_dimensions.append(len(group_shape))
            group_shape.append(pending_gap)
            pending_gap = 1

    for qubit in range(int(n_row_qubits)):
        if qubit in target_set:
            flush_gap()
            target_dimensions[qubit] = len(group_shape)
            group_shape.append(2)
        else:
            pending_gap *= 2
    flush_gap()

    batch = int(rows.numel() // rows.shape[0])
    batch_axis = len(group_shape)
    permutation = (
        tuple(target_dimensions[axis] for axis in axes)
        + tuple(gap_dimensions)
        + (batch_axis,)
    )
    reshaped = backend.reshape(rows, (*group_shape, batch))
    moved = _contiguous(backend.transpose(reshaped, permutation))
    flat = backend.reshape(moved, (1 << len(axes), -1))
    updated = backend.apply_local_matrix(local_matrix, flat)
    restored_shape = (
        (2,) * len(axes)
        + tuple(group_shape[index] for index in gap_dimensions)
        + (batch,)
    )
    restored = backend.reshape(updated, restored_shape)
    restored = _contiguous(
        backend.transpose(restored, _inverse_permutation(permutation))
    )
    return backend.reshape(restored, rows.shape)


class _MatrixKernel:
    """Execute row-sharded ``A rho A†`` transformations."""

    def __init__(self, backend):
        self._backend = backend

    def _apply_left_source(self, source, plan, source_rank):
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
        local_n_qubits = int(source.shape[0]).bit_length() - 1
        return _apply_matrix_to_rows(
            source,
            block,
            local_axes,
            local_n_qubits,
            self._backend,
        )

    def _apply_left(self, state, plan):
        output = self._apply_left_source(
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
            contribution = self._apply_left_source(incoming, plan, peer)
            output = self._backend.add(output, contribution)
        return output

    def apply_left(self, state: DistState, plan):
        if state.kind != "matrix":
            raise TypeError("_MatrixKernel 仅接受 matrix DistState")
        if getattr(state, "_pair", None) is not None:
            from .autograd._density import _PairMatrixKernel

            return _PairMatrixKernel(self._backend).apply_left(
                state, plan, operation_index=plan.instruction_index
            )
        return self._apply_left(state, plan)

    def apply_unitary(self, state: DistState, plan) -> DistState:
        if state.kind != "matrix":
            raise TypeError("_MatrixKernel 仅接受 matrix DistState")
        if getattr(state, "_pair", None) is not None:
            from .autograd._density import _PairMatrixKernel

            return _PairMatrixKernel(self._backend).apply_unitary(
                state, plan, operation_index=plan.instruction_index
            )
        left = self._apply_left(state, plan)
        columns_first = _contiguous(
            self._backend.transpose(left, (1, 0))
        )
        conjugate = self._backend.conj(plan.local_matrix)
        right_transposed = _apply_matrix_to_rows(
            columns_first,
            conjugate,
            plan.storage_axes,
            state.n_qubits,
            self._backend,
        )
        result = _contiguous(
            self._backend.transpose(right_transposed, (1, 0))
        )
        return DistState.from_local(
            result,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

    def promote_vector(self, state: DistState) -> DistState:
        if state.kind != "vector":
            return state
        if getattr(state, "_pair", None) is not None:
            from .autograd._density import _PairMatrixKernel

            return _PairMatrixKernel(self._backend).promote_vector(state)
        shards = self._backend.communicator.all_gather(state.local_data)
        full = torch.cat(shards, dim=0)
        local_rows = state.local_data
        density_rows = self._backend.matmul(
            local_rows,
            self._backend.dagger(full),
        )
        spec = _ShardSpec.build(
            state.n_qubits,
            state.world_size,
            state.rank,
            "matrix",
            state.layout,
        )
        return DistState.from_local(
            density_rows,
            spec=spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

    def apply_channel(
        self,
        state: DistState,
        channel,
        *,
        instruction_index: int,
    ) -> DistState:
        if getattr(state, "_pair", None) is not None:
            from .autograd._density import _PairMatrixKernel

            return _PairMatrixKernel(self._backend).apply_channel(
                state, channel, instruction_index=instruction_index
            )
        state = self.promote_vector(state)
        planner = _GatePlanner(
            self._backend,
            state.layout,
            state.n_qubits,
        )
        accumulator = self._backend.zeros(state.local_shape)
        for offset, (matrix, logical_axes) in enumerate(
            channel._local_kraus(state.n_qubits, self._backend)
        ):
            plan = planner.plan_matrix(
                matrix,
                logical_axes,
                instruction_index=instruction_index * 256 + offset,
            )
            contribution = self.apply_unitary(state, plan)
            accumulator = self._backend.add(
                accumulator,
                contribution.local_data,
            )
        return DistState.from_local(
            accumulator,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

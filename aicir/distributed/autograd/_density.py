"""Paired-real row-sharded density-matrix evolution.

The kernel deliberately mirrors the established density storage contract:
only rows are sharded while every rank owns all columns.  A left action uses
the gate planner's partner exchanges; the right action is consequently a
local action on the transposed, full column dimension.  No density matrix is
materialized or gathered by this module.
"""

from __future__ import annotations

import torch

from ...core.gates import _flat_local_state_indices
from ..gates import _GatePlan
from ..layout import _ShardSpec
from ..state import DistState
from ._collectives import _exchange_pair
from ._pair import _Pair
from ._vector import _as_pair_matrix, _pair_block


def _scatter_rows(values: torch.Tensor, indices: torch.Tensor, shape) -> torch.Tensor:
    """Restore gathered row blocks with a real scatter-add operation."""

    columns = int(shape[1])
    output = torch.zeros(shape, dtype=torch.float32, device=values.device)
    expanded = indices.reshape(-1, 1).expand(-1, columns)
    return output.scatter_add_(0, expanded, values.reshape(-1, columns))


def _apply_rows(source: _Pair, matrix: _Pair, axes, n_qubits: int) -> _Pair:
    """Apply ``matrix`` to selected row axes, retaining all trailing columns."""

    axes = tuple(int(axis) for axis in axes)
    if not axes:
        scalar = _Pair(matrix.real.reshape(()), matrix.imag.reshape(()))
        return source.mul(scalar)
    indices = torch.as_tensor(
        _flat_local_state_indices(axes, int(n_qubits)),
        dtype=torch.long,
        device=source.real.device,
    )
    target_dimension = int(indices.shape[0])
    columns = int(source.real.shape[1])
    gathered = _Pair(
        torch.index_select(source.real, 0, indices.reshape(-1)).reshape(target_dimension, -1),
        torch.index_select(source.imag, 0, indices.reshape(-1)).reshape(target_dimension, -1),
    )
    updated = matrix.matmul(gathered)
    return _Pair(
        _scatter_rows(updated.real, indices, (source.real.shape[0], columns)),
        _scatter_rows(updated.imag, indices, (source.imag.shape[0], columns)),
    )


class _PairMatrixKernel:
    """Apply planned density operations entirely on float32 paired buffers."""

    def __init__(self, backend):
        self._backend = backend

    def _apply_left_source(self, source: _Pair, plan: _GatePlan, matrix: _Pair, *, source_rank: int) -> _Pair:
        block = _pair_block(
            matrix,
            plan,
            output_rank=self._backend.rank,
            input_rank=source_rank,
        )
        local_axes = tuple(
            storage - plan.distributed_axes
            for storage in plan.storage_axes
            if storage >= plan.distributed_axes
        )
        local_n_qubits = int(source.real.shape[0]).bit_length() - 1
        return _apply_rows(source, block, local_axes, local_n_qubits)

    def apply_left(self, state: DistState, plan: _GatePlan, *, operation_index: int) -> _Pair:
        """Return ``A rho`` in row-sharded paired-real form."""

        if state.kind != "matrix" or getattr(state, "_pair", None) is None:
            raise TypeError("_PairMatrixKernel 仅接受 paired-real matrix DistState")
        operation_index = int(operation_index)
        if operation_index < 0:
            raise ValueError("operation_index 必须非负")
        matrix = _as_pair_matrix(plan.local_matrix, reference=state._pair.real)
        output = self._apply_left_source(
            state._pair, plan, matrix, source_rank=self._backend.rank
        )
        for offset, mask in enumerate(plan.partner_masks, start=1):
            peer = plan.partner_for(rank=self._backend.rank, mask=mask)
            incoming = _exchange_pair(
                state._pair,
                communicator=self._backend.communicator,
                peer=peer,
                operation_index=operation_index * self._backend.world_size + offset,
                phase="forward",
            )
            output = output.add(
                self._apply_left_source(incoming, plan, matrix, source_rank=peer)
            )
        return output

    def apply_unitary(self, state: DistState, plan: _GatePlan, *, operation_index: int) -> DistState:
        """Apply ``U rho U†`` without gathering the row-sharded density matrix."""

        left = self.apply_left(state, plan, operation_index=operation_index)
        matrix = _as_pair_matrix(plan.local_matrix, reference=left.real)
        columns_first = _Pair(left.real.transpose(0, 1), left.imag.transpose(0, 1))
        right_transposed = _apply_rows(
            columns_first,
            _Pair(matrix.real, -matrix.imag),
            plan.storage_axes,
            state.n_qubits,
        )
        result = _Pair(
            right_transposed.real.transpose(0, 1),
            right_transposed.imag.transpose(0, 1),
        )
        return DistState.from_pair(
            result,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

    def promote_vector(self, state: DistState) -> DistState:
        """Promote a paired-real vector to a row-sharded ``|psi><psi|`` matrix."""

        if state.kind == "matrix":
            return state
        if state.kind != "vector" or getattr(state, "_pair", None) is None:
            raise TypeError("_PairMatrixKernel 仅接受 paired-real vector DistState")
        full = state._pair
        # A hypercube all-gather made of the existing differentiable P2P
        # primitive keeps both transport and its VJP in paired float32 form.
        for stage in range(state.layout.distributed_axes):
            mask = 1 << stage
            peer = state.rank ^ mask
            incoming = _exchange_pair(
                full,
                communicator=self._backend.communicator,
                peer=peer,
                operation_index=900_000 + stage,
                phase="forward",
            )
            if state.rank & mask:
                full = _Pair(
                    torch.cat((incoming.real, full.real), dim=0),
                    torch.cat((incoming.imag, full.imag), dim=0),
                )
            else:
                full = _Pair(
                    torch.cat((full.real, incoming.real), dim=0),
                    torch.cat((full.imag, incoming.imag), dim=0),
                )
        local = state._pair
        density = _Pair(
            local.real @ full.real.t() + local.imag @ full.imag.t(),
            local.imag @ full.real.t() - local.real @ full.imag.t(),
        )
        spec = _ShardSpec.build(
            state.n_qubits,
            state.world_size,
            state.rank,
            "matrix",
            state.layout,
        )
        return DistState.from_pair(
            density, spec=spec, backend=self._backend, bit_order=state.bit_order
        )

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
from ._collectives import _descriptor, _exchange_pair, _safe_int, _synchronize_preflight
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


def _preflight_density_operation(state: DistState, plan: _GatePlan, *, operation_index: int) -> None:
    """Reject divergent density metadata before any rank enters data P2P."""

    operation = _safe_int(operation_index)
    try:
        storage_axes = tuple(int(axis) for axis in plan.storage_axes)
        valid_plan = (
            len(storage_axes) <= 7
            and int(plan.distributed_axes) == state.layout.distributed_axes
            and int(plan.instruction_index) >= 0
        )
    except Exception:  # noqa: BLE001 - collective preflight must be uniform
        storage_axes, valid_plan = (), False
    valid_state = (
        isinstance(state, DistState)
        and state.kind == "matrix"
        and isinstance(getattr(state, "_pair", None), _Pair)
        and operation is not None
        and operation >= 0
    )
    descriptor = _descriptor(
        valid=valid_state and valid_plan,
        code=1,
        values=(
            operation if operation is not None else 0,
            state.n_qubits if isinstance(state, DistState) else 0,
            state.local_shape[0] if isinstance(state, DistState) else 0,
            state.local_shape[1] if isinstance(state, DistState) else 0,
            int(plan.instruction_index) if valid_plan else 0,
            int(plan.distributed_axes) if valid_plan else 0,
            len(storage_axes),
            *storage_axes,
        ),
        communicator=state.backend.communicator if isinstance(state, DistState) else plan._backend.communicator,
    )
    _synchronize_preflight(
        state.backend.communicator,
        descriptor,
        names={1: "density plan"},
        fields=tuple(range(2, 16)),
        field_names={field: "density plan" for field in range(2, 16)},
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

        _preflight_density_operation(state, plan, operation_index=operation_index)
        operation_index = int(operation_index)
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
            # The next cross-shard left action uses this pair as a P2P source.
            # Transposing the right action leaves a non-contiguous view; make
            # the *real* paired buffers contiguous here so a replayed segment
            # has the identical transport contract on Gloo/HCCL.
            right_transposed.real.transpose(0, 1).contiguous(),
            right_transposed.imag.transpose(0, 1).contiguous(),
        )
        return DistState.from_pair(
            result,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

    def apply_channel(self, state: DistState, channel, *, instruction_index: int) -> DistState:
        """Apply an analytic Kraus channel in paired-real row-sharded form."""

        from ._channels import _PairChannelKernel

        return _PairChannelKernel(self._backend).apply_channel(
            state, channel, instruction_index=instruction_index
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

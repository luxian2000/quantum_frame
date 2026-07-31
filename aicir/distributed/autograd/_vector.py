"""Paired-real, shard-local statevector evolution.

This module is intentionally separate from the forward-only distributed
simulator.  It consumes the existing gate planner's rank and local-axis plan,
but never creates a complex tensor in the execution graph.
"""

from __future__ import annotations

import numpy as np
import torch

from ...core.gates import _flat_local_state_indices
from ..gates import _GatePlan, _matrix_block
from ._collectives import _exchange_pair
from ._pair import _Pair


def _as_pair_matrix(matrix, *, reference: torch.Tensor) -> _Pair:
    """Convert a planning boundary matrix to float32 real and imaginary parts."""

    if isinstance(matrix, _Pair):
        return matrix
    if isinstance(matrix, torch.Tensor):
        if matrix.device != reference.device:
            matrix = matrix.to(device=reference.device)
        if matrix.dtype == torch.float32:
            return _Pair(matrix, torch.zeros_like(matrix))
        return _Pair(torch.real(matrix).to(torch.float32), torch.imag(matrix).to(torch.float32))
    array = np.asarray(matrix)
    return _Pair(
        torch.as_tensor(np.real(array), dtype=torch.float32, device=reference.device),
        torch.as_tensor(np.imag(array), dtype=torch.float32, device=reference.device),
    )


def _pair_block(matrix: _Pair, plan: _GatePlan, *, output_rank: int, input_rank: int) -> _Pair:
    """Select the planner-defined source/output rank block from both parts."""

    return _Pair(
        _matrix_block(
            matrix.real,
            plan.storage_axes,
            output_rank=output_rank,
            input_rank=input_rank,
            distributed_axes=plan.distributed_axes,
        ).to(dtype=torch.float32),
        _matrix_block(
            matrix.imag,
            plan.storage_axes,
            output_rank=output_rank,
            input_rank=input_rank,
            distributed_axes=plan.distributed_axes,
        ).to(dtype=torch.float32),
    )


def _scatter_flat(values: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
    """Restore flat local amplitudes through a float scatter-add kernel.

    ``indices`` partitions the destination, nevertheless ``scatter_add_`` is
    deliberate: its backward is a real gather and remains valid if a backend
    coalesces index maps.
    """

    output = torch.zeros(size, dtype=torch.float32, device=values.device)
    return output.scatter_add_(0, indices.reshape(-1), values.reshape(-1))


class _PairVectorKernel:
    """Apply one planned gate to paired-real statevector amplitudes."""

    def __init__(self, backend):
        self._backend = backend

    def _apply_source(
        self,
        source: _Pair,
        plan: _GatePlan,
        matrix: _Pair,
        *,
        source_rank: int,
    ) -> _Pair:
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
        if not local_axes:
            return source.mul(_Pair(block.real.reshape(()), block.imag.reshape(())))

        local_n_qubits = int(source.real.shape[0]).bit_length() - 1
        indices = torch.as_tensor(
            _flat_local_state_indices(local_axes, local_n_qubits),
            dtype=torch.long,
            device=source.real.device,
        )
        gathered = _Pair(
            torch.index_select(source.real.reshape(-1), 0, indices.reshape(-1)).reshape(indices.shape),
            torch.index_select(source.imag.reshape(-1), 0, indices.reshape(-1)).reshape(indices.shape),
        )
        updated = block.matmul(gathered)
        return _Pair(
            _scatter_flat(updated.real, indices, source.real.numel()).reshape_as(source.real),
            _scatter_flat(updated.imag, indices, source.imag.numel()).reshape_as(source.imag),
        )

    def apply(
        self,
        state_pair: _Pair,
        plan: _GatePlan,
        *,
        operation_index: int,
    ) -> _Pair:
        """Apply ``plan`` without full-state gathering or complex transport."""

        if not isinstance(state_pair, _Pair):
            raise TypeError("_PairVectorKernel 仅接受 _Pair state_pair")
        operation_index = int(operation_index)
        if operation_index < 0:
            raise ValueError("operation_index 必须非负")
        matrix = _as_pair_matrix(plan.local_matrix, reference=state_pair.real)
        output = self._apply_source(
            state_pair,
            plan,
            matrix,
            source_rank=self._backend.rank,
        )
        for offset, mask in enumerate(plan.partner_masks, start=1):
            peer = plan.partner_for(rank=self._backend.rank, mask=mask)
            incoming = _exchange_pair(
                state_pair,
                communicator=self._backend.communicator,
                peer=peer,
                operation_index=operation_index * self._backend.world_size + offset,
                phase="forward",
            )
            output = output.add(
                self._apply_source(
                    incoming,
                    plan,
                    matrix,
                    source_rank=peer,
                )
            )
        return output

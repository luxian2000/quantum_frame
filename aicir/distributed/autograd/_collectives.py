"""Differentiable paired-real distributed communication primitives.

All process-group payloads in this module are real ``torch.float32`` tensors.
The forward-only distributed simulator deliberately does not import this file.
"""

from __future__ import annotations

import torch

from ._pair import _Pair


_PHASE_IDS = {"forward": 0, "backward": 1}


def _tag(operation_index: int, *, phase: str, direction: int, component: int) -> int:
    """Allocate a deterministic paired-real P2P tag."""

    try:
        phase_id = _PHASE_IDS[phase]
    except KeyError as error:
        raise ValueError("phase 必须是 'forward' 或 'backward'") from error
    operation_index = int(operation_index)
    if operation_index < 0 or direction not in {0, 1} or component not in {0, 1}:
        raise ValueError("operation_index、direction 和 component 必须有效")
    return operation_index * 8 + phase_id * 4 + direction * 2 + component


def _zero_if_none(gradient, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference) if gradient is None else gradient


class _PairExchangeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real, imag, communicator, peer, operation_index, phase):
        ctx.communicator = communicator
        ctx.peer = int(peer)
        ctx.operation_index = int(operation_index)
        return (
            communicator.exchange_real(
                real,
                peer=peer,
                tag=_tag(operation_index, phase=phase, direction=0, component=0),
            ),
            communicator.exchange_real(
                imag,
                peer=peer,
                tag=_tag(operation_index, phase=phase, direction=0, component=1),
            ),
        )

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        grad_real = _zero_if_none(grad_real, grad_imag)
        grad_imag = _zero_if_none(grad_imag, grad_real)
        return (
            ctx.communicator.exchange_real(
                grad_real,
                peer=ctx.peer,
                tag=_tag(ctx.operation_index, phase="backward", direction=0, component=0),
            ),
            ctx.communicator.exchange_real(
                grad_imag,
                peer=ctx.peer,
                tag=_tag(ctx.operation_index, phase="backward", direction=0, component=1),
            ),
            None,
            None,
            None,
            None,
        )


class _ReplicatedAllReduceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, communicator):
        ctx.communicator = communicator
        return communicator.all_reduce_sum_real(tensor) / communicator.world_size

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ctx.communicator.all_reduce_sum_real(grad_output)
            / ctx.communicator.world_size,
            None,
        )


class _RootScatterPairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, root_real, root_imag, communicator, root, local_shape):
        ctx.communicator = communicator
        ctx.root = int(root)
        ctx.local_shape = tuple(local_shape)
        ctx.is_root = communicator.rank == ctx.root
        real_parts = list(root_real.unbind(0)) if ctx.is_root else None
        imag_parts = list(root_imag.unbind(0)) if ctx.is_root else None
        return (
            communicator.scatter_from_root_real(
                real_parts,
                root=ctx.root,
                shape=ctx.local_shape,
            ),
            communicator.scatter_from_root_real(
                imag_parts,
                root=ctx.root,
                shape=ctx.local_shape,
            ),
        )

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        grad_real = _zero_if_none(
            grad_real,
            torch.empty(ctx.local_shape, dtype=torch.float32, device=ctx.communicator.device),
        )
        grad_imag = _zero_if_none(
            grad_imag,
            torch.empty(ctx.local_shape, dtype=torch.float32, device=ctx.communicator.device),
        )
        real_parts = ctx.communicator.gather_to_root_real(grad_real, root=ctx.root)
        imag_parts = ctx.communicator.gather_to_root_real(grad_imag, root=ctx.root)
        if not ctx.is_root:
            return None, None, None, None, None
        return torch.stack(real_parts), torch.stack(imag_parts), None, None, None


def _validate_matching_shape(shape, *, communicator) -> tuple[int, ...]:
    shape = tuple(int(size) for size in shape)
    if len(shape) > 7 or any(size < 0 for size in shape):
        raise ValueError("local_shape 必须是至多七维的非负整数 shape")
    descriptor = torch.zeros(8, dtype=torch.float32, device=communicator.device)
    descriptor[0] = len(shape)
    if shape:
        descriptor[1 : len(shape) + 1] = torch.tensor(
            shape,
            dtype=torch.float32,
            device=communicator.device,
        )
    descriptors = communicator.all_gather_real(descriptor)
    if any(not torch.equal(descriptor, candidate) for candidate in descriptors):
        raise ValueError("所有 rank 的 local_shape 必须一致")
    return shape


def _exchange_pair(pair, *, communicator, peer, operation_index, phase) -> _Pair:
    """Exchange one paired-real value with custom real-valued backward P2P."""

    if not isinstance(pair, _Pair):
        raise TypeError("pair 必须是 _Pair")
    real, imag = _PairExchangeFn.apply(
        pair.real,
        pair.imag,
        communicator,
        int(peer),
        int(operation_index),
        phase,
    )
    return _Pair(real, imag)


def _replicated_all_reduce(tensor, *, communicator) -> torch.Tensor:
    """Return the replicated mean with a world-size-normalized backward."""

    communicator._require_real_float32(tensor)
    return _ReplicatedAllReduceFn.apply(tensor, communicator)


def _scatter_root_pair(pair_or_none, *, communicator, root, local_shape) -> _Pair:
    """Scatter root-owned paired-real shards and gather their gradients to root."""

    root = int(root)
    if not 0 <= root < communicator.world_size:
        raise ValueError(f"root={root} 必须位于 [0, {communicator.world_size})")
    local_shape = _validate_matching_shape(local_shape, communicator=communicator)
    is_root = communicator.rank == root
    valid_input = is_root and isinstance(pair_or_none, _Pair)
    if is_root and valid_input:
        expected_shape = (communicator.world_size,) + local_shape
        valid_input = pair_or_none.real.shape == expected_shape
    valid = torch.tensor(
        [float(valid_input if is_root else pair_or_none is None)],
        dtype=torch.float32,
        device=communicator.device,
    )
    valid_count = communicator.all_reduce_sum_real(valid)
    if int(valid_count.item()) != communicator.world_size:
        raise ValueError("root 必须提供 shape 为 (world_size, *local_shape) 的 _Pair")

    if is_root:
        root_real, root_imag = pair_or_none.real, pair_or_none.imag
    else:
        root_real = torch.zeros(
            local_shape,
            dtype=torch.float32,
            device=communicator.device,
            requires_grad=True,
        )
        root_imag = torch.zeros(
            local_shape,
            dtype=torch.float32,
            device=communicator.device,
            requires_grad=True,
        )
    real, imag = _RootScatterPairFn.apply(
        root_real,
        root_imag,
        communicator,
        root,
        local_shape,
    )
    return _Pair(real, imag)


def _gather_root_pair(pair, *, communicator, root) -> _Pair | None:
    """Gather paired-real local values to root without complex transport."""

    if not isinstance(pair, _Pair):
        raise TypeError("pair 必须是 _Pair")
    root = int(root)
    real_parts = communicator.gather_to_root_real(pair.real, root=root)
    imag_parts = communicator.gather_to_root_real(pair.imag, root=root)
    if communicator.rank != root:
        return None
    return _Pair(torch.stack(real_parts), torch.stack(imag_parts))

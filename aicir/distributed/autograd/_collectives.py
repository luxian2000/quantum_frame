"""Differentiable paired-real distributed communication primitives.

All process-group payloads in this module are real ``torch.float32`` tensors.
The forward-only distributed simulator deliberately does not import this file.
"""

from __future__ import annotations

import torch
import weakref

from ._pair import _Pair


_FORWARD_PHASE = "forward"
_MAX_DESCRIPTOR_DIMENSIONS = 7
_MAX_DESCRIPTOR_INTEGER = 2**24 - 1
_DESCRIPTOR_WIDTH = 16
_DESCRIPTOR_VALUE_START = 2


class _AsyncPairExchange:
    """Paired-real receive whose value is unavailable until both works finish."""

    def __init__(self, pair: _Pair, real_work, imag_work):
        self._pair = pair
        self._real_work = real_work
        self._imag_work = imag_work

    def wait(self) -> _Pair:
        try:
            self._real_work.wait()
        finally:
            # A failed real component must not leave the imag component live.
            self._imag_work.wait()
        return self._pair


class _PairBufferPool:
    """Reusable paired-real buffers keyed by transport identity.

    Pending buffers remain unavailable until both component work handles
    report completion, preventing an in-flight receive from being overwritten.
    """

    def __init__(self):
        self._available = {}
        self._pending = []
        self._checked_out = {}
        self.reuse_count = 0

    @staticmethod
    def _key(shape, dtype, device, peer, phase):
        return (tuple(shape), dtype, torch.device(device), int(peer), str(phase))

    def _reap(self):
        pending = []
        for key, pair, real_work, imag_work in self._pending:
            if bool(real_work.is_completed()) and bool(imag_work.is_completed()):
                self._available.setdefault(key, []).append(pair)
            else:
                pending.append((key, pair, real_work, imag_work))
        self._pending = pending

    def acquire(self, shape, dtype, device, peer, phase) -> _Pair:
        if dtype != torch.float32:
            raise TypeError("paired buffer 必须使用 torch.float32")
        self._reap()
        key = self._key(shape, dtype, device, peer, phase)
        available = self._available.get(key, [])
        if available:
            self.reuse_count += 1
            pair = available.pop()
            self._checked_out[id(pair)] = key
            return pair
        real = torch.empty(tuple(shape), dtype=dtype, device=device)
        pair = _Pair(real, torch.empty_like(real))
        self._checked_out[id(pair)] = key
        return pair

    def release(self, pair: _Pair, *, real_work=None, imag_work=None) -> None:
        key = self._checked_out.pop(id(pair), None)
        if key is None:
            return
        if real_work is not None or imag_work is not None:
            if real_work is None or imag_work is None:
                raise ValueError("paired buffer release 必须同时提供 real/imag work")
            self._pending.append((key, pair, real_work, imag_work))
        else:
            self._available.setdefault(key, []).append(pair)

    def discard(self, pair_id: int) -> None:
        """Forget an output that was dropped without an autograd backward."""

        self._checked_out.pop(int(pair_id), None)


def _pair_buffer_pool_for(communicator) -> _PairBufferPool:
    """Return the communicator-owned pool without altering public routing."""

    pool = getattr(communicator, "_autograd_pair_buffer_pool", None)
    if pool is None:
        pool = _PairBufferPool()
        communicator._autograd_pair_buffer_pool = pool
    return pool


def _is_forward_phase(value) -> bool:
    """Reject non-string phase controls without invoking user equality hooks."""

    return type(value) is str and value == _FORWARD_PHASE


def _safe_int(value):
    """Convert a control value without letting one rank skip preflight."""

    try:
        return int(value)
    except Exception:  # noqa: BLE001 - malformed controls must synchronize
        return None


def _safe_shape(value):
    """Return a compact, float32-exact shape or a deterministic failure."""

    try:
        shape = tuple(value)
    except Exception:  # noqa: BLE001 - malformed controls must synchronize
        return None
    if len(shape) > _MAX_DESCRIPTOR_DIMENSIONS:
        return None
    converted = []
    for size in shape:
        converted_size = _safe_int(size)
        if converted_size is None or not 0 <= converted_size <= _MAX_DESCRIPTOR_INTEGER:
            return None
        converted.append(converted_size)
    return tuple(converted)


def _descriptor(*, valid, code=0, values=(), communicator) -> torch.Tensor:
    """Build a fixed-size float32 control-plane payload."""

    result = torch.zeros(
        _DESCRIPTOR_WIDTH,
        dtype=torch.float32,
        device=communicator.device,
    )
    result[0] = float(bool(valid))
    result[1] = float(code)
    for index, value in enumerate(values, start=_DESCRIPTOR_VALUE_START):
        result[index] = float(value)
    return result


def _shape_fields(length_field: int) -> tuple[int, ...]:
    """Return the length slot plus every fixed-width encoded shape dimension."""

    return tuple(
        range(length_field, length_field + 1 + _MAX_DESCRIPTOR_DIMENSIONS)
    )


def _shape_field_names(length_field: int, name: str) -> dict[int, str]:
    return {field: name for field in _shape_fields(length_field)}


def _raise_preflight_failure(descriptors, *, names) -> None:
    """Raise one deterministic error after every rank has joined preflight."""

    first_invalid = next(
        descriptor for descriptor in descriptors if not bool(descriptor[0].item())
    )
    code = int(first_invalid[1].item())
    name = names.get(code, "参数")
    raise ValueError(f"分布式 autograd collective 参数无效: {name}")


def _synchronize_preflight(
    communicator,
    descriptor,
    *,
    names,
    fields=(),
    field_names=None,
):
    """Synchronize validation before any data-plane collective is entered."""

    descriptors = communicator.all_gather_real(descriptor)
    if any(not bool(candidate[0].item()) for candidate in descriptors):
        _raise_preflight_failure(descriptors, names=names)
    for field in fields:
        values = [candidate[field].item() for candidate in descriptors]
        if any(value != values[0] for value in values[1:]):
            name = (field_names or {}).get(field, "参数")
            raise ValueError(f"分布式 autograd collective 参数不一致: {name}")
    return descriptors


def _pair_shape(pair, *, communicator, expected_shape=None):
    """Validate a possibly externally-mutated paired-real value without raising."""

    if not isinstance(pair, _Pair):
        return None
    real = getattr(pair, "real", None)
    imag = getattr(pair, "imag", None)
    if not isinstance(real, torch.Tensor) or not isinstance(imag, torch.Tensor):
        return None
    if (
        real.dtype != torch.float32
        or imag.dtype != torch.float32
        or torch.is_complex(real)
        or torch.is_complex(imag)
        or real.device != imag.device
        or real.device != communicator.device
        or tuple(real.shape) != tuple(imag.shape)
    ):
        return None
    actual_shape = tuple(real.shape)
    if expected_shape is not None:
        return actual_shape if actual_shape == expected_shape else None
    return _safe_shape(actual_shape)


def _real_tensor_shape(tensor, *, communicator):
    if not isinstance(tensor, torch.Tensor):
        return None
    if (
        tensor.dtype != torch.float32
        or torch.is_complex(tensor)
        or tensor.device != communicator.device
    ):
        return None
    return _safe_shape(tensor.shape)


def _tag(operation_index: int, *, phase: str, direction: int, component: int) -> int:
    """Allocate a deterministic paired-real P2P tag."""

    if phase == _FORWARD_PHASE:
        phase_id = 0
    elif phase == "backward":
        phase_id = 1
    else:
        raise ValueError("phase 必须是 'forward' 或 'backward'")
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
                tag=_tag(operation_index, phase=_FORWARD_PHASE, direction=0, component=0),
            ),
            communicator.exchange_real(
                imag,
                peer=peer,
                tag=_tag(operation_index, phase=_FORWARD_PHASE, direction=0, component=1),
            ),
        )

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        grad_real = _zero_if_none(grad_real, grad_imag)
        grad_imag = _zero_if_none(grad_imag, grad_real)
        # Density right actions transpose paired buffers.  Their real VJPs can
        # therefore be non-contiguous; P2P requires contiguous float32 input.
        grad_real = grad_real.contiguous()
        grad_imag = grad_imag.contiguous()
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


class _LaunchedPairExchangeFn(torch.autograd.Function):
    """Attach a paired-real VJP after a kernel overlaps the receive launch."""

    @staticmethod
    def forward(
        ctx,
        source_real,
        source_imag,
        received_real,
        received_imag,
        communicator,
        peer,
        operation_index,
        pool,
        buffer,
    ):
        ctx.communicator = communicator
        ctx.peer = int(peer)
        ctx.operation_index = int(operation_index)
        ctx.pool = pool
        ctx.buffer = buffer
        return received_real, received_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        # The completed receive is no longer needed once its VJP is known.
        ctx.pool.release(ctx.buffer)
        grad_real = _zero_if_none(grad_real, grad_imag).contiguous()
        grad_imag = _zero_if_none(grad_imag, grad_real).contiguous()
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
            None,
            None,
            None,
        )


class _LaunchedPairExchange:
    """Own an in-flight paired receive until a cross-shard kernel consumes it."""

    def __init__(
        self,
        source: _Pair,
        buffer: _Pair,
        exchange: _AsyncPairExchange,
        *,
        communicator,
        peer,
        operation_index,
        pool: _PairBufferPool,
    ):
        self._source = source
        self._buffer = buffer
        self._exchange = exchange
        self._communicator = communicator
        self._peer = int(peer)
        self._operation_index = int(operation_index)
        self._pool = pool
        self._waited = False

    def wait(self) -> _Pair:
        if self._waited:
            raise RuntimeError("in-flight paired receive 只能 wait 一次")
        self._waited = True
        try:
            received = self._exchange.wait()
        except Exception:
            # Both work groups have been drained by _AsyncPairExchange.wait.
            self._pool.release(self._buffer)
            raise
        if received is not self._buffer:
            self._pool.release(self._buffer)
            raise RuntimeError("paired P2P receive 未写入所申请的缓冲区")
        if not (self._source.real.requires_grad or self._source.imag.requires_grad):
            # Forward-only results may outlive this primitive indefinitely;
            # detach them from the reusable destination before returning it.
            result = _Pair(received.real.clone(), received.imag.clone())
            self._pool.release(received)
            return result
        result_real, result_imag = _LaunchedPairExchangeFn.apply(
            self._source.real,
            self._source.imag,
            received.real,
            received.imag,
            self._communicator,
            self._peer,
            self._operation_index,
            self._pool,
            received,
        )
        # If callers intentionally drop a differentiable result without
        # calling backward, do not leave an unbounded checked-out entry.
        weakref.finalize(result_real, self._pool.discard, id(received))
        return _Pair(result_real, result_imag)


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


def _validate_pair_exchange(pair, *, communicator, peer, operation_index, phase):
    """Run the fixed-shape all-rank preflight for one P2P exchange."""

    parsed_peer = _safe_int(peer)
    parsed_operation = _safe_int(operation_index)
    pair_shape = _pair_shape(pair, communicator=communicator)
    valid = (
        pair_shape is not None
        and parsed_peer is not None
        and 0 <= parsed_peer < communicator.world_size
        and parsed_peer != communicator.rank
        and parsed_operation is not None
        and 0 <= parsed_operation <= _MAX_DESCRIPTOR_INTEGER
        and _is_forward_phase(phase)
    )
    descriptor = _descriptor(
        valid=valid,
        code=(
            1
            if pair_shape is None
            else 2
            if parsed_peer is None
            else 3
            if not 0 <= parsed_peer < communicator.world_size
            else 4
            if parsed_peer == communicator.rank
            else 5
            if parsed_operation is None or not 0 <= parsed_operation <= _MAX_DESCRIPTOR_INTEGER
            else 6
            if not _is_forward_phase(phase)
            else 0
        ),
        values=(
            parsed_peer if parsed_peer is not None else 0,
            parsed_operation if parsed_operation is not None else 0,
            0 if _is_forward_phase(phase) else 1,
            len(pair_shape) if pair_shape is not None else 0,
            *(pair_shape or ()),
        ),
        communicator=communicator,
    )
    descriptors = _synchronize_preflight(
        communicator,
        descriptor,
        names={
            1: "pair",
            2: "peer",
            3: "peer",
            4: "peer",
            5: "operation_index",
            6: "phase",
        },
        fields=(3, 4, *_shape_fields(5)),
        field_names={
            3: "operation_index",
            4: "phase",
            **_shape_field_names(5, "pair shape"),
        },
    )
    for rank, candidate in enumerate(descriptors):
        candidate_peer = int(candidate[2].item())
        if int(descriptors[candidate_peer][2].item()) != rank:
            raise ValueError("分布式 autograd collective 参数不一致: peer")
    return parsed_peer, parsed_operation


def _launch_validated_pair_exchange(
    pair,
    *,
    communicator,
    peer,
    operation_index,
) -> _LaunchedPairExchange:
    """Launch a reusable paired receive after collective preflight succeeds."""

    pool = _pair_buffer_pool_for(communicator)
    buffer = pool.acquire(
        pair.real.shape,
        torch.float32,
        pair.real.device,
        peer,
        _FORWARD_PHASE,
    )
    real_work = imag_work = None
    try:
        received_real, real_work = communicator.exchange_real_async(
            pair.real,
            peer=peer,
            tag=_tag(operation_index, phase=_FORWARD_PHASE, direction=0, component=0),
            receive=buffer.real,
        )
        received_imag, imag_work = communicator.exchange_real_async(
            pair.imag,
            peer=peer,
            tag=_tag(operation_index, phase=_FORWARD_PHASE, direction=0, component=1),
            receive=buffer.imag,
        )
    except Exception:
        if real_work is not None:
            try:
                real_work.wait()
            finally:
                pool.release(buffer)
        else:
            pool.release(buffer)
        raise
    if received_real is not buffer.real or received_imag is not buffer.imag:
        # This is a communicator contract violation; drain before reuse.
        _AsyncPairExchange(buffer, real_work, imag_work).wait()
        pool.release(buffer)
        raise RuntimeError("paired P2P receive 未写入所申请的缓冲区")
    return _LaunchedPairExchange(
        pair,
        buffer,
        _AsyncPairExchange(buffer, real_work, imag_work),
        communicator=communicator,
        peer=peer,
        operation_index=operation_index,
        pool=pool,
    )


def _launch_pair_exchange(pair, *, communicator, peer, operation_index, phase):
    """Start an overlap-mode exchange without exposing its destination yet."""

    parsed_peer, parsed_operation = _validate_pair_exchange(
        pair,
        communicator=communicator,
        peer=peer,
        operation_index=operation_index,
        phase=phase,
    )
    if getattr(communicator, "autograd_communication_mode", "baseline") != "overlap":
        raise RuntimeError("_launch_pair_exchange 仅适用于 overlap 通信模式")
    return _launch_validated_pair_exchange(
        pair,
        communicator=communicator,
        peer=parsed_peer,
        operation_index=parsed_operation,
    )


def _exchange_pair(pair, *, communicator, peer, operation_index, phase) -> _Pair:
    """Exchange one paired-real value with a real-valued backward P2P VJP."""

    parsed_peer, parsed_operation = _validate_pair_exchange(
        pair,
        communicator=communicator,
        peer=peer,
        operation_index=operation_index,
        phase=phase,
    )
    mode = getattr(communicator, "autograd_communication_mode", "baseline")
    if mode == "baseline":
        real, imag = _PairExchangeFn.apply(
            pair.real,
            pair.imag,
            communicator,
            parsed_peer,
            parsed_operation,
            _FORWARD_PHASE,
        )
        return _Pair(real, imag)
    if mode in {"reuse", "overlap"}:
        return _launch_validated_pair_exchange(
            pair,
            communicator=communicator,
            peer=parsed_peer,
            operation_index=parsed_operation,
        ).wait()
    raise ValueError("通信模式必须是 baseline、reuse 或 overlap")


def _replicated_all_reduce(tensor, *, communicator) -> torch.Tensor:
    """Return the replicated mean with a world-size-normalized backward."""

    shape = _real_tensor_shape(tensor, communicator=communicator)
    descriptor = _descriptor(
        valid=shape is not None,
        code=1,
        values=(len(shape) if shape is not None else 0, *(shape or ())),
        communicator=communicator,
    )
    _synchronize_preflight(
        communicator,
        descriptor,
        names={1: "tensor", 2: "tensor shape", 3: "tensor shape", 4: "tensor shape", 5: "tensor shape", 6: "tensor shape", 7: "tensor shape", 8: "tensor shape", 9: "tensor shape"},
        fields=_shape_fields(2),
        field_names=_shape_field_names(2, "tensor shape"),
    )
    return _ReplicatedAllReduceFn.apply(tensor, communicator)


def _scatter_root_pair(pair_or_none, *, communicator, root, local_shape) -> _Pair:
    """Scatter root-owned paired-real shards and gather their gradients to root."""

    parsed_root = _safe_int(root)
    parsed_shape = _safe_shape(local_shape)
    root_is_valid = parsed_root is not None and 0 <= parsed_root < communicator.world_size
    expected_shape = (
        (communicator.world_size,) + parsed_shape
        if root_is_valid and parsed_shape is not None
        else None
    )
    if root_is_valid and communicator.rank == parsed_root:
        input_is_valid = _pair_shape(
            pair_or_none,
            communicator=communicator,
            expected_shape=expected_shape,
        ) is not None
    else:
        input_is_valid = pair_or_none is None
    valid = root_is_valid and parsed_shape is not None and input_is_valid
    descriptor = _descriptor(
        valid=valid,
        code=(
            1
            if parsed_root is None
            else 2
            if not root_is_valid
            else 3
            if parsed_shape is None
            else 4
        ),
        values=(
            parsed_root if parsed_root is not None else 0,
            len(parsed_shape) if parsed_shape is not None else 0,
            *(parsed_shape or ()),
        ),
        communicator=communicator,
    )
    _synchronize_preflight(
        communicator,
        descriptor,
        names={1: "root", 2: "root", 3: "local_shape", 4: "root pair"},
        fields=(2, *_shape_fields(3)),
        field_names={2: "root", **_shape_field_names(3, "local_shape")},
    )
    root = parsed_root
    local_shape = parsed_shape
    is_root = communicator.rank == root

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

    parsed_root = _safe_int(root)
    pair_shape = _pair_shape(pair, communicator=communicator)
    valid = (
        parsed_root is not None
        and 0 <= parsed_root < communicator.world_size
        and pair_shape is not None
    )
    descriptor = _descriptor(
        valid=valid,
        code=(1 if parsed_root is None else 2 if not 0 <= parsed_root < communicator.world_size else 3),
        values=(
            parsed_root if parsed_root is not None else 0,
            len(pair_shape) if pair_shape is not None else 0,
            *(pair_shape or ()),
        ),
        communicator=communicator,
    )
    _synchronize_preflight(
        communicator,
        descriptor,
        names={1: "root", 2: "root", 3: "pair", 4: "pair shape", 5: "pair shape", 6: "pair shape", 7: "pair shape", 8: "pair shape", 9: "pair shape"},
        fields=(2, *_shape_fields(3)),
        field_names={2: "root", **_shape_field_names(3, "pair shape")},
    )
    root = parsed_root
    real_parts = communicator.gather_to_root_real(pair.real, root=root)
    imag_parts = communicator.gather_to_root_real(pair.imag, root=root)
    if communicator.rank != root:
        return None
    return _Pair(torch.stack(real_parts), torch.stack(imag_parts))

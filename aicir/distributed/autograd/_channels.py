"""Paired-real differentiable density-matrix channels.

The public distributed runner remains forward-only until its release gate is
opened.  This private module never creates a complex execution tensor.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral

import torch

from ...ir import instruction_name
from ...noise.channels import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    PhaseFlipChannel,
    differentiable_probability,
)
from ..gates import _AutogradExecutionContext, _GatePlanner
from ..state import DistState
from ._collectives import _descriptor, _synchronize_preflight
from ._pair import _Pair
from ._parameters import StinespringParam


def _probability(value, *, name: str, reference: torch.Tensor) -> torch.Tensor:
    """Return one bounded float32 probability without detaching a leaf."""

    differentiable_probability(value, name)
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or torch.is_complex(value) or value.numel() != 1:
            raise TypeError(f"{name} 必须是标量 torch.float32")
        if value.device != reference.device:
            raise ValueError(f"{name} 与密度矩阵必须位于同一设备")
        return value
    return torch.tensor(float(value), dtype=torch.float32, device=reference.device)


def _safe_probability(value) -> float | None:
    """Parse an untrusted scalar without allowing its exception to escape."""

    try:
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float32 or torch.is_complex(value) or value.numel() != 1:
                return None
            value = float(value.detach().cpu())
        else:
            value = float(value)
    except Exception:  # noqa: BLE001 - collective preflight absorbs hostile metadata
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _channel_metadata(channel, *, n_qubits: int) -> tuple[bool, int, tuple[object, ...]]:
    """Return only primitive metadata; no channel planning happens here."""

    try:
        if isinstance(channel, (BitFlipChannel, PhaseFlipChannel, DepolarizingChannel, AmplitudeDampingChannel)):
            target = channel.target_qubit
            if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= int(target) < n_qubits:
                return False, 3, ()
            field = "gamma" if isinstance(channel, AmplitudeDampingChannel) else "p"
            probability = _safe_probability(getattr(channel, field))
            if probability is None:
                return False, 2, ()
            kind = {
                BitFlipChannel: 1,
                PhaseFlipChannel: 2,
                DepolarizingChannel: 3,
                AmplitudeDampingChannel: 4,
            }[type(channel)]
            terms = 4 if isinstance(channel, DepolarizingChannel) else 2
            return True, 0, (kind, int(target), probability, terms)
        if isinstance(channel, StinespringParam):
            dimension = int(channel.output_dim) * int(channel.environment_dim)
            qubits = int(math.log2(int(channel.input_dim)))
            valid = (
                channel.input_dim == channel.output_dim
                and (1 << qubits) == channel.input_dim
                and channel.environment_dim > 0
                and tuple(channel.real.shape) == (dimension, dimension)
                and tuple(channel.imag.shape) == (dimension, dimension)
                and len(channel.target_qubits) == qubits
                and len(set(channel.target_qubits)) == qubits
                and all(isinstance(axis, Integral) and not isinstance(axis, bool) and 0 <= int(axis) < n_qubits for axis in channel.target_qubits)
            )
            if not valid:
                return False, 4, ()
            return True, 0, (5, channel.input_dim, channel.output_dim, channel.environment_dim, *channel.target_qubits)
    except Exception:  # noqa: BLE001 - no rank may skip the control collective
        return False, 1, ()
    return False, 1, ()


def _metadata_digest(metadata: tuple[object, ...]) -> int:
    encoded = "|".join(str(value) for value in metadata).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:3], byteorder="big")


def _preflight_channel(channel, *, n_qubits: int, communicator) -> bool:
    """Synchronize all channel metadata before density promotion or P2P.

    The payload has fixed float32 width and only contains bounded primitive
    values.  Invalid or divergent rank metadata raises one identical exception
    after every rank joins a barrier, leaving no data-plane P2P record.
    """

    try:
        valid_n_qubits = isinstance(n_qubits, Integral) and not isinstance(n_qubits, bool) and int(n_qubits) > 0
        valid, code, metadata = _channel_metadata(channel, n_qubits=int(n_qubits) if valid_n_qubits else 0)
    except Exception:  # noqa: BLE001 - hostile n_qubits is a collective error
        valid_n_qubits, valid, code, metadata = False, False, 1, ()
    valid = bool(valid_n_qubits and valid)
    if communicator is None:
        return valid
    descriptor = _descriptor(
        valid=valid,
        code=code if not valid else 0,
        values=(int(n_qubits) if valid_n_qubits else 0, _metadata_digest(metadata) if valid else 0),
        communicator=communicator,
    )
    try:
        _synchronize_preflight(
            communicator,
            descriptor,
            names={1: "channel type", 2: "channel probability", 3: "channel target", 4: "Stinespring channel"},
            fields=(2, 3),
            field_names={2: "channel n_qubits", 3: "channel metadata"},
        )
    except ValueError:
        if communicator.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        raise
    return True


def _matrix(real_rows, imag_rows=None, *, reference: torch.Tensor) -> _Pair:
    def convert(rows):
        return torch.stack(tuple(torch.stack(tuple(
            value if isinstance(value, torch.Tensor) else torch.as_tensor(
                value, dtype=torch.float32, device=reference.device
            )
            for value in row
        )) for row in rows)).to(dtype=torch.float32, device=reference.device)

    real = convert(real_rows)
    imag = torch.zeros_like(real) if imag_rows is None else convert(imag_rows)
    return _Pair(real, imag)


def _builtin_kraus(channel, *, reference: torch.Tensor) -> tuple[tuple[_Pair, tuple[int, ...]], ...] | None:
    """Canonical local Kraus matrices with deterministic order."""

    target = getattr(channel, "target_qubit", None)
    if target is None:
        return None
    target = (int(target),)
    zero = torch.zeros((), dtype=torch.float32, device=reference.device)
    one = torch.ones((), dtype=torch.float32, device=reference.device)
    if isinstance(channel, BitFlipChannel):
        p = _probability(channel.p, name="bit flip p", reference=reference)
        root = torch.sqrt(one - p)
        return ((_matrix(((root, zero), (zero, root)), reference=reference), target), (_matrix(((zero, torch.sqrt(p)), (torch.sqrt(p), zero)), reference=reference), target))
    if isinstance(channel, PhaseFlipChannel):
        p = _probability(channel.p, name="phase flip p", reference=reference)
        root = torch.sqrt(one - p)
        return ((_matrix(((root, zero), (zero, root)), reference=reference), target), (_matrix(((torch.sqrt(p), zero), (zero, -torch.sqrt(p))), reference=reference), target))
    if isinstance(channel, DepolarizingChannel):
        p = _probability(channel.p, name="depolarizing p", reference=reference)
        identity, pauli = torch.sqrt(one - p), torch.sqrt(p / 3.0)
        return (
            (_matrix(((identity, zero), (zero, identity)), reference=reference), target),
            (_matrix(((zero, pauli), (pauli, zero)), reference=reference), target),
            (_matrix(((zero, zero), (zero, zero)), ((zero, -pauli), (pauli, zero)), reference=reference), target),
            (_matrix(((pauli, zero), (zero, -pauli)), reference=reference), target),
        )
    if isinstance(channel, AmplitudeDampingChannel):
        gamma = _probability(channel.gamma, name="amplitude damping gamma", reference=reference)
        return ((_matrix(((one, zero), (zero, torch.sqrt(one - gamma))), reference=reference), target), (_matrix(((zero, torch.sqrt(gamma)), (zero, zero)), reference=reference), target))
    return None


def _householder_isometry(parameter: StinespringParam) -> _Pair:
    """Build an isometry with fixed paired-real Householder reflections."""

    dimension = int(parameter.output_dim) * int(parameter.environment_dim)
    if int(parameter.input_dim) > dimension or tuple(parameter.real.shape) != (dimension, dimension):
        raise ValueError("Stinespring 原始参数不满足固定 Householder 形状")
    real, imag = parameter.real.reshape(dimension, dimension), parameter.imag.reshape(dimension, dimension)
    current = _Pair(
        torch.eye(dimension, int(parameter.input_dim), dtype=torch.float32, device=real.device),
        torch.zeros((dimension, int(parameter.input_dim)), dtype=torch.float32, device=real.device),
    )
    for index in range(dimension):
        vr, vi = real[index:index + 1], imag[index:index + 1]
        norm_squared = vr.square().sum() + vi.square().sum()
        safe_denominator = torch.where(norm_squared > 0, norm_squared, torch.ones_like(norm_squared))
        scale = torch.where(norm_squared > 0, 2.0 / safe_denominator, torch.zeros_like(norm_squared))
        inner_real, inner_imag = vr @ current.real + vi @ current.imag, vr @ current.imag - vi @ current.real
        product_real = vr.t() * inner_real - vi.t() * inner_imag
        product_imag = vr.t() * inner_imag + vi.t() * inner_real
        current = _Pair(
            current.real - scale * product_real,
            current.imag - scale * product_imag,
        )
    return current


def _stinespring_kraus(parameter: StinespringParam) -> tuple[_Pair, ...]:
    """Split environment blocks in ascending environment index order."""

    isometry, block = _householder_isometry(parameter), int(parameter.output_dim)
    return tuple(
        _Pair(isometry.real[index * block:(index + 1) * block], isometry.imag[index * block:(index + 1) * block])
        for index in range(int(parameter.environment_dim))
    )


def _stinespring_terms(parameter: StinespringParam, state: DistState) -> tuple[tuple[_Pair, tuple[int, ...]], ...]:
    axes = tuple(parameter.target_qubits)
    if any(axis >= state.n_qubits for axis in axes):
        raise ValueError("Stinespring target_qubits 超出范围")
    return tuple((matrix, axes) for matrix in _stinespring_kraus(parameter))


def _channel_terms(channel, state: DistState) -> tuple[tuple[_Pair, tuple[int, ...]], ...]:
    if isinstance(channel, StinespringParam):
        return _stinespring_terms(channel, state)
    builtin = _builtin_kraus(channel, reference=state._pair.real)
    if builtin is not None:
        return builtin
    raise TypeError("分布式 paired-real autograd 不支持该噪声信道")


def _selected_noise_rules(model, gate) -> tuple[object, ...]:
    """Select rules identically to ``NoiseModel.apply`` without applying them."""

    gate_type = instruction_name(gate) if gate is not None else None
    return tuple(rule.channel for rule in model.rules if model._match_rule(rule, gate_type) and model._should_apply_to_gate(rule, gate))


class _PairChannelKernel:
    """Apply analytic Kraus channels using only paired float32 buffers."""

    def __init__(self, backend):
        self._backend = backend

    def apply_channel(self, state: DistState, channel, *, instruction_index: int) -> DistState:
        from ._density import _PairMatrixKernel

        _preflight_channel(
            channel,
            n_qubits=state.n_qubits,
            communicator=self._backend.communicator,
        )
        state = _PairMatrixKernel(self._backend).promote_vector(state)
        if state.kind != "matrix" or not isinstance(getattr(state, "_pair", None), _Pair):
            raise TypeError("_PairChannelKernel 仅接受 paired-real matrix DistState")
        planner = _GatePlanner(self._backend, state.layout, state.n_qubits, execution_context=_AutogradExecutionContext())
        accumulator = _Pair(torch.zeros_like(state._pair.real), torch.zeros_like(state._pair.imag))
        kernel = _PairMatrixKernel(self._backend)
        for offset, (matrix, axes) in enumerate(_channel_terms(channel, state)):
            plan = planner.plan_matrix(matrix, axes, instruction_index=int(instruction_index) * 256 + offset)
            contribution = kernel.apply_unitary(state, plan, operation_index=plan.instruction_index)
            accumulator = accumulator.add(contribution._pair)
        return DistState.from_pair(accumulator, spec=state.spec, backend=self._backend, bit_order=state.bit_order)


__all__ = ["_PairChannelKernel", "_householder_isometry", "_preflight_channel", "_selected_noise_rules", "_stinespring_kraus"]

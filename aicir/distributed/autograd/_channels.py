"""Paired-real differentiable density-matrix channels.

The public distributed runner remains forward-only until its release gate is
opened.  This private module never creates a complex execution tensor.
"""

from __future__ import annotations

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
    if int(parameter.input_dim) > dimension:
        raise ValueError("Stinespring 输出与环境维度之积必须不小于输入维度")
    if parameter.real.numel() != dimension * dimension:
        raise ValueError("Stinespring 原始参数必须包含固定数量的 Householder 向量")
    real, imag = parameter.real.reshape(dimension, dimension), parameter.imag.reshape(dimension, dimension)
    current = _Pair(
        torch.eye(dimension, int(parameter.input_dim), dtype=torch.float32, device=real.device),
        torch.zeros((dimension, int(parameter.input_dim)), dtype=torch.float32, device=real.device),
    )
    epsilon = torch.tensor(1e-12, dtype=torch.float32, device=real.device)
    for index in range(dimension):
        vr, vi = real[index:index + 1], imag[index:index + 1]
        denominator = vr.square().sum() + vi.square().sum() + epsilon
        inner_real, inner_imag = vr @ current.real + vi @ current.imag, vr @ current.imag - vi @ current.real
        product_real = vr.t() * inner_real - vi.t() * inner_imag
        product_imag = vr.t() * inner_imag + vi.t() * inner_real
        current = _Pair(
            current.real - 2.0 * product_real / denominator,
            current.imag - 2.0 * product_imag / denominator,
        )
    return current


def _stinespring_kraus(parameter: StinespringParam) -> tuple[_Pair, ...]:
    """Split environment blocks in ascending environment index order."""

    if parameter.input_dim != parameter.output_dim:
        raise ValueError("当前分布式 Stinespring 信道要求 input_dim == output_dim")
    isometry, block = _householder_isometry(parameter), int(parameter.output_dim)
    return tuple(
        _Pair(isometry.real[index * block:(index + 1) * block], isometry.imag[index * block:(index + 1) * block])
        for index in range(int(parameter.environment_dim))
    )


def _stinespring_terms(parameter: StinespringParam, state: DistState) -> tuple[tuple[_Pair, tuple[int, ...]], ...]:
    if parameter.input_dim == 2:
        axes = (int(getattr(parameter, "target_qubit", 0)),)
    elif parameter.input_dim == (1 << state.n_qubits):
        axes = tuple(range(state.n_qubits))
    else:
        raise ValueError("Stinespring 输入维度必须为 2 或完整分布式希尔伯特维度")
    if any(axis < 0 or axis >= state.n_qubits for axis in axes):
        raise ValueError("Stinespring target_qubit 超出范围")
    return tuple((matrix, axes) for matrix in _stinespring_kraus(parameter))


def _channel_terms(channel, state: DistState) -> tuple[tuple[_Pair, tuple[int, ...]], ...]:
    if isinstance(channel, StinespringParam):
        return _stinespring_terms(channel, state)
    builtin = _builtin_kraus(channel, reference=state._pair.real)
    if builtin is not None:
        return builtin
    from ._vector import _as_pair_matrix
    return tuple(
        (_as_pair_matrix(matrix, reference=state._pair.real), tuple(int(axis) for axis in axes))
        for matrix, axes in channel._local_kraus(state.n_qubits, state.backend)
    )


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


__all__ = ["_PairChannelKernel", "_householder_isometry", "_selected_noise_rules", "_stinespring_kraus"]

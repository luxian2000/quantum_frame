"""Physical parameter containers backed only by real-valued tensor leaves."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import math

import torch

from ._pair import _Pair


class _ReplicatedParameterFn(torch.autograd.Function):
    """Identity forward with one real global parameter-adjoint reduction."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, communicator):
        ctx.communicator = communicator
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return ctx.communicator.all_reduce_sum_real(gradient), None


def replicated_parameter(value, *, communicator):
    """Mark a real trainable gate leaf as replicated across all ranks.

    This wrapper belongs at the gate parameter leaf, before the parameter is
    expanded into multiple matrix entries.  Reducing individual matrix parts
    would count a shared angle once per real/imaginary entry.
    """

    if not isinstance(value, torch.Tensor) or not value.requires_grad:
        return value
    if value.dtype != torch.float32 or torch.is_complex(value):
        raise TypeError("分布式 statevector 参数必须是实数 torch.float32")
    if communicator.world_size == 1:
        return value
    return _ReplicatedParameterFn.apply(value, communicator)


@dataclass(frozen=True)
class PureStateParam:
    """Unconstrained paired-real amplitudes for a normalized pure state."""

    real: torch.Tensor
    imag: torch.Tensor

    def _raw_pair(self) -> _Pair:
        return _Pair(self.real, self.imag)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

    def normalized_pair(self) -> _Pair:
        """Return amplitudes normalized by one global real Euclidean norm."""

        pair = self._raw_pair()
        norm = torch.sqrt(pair.abs_sq().sum())
        if float(norm.detach().cpu()) == 0.0:
            raise ValueError("纯态参数的范数必须大于 0")
        return pair.div_real(norm)


@dataclass(frozen=True)
class DensityParam:
    """Paired-real factor whose normalized ``L L^H`` is a density matrix."""

    real: torch.Tensor
    imag: torch.Tensor

    def _raw_pair(self) -> _Pair:
        return _Pair(self.real, self.imag)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

    def density_pair(self) -> _Pair:
        """Build a positive semidefinite, trace-one paired-real density matrix."""

        density = self._raw_pair().matmul(self._raw_pair().dagger())
        trace = torch.diagonal(density.real, dim1=-2, dim2=-1).sum()
        if float(trace.detach().cpu()) == 0.0:
            raise ValueError("密度矩阵因子的迹必须大于 0")
        return density.div_real(trace)


@dataclass(frozen=True)
class StinespringParam:
    """Raw paired-real square-channel Stinespring parameters.

    ``target_qubits`` defaults to the first ``log2(input_dim)`` logical
    qubits.  It may instead name any unique non-negative logical targets; the
    active distributed state validates the upper bound before planning.
    """

    input_dim: int
    output_dim: int
    environment_dim: int
    real: torch.Tensor
    imag: torch.Tensor
    target_qubits: tuple[int, ...] | None = None

    def __post_init__(self):
        dimensions = (self.input_dim, self.output_dim, self.environment_dim)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in dimensions):
            raise TypeError("Stinespring 维度必须是正整数")
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Stinespring 维度必须为正整数")
        if int(self.input_dim) != int(self.output_dim):
            raise ValueError("Stinespring 要求 input_dim == output_dim")
        qubits = int(math.log2(int(self.input_dim)))
        if (1 << qubits) != int(self.input_dim):
            raise ValueError("Stinespring input_dim 必须是 2 的幂")
        _Pair(self.real, self.imag)
        dimension = int(self.output_dim) * int(self.environment_dim)
        if tuple(self.real.shape) != (dimension, dimension):
            raise ValueError("Stinespring 原始参数 shape 必须为 (output_dim * environment_dim, output_dim * environment_dim)")
        targets = tuple(range(qubits)) if self.target_qubits is None else tuple(self.target_qubits)
        if len(targets) != qubits:
            raise ValueError("Stinespring target_qubits 数量必须等于 log2(input_dim)")
        if any(isinstance(target, bool) or not isinstance(target, Integral) or int(target) < 0 for target in targets):
            raise ValueError("Stinespring target_qubits 必须是互异非负整数")
        targets = tuple(int(target) for target in targets)
        if len(set(targets)) != len(targets):
            raise ValueError("Stinespring target_qubits 必须互异")
        object.__setattr__(self, "input_dim", int(self.input_dim))
        object.__setattr__(self, "output_dim", int(self.output_dim))
        object.__setattr__(self, "environment_dim", int(self.environment_dim))
        object.__setattr__(self, "target_qubits", targets)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

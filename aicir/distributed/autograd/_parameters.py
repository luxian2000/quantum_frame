"""Physical parameter containers backed only by real-valued tensor leaves."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._pair import _Pair


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
    """Raw paired-real Stinespring parameters; isometry construction is deferred."""

    input_dim: int
    output_dim: int
    environment_dim: int
    real: torch.Tensor
    imag: torch.Tensor

    def __post_init__(self):
        if self.input_dim <= 0 or self.output_dim <= 0 or self.environment_dim <= 0:
            raise ValueError("Stinespring 维度必须为正整数")
        _Pair(self.real, self.imag)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

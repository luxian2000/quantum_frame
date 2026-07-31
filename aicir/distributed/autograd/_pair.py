"""A complex value represented by two real ``float32`` tensors.

The distributed autograd engine only consumes this representation.  ``combine``
is an explicit boundary helper for CPU references and diagnostics; its complex
result is never an engine input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class _Combine(torch.autograd.Function):
    """Create a complex boundary value while preserving real leaf gradients."""

    @staticmethod
    def forward(ctx, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.real, gradient.imag


@dataclass(frozen=True)
class _Pair:
    """Paired-real complex arithmetic implemented entirely with real kernels."""

    real: torch.Tensor
    imag: torch.Tensor

    def __post_init__(self):
        if self.real.dtype != torch.float32:
            raise TypeError("_Pair.real 必须是 torch.float32")
        if self.imag.dtype != torch.float32:
            raise TypeError("_Pair.imag 必须是 torch.float32")
        if self.real.shape != self.imag.shape:
            raise ValueError("_Pair 的 real/imag shape 必须一致")
        if self.real.device != self.imag.device:
            raise ValueError("_Pair 的 real/imag device 必须一致")

    def add(self, other: "_Pair") -> "_Pair":
        return _Pair(self.real + other.real, self.imag + other.imag)

    def mul(self, other: "_Pair") -> "_Pair":
        return _Pair(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def div_real(self, denominator) -> "_Pair":
        return _Pair(self.real / denominator, self.imag / denominator)

    def matmul(self, other: "_Pair") -> "_Pair":
        return _Pair(
            self.real @ other.real - self.imag @ other.imag,
            self.real @ other.imag + self.imag @ other.real,
        )

    def dagger(self) -> "_Pair":
        return _Pair(self.real.t(), -self.imag.t())

    def abs_sq(self) -> torch.Tensor:
        return self.real.square() + self.imag.square()

    def index_select(self, axis: int, index: torch.Tensor) -> "_Pair":
        return _Pair(
            torch.index_select(self.real, axis, index),
            torch.index_select(self.imag, axis, index),
        )

    def combine(self) -> torch.Tensor:
        """Return a complex boundary tensor; do not feed it to the engine."""

        if self.real.device.type == "meta":
            raise RuntimeError("combine() 仅支持 CPU 诊断/参考边界")
        return _Combine.apply(self.real, self.imag)

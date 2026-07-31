"""Paired-real arithmetic agrees with a complex128 CPU reference."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.distributed.autograd._pair import _Pair


def _pair(real, imag):
    return _Pair(
        torch.tensor(real, dtype=torch.float32, requires_grad=True),
        torch.tensor(imag, dtype=torch.float32, requires_grad=True),
    )


def _complex128(pair):
    return pair.real.detach().numpy().astype(np.float64) + 1j * pair.imag.detach().numpy().astype(np.float64)


def test_pair_arithmetic_matches_cpu_complex128_and_preserves_real_gradients():
    left = _pair([[1.5, -2.0], [0.25, 3.0]], [[-0.5, 1.0], [2.0, -1.25]])
    right = _pair([[0.75, 1.25], [-1.0, 0.5]], [[1.0, -0.25], [0.5, 2.0]])
    denominator = torch.tensor(2.5, dtype=torch.float32)

    np.testing.assert_allclose(_complex128(left.add(right)), _complex128(left) + _complex128(right), atol=1e-6)
    np.testing.assert_allclose(_complex128(left.mul(right)), _complex128(left) * _complex128(right), atol=1e-6)
    np.testing.assert_allclose(_complex128(left.matmul(right)), _complex128(left) @ _complex128(right), atol=1e-6)
    np.testing.assert_allclose(_complex128(left.dagger()), _complex128(left).conj().T, atol=1e-6)
    np.testing.assert_allclose(_complex128(left.div_real(denominator)), _complex128(left) / 2.5, atol=1e-6)
    np.testing.assert_allclose(left.abs_sq().detach().numpy(), np.abs(_complex128(left)) ** 2, atol=1e-6)

    index = torch.tensor([1, 0], dtype=torch.long)
    np.testing.assert_allclose(
        _complex128(left.index_select(0, index)),
        np.take(_complex128(left), [1, 0], axis=0),
        atol=1e-6,
    )

    loss = left.abs_sq().sum()
    loss.backward()
    np.testing.assert_allclose(left.real.grad.numpy(), 2.0 * left.real.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(left.imag.grad.numpy(), 2.0 * left.imag.detach().numpy(), atol=1e-6)


def test_pair_combine_uses_a_complex_view_only_at_the_explicit_boundary():
    pair = _pair([1.5, -2.0], [0.25, 3.0])

    combined = pair.combine()
    loss = combined.real.sum() + combined.imag.sum()
    loss.backward()

    np.testing.assert_allclose(combined.detach().numpy(), _complex128(pair), atol=1e-6)
    np.testing.assert_allclose(pair.real.grad.numpy(), np.ones(2), atol=1e-6)
    np.testing.assert_allclose(pair.imag.grad.numpy(), np.ones(2), atol=1e-6)


def test_pair_combine_rejects_non_cpu_boundary_tensor():
    pair = _Pair(
        torch.ones(2, dtype=torch.float32, device="meta"),
        torch.zeros(2, dtype=torch.float32, device="meta"),
    )

    with pytest.raises(RuntimeError, match=r"combine\(\) 仅支持 CPU 诊断/参考边界"):
        pair.combine()


def test_pair_rejects_non_float32_or_mismatched_components():
    with pytest.raises(TypeError, match="_Pair.real 必须是 torch.float32"):
        _Pair(torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="_Pair 的 real/imag shape 必须一致"):
        _Pair(torch.ones(2, dtype=torch.float32), torch.ones(3, dtype=torch.float32))

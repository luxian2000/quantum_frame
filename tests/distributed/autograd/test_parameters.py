"""Physical parameter containers preserve their stated constraints."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.distributed import DensityParam, PureStateParam, StinespringParam


def _leaf(values):
    return torch.tensor(values, dtype=torch.float32, requires_grad=True)


def test_pure_state_param_normalizes_one_global_paired_real_norm():
    real = _leaf([3.0, 0.0])
    imag = _leaf([0.0, 4.0])
    parameter = PureStateParam(real, imag)

    normalized = parameter.normalized_pair()

    assert parameter.real is real
    assert parameter.imag is imag
    assert parameter.parameters() == (real, imag)
    assert float(normalized.abs_sq().sum()) == pytest.approx(1.0, abs=1e-6)


def test_pure_state_param_rejects_zero_norm_with_exact_error():
    parameter = PureStateParam(_leaf([0.0, 0.0]), _leaf([0.0, 0.0]))

    with pytest.raises(ValueError, match=r"^纯态参数的范数必须大于 0$"):
        parameter.normalized_pair()


def test_density_param_builds_hermitian_positive_trace_one_density_matrix():
    parameter = DensityParam(
        _leaf([[1.0, 0.5], [-0.25, 2.0]]),
        _leaf([[0.0, 1.0], [0.75, -0.5]]),
    )

    density = parameter.density_pair().combine().detach().numpy()

    np.testing.assert_allclose(density, density.conj().T, atol=1e-6)
    assert np.linalg.eigvalsh(density).min() >= -1e-6
    assert np.trace(density) == pytest.approx(1.0, abs=1e-6)


def test_density_param_rejects_zero_trace_factor_with_exact_error():
    parameter = DensityParam(_leaf([[0.0, 0.0]]), _leaf([[0.0, 0.0]]))

    with pytest.raises(ValueError, match=r"^密度矩阵因子的迹必须大于 0$"):
        parameter.density_pair()


def test_stinespring_param_keeps_dimensions_and_only_real_leaves():
    real = _leaf([[1.0, 0.0], [0.0, 1.0]])
    imag = _leaf([[0.0, 0.0], [0.0, 0.0]])

    parameter = StinespringParam(2, 2, 1, real, imag)

    assert (parameter.input_dim, parameter.output_dim, parameter.environment_dim) == (2, 2, 1)
    assert parameter.real is real
    assert parameter.imag is imag
    assert parameter.parameters() == (real, imag)

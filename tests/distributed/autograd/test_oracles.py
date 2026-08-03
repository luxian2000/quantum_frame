import numpy as np

from aicir.distributed import (
    finite_difference_gradient,
    parameter_shift_gradient,
    parameter_shift_jacobian,
)


def test_parameter_shift_gradient_matches_sine():
    theta = np.array([0.2, -0.4], dtype=np.float64)

    actual = parameter_shift_gradient(
        lambda values: np.sin(values).sum(),
        theta,
    )

    np.testing.assert_allclose(actual, np.cos(theta), atol=1e-12)


def test_parameter_shift_jacobian_preserves_output_and_parameter_axes():
    theta = np.array([0.2, -0.4], dtype=np.float64)

    actual = parameter_shift_jacobian(
        lambda values: np.stack(
            [np.sin(values[0]), np.cos(values[1])]
        ),
        theta,
    )

    expected = np.array(
        [[np.cos(theta[0]), 0.0], [0.0, -np.sin(theta[1])]]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_finite_difference_handles_non_shift_rule_parameters():
    theta = np.array([0.3, -0.1], dtype=np.float64)

    actual = finite_difference_gradient(
        lambda values: np.sum(values**3),
        theta,
    )

    np.testing.assert_allclose(actual, 3.0 * theta**2, atol=1e-6)

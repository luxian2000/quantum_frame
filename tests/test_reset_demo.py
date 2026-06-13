import numpy as np

from demos.reset_demo import run_demo


def test_reset_demo_verifies_reset_with_snap():
    report = run_demo(verbose=False)

    assert report["circuit"].n_qubits == 3
    np.testing.assert_allclose(
        report["after_hadamard"],
        _state([0, 4], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        report["after_cnot_10"],
        _state([0, 6], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        report["after_cnot_21"],
        _state([0, 7], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(report["after_measure"], report["after_cnot_21"], atol=1e-6)
    np.testing.assert_allclose(
        report["after_reset"],
        _state([0, 5], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        report["after_cnot_12"],
        _state([0, 7], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        atol=1e-6,
    )
    assert report["reset_verified"] is True


def _state(indices, values):
    state = np.zeros(8, dtype=np.complex64)
    for index, value in zip(indices, values):
        state[index] = value
    return state

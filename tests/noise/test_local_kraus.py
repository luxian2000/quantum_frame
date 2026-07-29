import numpy as np

from aicir import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    KrausChannel,
    TwoQubitDepolarizingChannel,
)
from aicir.backends import NumpyBackend


def test_single_qubit_local_kraus_does_not_embed_full_system():
    backend = NumpyBackend()
    channel = AmplitudeDampingChannel(target_qubit=2, gamma=0.2)

    local = channel._local_kraus(4, backend)

    assert len(local) == 2
    assert all(targets == (2,) for _, targets in local)
    assert all(tuple(matrix.shape) == (2, 2) for matrix, _ in local)


def test_local_kraus_preserves_existing_single_qubit_values():
    backend = NumpyBackend()
    channel = DepolarizingChannel(target_qubit=0, p=0.3)

    local = [matrix for matrix, _ in channel._local_kraus(1, backend)]
    embedded = channel.kraus_operators(1, backend)

    for actual, expected in zip(local, embedded):
        np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_two_qubit_channel_returns_four_by_four_local_operators():
    backend = NumpyBackend()
    channel = TwoQubitDepolarizingChannel(1, 3, p=0.1)

    local = channel._local_kraus(4, backend)

    assert len(local) == 16
    assert all(targets == (1, 3) for _, targets in local)
    assert all(tuple(matrix.shape) == (4, 4) for matrix, _ in local)


def test_targeted_custom_kraus_returns_local_operators():
    backend = NumpyBackend()
    channel = KrausChannel(
        [np.eye(2, dtype=np.complex64)],
        target_qubits=(2,),
    )

    local = channel._local_kraus(3, backend)

    assert local[0][1] == (2,)
    np.testing.assert_array_equal(local[0][0], np.eye(2, dtype=np.complex64))


def test_full_system_custom_kraus_is_not_treated_as_local():
    backend = NumpyBackend()
    channel = KrausChannel([np.eye(4, dtype=np.complex64)])

    try:
        channel._local_kraus(2, backend)
    except NotImplementedError as exc:
        assert "target_qubits" in str(exc)
    else:
        raise AssertionError("full-system Kraus operators must be rejected")


import unittest
from unittest import mock

import numpy as np

from aicir import NumpyBackend, State, StateVector
try:
    from aicir import TorchBackend
except ImportError:
    TorchBackend = None


class TestState(unittest.TestCase):
    def test_state_string_defaults_to_msb(self):
        backend = NumpyBackend()
        amplitude = 1 / np.sqrt(2)
        state = State.from_array([0, amplitude, amplitude, 0], n_qubits=2, backend=backend)

        self.assertEqual(str(state), "1/\\sqrt{2}|01>+1/\\sqrt{2}|10>")
        self.assertEqual(state.format(bit_order="lsb"), "1/\\sqrt{2}|10>+1/\\sqrt{2}|01>")

    def test_endianness_round_trip(self):
        backend = NumpyBackend()
        state = State.from_array([0, 1, 0, 0], n_qubits=2, backend=backend)

        self.assertEqual(str(state), "1|01>")

        big_endian_state = state.msb()
        self.assertEqual(big_endian_state.bit_order, "msb")
        self.assertEqual(str(big_endian_state), "1|01>")
        np.testing.assert_allclose(
            big_endian_state.to_numpy(),
            np.array([0, 1, 0, 0], dtype=np.complex64),
        )

        little_endian_state = big_endian_state.lsb()
        self.assertEqual(little_endian_state.bit_order, "lsb")
        np.testing.assert_allclose(
            little_endian_state.to_numpy(),
            np.array([0, 0, 1, 0], dtype=np.complex64),
        )

        restored = little_endian_state.msb()
        self.assertEqual(restored.bit_order, "msb")
        np.testing.assert_allclose(restored.to_numpy(), state.to_numpy())

    def test_statevector_remains_alias(self):
        self.assertIs(StateVector, State)

    def test_backend_native_tensor_construction_does_not_round_trip_through_numpy(self):
        if TorchBackend is None:
            self.skipTest("TorchBackend test requires torch")
        backend = TorchBackend(device="cpu")

        with mock.patch.object(
            backend,
            "to_numpy",
            side_effect=AssertionError("State should not copy backend-native tensors through NumPy"),
        ):
            state = StateVector.zero_state(2, backend)

        self.assertEqual(tuple(state.data.shape), (4, 1))

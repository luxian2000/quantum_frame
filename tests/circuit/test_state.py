import unittest

import numpy as np

from nexq import NumpyBackend, State, StateVector


class TestState(unittest.TestCase):
    def test_state_string_defaults_to_lsb(self):
        backend = NumpyBackend()
        amplitude = 1 / np.sqrt(2)
        state = State.from_array([0, amplitude, amplitude, 0], n_qubits=2, backend=backend)

        self.assertEqual(str(state), "1/\\sqrt{2}|10>+1/\\sqrt{2}|01>")
        self.assertEqual(state.format(bit_order="msb"), "1/\\sqrt{2}|01>+1/\\sqrt{2}|10>")

    def test_endianness_round_trip(self):
        backend = NumpyBackend()
        state = State.from_array([0, 1, 0, 0], n_qubits=2, backend=backend)

        self.assertEqual(str(state), "1|10>")

        big_endian_state = state.to_big_endian()
        self.assertEqual(big_endian_state.bit_order, "msb")
        self.assertEqual(str(big_endian_state), "1|10>")
        np.testing.assert_allclose(
            big_endian_state.to_numpy(),
            np.array([0, 0, 1, 0], dtype=np.complex64),
        )

        restored = big_endian_state.to_little_endian()
        self.assertEqual(restored.bit_order, "lsb")
        np.testing.assert_allclose(restored.to_numpy(), state.to_numpy())

    def test_statevector_remains_alias(self):
        self.assertIs(StateVector, State)
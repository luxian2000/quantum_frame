import unittest
from unittest import mock

import numpy as np

from aicir import NumpyBackend, State, TorchBackend


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

    def test_legacy_names_removed(self):
        with self.assertRaises(ImportError):
            from aicir import StateVector  # noqa: F401
        with self.assertRaises(ImportError):
            from aicir import DensityMatrix  # noqa: F401

    def test_backend_native_tensor_construction_does_not_round_trip_through_numpy(self):
        backend = TorchBackend(device="cpu")

        with mock.patch.object(
            backend,
            "to_numpy",
            side_effect=AssertionError("State should not copy backend-native tensors through NumPy"),
        ):
            state = State.zero_state(2, backend)

        self.assertEqual(tuple(state.data.shape), (4, 1))


def test_reorder_endianness_vectorized_scales():
    # 向量化后 n=20 端序重排应亚秒完成（旧实现 2^n 次 Python 循环，数秒）
    import time
    import numpy as np
    from aicir.core.state import State

    n = 22
    rng = np.random.default_rng(1)
    vec = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    state = State.from_array(vec, n_qubits=n)
    t0 = time.perf_counter()
    out = state.lsb()
    elapsed = time.perf_counter() - t0
    assert out.bit_order == "lsb"
    assert elapsed < 0.5  # 旧 2^n Python 循环在本机约 1.3s


def test_reorder_endianness_matches_bit_reversal():
    # 与逐 index 位反转定义一致（小 n 手写对照）
    import numpy as np
    from aicir.core.state import State

    n = 3
    vec = np.arange(1, (1 << n) + 1, dtype=complex)
    vec = vec / np.linalg.norm(vec)
    out = State.from_array(vec, n_qubits=n).lsb().to_numpy()
    expected = np.empty_like(vec)
    for i in range(1 << n):
        rev = int(f"{i:0{n}b}"[::-1], 2)
        expected[rev] = vec[i]
    np.testing.assert_allclose(out, expected, atol=1e-6)

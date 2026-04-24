import unittest
from unittest import mock

import numpy as np
import torch

from nexq import NPUBackend, StateVector, npu_runtime_context_from_env
from nexq.channel.backends.npu_backend import is_npu_available


class TestNPUBackend(unittest.TestCase):
    def test_runtime_context_defaults(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            ctx = npu_runtime_context_from_env()
        self.assertEqual(ctx.world_size, 1)
        self.assertEqual(ctx.rank, 0)
        self.assertEqual(ctx.local_rank, 0)
        self.assertFalse(ctx.distributed)

    def test_runtime_context_from_env(self):
        with mock.patch.dict(
            "os.environ",
            {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "1"},
            clear=False,
        ):
            ctx = npu_runtime_context_from_env()
        self.assertEqual(ctx.world_size, 4)
        self.assertEqual(ctx.rank, 2)
        self.assertEqual(ctx.local_rank, 1)
        self.assertTrue(ctx.distributed)

    def test_runtime_context_invalid_env(self):
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "abc"}, clear=False):
            with self.assertRaises(ValueError):
                npu_runtime_context_from_env()

    def test_fallback_or_npu_device_selection(self):
        backend = NPUBackend(fallback_to_cpu=True)
        state = backend.zeros_state(1)

        # Verify backend tensor creation works regardless of runtime availability.
        self.assertEqual(tuple(state.shape), (2, 1))
        probs = backend.to_numpy(backend.measure_probs(state))
        self.assertTrue(np.allclose(probs, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6))

        if is_npu_available():
            self.assertIn("device=npu", backend.name.lower())
        else:
            self.assertIn("device=cpu", backend.name.lower())

    def test_raise_when_request_npu_without_fallback(self):
        if is_npu_available():
            backend = NPUBackend(device="npu:0", fallback_to_cpu=False)
            self.assertIn("npu", backend.name.lower())
            return

        with self.assertRaises(RuntimeError):
            NPUBackend(device="npu:0", fallback_to_cpu=False)

    def test_statevector_pipeline(self):
        backend = NPUBackend(fallback_to_cpu=True)
        sv = StateVector.zero_state(2, backend)

        probs = backend.to_numpy(sv.probabilities())
        self.assertAlmostEqual(float(probs[0]), 1.0, places=6)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_from_distributed_env(self):
        with mock.patch.dict(
            "os.environ",
            {"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "2"},
            clear=False,
        ):
            backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

        self.assertIsNotNone(backend.runtime_context)
        self.assertEqual(backend.runtime_context.world_size, 8)
        self.assertEqual(backend.runtime_context.rank, 3)
        self.assertEqual(backend.runtime_context.local_rank, 2)
        self.assertTrue(backend.runtime_context.distributed)

    def test_complex_matmul_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        a = torch.tensor(
            [[1 + 2j, 3 - 1j], [0.5 + 0.25j, -2j]],
            dtype=torch.complex64,
        )
        b = torch.tensor(
            [[2 - 0.5j], [1 + 4j]],
            dtype=torch.complex64,
        )

        actual = backend._complex_matmul_workaround(a, b)
        expected = torch.matmul(a, b)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_npu_complex_matmul_uses_workaround(self):
        backend = NPUBackend(fallback_to_cpu=True)
        backend._device = torch.device("cpu")
        a = torch.tensor([[1 + 1j]], dtype=torch.complex64)
        b = torch.tensor([[1 - 1j]], dtype=torch.complex64)

        with mock.patch.object(
            NPUBackend,
            "_complex_matmul_workaround",
            return_value=torch.tensor([[2 + 0j]], dtype=torch.complex64),
        ) as workaround:
            self.assertTrue(torch.allclose(backend.matmul(a, b), torch.matmul(a, b)))
            workaround.assert_not_called()

        backend._device = torch.device("meta")
        backend._device = type("FakeDevice", (), {"type": "npu"})()
        with mock.patch.object(
            NPUBackend,
            "_complex_matmul_workaround",
            return_value=torch.tensor([[2 + 0j]], dtype=torch.complex64),
        ) as workaround:
            result = backend.matmul(a, b)
            workaround.assert_called_once_with(a, b)
            self.assertTrue(torch.equal(result, torch.tensor([[2 + 0j]], dtype=torch.complex64)))


if __name__ == "__main__":
    unittest.main()

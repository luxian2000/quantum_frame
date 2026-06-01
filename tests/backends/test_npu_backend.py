import unittest
from unittest import mock

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("NPU backend tests require torch") from exc

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
            {"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "1"},
            clear=True,
        ):
            with mock.patch.object(NPUBackend, "_resolve_device", return_value=torch.device("cpu")):
                backend = NPUBackend.from_distributed_env(fallback_to_cpu=True, init_process_group=False)

        self.assertIsNotNone(backend.runtime_context)
        self.assertEqual(backend.runtime_context.world_size, 8)
        self.assertEqual(backend.runtime_context.rank, 3)
        self.assertEqual(backend.runtime_context.local_rank, 1)
        self.assertTrue(backend.runtime_context.distributed)
        self.assertFalse(backend.runtime_context.process_group_initialized)

    def test_from_distributed_env_initializes_process_group_when_rendezvous_env_exists(self):
        env = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            with mock.patch.object(NPUBackend, "_resolve_device", return_value=torch.device("cpu")):
                with mock.patch("torch.distributed.is_available", return_value=True):
                    with mock.patch("torch.distributed.is_initialized", side_effect=[False, True]):
                        with mock.patch("torch.distributed.init_process_group") as init_pg:
                            backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

        init_pg.assert_called_once_with(backend="gloo", rank=1, world_size=2)
        self.assertTrue(backend.runtime_context.process_group_initialized)
        self.assertEqual(backend.runtime_context.process_group_backend, "gloo")

    def test_distributed_batch_helpers_partition_and_gather(self):
        backend = NPUBackend(fallback_to_cpu=True)
        backend._runtime_context = backend.runtime_context or npu_runtime_context_from_env()
        backend._runtime_context = type(backend._runtime_context)(
            world_size=2,
            rank=1,
            local_rank=1,
            distributed=True,
            process_group_initialized=True,
            process_group_backend="gloo",
        )

        with mock.patch("torch.distributed.is_available", return_value=True):
            with mock.patch("torch.distributed.is_initialized", return_value=True):
                self.assertFalse(backend.should_run_batch_index(0))
                self.assertTrue(backend.should_run_batch_index(1))

                def fake_gather(output, local):
                    output[0] = [(0, "rank0")]
                    output[1] = local

                with mock.patch("torch.distributed.all_gather_object", side_effect=fake_gather):
                    gathered = backend.gather_indexed_results([(1, "rank1")])

        self.assertEqual(gathered, [(0, "rank0"), (1, "rank1")])

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

    def test_abs_sq_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        t = torch.tensor([1 + 2j, -3 + 0j, 0 - 4j], dtype=torch.complex64)
        expected = torch.abs(t) ** 2
        # Force device label to npu to exercise workaround
        backend._device = type("FakeDevice", (), {"type": "npu"})()
        result = backend.abs_sq(t)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_measure_probs_workaround_matches_parent(self):
        from nexq.channel.backends.torch_backend import TorchBackend

        ref_backend = TorchBackend()
        state = ref_backend.zeros_state(2)

        npu_backend = NPUBackend(fallback_to_cpu=True)
        state_cast = npu_backend.cast(npu_backend.to_numpy(state))

        # Force _is_npu_complex to return True so the workaround branch is taken
        with mock.patch.object(NPUBackend, "_is_npu_complex", return_value=True):
            probs_npu = npu_backend.to_numpy(npu_backend.measure_probs(state_cast))

        probs_ref = ref_backend.to_numpy(ref_backend.measure_probs(state))
        self.assertTrue(np.allclose(probs_npu, probs_ref, atol=1e-6))

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

    def test_npu_real_complex_matmul_uses_real_complex_path(self):
        backend = NPUBackend(fallback_to_cpu=True)
        backend._device = type("FakeDevice", (), {"type": "npu"})()
        real_gate = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
        complex_state = torch.tensor([[1 + 2j], [3 - 4j]], dtype=torch.complex64)

        with mock.patch.object(
            NPUBackend,
            "_real_complex_matmul",
            wraps=NPUBackend._real_complex_matmul,
        ) as real_complex:
            with mock.patch.object(NPUBackend, "_complex_matmul_workaround") as complex_workaround:
                result = backend.matmul(real_gate, complex_state)

        real_complex.assert_called_once()
        complex_workaround.assert_not_called()
        expected = torch.tensor([[1 + 2j], [-3 + 4j]], dtype=torch.complex64)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_cast_local_matrix_caches_by_key(self):
        backend = NPUBackend(fallback_to_cpu=True)
        matrix = np.array([[1, 0], [0, -1]], dtype=np.complex64)

        first = backend.cast_local_matrix(matrix, cache_key=("z",))
        second = backend.cast_local_matrix(matrix, cache_key=("z",))

        self.assertIs(first, second)

    # ──────── new operator workaround tests ────────

    def _npu_backend_forced(self):
        """Return an NPUBackend whose _is_npu_complex always returns True."""
        backend = NPUBackend(fallback_to_cpu=True)
        backend._patched_is_npu_complex = True
        return backend

    def _run_with_npu_forced(self, fn):
        with mock.patch.object(NPUBackend, "_is_npu_complex", return_value=True):
            return fn()

    def test_kron_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        a = torch.tensor([[1 + 2j, 3j], [-1j, 2 - 1j]], dtype=torch.complex64)
        b = torch.tensor([[0.5 - 1j], [1 + 0.5j]], dtype=torch.complex64)
        expected = torch.kron(a, b)
        result = self._run_with_npu_forced(lambda: backend.kron(a, b))
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_dagger_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        m = torch.tensor([[1 + 2j, 3 - 1j], [0 + 1j, -2 + 0j]], dtype=torch.complex64)
        expected = torch.conj(torch.transpose(m, -2, -1))
        result = self._run_with_npu_forced(lambda: backend.dagger(m))
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_trace_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        m = torch.tensor([[1 + 2j, 3j], [-1j, 4 - 1j]], dtype=torch.complex64)
        expected = torch.trace(m)
        result = self._run_with_npu_forced(lambda: backend.trace(m))
        self.assertTrue(torch.allclose(result.unsqueeze(0), expected.unsqueeze(0), atol=1e-5))

    def test_inner_product_workaround_matches_torch(self):
        backend = NPUBackend(fallback_to_cpu=True)
        bra = torch.tensor([[1 + 1j], [0 - 1j]], dtype=torch.complex64)
        ket = torch.tensor([[0.5 + 0j], [1 - 0.5j]], dtype=torch.complex64)
        expected = torch.tensor(1.0 + 0.5j, dtype=torch.complex64)
        result = self._run_with_npu_forced(lambda: backend.inner_product(bra, ket))
        self.assertTrue(torch.allclose(result.unsqueeze(0), expected.unsqueeze(0), atol=1e-5))

    def test_partial_trace_workaround_matches_parent(self):
        from nexq.channel.backends.torch_backend import TorchBackend
        ref = TorchBackend()
        # 2-qubit density matrix for |00><00|
        rho = ref.zeros_state(2)
        rho_dm = torch.matmul(rho, ref.dagger(rho))  # (4,4)

        backend = NPUBackend(fallback_to_cpu=True)
        rho_cast = backend.cast(backend.to_numpy(rho_dm))
        expected = ref.to_numpy(ref.partial_trace(rho_dm, keep=[0], n_qubits=2))
        result = self._run_with_npu_forced(
            lambda: backend.to_numpy(backend.partial_trace(rho_cast, keep=[0], n_qubits=2))
        )
        self.assertTrue(np.allclose(result, expected, atol=1e-5))

    def test_expectation_sv_workaround_matches_parent(self):
        from nexq.channel.backends.torch_backend import TorchBackend
        ref = TorchBackend()
        state = ref.zeros_state(1)
        # Z operator: [[1,0],[0,-1]]
        z_op = ref.cast(np.array([[1, 0], [0, -1]], dtype=np.complex64))

        backend = NPUBackend(fallback_to_cpu=True)
        state_cast = backend.cast(backend.to_numpy(state))
        op_cast = backend.cast(backend.to_numpy(z_op))

        expected = float(ref.to_numpy(ref.expectation_sv(state, z_op)))
        result = self._run_with_npu_forced(
            lambda: float(backend.to_numpy(backend.expectation_sv(state_cast, op_cast)))
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_expectation_dm_workaround_matches_parent(self):
        from nexq.channel.backends.torch_backend import TorchBackend
        ref = TorchBackend()
        state = ref.zeros_state(1)
        rho = torch.matmul(state, ref.dagger(state))
        z_op = ref.cast(np.array([[1, 0], [0, -1]], dtype=np.complex64))

        backend = NPUBackend(fallback_to_cpu=True)
        rho_cast = backend.cast(backend.to_numpy(rho))
        op_cast = backend.cast(backend.to_numpy(z_op))

        expected = float(ref.to_numpy(ref.expectation_dm(rho, z_op)))
        result = self._run_with_npu_forced(
            lambda: float(backend.to_numpy(backend.expectation_dm(rho_cast, op_cast)))
        )
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()

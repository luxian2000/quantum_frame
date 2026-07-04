import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from aicir import NPUBackend, State, npu_runtime_context_from_env
from aicir.backends.npu_backend import is_npu_available


class _BanComplexAddMul(TorchDispatchMode):
    """Reproduce Ascend NPU's missing complex64 kernels on CPU.

    NPU has no complex64 ``aclnnAdd``/``aclnnMul``; the real device raises if a
    backward pass ever adds or multiplies complex tensors (e.g. when autograd
    accumulates the gradient of a complex tensor reused in several places).
    Entering this dispatch mode around ``loss.backward()`` makes the same
    situation fail on CPU, so the test suite can guard the NPU path without a
    real device.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if "add" in name or "mul" in name or "sub" in name:
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, torch.Tensor) and torch.is_complex(value):
                    raise RuntimeError(f"unsupported complex op on NPU: {name}")
        return func(*args, **kwargs)


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
        sv = State.zero_state(2, backend)

        probs = backend.to_numpy(sv.probabilities())
        self.assertAlmostEqual(float(probs[0]), 1.0, places=6)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_from_distributed_env(self):
        with mock.patch.dict(
            "os.environ",
            {
                "WORLD_SIZE": "8",
                "RANK": "3",
                "LOCAL_RANK": "2",
                "MASTER_ADDR": "",
                "MASTER_PORT": "",
            },
            clear=False,
        ):
            backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

        self.assertIsNotNone(backend.runtime_context)
        self.assertEqual(backend.runtime_context.world_size, 8)
        self.assertEqual(backend.runtime_context.rank, 3)
        self.assertEqual(backend.runtime_context.local_rank, 2)
        self.assertTrue(backend.runtime_context.distributed)
        self.assertFalse(backend.runtime_context.process_group_initialized)

    def test_from_distributed_env_maps_ascend_visible_devices(self):
        requested_devices = []

        def fake_resolve(*, device=None, fallback_to_cpu=True):
            requested_devices.append(device)
            return torch.device("cpu")

        with mock.patch.dict(
            "os.environ",
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "2",
                "MASTER_ADDR": "",
                "MASTER_PORT": "",
                "ASCEND_RT_VISIBLE_DEVICES": "0,5,6,7",
            },
            clear=False,
        ):
            with mock.patch.object(NPUBackend, "_resolve_device", side_effect=fake_resolve):
                backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

        self.assertEqual(requested_devices, ["npu:6"])
        self.assertEqual(backend.runtime_context.local_rank, 2)

    def test_from_distributed_env_initializes_process_group_when_rendezvous_env_exists(self):
        env = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            with mock.patch("torch.distributed.is_available", return_value=True):
                with mock.patch("torch.distributed.is_initialized", side_effect=[False, True]):
                    with mock.patch("torch.distributed.init_process_group") as init_pg:
                        backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)

        expected_backend = "hccl" if getattr(backend._device, "type", None) == "npu" else "gloo"
        init_pg.assert_called_once_with(backend=expected_backend, rank=1, world_size=2)
        self.assertTrue(backend.runtime_context.process_group_initialized)
        self.assertEqual(backend.runtime_context.process_group_backend, expected_backend)

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
        from aicir.backends.gpu_backend import TorchBackend

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

    def test_inner_product_workaround_matches_numpy_reference(self):
        backend = NPUBackend(fallback_to_cpu=True)
        bra = torch.tensor([[1 + 1j], [0 - 1j]], dtype=torch.complex64)
        ket = torch.tensor([[0.5 + 0j], [1 - 0.5j]], dtype=torch.complex64)
        expected_np = np.vdot(
            np.asarray(bra.detach().cpu().numpy()).reshape(-1),
            np.asarray(ket.detach().cpu().numpy()).reshape(-1),
        ).astype(np.complex64)
        result = self._run_with_npu_forced(lambda: backend.inner_product(bra, ket))
        actual_np = np.asarray(backend.to_numpy(result)).reshape(())
        self.assertTrue(
            np.allclose(actual_np, expected_np, atol=1e-5),
            f"actual={actual_np!r}, expected={expected_np!r}",
        )

    def test_inner_product_workaround_does_not_use_torch_dot(self):
        backend = NPUBackend(fallback_to_cpu=True)
        bra = torch.tensor([[1 + 1j], [0 - 1j]], dtype=torch.complex64)
        ket = torch.tensor([[0.5 + 0j], [1 - 0.5j]], dtype=torch.complex64)

        def fail_dot(*args, **kwargs):
            raise RuntimeError("torch.dot is not NPU-safe for this workaround")

        with mock.patch("torch.dot", side_effect=fail_dot):
            result = self._run_with_npu_forced(lambda: backend.inner_product(bra, ket))

        actual_np = np.asarray(backend.to_numpy(result)).reshape(())
        expected_np = np.asarray(1 + 0.5j, dtype=np.complex64).reshape(())
        self.assertTrue(
            np.allclose(actual_np, expected_np, atol=1e-5),
            f"actual={actual_np!r}, expected={expected_np!r}",
        )

    def test_inner_product_workaround_uses_backend_matmul_path(self):
        backend = NPUBackend(fallback_to_cpu=True)
        bra = torch.tensor([[1 + 1j], [0 - 1j]], dtype=torch.complex64)
        ket = torch.tensor([[0.5 + 0j], [1 - 0.5j]], dtype=torch.complex64)

        with mock.patch.object(backend, "dagger", wraps=backend.dagger) as dagger:
            with mock.patch.object(backend, "matmul", wraps=backend.matmul) as matmul:
                result = self._run_with_npu_forced(lambda: backend.inner_product(bra, ket))

        dagger.assert_called_once()
        matmul.assert_called_once()
        actual_np = np.asarray(backend.to_numpy(result)).reshape(())
        expected_np = np.asarray(1 + 0.5j, dtype=np.complex64).reshape(())
        self.assertTrue(
            np.allclose(actual_np, expected_np, atol=1e-5),
            f"actual={actual_np!r}, expected={expected_np!r}",
        )

    def test_partial_trace_workaround_matches_parent(self):
        from aicir.backends.gpu_backend import TorchBackend
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
        from aicir.backends.gpu_backend import TorchBackend
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
        from aicir.backends.gpu_backend import TorchBackend
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

    # ──────── autograd-safe NPU Functions (value + gradient) ────────

    def test_npu_complex_matmul_function_matches_native_gradient(self):
        from aicir.backends.npu_backend import _NpuMatmulFn

        torch.manual_seed(0)
        a = torch.randn(4, 4, dtype=torch.complex64, requires_grad=True)
        b = torch.randn(4, 3, dtype=torch.complex64, requires_grad=True)
        c = torch.matmul(a, b)
        (c.real ** 2 + c.imag ** 2).sum().backward()
        ga, gb = a.grad.clone(), b.grad.clone()

        a2 = a.detach().clone().requires_grad_()
        b2 = b.detach().clone().requires_grad_()
        c2 = _NpuMatmulFn.apply(a2, b2)
        (c2.real ** 2 + c2.imag ** 2).sum().backward()

        self.assertTrue(torch.allclose(c, c2, atol=1e-5))
        self.assertTrue(torch.allclose(ga, a2.grad, atol=1e-4))
        self.assertTrue(torch.allclose(gb, b2.grad, atol=1e-4))

    def test_npu_real_complex_matmul_function_gradient(self):
        from aicir.backends.npu_backend import _NpuMatmulFn

        torch.manual_seed(1)
        real_gate = torch.randn(4, 4)  # constant real gate
        v = torch.randn(4, 1, dtype=torch.complex64, requires_grad=True)
        c = torch.matmul(real_gate.to(torch.complex64), v)
        (c.real ** 2 + c.imag ** 2).sum().backward()
        gv = v.grad.clone()

        v2 = v.detach().clone().requires_grad_()
        c2 = _NpuMatmulFn.apply(real_gate, v2)
        (c2.real ** 2 + c2.imag ** 2).sum().backward()
        self.assertTrue(torch.allclose(gv, v2.grad, atol=1e-4))

    def test_npu_expectation_function_matches_native_gradient(self):
        from aicir.backends.npu_backend import _NpuExpectationFn
        from aicir.backends.gpu_backend import GPUBackend

        torch.manual_seed(2)
        m = torch.randn(8, 8, dtype=torch.complex64)
        h = (m + m.conj().T) / 2  # Hermitian
        s = torch.randn(8, 1, dtype=torch.complex64, requires_grad=True)
        e = GPUBackend(device="cpu").expectation_sv(s, h)
        e.backward()
        gs = s.grad.clone()

        s2 = s.detach().clone().requires_grad_()
        e2 = _NpuExpectationFn.apply(s2, h)
        e2.backward()
        self.assertTrue(torch.allclose(e, e2, atol=1e-5))
        self.assertTrue(torch.allclose(gs, s2.grad, atol=1e-4))

    def test_npu_circuit_energy_gradient_matches_gpu(self):
        # End-to-end: a parameterised circuit's energy gradient via the forced
        # NPU complex path must match native GPUBackend autograd.
        import types
        from aicir.backends.npu_backend import NPUBackend, _NpuMatmulFn
        from aicir.backends.gpu_backend import GPUBackend
        from aicir.core.gates import apply_gate_to_state
        from aicir.core.circuit import rx, ry, rz, rzz, cx, hadamard

        torch.manual_seed(3)
        n = 3
        m = torch.randn(1 << n, 1 << n, dtype=torch.complex64)
        h = (m + m.conj().T) / 2

        def energy(backend, theta):
            gates = [
                hadamard(0), rx(theta[0], 0), ry(theta[1], 1), rz(theta[2], 2),
                cx(target_qubit=1, control_qubits=[0]),
                rzz(theta[3], qubit_1=1, qubit_2=2),
            ]
            state = backend.zeros_state(n)
            for g in gates:
                state = apply_gate_to_state(g, state, n, backend)
            return backend.expectation_sv(state, backend.cast(h.numpy()))

        init = [0.3, -0.7, 1.1, 0.4]
        t1 = torch.tensor(init, requires_grad=True)
        energy(GPUBackend(device="cpu"), t1).backward()
        g1 = t1.grad.clone()

        npu = NPUBackend(fallback_to_cpu=True)

        def forced_matmul(self, a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and (
                torch.is_complex(a) or torch.is_complex(b)
            ):
                return _NpuMatmulFn.apply(a, b)
            return GPUBackend.matmul(self, a, b)

        npu.matmul = types.MethodType(forced_matmul, npu)
        t2 = torch.tensor(init, requires_grad=True)
        with mock.patch.object(NPUBackend, "_is_npu_complex", return_value=True):
            energy(npu, t2).backward()

        self.assertTrue(torch.allclose(g1, t2.grad, atol=1e-4))

    def test_parameterized_gate_matrices_have_no_complex_backward(self):
        # Every parameterised gate matrix must backprop without a complex
        # add/mul, otherwise loss.backward() crashes on Ascend NPU. This is the
        # bug where rzz/rxx reused one complex tensor across cells (-> complex
        # add) and u2/u3 multiplied complex tensors (-> complex mul).
        from aicir.core.gates import (
            _rzz_backend,
            _rxx_backend,
            _single_qubit_base_for_gate_backend,
        )
        from aicir.backends.gpu_backend import GPUBackend

        backend = GPUBackend(device="cpu")

        def matrix(gate_type, value):
            param = torch.tensor(value, requires_grad=True)
            if gate_type == "rzz":
                m = _rzz_backend(param, backend)
            elif gate_type == "rxx":
                m = _rxx_backend(param, backend)
            else:
                m = _single_qubit_base_for_gate_backend(
                    {"type": gate_type, "parameter": param}, backend
                )
            return param, m

        cases = {
            "rx": 0.37, "ry": 0.37, "rz": 0.37, "rzz": 0.37, "rxx": 0.37,
            "u2": [0.3, 0.5], "u3": [0.3, 0.5, 0.7],
        }
        for gate_type, value in cases.items():
            with self.subTest(gate=gate_type):
                param, m = matrix(gate_type, value)
                # Consume the matrix exactly once into a real scalar (mimics a
                # gate applied to a state), so the only complex ops in backward
                # come from the matrix construction itself.
                vec = torch.randn(m.shape[0], 1, dtype=torch.complex64)
                loss = torch.real(m @ vec).sum()
                with _BanComplexAddMul():
                    loss.backward()  # must not raise
                self.assertIsNotNone(param.grad)

    def test_npu_circuit_backward_runs_without_complex_ops(self):
        # End-to-end guard: the full forced-NPU backward (custom matmul /
        # expectation Functions + parameterised gate construction) must not use
        # any complex add/mul, i.e. it would run on a real Ascend NPU.
        import types
        from aicir.backends.npu_backend import _NpuMatmulFn, _NpuExpectationFn
        from aicir.backends.gpu_backend import GPUBackend
        from aicir.core.gates import apply_gate_to_state
        from aicir.core.circuit import rx, ry, rz, rzz, cx, hadamard

        torch.manual_seed(4)
        n = 3
        m = torch.randn(1 << n, 1 << n, dtype=torch.complex64)
        h = (m + m.conj().T) / 2

        npu = NPUBackend(fallback_to_cpu=True)

        def forced_matmul(self, a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and (
                torch.is_complex(a) or torch.is_complex(b)
            ):
                return _NpuMatmulFn.apply(a, b)
            return GPUBackend.matmul(self, a, b)

        npu.matmul = types.MethodType(forced_matmul, npu)
        npu.expectation_sv = lambda state, op: _NpuExpectationFn.apply(state, op)

        theta = torch.tensor([0.3, -0.7, 1.1, 0.4], requires_grad=True)
        gates = [
            hadamard(0), rx(theta[0], 0), ry(theta[1], 1), rz(theta[2], 2),
            cx(target_qubit=1, control_qubits=[0]),
            rzz(theta[3], qubit_1=1, qubit_2=2),
        ]
        state = npu.zeros_state(n)
        for gate in gates:
            state = apply_gate_to_state(gate, state, n, npu)
        energy = npu.expectation_sv(state, npu.cast(h.numpy()))

        with _BanComplexAddMul():
            energy.backward()  # must not raise on the NPU restriction
        self.assertIsNotNone(theta.grad)

    def test_flat_local_gate_application_matches_tensor_path_for_12q(self):
        from aicir.backends.numpy_backend import NumpyBackend
        from aicir.core.gates import (
            _apply_local_matrix_to_state,
            _apply_local_matrix_to_state_flat,
        )

        backend = NumpyBackend()
        rng = np.random.default_rng(7)
        state = rng.normal(size=(1 << 12, 1)) + 1j * rng.normal(size=(1 << 12, 1))
        local = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.complex64,
        )

        expected = _apply_local_matrix_to_state(state, local, [2, 10], 12, backend)
        actual = _apply_local_matrix_to_state_flat(state, local, [2, 10], 12, backend)

        self.assertTrue(np.allclose(actual, expected))

    def test_npu_12q_local_gate_uses_flat_path(self):
        from aicir.core.gates import _is_npu_complex_tensor, _should_use_flat_local_apply

        backend = NPUBackend(fallback_to_cpu=True)
        backend._device = type("FakeDevice", (), {"type": "npu"})()

        self.assertTrue(_should_use_flat_local_apply(backend, 12))
        self.assertFalse(_should_use_flat_local_apply(backend, 8))
        self.assertTrue(_is_npu_complex_tensor(torch.zeros(4, dtype=torch.complex64), backend))


if __name__ == "__main__":
    unittest.main()

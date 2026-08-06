"""dtype 策略测试：后端是复数精度的单一真源。

策略要点（见 CHANGELOG 与 aicir/dtypes.py）：
- ``NumpyBackend`` 默认 ``complex128``；``GPUBackend`` 默认 ``complex64``
  （torch 生态 float32-native）；``NPUBackend`` 硬件锁定 ``complex64``。
- 门矩阵与 ``Circuit.unitary()`` 的 dtype 由后端决定，不再硬编码。
- ``aicir.set_default_dtype`` 提供全局默认覆盖，显式设置后 torch 后端亦服从。
- 演化热路径不得泄漏 RuntimeWarning。
"""

import unittest
import warnings

import numpy as np

import aicir as A
from aicir import Circuit, Measure, NumpyBackend
from aicir.dtypes import get_default_dtype, reset_default_dtype, set_default_dtype


def _bell_chain(n):
    """n 比特 H 链 + CNOT 链，末态范数恒为 1，用于精度体检。"""
    gates = [A.hadamard(q) for q in range(n)]
    gates += [A.cnot(q + 1, [q]) for q in range(n - 1)]
    return Circuit(*gates, n_qubits=n)


class TestBackendDtypeDefaults(unittest.TestCase):
    """后端默认精度与公开 dtype 属性。"""

    def test_numpy_backend_defaults_to_complex128(self):
        self.assertEqual(NumpyBackend().dtype, np.complex128)

    def test_numpy_backend_honours_explicit_complex64(self):
        self.assertEqual(NumpyBackend(dtype=np.complex64).dtype, np.complex64)


class TestGateMatrixFollowsBackend(unittest.TestCase):
    """门矩阵与酉矩阵的 dtype 来自后端，而非模块常量。"""

    def test_unitary_defaults_to_complex128(self):
        circuit = Circuit(A.hadamard(0), A.cnot(1, [0]), n_qubits=2)
        self.assertEqual(np.asarray(circuit.unitary()).dtype, np.complex128)

    def test_unitary_follows_complex64_backend(self):
        circuit = Circuit(A.hadamard(0), A.cnot(1, [0]), n_qubits=2)
        unitary = circuit.unitary(backend=NumpyBackend(dtype=np.complex64))
        self.assertEqual(np.asarray(unitary).dtype, np.complex64)


class TestStateConstructionPrecision(unittest.TestCase):
    """``State.from_array`` / ``from_matrix`` 必须遵循后端精度。

    历史 bug：两者都硬编码 ``np.complex64``，再由 ``backend.cast`` 拓宽回
    complex128——精度在拓宽前就已经丢了。表现是范数误差约 1e-8（单精度量级）
    而非 1e-16，且**不报错**。与 `Circuit.unitary()` 的旧 bug 同源：核心路径里
    不该出现裸 complex64 字面量，dtype 一律取自后端。
    """

    def test_from_array_preserves_double_precision(self):
        backend = NumpyBackend(dtype=np.complex128)
        rng = np.random.default_rng(5)
        vec = rng.normal(size=8) + 1j * rng.normal(size=8)
        vec = (vec / np.linalg.norm(vec)).astype(np.complex128)

        state = A.State.from_array(vec, backend=backend)
        data = np.asarray(state.to_numpy())
        self.assertEqual(data.dtype, np.complex128)
        norm_error = abs(float(np.linalg.norm(data)) - 1.0)
        self.assertLess(norm_error, 1e-14, f"from_array 范数误差 {norm_error:.3e} 达不到双精度")

    def test_from_array_honours_complex64_backend(self):
        backend = NumpyBackend(dtype=np.complex64)
        vec = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        state = A.State.from_array(vec, backend=backend)
        self.assertEqual(np.asarray(state.to_numpy()).dtype, np.complex64)

    def test_from_matrix_preserves_double_precision(self):
        backend = NumpyBackend(dtype=np.complex128)
        rng = np.random.default_rng(6)
        vec = rng.normal(size=4) + 1j * rng.normal(size=4)
        vec = (vec / np.linalg.norm(vec)).astype(np.complex128)
        rho = np.outer(vec, vec.conj())

        state = A.State.from_matrix(rho, backend=backend)
        data = np.asarray(state.to_numpy())
        self.assertEqual(data.dtype, np.complex128)
        trace_error = abs(complex(np.trace(data)).real - 1.0)
        self.assertLess(trace_error, 1e-14, f"from_matrix 迹误差 {trace_error:.3e} 达不到双精度")


class TestEvolutionPrecision(unittest.TestCase):
    """complex128 必须是真正的双精度，而不是被 complex64 门矩阵污染。"""

    def test_complex128_evolution_reaches_double_precision(self):
        result = Measure(backend=NumpyBackend(dtype=np.complex128)).run(_bell_chain(14), shots=None)
        state = np.asarray(result.final_state)
        self.assertEqual(state.dtype, np.complex128)
        norm_error = abs(float(np.linalg.norm(state)) - 1.0)
        self.assertLess(norm_error, 1e-14, f"complex128 路径范数误差 {norm_error:.3e} 达不到双精度")

    def test_complex64_evolution_stays_single_precision(self):
        result = Measure(backend=NumpyBackend(dtype=np.complex64)).run(_bell_chain(14), shots=None)
        state = np.asarray(result.final_state)
        self.assertEqual(state.dtype, np.complex64)
        norm_error = abs(float(np.linalg.norm(state)) - 1.0)
        self.assertLess(norm_error, 1e-6)


class TestNoRuntimeWarnings(unittest.TestCase):
    """演化热路径不得泄漏 divide/overflow/invalid 警告。"""

    def test_evolution_emits_no_runtime_warning(self):
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    Measure(backend=NumpyBackend(dtype=dtype)).run(_bell_chain(14), shots=None)


class TestGlobalDefaultDtype(unittest.TestCase):
    """全局默认覆盖（对标 TensorCircuit 的 tc.set_dtype）。"""

    def setUp(self):
        self._original = get_default_dtype()

    def tearDown(self):
        reset_default_dtype()

    def test_default_is_complex128(self):
        self.assertEqual(get_default_dtype(), np.complex128)

    def test_set_default_dtype_affects_new_backends(self):
        set_default_dtype(np.complex64)
        self.assertEqual(NumpyBackend().dtype, np.complex64)

    def test_set_default_dtype_rejects_non_complex(self):
        with self.assertRaises(ValueError):
            set_default_dtype(np.float64)

    def test_set_default_dtype_is_exported_at_top_level(self):
        self.assertIs(A.set_default_dtype, set_default_dtype)


class TestTorchBackendDefaultsToSinglePrecision(unittest.TestCase):
    """Torch 后端默认单精度：生态是 float32-native，且 GPU fp64 吞吐被大幅限制。"""

    def setUp(self):
        self._original = get_default_dtype()

    def tearDown(self):
        reset_default_dtype()

    def test_gpu_backend_defaults_to_complex64(self):
        torch = __import__("pytest").importorskip("torch")
        from aicir import GPUBackend

        self.assertEqual(GPUBackend(device="cpu").dtype, torch.complex64)

    def test_explicit_global_double_is_honoured_by_gpu_backend(self):
        torch = __import__("pytest").importorskip("torch")
        from aicir import GPUBackend

        set_default_dtype(np.complex128)
        self.assertEqual(GPUBackend(device="cpu").dtype, torch.complex128)

    def test_quantum_layer_readout_composes_with_float32_classical_layer(self):
        """混合模型回归：量子层读出必须能直接喂给默认 float32 的 nn.Linear。"""

        torch = __import__("pytest").importorskip("torch")
        from aicir import GPUBackend
        from aicir.qml import build_classifier

        model = build_classifier(
            n_features=2, n_classes=2, backend=GPUBackend(device="cpu"), n_qubits=2, layers=1
        )
        out = model(torch.zeros(3, 2))
        self.assertEqual(out.dtype, torch.float32)


class TestNPUPrecisionCapability(unittest.TestCase):
    """NPU 精度是硬件能力：Ascend 无 complex128 内核，必须显式拒绝。"""

    def test_npu_backend_rejects_complex128(self):
        torch = __import__("pytest").importorskip("torch")
        from aicir.backends.npu_backend import validate_npu_dtype

        with self.assertRaises(ValueError):
            validate_npu_dtype(torch.complex128)

    def test_npu_backend_accepts_complex64(self):
        torch = __import__("pytest").importorskip("torch")
        from aicir.backends.npu_backend import validate_npu_dtype

        self.assertEqual(validate_npu_dtype(torch.complex64), torch.complex64)

    def test_npu_backend_defaults_to_complex64_even_when_global_is_128(self):
        __import__("pytest").importorskip("torch")
        from aicir.backends.npu_backend import validate_npu_dtype

        import torch

        self.assertEqual(validate_npu_dtype(None), torch.complex64)


if __name__ == "__main__":
    unittest.main()

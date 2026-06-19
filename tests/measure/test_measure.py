"""
tests/measure/test_measure.py

已迁移至统一测量模型（Measure.run 单入口）。

已删除（测试已移除功能）：
- test_run_density_matrix_path           → run_density_matrix 已删除
- test_run_batch_multi_circuits          → run_batch 已删除
- test_run_batch_uses_distributed_backend_partition_and_gather → run_batch 已删除
- test_scan_parameters                   → scan_parameters 已删除
- test_approaches_are_mutually_exclusive → 旧"机制一/二互斥"规则已废除；
                                           内嵌 measure() 与 measure_qubits 现可共存
- test_reset_requires_previous_measure_on_same_qubit  → 新模型不再强制此约束
- test_reset_rejects_gate_between_measure_and_reset   → 新模型不再强制此约束
- test_build_initial_state_rejects_density_form       → _build_initial_state 私有方法已删除
- test_build_initial_density_matrix_rejects_vector_form → _build_initial_density_matrix 私有方法已删除

新模型参考：tests/measure/test_unified_run.py
"""
import unittest
import unittest.mock
from types import SimpleNamespace

import numpy as np
import torch

from aicir import Circuit, Measure, TorchBackend, cnot, hadamard, measure, pauli_x, reset, ry
from aicir.backends import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.core.circuit import crx, rxx, swap, toffoli
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.core.state import State
from aicir.measure.result import Result


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.backend = TorchBackend(device="cpu")
        self.measure = Measure(self.backend)
        self.bell = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)

    def test_run_state_vector_probabilities_and_counts(self):
        result = self.measure.run(self.bell, shots=2000)

        self.assertEqual(result.n_qubits, 2)
        # counts(-1) 返回 {"00": N, "11": M} 格式的字典
        counts = result.counts(-1)
        self.assertIsNotNone(counts)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)
        self.assertAlmostEqual(float(np.sum(result.probabilities)), 1.0, places=6)

    def test_run_state_vector_expectation_value(self):
        # 新模型在 shots=None（exact 模式）下计算期望值；不计算方差
        h = Hamiltonian(n_qubits=2, terms=[("ZZ", 1.0)])
        op = h.to_matrix(self.backend)

        result = self.measure.run(self.bell, shots=None, observables={"ZZ": op})

        self.assertIn("ZZ", result.expectation_values)
        self.assertAlmostEqual(result.expectation_values["ZZ"], 1.0, places=5)

    def test_build_result_converts_probabilities_through_backend(self):
        class DeviceArray:
            def __array__(self, dtype=None):
                raise TypeError("device tensor must be copied through backend")

        class BackendWithDeviceArray(NumpyBackend):
            def to_numpy(self, tensor):
                if isinstance(tensor, DeviceArray):
                    return np.array([1.0, 0.0])
                return super().to_numpy(tensor)

        pre = SimpleNamespace(probabilities=lambda: DeviceArray())
        trajectory = SimpleNamespace(pre=pre, post=pre, incircuit={}, terminal=None, snaps={})

        result = Measure(BackendWithDeviceArray())._build_result(
            [trajectory],
            1,
            BackendWithDeviceArray(),
            1,
            True,
            [],
            None,
            False,
            {},
            False,
        )

        self.assertTrue(np.allclose(result.probabilities, [1.0, 0.0]))

    def test_run_prefers_circuit_bound_backend(self):
        np_backend = NumpyBackend()
        circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=np_backend)

        result = self.measure.run(circ, shots=200)

        # 新模型后端名称为类名（不带括号参数）
        self.assertEqual(result.backend_name, "NumpyBackend")
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)

    def test_run_gatewise_path_does_not_require_unitary(self):
        class GateOnlyCircuit:
            def __init__(self):
                self.n_qubits = 2
                self.gates = [hadamard(0), cnot(1, [0])]

            def unitary(self, backend=None):
                raise AssertionError("gatewise path should not call unitary()")

        result = self.measure.run(GateOnlyCircuit(), shots=None)
        self.assertAlmostEqual(result.probabilities[0], 0.5, places=3)
        self.assertAlmostEqual(result.probabilities[3], 0.5, places=3)

    def test_local_gate_application_matches_full_matrix_path(self):
        rng = np.random.default_rng(7)
        gates = [
            hadamard(0),
            ry(0.3, 2),
            cnot(0, [2], [0]),
            crx(0.2, 2, [1]),
            swap(0, 2),
            rxx(0.4, 0, 2),
            toffoli(2, [0, 1]),
        ]

        for backend in (NumpyBackend(), TorchBackend(device="cpu")):
            for gate in gates:
                n_qubits = 3
                state_np = rng.normal(size=(1 << n_qubits, 1)) + 1j * rng.normal(size=(1 << n_qubits, 1))
                state_np = (state_np / np.linalg.norm(state_np)).astype(np.complex64)
                state = backend.cast(state_np)

                direct = apply_gate_to_state(gate, state, n_qubits, backend)
                full = backend.apply_unitary(state, gate_to_matrix(gate, n_qubits, backend=backend))

                self.assertTrue(
                    np.allclose(backend.to_numpy(direct), backend.to_numpy(full), atol=1e-5),
                    msg=f"local gate mismatch for {backend.name}: {gate}",
                )

    def test_local_gate_application_reuses_cached_backend_matrix(self):
        backend = NumpyBackend()
        state = backend.zeros_state(1)
        gate = hadamard(0)

        with unittest.mock.patch.object(backend, "cast", wraps=backend.cast) as cast:
            state = apply_gate_to_state(gate, state, 1, backend)
            state = apply_gate_to_state(gate, state, 1, backend)

        self.assertEqual(cast.call_count, 1)
        np.testing.assert_allclose(
            backend.to_numpy(state),
            np.array([[1.0 + 0j], [0.0 + 0j]], dtype=np.complex64),
            atol=1e-6,
        )

    def test_explicit_measure_qubits_reads_subset(self):
        # 通过 measure_qubits 指定末端读出子集（shots>=1）
        result = self.measure.run(self.bell, shots=2000, measure_qubits=[0])

        self.assertEqual(result.terminal_qubits, [0])
        # counts(-1) 键为裸比特串（无 |> 符号），长度为 1（仅读出 qubit0）
        counts = result.counts(-1)
        self.assertTrue(all(len(k) == 1 for k in counts))

    def test_explicit_measure_qubits_with_all_qubits(self):
        # 显式列出所有比特的 measure_qubits，效果等同于默认读出
        result = self.measure.run(self.bell, shots=200, measure_qubits=[0, 1])

        self.assertEqual(result.terminal_qubits, [0, 1])
        counts = result.counts(-1)
        self.assertTrue(all(len(k) == 2 for k in counts))

    def test_empty_measure_qubits_reads_all(self):
        # measure_qubits=[] 表示读出全部比特
        result = self.measure.run(self.bell, shots=200, measure_qubits=[])

        self.assertEqual(result.terminal_qubits, [0, 1])
        counts = result.counts(-1)
        self.assertTrue(all(len(k) == 2 for k in counts))

    def test_measure_qubits_validates_indices(self):
        with self.assertRaises(ValueError):
            self.measure.run(self.bell, shots=10, measure_qubits=[5])

    def test_in_circuit_measure_registers_spec(self):
        # 内嵌 measure(0) 在新模型中是投影测量；measurement_specs 记录该操作
        circ = Circuit(hadamard(0), cnot(1, [0]), measure(0), n_qubits=2)
        result = self.measure.run(circ, shots=1000)

        specs = result.measurement_specs
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].qubits, [0])

    def test_reset_sets_qubit_to_zero(self):
        # X 门翻转后 measure + reset，最终态应为 |0>
        circ = Circuit(pauli_x(0), measure(0), reset(0), n_qubits=1)

        result = self.measure.run(circ, shots=None)

        np.testing.assert_allclose(result.state.array, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.probabilities, [1.0, 0.0], atol=1e-6)

    def test_reset_with_density_matrix_initial_state(self):
        # 以密度矩阵初始态运行含 reset 的线路
        backend = NumpyBackend()
        m = Measure(backend)
        circ = Circuit(pauli_x(0), measure(0), reset(0), n_qubits=1)
        # 以 |0><0| 为初始密度矩阵
        rho0 = np.array([[1.0 + 0.0j, 0.0], [0.0, 0.0]], dtype=np.complex64)
        result = m.run(circ, shots=None, initial_density_matrix=rho0)

        np.testing.assert_allclose(result.probabilities, [1.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

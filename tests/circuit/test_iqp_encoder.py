import unittest

import numpy as np

from aicir.encoder.iqp import IQPEncoder


def _reference_state(x, reps=2):
    """按论文定义直接构造 (U_Phi(x) H^n)^reps |0>^n 作为参照。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    dim = 1 << n
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    Hn = H
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)

    # z 比特序与线路一致：qubit 0 为最高位
    phases = np.zeros(dim)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        signs = [1 - 2 * b for b in bits]
        phi = sum(x[q] * signs[q] for q in range(n))
        for i in range(n):
            for j in range(i + 1, n):
                phi += (np.pi - x[i]) * (np.pi - x[j]) * signs[i] * signs[j]
        phases[idx] = phi
    U_phi = np.diag(np.exp(1j * phases))

    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for _ in range(reps):
        state = U_phi @ (Hn @ state)
    return state


class TestIQPEncoder(unittest.TestCase):
    def test_matches_paper_definition(self):
        x = [0.7, 2.1]
        _, state = IQPEncoder().encode(x)
        got = state.to_numpy().ravel()
        want = _reference_state(x)
        # 默认 NumpyBackend 使用 complex64，取单精度容差
        self.assertTrue(np.allclose(got, want, atol=1e-5))

    def test_matches_paper_definition_three_qubits(self):
        x = [0.3, 1.9, 4.2]
        _, state = IQPEncoder().encode(x)
        got = state.to_numpy().ravel()
        want = _reference_state(x)
        self.assertTrue(np.allclose(got, want, atol=1e-5))

    def test_gate_structure(self):
        circuit = IQPEncoder().circuit([0.5, 1.5])
        names = [g.name for g in circuit.gates]
        # 每个 rep：2 个 hadamard + 2 个 rz + 1 个 rzz，reps=2
        self.assertEqual(names, ["hadamard", "hadamard", "rz", "rz", "rzz"] * 2)
        self.assertEqual(circuit.n_qubits, 2)

    def test_linear_entanglement(self):
        circuit = IQPEncoder(reps=1, entanglement="linear").circuit([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(sum(1 for g in circuit.gates if g.name == "rzz"), 3)

    def test_explicit_pairs(self):
        circuit = IQPEncoder(reps=1, entanglement=[(0, 2)]).circuit([0.1, 0.2, 0.3])
        rzz_gates = [g for g in circuit.gates if g.name == "rzz"]
        self.assertEqual(len(rzz_gates), 1)
        self.assertEqual(tuple(rzz_gates[0].qubits), (0, 2))

    def test_padding_and_length_check(self):
        encoder = IQPEncoder(n_qubits=3)
        circuit = encoder.circuit([0.5, 1.0])
        self.assertEqual(circuit.n_qubits, 3)
        with self.assertRaises(ValueError):
            encoder.circuit([0.1, 0.2, 0.3, 0.4])

    def test_kernel_properties(self):
        encoder = IQPEncoder()
        x, z = [0.7, 2.1], [1.3, 0.4]
        self.assertAlmostEqual(encoder.kernel(x, x), 1.0, places=5)
        kxz = encoder.kernel(x, z)
        self.assertAlmostEqual(kxz, encoder.kernel(z, x), places=5)
        self.assertTrue(0.0 <= kxz <= 1.0)

    def test_kernel_matrix(self):
        encoder = IQPEncoder()
        xs = [[0.7, 2.1], [1.3, 0.4], [5.0, 3.3]]
        K = encoder.kernel_matrix(xs)
        self.assertEqual(K.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(K), 1.0, atol=1e-5))
        self.assertTrue(np.allclose(K, K.T, atol=1e-5))
        self.assertAlmostEqual(K[0, 1], encoder.kernel(xs[0], xs[1]), places=5)

    def test_decode_unsupported(self):
        encoder = IQPEncoder()
        _, state = encoder.encode([0.5, 1.5])
        with self.assertRaises(NotImplementedError):
            encoder.decode(state)

    def test_custom_data_map(self):
        encoder = IQPEncoder(data_map=lambda vals: float(np.prod(vals)))
        x = [0.7, 2.1]
        _, state = encoder.encode(x)
        # 自定义映射改变双比特相位，结果应偏离论文默认映射
        default = IQPEncoder().encode(x)[1].to_numpy().ravel()
        self.assertFalse(np.allclose(state.to_numpy().ravel(), default, atol=1e-6))

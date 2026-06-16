import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("TorchBackend unitary tests require torch") from exc

from aicir import Circuit, TorchBackend, cnot, crz, hadamard, ms_gate, rx, rxx, rzz
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.backends.numpy_backend import NumpyBackend


class TestCircuitBackendUnitary(unittest.TestCase):
    def test_unitary_backend_matches_numpy(self):
        circ = Circuit(
            hadamard(0),
            cnot(1, [0]),
            rx(np.pi / 4, 1),
            n_qubits=2,
        )

        u_np = circ.unitary()

        backend = TorchBackend(device="cpu")
        u_backend = circ.unitary(backend=backend)
        u_backend_np = backend.to_numpy(u_backend)

        self.assertIsInstance(u_backend, torch.Tensor)
        self.assertEqual(tuple(u_backend.shape), u_np.shape)
        self.assertTrue(np.allclose(u_backend_np, u_np, atol=1e-6))

    def test_empty_circuit_returns_backend_identity(self):
        circ = Circuit(n_qubits=3)
        backend = TorchBackend(device="cpu")

        u_backend = circ.unitary(backend=backend)
        u_backend_np = backend.to_numpy(u_backend)

        self.assertIsInstance(u_backend, torch.Tensor)
        self.assertEqual(tuple(u_backend.shape), (8, 8))
        self.assertTrue(np.allclose(u_backend_np, np.eye(8, dtype=np.complex64), atol=1e-6))

    def test_circuit_bound_backend_used_by_default(self):
        backend = TorchBackend(device="cpu")
        circ = Circuit(
            hadamard(0),
            cnot(1, [0]),
            n_qubits=2,
            backend=backend,
        )

        u_backend = circ.unitary()
        self.assertIsInstance(u_backend, torch.Tensor)
        self.assertEqual(tuple(u_backend.shape), (4, 4))

    def test_controlled_gates_default_to_one_control_state(self):
        implicit = Circuit({"type": "cx", "target_qubit": 1, "control_qubits": [0]}, n_qubits=2)
        explicit = Circuit(cnot(1, [0]), n_qubits=2)

        self.assertTrue(np.allclose(implicit.unitary(), explicit.unitary(), atol=1e-6))

    def test_rzz_full_matrix_embeds_nonleading_qubits(self):
        circ = Circuit(rzz(0.7, qubit_1=1, qubit_2=2), n_qubits=3)

        unitary = circ.unitary()

        self.assertEqual(unitary.shape, (8, 8))
        self.assertTrue(np.allclose(unitary.conj().T @ unitary, np.eye(8), atol=1e-6))

    def test_rxx_matches_standard_xx_rotation_and_ms_alias(self):
        theta = 0.7
        cos = np.cos(theta / 2.0)
        neg_i_sin = -1j * np.sin(theta / 2.0)
        expected = np.array(
            [
                [cos, 0.0, 0.0, neg_i_sin],
                [0.0, cos, neg_i_sin, 0.0],
                [0.0, neg_i_sin, cos, 0.0],
                [neg_i_sin, 0.0, 0.0, cos],
            ],
            dtype=np.complex64,
        )

        self.assertEqual(ms_gate(theta, 0, 1), rxx(theta, 0, 1))
        np.testing.assert_allclose(gate_to_matrix(rxx(theta, 0, 1), cir_qubits=2), expected, atol=1e-6)

    def test_rxx_full_matrix_embeds_nonleading_qubits(self):
        circ = Circuit(rxx(0.7, qubit_1=1, qubit_2=2), n_qubits=3)

        unitary = circ.unitary()

        self.assertEqual(unitary.shape, (8, 8))
        self.assertTrue(np.allclose(unitary.conj().T @ unitary, np.eye(8), atol=1e-6))

    def test_identity_gate_expands_to_circuit_width(self):
        circ = Circuit({"type": "identity", "n_qubits": 1}, n_qubits=3)

        unitary = circ.unitary()

        self.assertEqual(unitary.shape, (8, 8))
        np.testing.assert_allclose(unitary, np.eye(8, dtype=np.complex64))

    def test_torch_parameterized_unitary_preserves_autograd(self):
        backend = TorchBackend(device="cpu")
        theta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        circ = Circuit(rx(theta, 0), n_qubits=1)

        unitary = circ.unitary(backend=backend)
        loss = torch.real(unitary[0, 0])
        loss.backward()

        self.assertTrue(unitary.requires_grad)
        self.assertIsNotNone(theta.grad)
        self.assertAlmostEqual(float(theta.grad), -0.5 * np.sin(0.25), places=6)

    def test_torch_custom_unitary_preserves_autograd(self):
        backend = TorchBackend(device="cpu")
        matrix = torch.eye(2, dtype=torch.complex64)
        matrix.requires_grad_()
        circ = Circuit({"type": "unitary", "parameter": matrix, "n_qubits": 1}, n_qubits=2)

        unitary = circ.unitary(backend=backend)
        loss = torch.real(unitary[0, 0])
        loss.backward()

        self.assertTrue(unitary.requires_grad)
        self.assertIsNotNone(matrix.grad)

    def test_torch_controlled_and_rzz_parameters_preserve_autograd(self):
        backend = TorchBackend(device="cpu")
        theta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        phi = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
        circ = Circuit(
            crz(theta, target_qubit=1, control_qubits=[0]),
            rzz(phi, qubit_1=0, qubit_2=2),
            n_qubits=3,
        )

        unitary = circ.unitary(backend=backend)
        loss = torch.real(unitary[7, 7])
        loss.backward()

        self.assertTrue(unitary.requires_grad)
        self.assertIsNotNone(theta.grad)
        self.assertIsNotNone(phi.grad)

    def test_torch_rxx_parameter_preserves_autograd(self):
        backend = TorchBackend(device="cpu")
        theta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        circ = Circuit(rxx(theta, qubit_1=0, qubit_2=1), n_qubits=2)

        unitary = circ.unitary(backend=backend)
        loss = torch.real(unitary[0, 0])
        loss.backward()

        self.assertTrue(unitary.requires_grad)
        self.assertIsNotNone(theta.grad)
        self.assertAlmostEqual(float(theta.grad), -0.5 * np.sin(0.25), places=6)

    def test_toffoli_full_matrix_supports_control_states_and_more_controls(self):
        backend = NumpyBackend()
        gates = [
            {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1], "control_states": [1, 0]},
            {"type": "toffoli", "target_qubit": 3, "control_qubits": [0, 1, 2], "control_states": [1, 0, 1]},
        ]

        for gate in gates:
            n_qubits = max([gate["target_qubit"]] + gate["control_qubits"]) + 1
            unitary = Circuit(gate, n_qubits=n_qubits).unitary()
            self.assertEqual(unitary.shape, (1 << n_qubits, 1 << n_qubits))

            for basis_index in range(1 << n_qubits):
                basis = np.zeros((1 << n_qubits, 1), dtype=np.complex64)
                basis[basis_index, 0] = 1.0
                local = apply_gate_to_state(gate, backend.cast(basis), n_qubits, backend)
                np.testing.assert_allclose(unitary @ basis, backend.to_numpy(local), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

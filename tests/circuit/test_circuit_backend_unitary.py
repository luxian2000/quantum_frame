import unittest

import numpy as np
import torch

from nexq import Circuit, TorchBackend, cnot, crz, hadamard, rx, rzz


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


if __name__ == "__main__":
    unittest.main()

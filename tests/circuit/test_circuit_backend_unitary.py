import unittest

import numpy as np
import torch

from nexq import Circuit, TorchBackend, cnot, hadamard, rx


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


if __name__ == "__main__":
    unittest.main()

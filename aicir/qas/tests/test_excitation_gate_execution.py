import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class ExcitationGateExecutionTests(unittest.TestCase):
    def test_single_excitation_matrix_is_trainable_givens_rotation(self):
        from aicir import NumpyBackend
        from aicir.core.gates import gate_to_matrix

        matrix = gate_to_matrix(
            {"type": "single_excitation", "qubit_1": 0, "qubit_2": 1, "parameter": 0.7},
            cir_qubits=2,
            backend=NumpyBackend(),
        )

        c, s = math.cos(0.35), math.sin(0.35)
        expected = np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
            dtype=complex,
        )
        self.assertTrue(np.allclose(matrix, expected))

    def test_double_excitation_matrix_couples_0011_and_1100(self):
        from aicir import NumpyBackend
        from aicir.core.gates import gate_to_matrix

        matrix = gate_to_matrix(
            {"type": "double_excitation", "qubits": [0, 1, 2, 3], "parameter": 0.8},
            cir_qubits=4,
            backend=NumpyBackend(),
        )

        c, s = math.cos(0.4), math.sin(0.4)
        expected = np.eye(16, dtype=complex)
        expected[3, 3] = c
        expected[3, 12] = -s
        expected[12, 3] = s
        expected[12, 12] = c
        self.assertTrue(np.allclose(matrix, expected))

    def test_apply_single_excitation_accepts_qubits_field_after_normalization(self):
        from aicir import NumpyBackend
        from aicir.core.gates import apply_gate_to_state

        backend = NumpyBackend()
        state = np.zeros((4, 1), dtype=np.complex64)
        state[1, 0] = 1.0

        updated = apply_gate_to_state(
            {"type": "single_excitation", "qubits": [0, 1], "parameter": 0.7},
            backend.cast(state),
            2,
            backend,
        )

        self.assertTrue(np.allclose(np.asarray(updated).reshape(4), [0.0, math.cos(0.35), math.sin(0.35), 0.0]))

if __name__ == "__main__":
    unittest.main()

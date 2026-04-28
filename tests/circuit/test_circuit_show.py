import io
import unittest
from contextlib import redirect_stdout

from nexq import Circuit, cnot, hadamard, rz, swap


class TestCircuitShow(unittest.TestCase):
    def test_show_prints_and_returns_ascii_diagram(self):
        circ = Circuit(
            hadamard(0),
            cnot(1, [0]),
            swap(0, 1),
            n_qubits=2,
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            diagram = circ.show()

        printed = buf.getvalue().strip()
        self.assertEqual(printed, diagram)
        self.assertIn("q0:", diagram)
        self.assertIn("q1:", diagram)
        self.assertIn("H", diagram)
        self.assertIn("●", diagram)
        self.assertIn("X", diagram)
        self.assertIn("x", diagram)
        self.assertIn("│", diagram)

    def test_show_handles_empty_circuit(self):
        circ = Circuit(n_qubits=3)
        diagram = circ.show(file=io.StringIO())

        self.assertIn("q0:", diagram)
        self.assertIn("q1:", diagram)
        self.assertIn("q2:", diagram)
        self.assertEqual(len(diagram.splitlines()), 5)

    def test_show_displays_rotation_angles(self):
        circ = Circuit(
            rz(0.5, 0),
            {"type": "crz", "target_qubit": 1, "control_qubits": [0], "parameter": 1.0, "control_states": [1]},
            {"type": "rzz", "qubit_1": 0, "qubit_2": 1, "parameter": 0.25},
            n_qubits=2,
        )

        diagram = circ.show(file=io.StringIO())

        self.assertIn("θ=0.500", diagram)
        self.assertIn("θ=1.000", diagram)
        self.assertIn("θ=0.250", diagram)

    def test_show_supports_all_gate_families_from_gates_py(self):
        circ = Circuit(
            {"type": "I", "n_qubits": 3},
            {"type": "u2", "target_qubit": 0, "parameter": [0.1, 0.2]},
            {"type": "u3", "target_qubit": 1, "parameter": [0.1, 0.2, 0.3]},
            {"type": "cy", "target_qubit": 2, "control_qubits": [1], "control_states": [1]},
            {"type": "rzz", "qubit_1": 0, "qubit_2": 2, "parameter": 0.4},
            {"type": "toffoli", "target_qubit": 2, "control_qubits": [0, 1], "control_states": [1, 1]},
            n_qubits=3,
        )

        diagram = circ.show(file=io.StringIO())

        self.assertIn("I", diagram)
        self.assertIn("U2", diagram)
        self.assertIn("U3", diagram)
        self.assertIn("Y", diagram)
        self.assertIn("ZZ", diagram)
        self.assertGreaterEqual(diagram.count("●"), 3)


if __name__ == "__main__":
    unittest.main()

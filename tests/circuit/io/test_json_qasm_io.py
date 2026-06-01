import os
import tempfile
import unittest

import numpy as np
import torch

from nexq import Circuit, Parameter, cnot, crx, cry, crz, hadamard, rx, rzz, swap, toffoli, u2, u3
from nexq.core.io.json_io import (
    circuit_from_json,
    circuit_to_json,
    load_circuit_json,
    save_circuit_json,
)
from nexq.core.io.qasm import (
    circuit_from_qasm,
    circuit_to_qasm,
    circuit_to_qasm3,
    load_circuit_qasm,
    save_circuit_qasm,
    save_circuit_qasm3,
)


class TestJsonQasmIO(unittest.TestCase):
    def setUp(self):
        self.circ = Circuit(
            hadamard(0),
            cnot(1, [0]),
            rx(np.pi / 3, 1),
            swap(0, 1),
            n_qubits=2,
        )

    def test_json_roundtrip_text(self):
        text = circuit_to_json(self.circ)
        rebuilt = circuit_from_json(text)

        self.assertEqual(rebuilt.n_qubits, self.circ.n_qubits)
        self.assertEqual(rebuilt.gates, self.circ.gates)

    def test_json_roundtrip_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "circuit.json")
            save_circuit_json(self.circ, path)
            rebuilt = load_circuit_json(path)
            self.assertEqual(rebuilt.n_qubits, 2)
            self.assertEqual(rebuilt.gates, self.circ.gates)

    def test_json_roundtrip_parameter_and_unitary_values(self):
        theta = Parameter("theta")
        matrix = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex64)
        circ = Circuit(
            rx(theta, 0),
            {"type": "unitary", "parameter": matrix, "n_qubits": 1},
            n_qubits=1,
        )

        rebuilt = circuit_from_json(circuit_to_json(circ))

        self.assertEqual(rebuilt.gates[0], rx(theta, 0))
        self.assertEqual(rebuilt.gates[1]["type"], "unitary")
        self.assertEqual(rebuilt.gates[1]["parameter"].dtype, matrix.dtype)
        np.testing.assert_allclose(rebuilt.gates[1]["parameter"], matrix)

    def test_qasm_roundtrip_text(self):
        qasm = circuit_to_qasm(self.circ)
        rebuilt = circuit_from_qasm(qasm)

        self.assertEqual(rebuilt.n_qubits, self.circ.n_qubits)
        self.assertEqual(len(rebuilt.gates), len(self.circ.gates))

        # 重点检查控制门和 swap 是否保真
        self.assertEqual(rebuilt.gates[1]["type"], "cx")
        self.assertEqual(rebuilt.gates[1]["control_qubits"], [0])
        self.assertEqual(rebuilt.gates[1]["target_qubit"], 1)
        self.assertEqual(rebuilt.gates[3]["type"], "swap")
        self.assertEqual(rebuilt.gates[3]["qubit_1"], 0)
        self.assertEqual(rebuilt.gates[3]["qubit_2"], 1)

        # 参数门数值允许微小误差
        self.assertAlmostEqual(rebuilt.gates[2]["parameter"], np.pi / 3, places=10)

    def test_qasm_roundtrip_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "circuit.qasm")
            save_circuit_qasm(self.circ, path)
            rebuilt = load_circuit_qasm(path)
            self.assertEqual(rebuilt.n_qubits, 2)
            self.assertEqual(len(rebuilt.gates), len(self.circ.gates))

    def test_qasm3_roundtrip_text(self):
        qasm3 = circuit_to_qasm(self.circ, version="3.0")
        self.assertIn("OPENQASM 3.0;", qasm3)
        self.assertIn("qubit[2] q;", qasm3)

        rebuilt = circuit_from_qasm(qasm3)
        self.assertEqual(rebuilt.n_qubits, self.circ.n_qubits)
        self.assertEqual(len(rebuilt.gates), len(self.circ.gates))
        self.assertAlmostEqual(rebuilt.gates[2]["parameter"], np.pi / 3, places=10)

    def test_qasm3_roundtrip_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "circuit_v3.qasm")
            save_circuit_qasm(self.circ, path, version="3.0")
            rebuilt = load_circuit_qasm(path)
            self.assertEqual(rebuilt.n_qubits, 2)
            self.assertEqual(len(rebuilt.gates), len(self.circ.gates))

    def test_qasm3_convenience_helpers(self):
        qasm3 = circuit_to_qasm3(self.circ)
        self.assertTrue(qasm3.startswith("OPENQASM 3.0;"))

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "helper_v3.qasm")
            save_circuit_qasm3(self.circ, path)
            rebuilt = load_circuit_qasm(path)
            self.assertEqual(rebuilt.n_qubits, 2)

    def test_qasm_extended_gates_export_and_import(self):
        ext = Circuit(
            {"type": "u2", "target_qubit": 0, "parameter": [np.pi / 7, np.pi / 9]},
            u3(np.pi / 3, np.pi / 5, np.pi / 8, target_qubit=1),
            crx(torch.tensor(np.pi / 4), target_qubit=1, control_qubits=[0]),
            cry(torch.tensor(np.pi / 6), target_qubit=2, control_qubits=[1]),
            crz(torch.tensor(np.pi / 10), target_qubit=0, control_qubits=[2]),
            toffoli(target_qubit=2, control_qubits=[0, 1]),
            n_qubits=3,
        )

        qasm2 = circuit_to_qasm(ext, version="2.0")
        self.assertIn("u2(", qasm2)
        self.assertIn("u3(", qasm2)
        self.assertIn("crx(", qasm2)
        self.assertIn("cry(", qasm2)
        self.assertIn("crz(", qasm2)
        self.assertIn("ccx ", qasm2)

        rebuilt = circuit_from_qasm(qasm2)
        self.assertEqual(rebuilt.n_qubits, 3)
        self.assertEqual(len(rebuilt.gates), 6)
        self.assertEqual(rebuilt.gates[-1]["type"], "toffoli")

    def test_u2_constructor_preserves_gate_type(self):
        gate = u2(np.pi / 7, np.pi / 9, target_qubit=0)

        self.assertEqual(gate["type"], "u2")
        self.assertEqual(gate["parameter"], [np.pi / 7, np.pi / 9])

        qasm2 = circuit_to_qasm(Circuit(gate, n_qubits=1), version="2.0")
        self.assertIn("u2(", qasm2)

    def test_public_rzz_constructor_builds_unitary(self):
        circ = Circuit(rzz(np.pi / 5, qubit_1=0, qubit_2=2), n_qubits=3)

        unitary = circ.unitary()
        rebuilt = circuit_from_qasm(circuit_to_qasm(circ, version="2.0"))

        self.assertEqual(circ.gates[0]["type"], "rzz")
        self.assertEqual(unitary.shape, (8, 8))
        self.assertEqual(rebuilt.gates[0]["type"], "rzz")
        self.assertEqual(rebuilt.gates[0]["qubit_1"], 0)
        self.assertEqual(rebuilt.gates[0]["qubit_2"], 2)
        self.assertAlmostEqual(rebuilt.gates[0]["parameter"], np.pi / 5)

    def test_qasm3_multi_register_and_u_alias(self):
        qasm3 = """OPENQASM 3.0;
include \"stdgates.inc\";
qubit[2] a;
qubit b;
bit[3] c;
u(pi/2,pi/3,pi/5) a[0];
cx a[0], b;
crz(pi/7) b, a[1];
ccx a[0], a[1], b;
c[0] = measure a[0];
"""
        circ = circuit_from_qasm(qasm3)
        self.assertEqual(circ.n_qubits, 3)
        self.assertEqual(len(circ.gates), 4)
        self.assertEqual(circ.gates[0]["type"], "u3")
        self.assertEqual(circ.gates[1]["type"], "cx")
        self.assertEqual(circ.gates[2]["type"], "crz")
        self.assertEqual(circ.gates[3]["type"], "toffoli")

    def test_qasm_import_phase_aliases(self):
        qasm2 = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[1];
p(pi/3) q[0];
u1(pi/6) q[0];
"""
        circ = circuit_from_qasm(qasm2)
        self.assertEqual(circ.n_qubits, 1)
        self.assertEqual(len(circ.gates), 2)
        self.assertEqual(circ.gates[0]["type"], "rz")
        self.assertEqual(circ.gates[1]["type"], "rz")

    def test_qasm3_multi_control_cry_decomposition_preserves_action(self):
        circ = Circuit(
            {
                "type": "cry",
                "target_qubit": 3,
                "control_qubits": [0, 1, 2],
                "control_states": [0, 1, 0],
                "parameter": np.pi / 6,
            },
            n_qubits=4,
        )

        qasm3 = circuit_to_qasm3(circ)
        self.assertIn("qubit[4] q;", qasm3)
        self.assertIn("qubit[2] anc;", qasm3)
        self.assertIn("ccx q[0],q[1],anc[0];", qasm3)
        self.assertRegex(qasm3, r"cry\((pi/6|0\.523598775598299)\) anc\[1\],q\[3\];")
        self.assertIn("x q[0];", qasm3)
        self.assertIn("x q[2];", qasm3)

        rebuilt = circuit_from_qasm(qasm3)
        self.assertEqual(rebuilt.n_qubits, 6)

        original_unitary = circ.unitary()
        rebuilt_unitary = rebuilt.unitary()
        ancilla_dim = 1 << (rebuilt.n_qubits - circ.n_qubits)

        for basis_index in range(1 << circ.n_qubits):
            basis_state = np.zeros(1 << circ.n_qubits, dtype=np.complex64)
            basis_state[basis_index] = 1.0
            expected = original_unitary @ basis_state

            embedded = np.zeros(1 << rebuilt.n_qubits, dtype=np.complex64)
            embedded[basis_index << (rebuilt.n_qubits - circ.n_qubits)] = 1.0
            actual = rebuilt_unitary @ embedded
            actual = actual.reshape(1 << circ.n_qubits, ancilla_dim)

            np.testing.assert_allclose(actual[:, 0], expected, atol=1e-6)
            np.testing.assert_allclose(actual[:, 1:], 0.0, atol=1e-6)

    def test_qasm3_single_control_cry_not_decomposed(self):
        circ = Circuit(
            {
                "type": "cry",
                "target_qubit": 1,
                "control_qubits": [0],
                "control_states": [1],
                "parameter": np.pi / 4,
            },
            n_qubits=2,
        )

        qasm3 = circuit_to_qasm3(circ)
        self.assertIn("qubit[2] q;", qasm3)
        self.assertNotIn("qubit[", "\n".join([line for line in qasm3.splitlines() if "anc" in line]))
        self.assertRegex(qasm3, r"cry\((pi/4|0\.785398163397448)\) q\[0\],q\[1\];")
        self.assertNotIn("ccx ", qasm3)

    def test_qasm3_multi_control_crx_crz_decomposition_preserves_action(self):
        for gate_name, angle in (("crx", np.pi / 7), ("crz", np.pi / 5)):
            circ = Circuit(
                {
                    "type": gate_name,
                    "target_qubit": 3,
                    "control_qubits": [0, 1, 2],
                    "control_states": [1, 0, 1],
                    "parameter": angle,
                },
                n_qubits=4,
            )

            qasm3 = circuit_to_qasm3(circ)
            self.assertIn("qubit[2] anc;", qasm3)
            self.assertIn(f"{gate_name}(", qasm3)
            self.assertIn("ccx q[0],q[1],anc[0];", qasm3)

            rebuilt = circuit_from_qasm(qasm3)
            self.assertEqual(rebuilt.n_qubits, 6)

            original_unitary = circ.unitary()
            rebuilt_unitary = rebuilt.unitary()
            ancilla_dim = 1 << (rebuilt.n_qubits - circ.n_qubits)

            for basis_index in range(1 << circ.n_qubits):
                basis_state = np.zeros(1 << circ.n_qubits, dtype=np.complex64)
                basis_state[basis_index] = 1.0
                expected = original_unitary @ basis_state

                embedded = np.zeros(1 << rebuilt.n_qubits, dtype=np.complex64)
                embedded[basis_index << (rebuilt.n_qubits - circ.n_qubits)] = 1.0
                actual = rebuilt_unitary @ embedded
                actual = actual.reshape(1 << circ.n_qubits, ancilla_dim)

                np.testing.assert_allclose(actual[:, 0], expected, atol=1e-6)
                np.testing.assert_allclose(actual[:, 1:], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

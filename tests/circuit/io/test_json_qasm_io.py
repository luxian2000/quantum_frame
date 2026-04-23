import os
import tempfile
import unittest

import numpy as np
import torch

from quantum_sim import Circuit, cnot, crx, cry, crz, hadamard, rx, swap, toffoli, u3
from quantum_sim.circuit.io.json_io import (
    circuit_from_json,
    circuit_to_json,
    load_circuit_json,
    save_circuit_json,
)
from quantum_sim.circuit.io.qasm import (
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


if __name__ == "__main__":
    unittest.main()

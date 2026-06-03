import unittest

from aicir.core import Circuit
from aicir.metrics.hardware import native_depth_twoq_efficiency_details
from aicir.metrics._utils import depth_proxy


class TestHardwareMetrics(unittest.TestCase):
    def test_depth_proxy_uses_swap_qubit_fields(self):
        circuit = Circuit(
            {"type": "swap", "qubit_1": 0, "qubit_2": 1},
            {"type": "rz", "target_qubit": 2, "parameter": 0.25},
            n_qubits=3,
        )

        self.assertEqual(depth_proxy(circuit), 1.0)
        self.assertEqual(native_depth_twoq_efficiency_details(circuit)["depth_proxy"], 1.0)

    def test_depth_proxy_uses_rzz_qubit_fields(self):
        circuit = Circuit(
            {"type": "rzz", "qubit_1": 0, "qubit_2": 2, "parameter": 0.5},
            {"type": "rx", "target_qubit": 1, "parameter": 0.25},
            {"type": "rz", "target_qubit": 2, "parameter": 0.75},
            n_qubits=3,
        )

        self.assertEqual(depth_proxy(circuit), 2.0)
        self.assertEqual(native_depth_twoq_efficiency_details(circuit)["depth_proxy"], 2.0)


if __name__ == "__main__":
    unittest.main()

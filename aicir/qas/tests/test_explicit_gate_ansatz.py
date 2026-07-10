import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class ExplicitGateAnsatzTests(unittest.TestCase):
    def test_explicit_gate_gene_round_trips_and_builds_architecture(self):
        from aicir.qas.library.ansatz import (
            ExplicitGateAnsatzGene,
            architecture_from_explicit_gate_gene,
        )

        gene = ExplicitGateAnsatzGene.from_jsonable(
            {
                "kind": "explicit_gate_sequence",
                "n_qubits": 2,
                "gates": [
                    {"type": "ry", "target_qubit": 0, "parameter": 0.1},
                    {"type": "cx", "control_qubits": [0], "control_states": [1], "target_qubit": 1},
                    {"type": "rzz", "qubit_1": 0, "qubit_2": 1, "parameter": 0.2},
                ],
            }
        )

        architecture = architecture_from_explicit_gate_gene(gene)

        self.assertEqual(gene.to_jsonable()["kind"], "explicit_gate_sequence")
        self.assertEqual(architecture.n_qubits, 2)
        self.assertEqual(architecture.parameter_count, 2)
        self.assertEqual(architecture.two_qubit_gate_count, 2)
        self.assertEqual(architecture.metadata["family"], "explicit_gate_sequence")

    def test_labeling_accepts_explicit_gate_gene_rows(self):
        from aicir.qas.vqe_loop.fair_labeling import _architecture_from_row

        row = {
            "architecture_id": "adapt_seed",
            "ansatz_gene": json.dumps(
                {
                    "kind": "explicit_gate_sequence",
                    "n_qubits": 2,
                    "gates": [
                        {"type": "rx", "target_qubit": 0, "parameter": 0.1},
                        {"type": "rxx", "qubit_1": 0, "qubit_2": 1, "parameter": 0.2},
                    ],
                }
            ),
        }

        architecture = _architecture_from_row(row)

        self.assertEqual(architecture.n_qubits, 2)
        self.assertEqual(architecture.parameter_count, 2)
        self.assertEqual(architecture.metadata["family"], "explicit_gate_sequence")


class OperatorSequenceAnsatzTests(unittest.TestCase):
    def test_operator_sequence_gene_round_trips_and_builds_one_parameter_per_operator(self):
        from aicir.qas.library.ansatz import (
            OperatorSequenceAnsatzGene,
            architecture_from_operator_sequence_gene,
        )

        gene = OperatorSequenceAnsatzGene.from_jsonable(
            {
                "kind": "operator_sequence",
                "n_qubits": 3,
                "operators": ["XII", "YYI", "IZZ"],
                "name": "adapt_like_seed",
            }
        )

        architecture = architecture_from_operator_sequence_gene(gene)

        self.assertEqual(gene.to_jsonable()["kind"], "operator_sequence")
        self.assertEqual(gene.layers, 3)
        self.assertEqual(architecture.n_qubits, 3)
        self.assertEqual(architecture.parameter_count, 3)
        self.assertGreaterEqual(architecture.two_qubit_gate_count, 4)
        self.assertEqual(architecture.metadata["family"], "operator_sequence")

    def test_labeling_accepts_operator_sequence_gene_rows(self):
        from aicir.qas.vqe_loop.fair_labeling import _architecture_from_row

        row = {
            "architecture_id": "operator_seed",
            "ansatz_gene": json.dumps(
                {
                    "kind": "operator_sequence",
                    "n_qubits": 2,
                    "operators": ["XI", "YY"],
                }
            ),
        }

        architecture = _architecture_from_row(row)

        self.assertEqual(architecture.n_qubits, 2)
        self.assertEqual(architecture.parameter_count, 2)
        self.assertEqual(architecture.metadata["family"], "operator_sequence")

    def test_cheap_eval_accepts_operator_sequence_gene_rows(self):
        from aicir.qas.vqe_loop.cheap_eval_experiment import _architecture_from_experiment_row

        row = {
            "architecture_id": "operator_seed",
            "ansatz_gene": json.dumps(
                {
                    "kind": "operator_sequence",
                    "n_qubits": 2,
                    "operators": ["ZI", "XX"],
                }
            ),
        }

        architecture = _architecture_from_experiment_row(row)

        self.assertEqual(architecture.parameter_count, 2)
        self.assertEqual(architecture.metadata["family"], "operator_sequence")


if __name__ == "__main__":
    unittest.main()

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene


def operator_row(operators, fair=None):
    gene = OperatorSequenceAnsatzGene(n_qubits=2, operators=tuple(operators))
    row = {
        "architecture_id": "_".join(operators),
        "family": "operator_sequence",
        "n_qubits": "2",
        "hamiltonian_terms": json.dumps([[-1.0, "XI"], [-0.5, "YY"]]),
        "ansatz_gene": json.dumps(gene.to_jsonable()),
    }
    if fair is not None:
        row["fair_best_energy"] = str(fair)
    return row


class VQETaskProxyTests(unittest.TestCase):
    def test_hamiltonian_overlap_rewards_matching_operator_sequence_terms(self):
        from aicir.qas.vqe_loop.task_proxy import hamiltonian_ansatz_overlap

        matched = operator_row(("XI",))
        unmatched = operator_row(("ZZ",))

        self.assertGreater(hamiltonian_ansatz_overlap(matched), hamiltonian_ansatz_overlap(unmatched))
        self.assertEqual(hamiltonian_ansatz_overlap(matched), 0.5)

    def test_task_proxy_evaluator_returns_ranking_score_and_features(self):
        from aicir.qas.vqe_loop.task_proxy import build_vqe_task_proxy_evaluator

        evaluator = build_vqe_task_proxy_evaluator(operator_pool=("XI", "YY", "ZZ"))
        result = evaluator(operator_row(("XI",)))

        self.assertIn("VQE_TASK_PROXY", result)
        self.assertLess(float(result["VQE_TASK_PROXY"]), 0.0)
        self.assertGreater(float(result["task_proxy_hamiltonian_overlap"]), 0.0)
        self.assertGreaterEqual(float(result["task_proxy_adapt_growth_potential"]), 0.0)

    def test_graph_feature_dataset_keeps_targets_and_feature_names(self):
        from aicir.qas.vqe_loop.graph_predictor import build_graph_feature_dataset

        rows = [operator_row(("XI",), fair=-1.0), operator_row(("YY",), fair=-0.5)]
        dataset = build_graph_feature_dataset(rows)

        self.assertEqual(dataset.row_count, 2)
        self.assertEqual(dataset.targets, [-1.0, -0.5])
        self.assertEqual(len(dataset.features), 2)
        self.assertEqual(len(dataset.features[0]), len(dataset.feature_names))
        self.assertIn("n_qubits", dataset.feature_names)

    def test_graph_predictor_cross_validation_reports_fold_metrics(self):
        from aicir.qas.vqe_loop.graph_predictor import cross_validate_graph_predictor

        rows = [
            operator_row(("XI",), fair=-1.0),
            operator_row(("YY",), fair=-0.7),
            operator_row(("ZZ",), fair=-0.2),
            operator_row(("XI", "YY"), fair=-1.4),
        ]

        result = cross_validate_graph_predictor(rows, k=2, alpha=0.1)

        self.assertEqual(result["folds"], 2)
        self.assertEqual(len(result["fold_metrics"]), 2)
        self.assertGreaterEqual(result["mean_mae"], 0.0)
        self.assertIn("mean_rmse", result)
    def test_graph_energy_predictor_fits_labeled_rows_and_predicts_child(self):
        from aicir.qas.vqe_loop.graph_predictor import GraphEnergyPredictor

        predictor = GraphEnergyPredictor()
        predictor.fit([operator_row(("XI",), fair=-1.0), operator_row(("ZZ",), fair=0.0)])
        prediction = predictor.predict_row(operator_row(("XI", "YY")))

        self.assertIsNotNone(prediction.prediction)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLess(float(prediction.prediction), 0.5)


if __name__ == "__main__":
    unittest.main()

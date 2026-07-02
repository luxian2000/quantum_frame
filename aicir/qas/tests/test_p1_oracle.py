import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.library.ansatz import ChemistryExcitationAnsatzGene, OperatorSequenceAnsatzGene, SupernetAnsatzGene


def make_gene(single_layers, two_layers):
    return SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=tuple(tuple(layer) for layer in single_layers),
        two_qubit_layers=tuple(tuple(layer) for layer in two_layers),
        two_qubit_pairs=((0, 1),),
    )


def row(architecture_id, gene, fair=None):
    payload = {
        "architecture_id": architecture_id,
        "ansatz_gene": json.dumps(gene.to_jsonable()),
    }
    if fair is not None:
        payload["fair_best_energy"] = str(fair)
    return payload



def make_chemistry_gene(excitations=None):
    return ChemistryExcitationAnsatzGene(
        n_qubits=4,
        hf_occupied_qubits=(1, 3),
        excitations=tuple(excitations or ({"type": "single_excitation", "qubits": [0, 1]},)),
        active_electrons=2,
        active_spatial_orbitals=2,
    )


def make_operator_gene(operators):
    return OperatorSequenceAnsatzGene(n_qubits=2, operators=tuple(operators))


class P1OracleTests(unittest.TestCase):
    def test_gene_aware_distance_orders_small_and_large_mutations(self):
        from aicir.qas.vqe_loop.oracle import gene_aware_distance

        parent = make_gene([("ry", "rz"), ("rx", "ry")], [("cx",), ("rzz",)])
        gate_child = make_gene([("rz", "rz"), ("rx", "ry")], [("cx",), ("rzz",)])
        layer_child = make_gene([("h", "h"), ("rx", "ry")], [("none",), ("rzz",)])
        depth_child = make_gene([("ry", "rz"), ("rx", "ry"), ("ry", "ry")], [("cx",), ("rzz",), ("cx",)])

        gate_distance = gene_aware_distance(parent, gate_child)
        layer_distance = gene_aware_distance(parent, layer_child)
        depth_distance = gene_aware_distance(parent, depth_child)

        self.assertGreater(gate_distance, 0.0)
        self.assertLess(gate_distance, layer_distance)
        self.assertLess(gate_distance, depth_distance)

    def test_oracle_exact_match_reuses_fair_label(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        parent = make_gene([("ry", "rz")], [("cx",)])
        prediction = predict_fair_energy(
            row("same", parent),
            labeled_rows=[row("labeled", parent, fair=-2.5)],
            k_min=1,
            d_max=0.1,
        )

        self.assertTrue(prediction.trusted)
        self.assertEqual(prediction.prediction, -2.5)
        self.assertEqual(prediction.reason, "exact_match")

    def test_oracle_abstains_when_neighbors_are_insufficient(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        parent = make_gene([("ry", "rz")], [("cx",)])
        child = make_gene([("rz", "rz")], [("cx",)])
        prediction = predict_fair_energy(
            row("child", child),
            labeled_rows=[row("parent", parent, fair=-2.0)],
            k_min=2,
            d_max=1.0,
        )

        self.assertFalse(prediction.trusted)
        self.assertIsNone(prediction.prediction)
        self.assertEqual(prediction.reason, "insufficient_neighbors")

    def test_oracle_predicts_weighted_fair_energy_inside_trust_region(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        parent = make_gene([("ry", "rz")], [("cx",)])
        child = make_gene([("rz", "rz")], [("cx",)])
        sibling = make_gene([("rx", "rz")], [("cx",)])
        far = make_gene([("h", "h")], [("rzz",)])

        prediction = predict_fair_energy(
            row("child", child),
            labeled_rows=[
                row("parent", parent, fair=-2.0),
                row("sibling", sibling, fair=-1.0),
                row("far", far, fair=1.0),
            ],
            k_min=2,
            d_max=1.0,
        )

        self.assertTrue(prediction.trusted)
        self.assertEqual(prediction.neighbor_count, 3)
        self.assertLess(prediction.prediction, -1.0)
        self.assertGreater(prediction.prediction, -2.1)

    def test_oracle_abstains_when_neighbor_fair_labels_are_high_variance(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        child = make_gene([("rz", "rz")], [("cx",)])
        near_a = make_gene([("ry", "rz")], [("cx",)])
        near_b = make_gene([("rx", "rz")], [("cx",)])
        near_c = make_gene([("h", "rz")], [("cx",)])

        prediction = predict_fair_energy(
            row("child", child),
            labeled_rows=[
                row("near_a", near_a, fair=-2.0),
                row("near_b", near_b, fair=0.0),
                row("near_c", near_c, fair=2.0),
            ],
            k_min=3,
            d_max=1.0,
            max_neighbor_std=0.5,
        )

        self.assertFalse(prediction.trusted)
        self.assertIsNone(prediction.prediction)
        self.assertEqual(prediction.reason, "high_neighbor_variance")
        self.assertGreater(prediction.neighbor_target_std, 0.5)

    def test_oracle_prediction_records_mutation_type_for_calibration(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        parent = make_gene([("ry", "rz")], [("cx",)])
        child = row("child", parent)
        child["mutation_type"] = "connectivity_mutation"

        prediction = predict_fair_energy(
            child,
            labeled_rows=[row("parent", parent, fair=-2.0)],
            k_min=1,
            d_max=0.1,
        )

        self.assertEqual(prediction.mutation_type, "connectivity_mutation")
        self.assertEqual(prediction.to_jsonable()["mutation_type"], "connectivity_mutation")

    def test_operator_sequence_exact_match_reuses_fair_label(self):
        from aicir.qas.vqe_loop.oracle import predict_fair_energy

        parent = make_operator_gene(("XI", "YY"))
        prediction = predict_fair_energy(
            row("same", parent),
            labeled_rows=[row("labeled", parent, fair=-1.25)],
            k_min=1,
            d_max=0.1,
        )

        self.assertTrue(prediction.trusted)
        self.assertEqual(prediction.reason, "exact_match")
        self.assertEqual(prediction.prediction, -1.25)

    def test_operator_sequence_insert_neighbor_is_inside_trust_region(self):
        from aicir.qas.vqe_loop.oracle import gene_aware_distance, predict_fair_energy

        parent = make_operator_gene(("XI",))
        child = make_operator_gene(("XI", "YY"))
        self.assertLess(gene_aware_distance(parent, child), 0.6)

        prediction = predict_fair_energy(
            row("child", child),
            labeled_rows=[row("parent", parent, fair=-0.75)],
            k_min=1,
            d_max=0.6,
        )

        self.assertTrue(prediction.trusted)
        self.assertEqual(prediction.reason, "trusted_knn")
        self.assertEqual(prediction.prediction, -0.75)

    def test_oracle_treats_supernet_and_operator_sequence_as_different_families(self):
        from aicir.qas.vqe_loop.oracle import gene_aware_distance, predict_fair_energy

        supernet = make_gene([("ry", "rz")], [("cx",)])
        operator = make_operator_gene(("XI",))
        self.assertEqual(gene_aware_distance(supernet, operator), 1.0)

        prediction = predict_fair_energy(
            row("operator", operator),
            labeled_rows=[row("supernet", supernet, fair=-2.0)],
            k_min=1,
            d_max=0.2,
        )

        self.assertFalse(prediction.trusted)
        self.assertEqual(prediction.reason, "insufficient_neighbors")


    def test_chemistry_excitation_exact_match_reuses_fair_label(self):
        from aicir.qas.vqe_loop.oracle import gene_aware_distance, predict_fair_energy

        parent = make_chemistry_gene()
        child = make_chemistry_gene(({"type": "single_excitation", "qubits": [2, 3]},))
        self.assertGreater(gene_aware_distance(parent, child), 0.0)
        self.assertLess(gene_aware_distance(parent, child), 1.0)

        prediction = predict_fair_energy(
            row("same", parent),
            labeled_rows=[row("labeled", parent, fair=-1.75)],
            k_min=1,
            d_max=0.1,
        )

        self.assertTrue(prediction.trusted)
        self.assertEqual(prediction.reason, "exact_match")
        self.assertEqual(prediction.prediction, -1.75)
if __name__ == "__main__":
    unittest.main()


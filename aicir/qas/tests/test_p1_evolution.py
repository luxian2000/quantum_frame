import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.primitives.ansatz import ChemistryExcitationAnsatzGene, OperatorSequenceAnsatzGene, SupernetAnsatzGene


def make_gene(layers=2):
    single_layers = tuple(("ry", "rz") for _ in range(layers))
    two_layers = tuple(("cx",) for _ in range(layers))
    return SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=single_layers,
        two_qubit_layers=two_layers,
        two_qubit_pairs=((0, 1),),
    )


def make_custom_gene(single_layers, two_layers):
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
        "canonical_arch_hash": json.dumps(gene.to_jsonable(), sort_keys=True),
        "n_qubits": str(gene.n_qubits),
        "depth_group": f"L{gene.layers}",
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


def make_operator_gene(operators=("XI", "YY")):
    return OperatorSequenceAnsatzGene(n_qubits=2, operators=tuple(operators))


class P1VariationTests(unittest.TestCase):
    def test_select_parent_rows_prefers_low_fair_energy_labels(self):
        from aicir.qas.vqe_loop.p1_evolution import select_parent_rows

        gene = make_gene()
        parents = select_parent_rows(
            [
                row("bad", gene, fair=-1.0),
                row("unlabeled", gene),
                row("best", gene, fair=-3.0),
                row("middle", gene, fair=-2.0),
            ],
            count=2,
        )

        self.assertEqual([item["architecture_id"] for item in parents], ["best", "middle"])

    def test_select_parent_rows_can_include_diverse_parent(self):
        from aicir.qas.vqe_loop.p1_evolution import select_parent_rows

        best = make_custom_gene([("ry", "rz"), ("ry", "rz")], [("cx",), ("cx",)])
        near = make_custom_gene([("rz", "rz"), ("ry", "rz")], [("cx",), ("cx",)])
        far = make_custom_gene([("h", "h"), ("rx", "rx")], [("rzz",), ("rzz",)])

        parents = select_parent_rows(
            [
                row("best", best, fair=-3.0),
                row("near", near, fair=-2.9),
                row("far", far, fair=-1.0),
            ],
            count=2,
            diversity_count=1,
        )

        self.assertEqual([item["architecture_id"] for item in parents], ["best", "far"])

    def test_gate_mutation_changes_one_single_qubit_slot(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_gene(layers=2)
        result = mutate_gene(parent, mutation_type="gate_mutation", seed=3)

        self.assertEqual(result.mutation_type, "gate_mutation")
        self.assertEqual(result.child.layers, parent.layers)
        differences = sum(
            left != right
            for left_layer, right_layer in zip(parent.single_qubit_layers, result.child.single_qubit_layers)
            for left, right in zip(left_layer, right_layer)
        )
        self.assertEqual(differences, 1)
        self.assertEqual(parent.two_qubit_layers, result.child.two_qubit_layers)

    def test_connectivity_mutation_changes_one_two_qubit_slot(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_gene(layers=3)
        result = mutate_gene(parent, mutation_type="connectivity_mutation", seed=5)

        self.assertEqual(result.mutation_type, "connectivity_mutation")
        self.assertEqual(result.child.layers, parent.layers)
        differences = sum(
            left != right
            for left_layer, right_layer in zip(parent.two_qubit_layers, result.child.two_qubit_layers)
            for left, right in zip(left_layer, right_layer)
        )
        self.assertEqual(differences, 1)
        self.assertEqual(parent.single_qubit_layers, result.child.single_qubit_layers)

    def test_layer_mutation_replaces_one_layer_without_changing_depth(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_gene(layers=3)
        result = mutate_gene(parent, mutation_type="layer_mutation", seed=7)

        self.assertEqual(result.mutation_type, "layer_mutation")
        self.assertEqual(result.child.layers, parent.layers)
        changed_layers = [
            index
            for index, (left_single, right_single, left_two, right_two) in enumerate(
                zip(
                    parent.single_qubit_layers,
                    result.child.single_qubit_layers,
                    parent.two_qubit_layers,
                    result.child.two_qubit_layers,
                )
            )
            if left_single != right_single or left_two != right_two
        ]
        self.assertEqual(len(changed_layers), 1)

    def test_depth_mutation_changes_depth_by_one(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_gene(layers=2)
        result = mutate_gene(parent, mutation_type="depth_mutation", seed=11, min_layers=1, max_layers=4)

        self.assertEqual(result.mutation_type, "depth_mutation")
        self.assertEqual(abs(result.child.layers - parent.layers), 1)

    def test_depth_mutation_raises_when_depth_cannot_change(self):
        from aicir.qas.vqe_loop.p1_evolution import MutationUnavailable, mutate_gene

        parent = make_gene(layers=2)

        with self.assertRaises(MutationUnavailable):
            mutate_gene(parent, mutation_type="depth_mutation", seed=11, min_layers=2, max_layers=2)

    def test_generate_mutation_children_retries_when_depth_mutation_is_unavailable(self):
        from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children

        parent_gene = make_gene(layers=2)
        parent = row("parent", parent_gene, fair=-2.0)
        children = generate_mutation_children(
            [parent],
            children_per_parent=1,
            mutation_types=("depth_mutation", "gate_mutation"),
            seed=13,
            min_layers=2,
            max_layers=2,
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["mutation_type"], "gate_mutation")
        self.assertNotEqual(children[0]["canonical_arch_hash"], json.dumps(parent_gene.to_jsonable(), sort_keys=True))

    def test_layer_crossover_splices_two_parents(self):
        from aicir.qas.vqe_loop.p1_evolution import layer_crossover

        left = make_custom_gene([("ry", "rz"), ("rx", "rx"), ("h", "h")], [("cx",), ("none",), ("cx",)])
        right = make_custom_gene([("rz", "rz"), ("h", "h"), ("ry", "ry")], [("rzz",), ("rzz",), ("none",)])

        result = layer_crossover(left, right, cut=1)

        self.assertEqual(result.mutation_type, "layer_crossover")
        self.assertEqual(result.child.single_qubit_layers[0], left.single_qubit_layers[0])
        self.assertEqual(result.child.single_qubit_layers[1:], right.single_qubit_layers[1:])
        self.assertEqual(result.child.two_qubit_layers[0], left.two_qubit_layers[0])
        self.assertEqual(result.child.two_qubit_layers[1:], right.two_qubit_layers[1:])

    def test_generate_mutation_children_outputs_rows_with_parent_and_type(self):
        from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children

        parent = row("parent", make_gene(layers=2), fair=-2.0)
        children = generate_mutation_children(
            [parent],
            children_per_parent=2,
            mutation_types=("gate_mutation", "connectivity_mutation"),
            seed=13,
        )

        self.assertEqual(len(children), 2)
        self.assertEqual({child["parent_architecture_id"] for child in children}, {"parent"})
        self.assertEqual({child["mutation_type"] for child in children}, {"gate_mutation", "connectivity_mutation"})
        for child in children:
            self.assertIn("ansatz_gene", child)
            self.assertIn("canonical_arch_hash", child)
            self.assertTrue(child["architecture_id"].startswith("p1_child_"))

    def test_generate_mutation_children_supports_weighted_mutation_choice(self):
        from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children

        parent = row("parent", make_gene(layers=2), fair=-2.0)
        children = generate_mutation_children(
            [parent],
            children_per_parent=3,
            mutation_types=("gate_mutation", "connectivity_mutation"),
            mutation_weights={"gate_mutation": 1.0, "connectivity_mutation": 0.0},
            seed=13,
        )

        self.assertEqual([child["mutation_type"] for child in children], ["gate_mutation", "gate_mutation", "gate_mutation"])

    def test_operator_insert_adds_pool_operator(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_operator_gene(("XI",))
        result = mutate_gene(
            parent,
            mutation_type="operator_insert",
            operator_pool=("YY",),
            seed=3,
        )

        self.assertEqual(result.mutation_type, "operator_insert")
        self.assertEqual(result.child.layers, 2)
        self.assertIn("YY", result.child.operators)

    def test_operator_delete_removes_one_operator(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_operator_gene(("XI", "YY", "ZZ"))
        result = mutate_gene(parent, mutation_type="operator_delete", seed=5)

        self.assertEqual(result.mutation_type, "operator_delete")
        self.assertEqual(result.child.layers, 2)
        self.assertTrue(set(result.child.operators).issubset(set(parent.operators)))

    def test_operator_swap_changes_order_only(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_operator_gene(("XI", "YY", "ZZ"))
        result = mutate_gene(parent, mutation_type="operator_swap", seed=1)

        self.assertEqual(result.mutation_type, "operator_swap")
        self.assertNotEqual(result.child.operators, parent.operators)
        self.assertEqual(sorted(result.child.operators), sorted(parent.operators))

    def test_generate_mutation_children_outputs_operator_sequence_rows(self):
        from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children

        parent = row("operator_parent", make_operator_gene(("XI",)), fair=-1.0)
        parent["n_params"] = "1"
        parent["two_q_count"] = "0"
        children = generate_mutation_children(
            [parent],
            children_per_parent=1,
            mutation_types=("operator_insert",),
            operator_pool=("YY",),
            seed=7,
        )

        child_gene = OperatorSequenceAnsatzGene.from_jsonable(json.loads(children[0]["ansatz_gene"]))
        self.assertEqual(children[0]["mutation_type"], "operator_insert")
        self.assertEqual(children[0]["depth_group"], "L2")
        self.assertEqual(children[0]["family"], "operator_sequence")
        self.assertEqual(children[0]["n_params"], "2")
        self.assertEqual(children[0]["two_q_count"], "2")
        self.assertEqual(child_gene.layers, 2)
        self.assertEqual(set(child_gene.operators), {"XI", "YY"})

    def test_operator_adapt_growth_scans_pool_and_picks_best_scored_operator(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_operator_gene(("XI",))
        seen = []

        def scorer(gene, candidate_operator):
            seen.append((tuple(gene.operators), candidate_operator))
            return {"YY": -0.2, "ZZ": -1.5, "XX": -0.7}[candidate_operator]

        result = mutate_gene(
            parent,
            mutation_type="operator_adapt_growth",
            operator_pool=("YY", "ZZ", "XX"),
            operator_growth_evaluator=scorer,
        )

        self.assertEqual(result.mutation_type, "operator_adapt_growth")
        self.assertEqual(result.child.operators, ("XI", "ZZ"))
        self.assertEqual([item[1] for item in seen], ["YY", "ZZ", "XX"])

    def test_operator_adapt_growth_avoids_repeating_existing_operator_when_scored_best(self):
        from aicir.qas.vqe_loop.p1_evolution import mutate_gene

        parent = make_operator_gene(("XI",))

        def scorer(_gene, candidate_operator):
            return {"XI": -10.0, "YY": -1.0, "ZZ": -0.5}[candidate_operator]

        result = mutate_gene(
            parent,
            mutation_type="operator_adapt_growth",
            operator_pool=("XI", "YY", "ZZ"),
            operator_growth_evaluator=scorer,
        )

        self.assertEqual(result.child.operators, ("XI", "YY"))

    def test_operator_growth_evaluator_uses_parent_best_parameters_for_finite_difference(self):
        from aicir.qas.vqe_loop.p1_evolution import _operator_growth_evaluator_from_row

        parent = row("operator_parent", make_operator_gene(("XI",)), fair=-1.0)
        parent["hamiltonian_terms"] = json.dumps([[1.0, "ZI"]])
        parent["best_trace"] = json.dumps([{"best_parameters": [1.25]}])
        evaluator = _operator_growth_evaluator_from_row(parent)
        calls = []

        def fake_energy(_architecture, _problem, *, parameters):
            calls.append(list(parameters))
            return parameters[-1] * 2.0

        with patch("aicir.qas.vqe_loop.fair_vqe.evaluate_vqe_energy", side_effect=fake_energy):
            score = evaluator(make_operator_gene(("XI",)), "YY")

        self.assertEqual(calls, [[1.25, 0.05], [1.25, -0.05]])
        self.assertAlmostEqual(score, -2.0)


    def test_operator_growth_evaluator_lives_in_p1_evolution(self):
        from aicir.qas.vqe_loop.p1_evolution import _operator_growth_evaluator_from_row

        parent = row("operator_parent", make_operator_gene(("XI",)), fair=-1.0)
        parent["hamiltonian_terms"] = json.dumps([[1.0, "ZI"]])

        self.assertIsNotNone(_operator_growth_evaluator_from_row(parent))


    def test_generate_mutation_children_outputs_chemistry_excitation_rows(self):
        from aicir.qas.vqe_loop.p1_evolution import generate_mutation_children

        parent = row("chem_parent", make_chemistry_gene(), fair=-1.0)
        children = generate_mutation_children(
            [parent],
            children_per_parent=1,
            mutation_types=("chemistry_insert",),
            seed=7,
        )

        child_gene = ChemistryExcitationAnsatzGene.from_jsonable(json.loads(children[0]["ansatz_gene"]))
        self.assertEqual(children[0]["mutation_type"], "chemistry_insert")
        self.assertEqual(children[0]["family"], "chemistry_excitation")
        self.assertEqual(children[0]["depth_group"], "L2")
        self.assertEqual(children[0]["n_params"], "2")
        self.assertEqual(child_gene.layers, 2)
        self.assertEqual(child_gene.hf_occupied_qubits, (1, 3))
if __name__ == "__main__":
    unittest.main()



import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class ProblemAwareSamplingTests(unittest.TestCase):
    def test_pauli_coupling_weights_sum_two_body_activity(self):
        from aicir.qas.vqe_loop.p0_problem_aware import pauli_coupling_weights, prioritized_two_qubit_pairs

        terms = [(-1.0, "ZZII"), (-0.5, "IZZI"), (-0.25, "XIIX")]

        weights = pauli_coupling_weights(terms)

        self.assertEqual(weights[(0, 1)], 1.0)
        self.assertEqual(weights[(1, 2)], 0.5)
        self.assertEqual(weights[(0, 3)], 0.25)
        self.assertEqual(
            prioritized_two_qubit_pairs(terms, candidate_pairs=((1, 2), (0, 3), (0, 1))),
            ((0, 1), (1, 2), (0, 3)),
        )

    def test_problem_aware_sampler_activates_high_weight_pairs(self):
        from aicir.qas.vqe_loop.p0_problem_aware import sample_problem_aware_supernet_gene

        gene = sample_problem_aware_supernet_gene(
            n_qubits=3,
            depth=3,
            pairs=((0, 1), (1, 2)),
            hamiltonian_terms=[(-1.0, "ZZI")],
            seed=7,
            entangler_probability_floor=0.0,
        )

        self.assertTrue(all(layer[0] != "none" for layer in gene.two_qubit_layers))
        self.assertTrue(all(layer[1] == "none" for layer in gene.two_qubit_layers))


if __name__ == "__main__":
    unittest.main()


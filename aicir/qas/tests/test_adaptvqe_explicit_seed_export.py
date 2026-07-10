import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class AdaptVQEExplicitSeedExportTests(unittest.TestCase):
    def test_edge_two_local_pool_matches_heisenberg6_cycle_size(self):
        from aicir.qas.demos.export_adaptvqe_explicit_seed import build_operator_pool_paulis

        terms = []
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)):
            for axis in "XYZ":
                pauli = ["I"] * 6
                pauli[left] = axis
                pauli[right] = axis
                terms.append((1.0, "".join(pauli)))

        pool = build_operator_pool_paulis(terms, pool_kind="edge_two_local")

        self.assertEqual(len(pool), 72)
        self.assertEqual(len(set(pool)), 72)
        self.assertIn("XIIIII", pool)
        self.assertIn("XXIIII", pool)

    def test_support_generic_pool_uses_unique_hamiltonian_supports(self):
        from aicir.qas.demos.export_adaptvqe_explicit_seed import build_operator_pool_paulis

        terms = [
            (1.0, "XXI"),
            (1.0, "YYI"),
            (1.0, "IZZ"),
        ]

        pool = build_operator_pool_paulis(terms, pool_kind="support_generic")

        self.assertEqual(len(pool), 27)
        self.assertEqual(len(set(pool)), 27)
        self.assertIn("XYI", pool)
        self.assertIn("IYX", pool)

    def test_support_generic_pool_matches_random6b_legacy_size(self):
        from aicir.qas.demos.export_adaptvqe_explicit_seed import build_operator_pool_paulis

        terms = [
            (0.2688278688486707, "IIIIXI"),
            (0.044751541190802636, "IIIIXY"),
            (-0.19546657170671405, "IIIIZI"),
            (0.07913139701716122, "IIIIZX"),
            (0.036639695600324027, "IIIXYZ"),
            (-0.04785739812809809, "IIIYII"),
            (-0.16836626246080086, "IIIZIX"),
            (0.06079777831213455, "IIIZIY"),
            (-0.04214393141052272, "IIXIII"),
            (-0.09821504306448303, "IIXIIX"),
            (0.13801605795765365, "IIXIXI"),
            (-0.12235883113980893, "IIYZYI"),
            (0.17894091790401626, "IIZIII"),
            (-0.04971540881644459, "IIZIIX"),
            (-0.10441144984627891, "IXIZIX"),
            (0.04322446600744428, "IXYYII"),
            (0.05208264489324117, "IZYZII"),
            (0.10702260154315482, "XIIIYI"),
            (0.22831878756105015, "XIIIZZ"),
            (0.10635563825634936, "XIIZII"),
            (-0.306061751036585, "YIIIII"),
            (0.07025784890462769, "YIIXII"),
            (0.20819862948692108, "YIIZIZ"),
            (-0.00035965975961671254, "ZIIIYI"),
        ]

        pool = build_operator_pool_paulis(terms, pool_kind="support_generic")

        self.assertEqual(len(pool), 133)

    def test_operator_sequence_seed_row_uses_selected_adapt_paulis(self):
        from aicir.qas.demos.export_adaptvqe_explicit_seed import build_operator_sequence_seed_row
        from aicir.qas.library.ansatz import OperatorSequenceAnsatzGene

        row = build_operator_sequence_seed_row(
            selected_paulis=("XI", "YY"),
            n_qubits=2,
            architecture_id="adapt_operator_seed",
            batch_id="adapt_batch",
            protocol_version="fair_vqe_protocol_v2",
            hamiltonian_id="toy",
            hamiltonian_class="pauli_terms",
            terms=((1.0, "ZI"), (-0.5, "XX")),
            reference_energy=-1.0,
            screening_energy=-0.75,
        )

        gene = OperatorSequenceAnsatzGene.from_jsonable(row["ansatz_gene"])
        self.assertEqual(gene.operators, ("XI", "YY"))
        self.assertEqual(row["family"], "operator_sequence")
        self.assertEqual(row["source"], "initial_train")
        self.assertEqual(row["screening_energy_is_final_label"], "false")


if __name__ == "__main__":
    unittest.main()

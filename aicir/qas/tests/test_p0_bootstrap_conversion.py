import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.primitives.ansatz import SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS


def make_gene():
    return SupernetAnsatzGene(
        n_qubits=2,
        single_qubit_layers=(("ry", "rz"),),
        two_qubit_layers=(("cx",),),
        two_qubit_pairs=((0, 1),),
    )


class P0BootstrapConversionTests(unittest.TestCase):
    def test_convert_p0_rows_maps_fair_high_and_preserves_hamiltonian_terms(self):
        from aicir.qas.demos.convert_p0_rows_to_bootstrap_labels import convert_p0_rows

        gene = make_gene()
        rows = [
            {
                "problem_id": "toy_h2",
                "architecture_id": "p0_arch_001",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": "2",
                "depth": "1",
                "fair_high": "-1.2345",
                "reference_energy": "-1.5",
                "hamiltonian_class": "molecular_preset",
            }
        ]
        terms = ((-1.0, "ZI"), (-0.5, "IZ"))

        converted = convert_p0_rows(
            rows,
            preset="toy_h2",
            terms=terms,
            n_qubits=2,
            reference_energy=-1.5,
        )

        self.assertEqual(len(converted), 1)
        row = converted[0]
        self.assertEqual(set(row), set(BENCHMARK_TABLE_FIELDS))
        self.assertEqual(row["fair_best_energy"], "-1.234500000000")
        self.assertEqual(row["label_status"], "completed")
        self.assertEqual(row["source"], "initial_train")
        self.assertEqual(row["protocol_version"], "fair_vqe_protocol_v2")
        self.assertEqual(row["hamiltonian_id"], "toy_h2")
        self.assertEqual(json.loads(row["hamiltonian_terms"]), [[-1.0, "ZI"], [-0.5, "IZ"]])
        self.assertNotEqual(row["canonical_arch_hash"], "")

    def test_cli_writes_benchmark_table_csv(self):
        from aicir.qas.demos import convert_p0_rows_to_bootstrap_labels as converter

        gene = make_gene()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "rows.csv"
            output_path = root / "bootstrap_labels.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["problem_id", "architecture_id", "ansatz_gene", "n_qubits", "depth", "fair_high"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "problem_id": "toy_h2",
                        "architecture_id": "p0_arch_001",
                        "ansatz_gene": json.dumps(gene.to_jsonable()),
                        "n_qubits": "2",
                        "depth": "1",
                        "fair_high": "-1.2345",
                    }
                )

            result = converter.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--preset",
                    "toy_h2",
                ],
                hamiltonian_loader=lambda *_args, **_kwargs: (((-1.0, "ZI"), (-0.5, "IZ")), 2, -1.5),
            )

            with output_path.open(newline="", encoding="utf-8-sig") as handle:
                output_rows = list(csv.DictReader(handle))

        self.assertEqual(result["written"], 1)
        self.assertEqual(output_rows[0]["fair_best_energy"], "-1.234500000000")
        self.assertEqual(output_rows[0]["hamiltonian_terms"], json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]))


if __name__ == "__main__":
    unittest.main()

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class ChemistryExcitationAnsatzTests(unittest.TestCase):
    def test_closed_shell_excitation_pools_match_beh2_convention(self):
        from aicir.qas.vqe_loop.p0_chemistry_excitation import closed_shell_excitation_pools

        hf, singles, doubles = closed_shell_excitation_pools(
            active_electrons=2,
            active_spatial_orbitals=3,
        )

        self.assertEqual(hf, (2, 5))
        self.assertEqual(singles, ((4, 5), (1, 2), (3, 5), (0, 2)))
        self.assertEqual(doubles, ((1, 4, 2, 5), (0, 3, 2, 5)))

    def test_chemistry_excitation_gene_round_trips_and_builds_architecture(self):
        from aicir.qas.library.ansatz import (
            ChemistryExcitationAnsatzGene,
            architecture_from_chemistry_excitation_gene,
        )

        gene = ChemistryExcitationAnsatzGene.from_jsonable(
            {
                "kind": "chemistry_excitation",
                "n_qubits": 4,
                "hf_occupied_qubits": [1, 3],
                "excitations": [
                    {"type": "single_excitation", "qubits": [0, 1]},
                    {"type": "double_excitation", "qubits": [0, 2, 1, 3]},
                ],
                "active_electrons": 2,
                "active_spatial_orbitals": 2,
            }
        )

        architecture = architecture_from_chemistry_excitation_gene(gene)

        self.assertEqual(gene.to_jsonable()["kind"], "chemistry_excitation")
        self.assertEqual(gene.layers, 2)
        self.assertEqual(architecture.n_qubits, 4)
        self.assertEqual(architecture.parameter_count, 2)
        self.assertEqual(architecture.two_qubit_gate_count, 2)
        self.assertEqual(architecture.metadata["family"], "chemistry_excitation")
        from aicir.ir import instruction_name

        self.assertEqual(
            [instruction_name(gate) for gate in architecture.circuit.gates],
            ["pauli_x", "pauli_x", "single_excitation", "double_excitation"],
        )

    def test_public_candidate_row_parser_accepts_chemistry_excitation_rows(self):
        from aicir.qas.vqe_loop.benchmark_table import architecture_from_candidate_row

        row = {
            "architecture_id": "chem_seed",
            "ansatz_gene": json.dumps(
                {
                    "kind": "chemistry_excitation",
                    "n_qubits": 2,
                    "hf_occupied_qubits": [1],
                    "excitations": [{"type": "single_excitation", "qubits": [0, 1]}],
                    "active_electrons": 1,
                    "active_spatial_orbitals": 1,
                }
            ),
        }

        architecture = architecture_from_candidate_row(dict(row))

        self.assertEqual(architecture.metadata["family"], "chemistry_excitation")
        self.assertEqual(architecture.parameter_count, 1)

    def test_labeling_and_cheap_eval_accept_chemistry_excitation_rows(self):
        from aicir.qas.vqe_loop.cheap_eval_experiment import _architecture_from_experiment_row
        from aicir.qas.vqe_loop.fair_labeling import _architecture_from_row

        row = {
            "architecture_id": "chem_seed",
            "ansatz_gene": json.dumps(
                {
                    "kind": "chemistry_excitation",
                    "n_qubits": 2,
                    "hf_occupied_qubits": [1],
                    "excitations": [{"type": "single_excitation", "qubits": [0, 1]}],
                    "active_electrons": 1,
                    "active_spatial_orbitals": 1,
                }
            ),
        }

        labeled_architecture = _architecture_from_row(dict(row))
        cheap_architecture = _architecture_from_experiment_row(dict(row))

        self.assertEqual(labeled_architecture.metadata["family"], "chemistry_excitation")
        self.assertEqual(cheap_architecture.parameter_count, 1)

    def test_build_chemistry_excitation_rows_emit_benchmark_schema(self):
        from aicir.qas.vqe_loop.p0_chemistry_excitation import build_chemistry_excitation_rows
        from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS

        rows, summary = build_chemistry_excitation_rows(
            active_electrons=2,
            active_spatial_orbitals=2,
            hamiltonian_id="toy_h2",
            hamiltonian_class="molecular",
            count=3,
            max_excitations=2,
            seed=7,
        )

        self.assertEqual(len(rows), 3)
        self.assertIs(summary["enabled"], True)
        self.assertEqual(summary["family"], "chemistry_excitation")
        self.assertEqual(summary["single_excitation_count"], 2)
        self.assertEqual(summary["double_excitation_count"], 1)
        for row in rows:
            self.assertTrue(set(row).issubset(set(BENCHMARK_TABLE_FIELDS)))
            self.assertEqual(row["family"], "chemistry_excitation")
            self.assertEqual(row["source"], "trackB_chemistry_excitation")
            self.assertEqual(row["screening_energy_is_final_label"], "false")
            self.assertEqual(row["zero_cost_status"], "pass")
            payload = json.loads(row["ansatz_gene"])
            self.assertEqual(payload["kind"], "chemistry_excitation")
            self.assertEqual(payload["hf_occupied_qubits"], [1, 3])

    def test_build_chemistry_excitation_rows_starts_with_hf_empty_gene(self):
        from aicir.qas.vqe_loop.p0_chemistry_excitation import build_chemistry_excitation_rows

        rows, _summary = build_chemistry_excitation_rows(
            active_electrons=2,
            active_spatial_orbitals=2,
            hamiltonian_id="toy",
            hamiltonian_class="molecular",
            count=2,
            max_excitations=1,
            seed=7,
        )

        first_gene = json.loads(rows[0]["ansatz_gene"])
        self.assertEqual(first_gene["excitations"], [])
        self.assertEqual(rows[0]["depth_group"], "L0")

    def test_ansatz_family_capabilities_make_chemistry_boundary_explicit(self):
        from aicir.qas.vqe_loop.ansatz_family import ansatz_family_capabilities, summarize_ansatz_families

        capabilities = ansatz_family_capabilities("chemistry_excitation")
        self.assertTrue(capabilities["candidate_generation"])
        self.assertTrue(capabilities["mutation"])
        self.assertTrue(capabilities["oracle"])
        self.assertTrue(capabilities["evaluators"]["E2"])
        self.assertTrue(capabilities["evaluators"]["fair_label"])
        self.assertFalse(capabilities["evaluators"]["E5"])
        self.assertFalse(capabilities["native_supernet_screening"])

        summary = summarize_ansatz_families([{"family": "chemistry_excitation"}])
        self.assertEqual(summary["chemistry_excitation"]["count"], 1)
        self.assertFalse(summary["chemistry_excitation"]["capabilities"]["evaluators"]["E5"])
    def test_write_chemistry_excitation_bootstrap_queue_uses_fair_label_schema(self):
        from aicir.qas.vqe_loop.p0_bootstrap_fair import (
            ClosedLoopConfig,
            write_chemistry_excitation_bootstrap_queue,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path, oracle_path, summary_path = write_chemistry_excitation_bootstrap_queue(
                ClosedLoopConfig(
                    output_dir=Path(tmpdir),
                    n_qubits=4,
                    hamiltonian_terms=[(1.0, "ZIII")],
                    hamiltonian_id="toy_lih",
                    hamiltonian_class="molecular",
                    use_chemistry_excitation_pool=True,
                    active_electrons=2,
                    active_spatial_orbitals=2,
                    chemistry_excitation_count=2,
                ),
                output_dir=Path(tmpdir),
            )

            self.assertTrue(queue_path.exists())
            self.assertTrue(oracle_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["source"], "trackB_chemistry_excitation")
            self.assertEqual(summary["planned_total"], 2)
            self.assertTrue(summary["training_free"]["enabled"])
            self.assertEqual(summary["training_free"]["hard_reject_count"], 0)
            self.assertEqual(summary["training_free"]["soft_flag_count"], 0)
            self.assertEqual(summary["training_free"]["pass_count"], 2)
            self.assertEqual(summary["training_free"]["zero_cost_status_counts"], {"hard_reject": 0, "soft_flag": 0, "pass": 2})
            with queue_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["source"], "trackB_chemistry_excitation")
            self.assertEqual(json.loads(rows[0]["hamiltonian_terms"]), [[1.0, "ZIII"]])
            self.assertEqual(json.loads(rows[0]["ansatz_gene"])["kind"], "chemistry_excitation")

if __name__ == "__main__":
    unittest.main()








"""Tests for shared VQE-loop row parsing helpers."""

from __future__ import annotations

import csv
import json
import math
import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.benchmark_table import (
    as_float,
    decoded_ansatz_gene_payload,
    hamiltonian_from_terms,
    is_empty,
    parse_pauli_hamiltonian_terms,
    problem_from_row_terms,
    read_csv_rows,
    read_csv_with_fieldnames,
    row_hamiltonian_terms,
    write_csv_rows,
)


class SharedRowHelpersTest(unittest.TestCase):
    def test_row_values_parse_empty_and_float_fields(self) -> None:
        self.assertTrue(is_empty(None))
        self.assertTrue(is_empty("  "))
        self.assertIsNone(as_float(""))
        self.assertEqual(as_float("1.25"), 1.25)
        self.assertTrue(math.isnan(as_float("nan")))

    def test_csv_rows_preserve_fieldnames_and_fill_missing_values(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv_rows(path, [{"b": 2, "a": 1}, {"a": 3}], fieldnames=["a", "b"])
            fieldnames, rows = read_csv_with_fieldnames(path)
            self.assertEqual(fieldnames, ["a", "b"])
            self.assertEqual(rows, [{"a": "1", "b": "2"}, {"a": "3", "b": ""}])
            self.assertEqual(read_csv_rows(path), rows)

    def test_decoded_ansatz_gene_payload_handles_double_encoded_json(self) -> None:
        payload = {"kind": "operator_sequence", "n_qubits": 2, "operators": ["XI"]}
        row = {"ansatz_gene": json.dumps(json.dumps(payload))}
        self.assertEqual(decoded_ansatz_gene_payload(row), payload)

    def test_hamiltonian_rows_parse_problem_and_operator_hamiltonian(self) -> None:
        row = {
            "n_qubits": "2",
            "hamiltonian_id": "toy",
            "hamiltonian_terms": json.dumps([[1.0, "ZI"], [-0.5, "IZ"]]),
        }
        parsed_directly = parse_pauli_hamiltonian_terms([[1.0, "ZI"], {"coeff": -0.5, "pauli": "iz"}])
        self.assertEqual(parsed_directly, ((1.0, "ZI"), (-0.5, "IZ")))
        terms = row_hamiltonian_terms(row)
        self.assertEqual(terms, parsed_directly)
        problem = problem_from_row_terms(row, n_qubits=2)
        self.assertIsInstance(problem, VQEProblem)
        self.assertEqual(problem.name, "toy")
        operator_hamiltonian = hamiltonian_from_terms(terms, n_qubits=2)
        self.assertEqual(operator_hamiltonian.n_qubits, 2)


if __name__ == "__main__":
    unittest.main()


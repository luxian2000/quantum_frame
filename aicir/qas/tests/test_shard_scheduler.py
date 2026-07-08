import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, read_csv_rows, write_csv_rows
from aicir.qas.vqe_loop.shard_scheduler import _format_shard_failure, _is_completed_label, _merge_shard_outputs


class ShardSchedulerTests(unittest.TestCase):
    def test_completed_label_requires_status_or_fair_energy(self):
        self.assertTrue(_is_completed_label({"label_status": "completed"}))
        self.assertTrue(_is_completed_label({"fair_best_energy": "-1.25"}))
        self.assertFalse(_is_completed_label({"label_status": "running", "fair_best_energy": ""}))
        self.assertFalse(_is_completed_label({"fair_best_energy": "nan"}))

    def test_format_shard_failure_marks_sigkill_as_possible_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_output = root / "out.shard00of01.csv"
            write_csv_rows(
                shard_output,
                [{"architecture_id": "a", "label_status": "running", "fair_best_energy": ""}],
                fieldnames=BENCHMARK_TABLE_FIELDS,
            )

            failure = _format_shard_failure(
                shard_index=0,
                start=0,
                end=1,
                return_code=-9,
                shard_output=shard_output,
            )

        self.assertEqual(failure["signal"], "SIGKILL")
        self.assertTrue(failure["possible_oom"])
        self.assertIn("running=1", failure["shard_output_status"])

    def test_merge_shard_outputs_can_keep_completed_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0 = root / "out.shard00of02.csv"
            shard1 = root / "out.shard01of02.csv"
            output = root / "merged.csv"
            rows0 = [
                {"architecture_id": "a", "label_status": "completed", "fair_best_energy": "-1.0"},
                {"architecture_id": "b", "label_status": "running", "fair_best_energy": ""},
            ]
            rows1 = [
                {"architecture_id": "c", "label_status": "completed", "fair_best_energy": "-2.0"},
            ]
            write_csv_rows(shard0, rows0, fieldnames=BENCHMARK_TABLE_FIELDS)
            write_csv_rows(shard1, rows1, fieldnames=BENCHMARK_TABLE_FIELDS)

            merged = _merge_shard_outputs([shard0, shard1], output, completed_only=True)

            self.assertEqual([row["architecture_id"] for row in merged], ["a", "c"])
            self.assertEqual(
                [row["architecture_id"] for row in read_csv_rows(output)],
                ["a", "c"],
            )


if __name__ == "__main__":
    unittest.main()

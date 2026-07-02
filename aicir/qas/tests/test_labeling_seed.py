import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class LabelingSeedTests(unittest.TestCase):
    def test_architecture_seed_is_stable_across_queue_positions(self):
        from aicir.qas.vqe_loop.fair_labeling import _label_seed_for_row

        row = {
            "architecture_id": "arch_a",
            "canonical_arch_hash": "same-architecture",
        }

        self.assertEqual(
            _label_seed_for_row(base_seed=2026, row_index=0, row=row, seed_by_architecture_id=True),
            _label_seed_for_row(base_seed=2026, row_index=7, row=row, seed_by_architecture_id=True),
        )

    def test_legacy_row_index_seed_remains_available(self):
        from aicir.qas.vqe_loop.fair_labeling import _label_seed_for_row

        self.assertEqual(
            _label_seed_for_row(base_seed=2026, row_index=2, row={}, seed_index_offset=1),
            5026,
        )


if __name__ == "__main__":
    unittest.main()

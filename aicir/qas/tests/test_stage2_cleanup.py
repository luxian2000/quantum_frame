import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class Stage2CleanupTests(unittest.TestCase):
    def test_closed_loop_no_longer_invokes_removed_stage2_module(self):
        source = Path(__file__).resolve().parents[1] / "vqe_loop" / "p0_bootstrap_fair.py"
        text = source.read_text(encoding="utf-8")

        self.assertNotIn("aicir.qas.vqe_loop.stage2", text)


if __name__ == "__main__":
    unittest.main()

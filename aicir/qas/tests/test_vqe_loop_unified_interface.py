import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class VQELoopUnifiedInterfaceTests(unittest.TestCase):
    def test_vqe_loop_is_available_through_qas_run_and_config(self):
        from aicir.qas import QASResult, config, run, available_qas_methods
        from aicir.qas.vqe_loop import ClosedLoopConfig, ClosedLoopResult, P0BootstrapConfig, P0BootstrapResult

        with tempfile.TemporaryDirectory() as temp:
            cfg = config.vqe_loop(
                output_dir=Path(temp) / "qas_loop",
                n_qubits=2,
                hamiltonian_terms=[(-1.0, "ZI")],
                rounds=0,
                initial_labels=0,
            )

            expected = ClosedLoopResult(
                output_dir=cfg.output_dir,
                candidates=cfg.output_dir / "stage0_candidates.csv",
                initial_queue=cfg.output_dir / "stage1_5_initial_label_queue.csv",
                initial_benchmark_table=cfg.output_dir / "benchmark_table_2q_v2.csv",
                final_benchmark_table=cfg.output_dir / "benchmark_table_2q_v2.csv",
                round_summaries=(),
            )

            self.assertIsInstance(cfg, ClosedLoopConfig)
            self.assertIsInstance(cfg, P0BootstrapConfig)
            self.assertIs(ClosedLoopResult, P0BootstrapResult)
            self.assertIn("vqe_loop", available_qas_methods())

            with patch("aicir.qas.vqe_loop.run_vqe_qas_closed_loop", return_value=expected) as runner:
                result = run("vqe_loop", config=cfg)

            runner.assert_called_once_with(config=cfg)
            # 3b：run() 统一返回 QASResult；ClosedLoopResult 只有输出路径（无内存态
            # circuit/energy），完整对象保留在 raw。
            self.assertIsInstance(result, QASResult)
            self.assertEqual(result.method, "vqe_loop")
            self.assertIs(result.raw, expected)
            self.assertIsNone(result.value)


if __name__ == "__main__":
    unittest.main()

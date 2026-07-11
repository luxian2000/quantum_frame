"""crlqas 不应污染 numpy/stdlib 全局随机数状态（局部 rng_py/rng_np 取代全局 seeding）。"""

import random
import unittest

import numpy as np

try:
    from aicir.qas import AdamSPSAConfig, CRLQASConfig, train_crlqas
except ImportError as exc:
    if "AdamSPSAConfig" not in str(exc):
        raise
    raise unittest.SkipTest("CRLQAS tests require torch") from exc


def _np_random_state_equal(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


class TestCRLQASGlobalRngUntouched(unittest.TestCase):
    def test_train_crlqas_does_not_mutate_global_numpy_or_random_state(self):
        np.random.seed(1234)
        random.seed(5678)
        np_state_before = np.random.get_state()
        py_state_before = random.getstate()

        hamiltonian = np.zeros((2, 2), dtype=np.complex64)
        config = CRLQASConfig(
            max_episodes=2,
            n_act=2,
            batch_size=2,
            replay_capacity=32,
            train_interval=1,
            target_update_interval=5,
            log_interval=0,
            adam_spsa=AdamSPSAConfig(iterations=2),
            seed=7,
        )
        train_crlqas(hamiltonian=hamiltonian, config=config)

        self.assertTrue(_np_random_state_equal(np.random.get_state(), np_state_before))
        self.assertEqual(random.getstate(), py_state_before)


if __name__ == "__main__":
    unittest.main()

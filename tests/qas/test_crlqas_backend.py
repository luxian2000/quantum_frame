"""crlqas 后端解析：CRLQASConfig.device 与 qas.core.backend_utils.make_torch_backend。"""

import random
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.backends.numpy_backend import NumpyBackend
from aicir.backends.gpu_backend import GPUBackend
from aicir.qas.algorithms.crlqas import CRLQASConfig, _resolve_crlqas_backend, train_crlqas
from aicir.qas.core.backend_utils import make_torch_backend


def test_crlqas_config_device_defaults_to_none():
    assert CRLQASConfig().device is None


def test_resolve_crlqas_backend_device_none_is_numpy_backend():
    backend = _resolve_crlqas_backend(None)

    assert isinstance(backend, NumpyBackend)


def test_resolve_crlqas_backend_device_cpu_is_torch_backend():
    backend = _resolve_crlqas_backend("cpu")

    assert isinstance(backend, GPUBackend)


def test_make_torch_backend_device_none_matches_original_make_backend_default():
    # 三份原 _make_backend 实现里 device=None 时 str(None) 不以 "npu" 开头，
    # 因此落到 GPUBackend(device=None) 分支——这里锁定该行为不被后续改动破坏。
    backend = make_torch_backend(None)

    assert isinstance(backend, GPUBackend)


def test_make_torch_backend_cpu_device_string_maps_to_gpu_backend_class():
    backend = make_torch_backend("cpu")

    assert isinstance(backend, GPUBackend)


def test_make_torch_backend_npu_prefixed_device_selects_npu_backend_or_falls_back():
    try:
        from aicir.backends.npu_backend import NPUBackend
    except Exception:
        pytest.skip("NPUBackend unavailable in this environment")

    backend = make_torch_backend("npu:0")

    # 真实 Ascend 硬件不可用时 NPUBackend 会透明回退到 CPU，但类型仍是 NPUBackend。
    assert isinstance(backend, NPUBackend)


def _np_random_state_equal(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


class TestCRLQASBackwardCompatBackend(unittest.TestCase):
    """device=None 时训练路径与改造前完全一致（NumpyBackend + 不污染全局 RNG）。"""

    def test_train_crlqas_device_none_keeps_numpy_backend_and_global_rng_untouched(self):
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
            seed=7,
        )

        result = train_crlqas(hamiltonian=hamiltonian, config=config)

        self.assertIsNotNone(result.circuit)
        self.assertTrue(_np_random_state_equal(np.random.get_state(), np_state_before))
        self.assertEqual(random.getstate(), py_state_before)


if __name__ == "__main__":
    unittest.main()

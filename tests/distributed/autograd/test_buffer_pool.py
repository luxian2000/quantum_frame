import torch

from aicir.distributed.autograd._collectives import _AsyncPairExchange, _PairBufferPool


class _Work:
    def __init__(self, done=False):
        self.done = done
        self.waits = 0

    def is_completed(self):
        return self.done

    def wait(self):
        self.waits += 1
        self.done = True


def test_pair_buffer_pool_reuses_only_after_both_component_handles_complete():
    pool = _PairBufferPool()
    first = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    real_work, imag_work = _Work(), _Work()
    pool.release(first, real_work=real_work, imag_work=imag_work)
    second = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    assert second is not first
    real_work.done = True
    third = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    assert third is not first
    imag_work.done = True
    pool.release(second)
    assert pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward") is first
    assert pool.reuse_count == 1


def test_async_pair_exchange_waits_for_both_real_components_before_returning():
    pair = _PairBufferPool().acquire((2,), torch.float32, "cpu", peer=1, phase="forward")
    real_work, imag_work = _Work(), _Work()

    assert _AsyncPairExchange(pair, real_work, imag_work).wait() is pair
    assert real_work.waits == imag_work.waits == 1

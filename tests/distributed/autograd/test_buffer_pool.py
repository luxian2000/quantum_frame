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


def test_release_still_returns_buffer_after_discard_trimmed_bookkeeping():
    """discard 先于 backward 触发时，release 仍必须把 buffer 归还池中。

    `_LaunchedPairExchange.wait` 给前向输出挂了 weakref finalizer 调用
    `discard`，而该输出与 buffer.real 不是同一个 Python 对象，因此 finalizer
    可能早于 `_LaunchedPairExchangeFn.backward` 触发。修复前 release 依赖
    `_checked_out` 取归还键，被 discard 清掉后就变成空操作，buffer 永远回不到
    池里——真机与 CPU 上都表现为 raw_state / density_factor 两条路径的
    buffer_reuse_count 恒为 0。
    """

    pool = _PairBufferPool()
    first = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    pool.discard(id(first))  # finalizer 抢跑
    pool.release(first)  # backward 真正归还
    assert pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward") is first
    assert pool.reuse_count == 1


def test_discard_alone_never_recycles_a_live_forward_output():
    """只有 discard、没有 release 时不得回收：前向输出可能仍被计算图引用。"""

    pool = _PairBufferPool()
    first = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    pool.discard(id(first))
    assert pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward") is not first
    assert pool.reuse_count == 0


def test_release_is_idempotent():
    """重复归还不能把同一个 buffer 放进池两次。"""

    pool = _PairBufferPool()
    first = pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward")
    pool.release(first)
    pool.release(first)
    assert pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward") is first
    assert pool.acquire((4,), torch.float32, "cpu", peer=1, phase="forward") is not first
    assert pool.reuse_count == 1


def test_async_pair_exchange_waits_for_both_real_components_before_returning():
    pair = _PairBufferPool().acquire((2,), torch.float32, "cpu", peer=1, phase="forward")
    real_work, imag_work = _Work(), _Work()

    assert _AsyncPairExchange(pair, real_work, imag_work).wait() is pair
    assert real_work.waits == imag_work.waits == 1

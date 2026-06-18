import pytest

torch = pytest.importorskip("torch")

from aicir.qas.primitives.sharding import (
    ShardContext,
    owned_indices,
    shard_context,
    all_gather,
)
from aicir.backends.npu_backend import NPUBackend
from aicir.backends.gpu_backend import GPUBackend


@pytest.mark.parametrize("n,world", [(10, 4), (8, 2), (5, 3), (3, 4)])
def test_owned_indices_partition_covers_range_once(n, world):
    seen = []
    for rank in range(world):
        seen.extend(owned_indices(n, rank, world))
    assert sorted(seen) == list(range(n))


def test_owned_indices_is_strided():
    assert owned_indices(10, 1, 4) == [1, 5, 9]


def test_shard_context_off_distributed_is_inactive():
    # No process group initialized in this test process.
    ctx = shard_context(NPUBackend(device="npu:0"))
    assert ctx == ShardContext(is_sharded=False, rank=0, world_size=1)


def test_shard_context_non_npu_backend_is_inactive():
    ctx = shard_context(GPUBackend(device="cpu"))
    assert ctx.is_sharded is False


def test_all_gather_without_process_group_returns_singleton():
    assert all_gather({"x": 1}) == [{"x": 1}]

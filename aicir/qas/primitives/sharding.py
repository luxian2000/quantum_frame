"""多 NPU 任务并行的分片原语。

这些工具只在"分布式 NPU 运行"下生效：backend 是 NPUBackend 且 torch.distributed
已初始化、world_size > 1。其它情况下全部退化为本地无操作，调用方无需分支判断
（只读 shard_context 的结果即可）。这里只搬运实数张量和 Python 对象，不做任何
复数 dtype 的集合通信（Ascend NPU 没有复数算子）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ...backends.npu_backend import NPUBackend


@dataclass(frozen=True)
class ShardContext:
    is_sharded: bool
    rank: int
    world_size: int


def _dist_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def shard_context(backend: Any) -> ShardContext:
    """判定当前是否为分布式 NPU 运行，并返回 rank/world_size。"""
    if not isinstance(backend, NPUBackend) or not _dist_ready():
        return ShardContext(is_sharded=False, rank=0, world_size=1)
    world_size = dist.get_world_size()
    if world_size <= 1:
        return ShardContext(is_sharded=False, rank=0, world_size=1)
    return ShardContext(is_sharded=True, rank=dist.get_rank(), world_size=world_size)


def owned_indices(n: int, rank: int, world_size: int) -> list[int]:
    """跨 rank 的等距切分；各 rank 的并集恰好覆盖 range(n) 一次。"""
    return list(range(rank, n, world_size))


def all_gather(obj: Any) -> list[Any]:
    """按 rank 顺序收集每个 rank 的 Python 对象；未分布式时返回 [obj]。"""
    if not _dist_ready():
        return [obj]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def all_reduce_mean(tensors: Sequence[torch.Tensor]) -> None:
    """对一组实数张量做跨 rank 求和再除以 world_size（原地）。未分布式时不操作。"""
    if not _dist_ready():
        return
    world_size = dist.get_world_size()
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)


def broadcast_parameters(params: Mapping[Any, torch.Tensor], src: int = 0) -> None:
    """把 src rank 的实数参数张量广播到所有 rank（原地）。未分布式时不操作。"""
    if not _dist_ready():
        return
    for _, tensor in sorted(params.items(), key=lambda item: str(item[0])):
        dist.broadcast(tensor, src=src)

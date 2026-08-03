"""root-owned scatter 的 requires_grad 必须在所有 rank 上一致。

非 root rank 拿不到 root 的实际参数，只能用占位张量参与 scatter。占位张量过去
硬编码 ``requires_grad=True``：当 root 其实不需要梯度（纯前向求值）时，root 走
forward-only 分支、其余 rank 却建出一张永远不会 backward 的图，
`_LaunchedPairExchange.wait` 因此在非 root rank 上永远等不到归还时机，pooled
buffer 只借不还（真机与 CPU/gloo 上都表现为 density 前向路径在非 root rank 的
``buffer_reuse_count`` 恒为 0）。
"""

import json
import os
from pathlib import Path
import socket

import torch
import torch.multiprocessing as mp

from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._collectives import (
    _PairBufferPool,
    _scatter_root_pair,
)
from aicir.distributed.autograd._pair import _Pair


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    local_shape = (2, 1)
    observed = {}
    for flag in (False, True):
        root_pair = None
        if backend.rank == 0:
            real = torch.ones(
                (world_size, *local_shape),
                dtype=torch.float32,
                requires_grad=flag,
            )
            root_pair = _Pair(real, torch.zeros_like(real))
        scattered = _scatter_root_pair(
            root_pair,
            communicator=backend.communicator,
            root=0,
            local_shape=local_shape,
            root_requires_grad=flag,
        )
        observed[str(flag)] = [
            bool(scattered.real.requires_grad),
            bool(scattered.imag.requires_grad),
        ]

    Path(output_dir, f"rank{rank}.json").write_text(json.dumps(observed, sort_keys=True))
    torch.distributed.destroy_process_group()


def test_scattered_requires_grad_matches_root_on_every_rank(tmp_path):
    world_size = 2
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    reports = [
        json.loads(Path(tmp_path, f"rank{rank}.json").read_text())
        for rank in range(world_size)
    ]
    # 每个 rank 都必须跟随 root：False 时纯前向，True 时全体建图。
    for report in reports:
        assert report["False"] == [False, False]
        assert report["True"] == [True, True]
    assert reports[0] == reports[1], "scatter 后的 requires_grad 在各 rank 间不对称"


def test_forward_only_scatter_returns_pooled_buffers():
    """root 不需要梯度时，pooled buffer 必须当场归还而不是等一个不会来的 backward。"""

    pool = _PairBufferPool()
    first = pool.acquire((2, 1), torch.float32, "cpu", peer=1, phase="forward")
    pool.release(first)
    assert pool.acquire((2, 1), torch.float32, "cpu", peer=1, phase="forward") is first
    assert pool.reuse_count == 1

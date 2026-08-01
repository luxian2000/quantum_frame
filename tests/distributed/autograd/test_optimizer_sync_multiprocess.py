"""Replicated optimizer agreement is driven solely by GradientBucketFn."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._parameters import _bucket_parameters


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join(context, *, timeout=60):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "gradient-bucket optimizer test timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5)
    assert all(process.exitcode == 0 for process in context.processes)


def _digest(parameter, optimizer):
    state = optimizer.state_dict()
    payload = parameter.detach().cpu().numpy().tobytes()
    for key in sorted(state["state"]):
        for field, value in sorted(state["state"][key].items()):
            payload += field.encode("ascii")
            if isinstance(value, torch.Tensor):
                payload += value.detach().cpu().numpy().tobytes()
            else:
                payload += repr(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        results = {}
        for count in (32, 128):
            for optimizer_name, factory in (
                ("sgd", lambda values: torch.optim.SGD([values], lr=0.01, momentum=0.9)),
                ("adam", lambda values: torch.optim.Adam([values], lr=0.01)),
            ):
                parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.5, count, dtype=torch.float32))
                optimizer = factory(parameter)
                before = len(backend.communicator.communication_records)
                for _ in range(100):
                    optimizer.zero_grad(set_to_none=True)
                    (alias,) = _bucket_parameters((parameter,), communicator=backend.communicator)
                    # Every rank supplies a distinct local VJP; the one bucket
                    # makes the optimizer see the exact global gradient.
                    (alias * float(rank + 1)).sum().backward()
                    optimizer.step()
                    digests = [None] * world_size
                    torch.distributed.all_gather_object(digests, _digest(parameter, optimizer))
                    assert len(set(digests)) == 1
                records = backend.communicator.communication_records[before:]
                results[f"{optimizer_name}-{count}"] = {
                    "all_reduce_count": sum(record["kind"] == "all_reduce" for record in records),
                    "all_float32": all(record["dtype"] == "torch.float32" for record in records),
                    "digest": _digest(parameter, optimizer),
                }
        # Initial-state leaves are not passed to the replicated bucket.  Root
        # ownership and sharded ownership remain local contracts (covered by
        # the initial-state multiprocess suite) rather than optimizer aliases.
        root_owned = torch.tensor([1.0], dtype=torch.float32, requires_grad=True) if rank == 0 else None
        sharded = torch.tensor([float(rank + 1)], dtype=torch.float32, requires_grad=True)
        (3.0 * sharded).backward()
        if rank == 0:
            (2.0 * root_owned).backward()
            Path(output_path).write_text(json.dumps({"results": results, "root_grad": root_owned.grad.tolist()}), encoding="utf-8")
        assert sharded.grad.item() == pytest.approx(3.0)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_replicated_sgd_and_adam_agree_for_100_steps_on_two_and_four_ranks(tmp_path, world_size):
    output = tmp_path / f"optimizer-{world_size}.json"
    context = mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=False)
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["root_grad"] == [2.0]
    assert set(result["results"]) == {"sgd-32", "sgd-128", "adam-32", "adam-128"}
    for metrics in result["results"].values():
        assert metrics["all_reduce_count"] == 100
        assert metrics["all_float32"]

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

from aicir import Circuit
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, PureStateParam
from aicir.distributed._contracts import PARAMETER_STRUCTURE_ERROR
from aicir.distributed.autograd._parameters import (
    _bind_replicated_gradient_bucket,
    _bucket_parameters,
)


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
        # Initial-state leaves deliberately bypass the replicated bucket.
        # Root-owned optimizer state exists only on rank 0; shard-local leaves
        # have independent optimizer state but are globally normalized after
        # every step through their explicit paired-real forward.
        root_real = torch.nn.Parameter(torch.tensor([1.0, 0.5], dtype=torch.float32)) if rank == 0 else None
        root_imag = torch.nn.Parameter(torch.tensor([0.0, 0.25], dtype=torch.float32)) if rank == 0 else None
        root_optimizer = torch.optim.Adam([root_real, root_imag], lr=0.01) if rank == 0 else None
        sharded_real = torch.nn.Parameter(torch.tensor([float(rank + 1)], dtype=torch.float32))
        sharded_imag = torch.nn.Parameter(torch.tensor([float(rank)], dtype=torch.float32))
        sharded_optimizer = torch.optim.SGD([sharded_real, sharded_imag], lr=0.01, momentum=0.9)
        sharded_normalized = True
        for _ in range(100):
            if rank == 0:
                root_optimizer.zero_grad(set_to_none=True)
                root_pair = PureStateParam(root_real, root_imag).normalized_pair()
                root_pair.abs_sq().sum().backward()
                root_optimizer.step()
            sharded_optimizer.zero_grad(set_to_none=True)
            (sharded_real.square() + sharded_imag.square()).sum().backward()
            sharded_optimizer.step()
            total = backend.communicator.all_reduce_sum(
                (sharded_real.square() + sharded_imag.square()).sum().reshape(())
            )
            local_probability = (sharded_real.square() + sharded_imag.square()).sum() / total
            sharded_normalized = sharded_normalized and bool(torch.isfinite(local_probability))
        ownership = [None] * world_size
        torch.distributed.all_gather_object(ownership, {
            "root_has_state": bool(root_optimizer is not None and root_optimizer.state),
            "sharded_has_state": bool(sharded_optimizer.state),
            "sharded_normalized": sharded_normalized,
        })
        if rank == 0:
            Path(output_path).write_text(json.dumps({"results": results, "ownership": ownership}), encoding="utf-8")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_replicated_sgd_and_adam_agree_for_100_steps_on_two_and_four_ranks(tmp_path, world_size):
    output = tmp_path / f"optimizer-{world_size}.json"
    context = mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=False)
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ownership"][0]["root_has_state"]
    assert all(not item["root_has_state"] for item in result["ownership"][1:])
    assert all(item["sharded_has_state"] and item["sharded_normalized"] for item in result["ownership"])
    assert set(result["results"]) == {"sgd-32", "sgd-128", "adam-32", "adam-128"}
    for metrics in result["results"].values():
        assert metrics["all_reduce_count"] == 100
        assert metrics["all_float32"]


def _schema_mismatch_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=rank == 0)
        circuit = Circuit(ry(theta, 0), n_qubits=1)
        backend.communicator.clear_communication_records()
        try:
            _bind_replicated_gradient_bucket(circuit, communicator=backend.communicator)
        except ValueError as error:
            message = str(error)
        else:
            message = "NO_ERROR"
        records = backend.communicator.communication_records
        observed = [None] * world_size
        torch.distributed.all_gather_object(
            observed,
            {
                "message": message,
                "gradient_all_reduces": sum(record["kind"] == "all_reduce" for record in records),
            },
        )
        if rank == 0:
            Path(output_path).write_text(json.dumps(observed), encoding="utf-8")
    finally:
        torch.distributed.destroy_process_group()


def test_requires_grad_schema_mismatch_is_collective_safe_before_bucket_allreduce(tmp_path):
    output = tmp_path / "schema-mismatch.json"
    context = mp.spawn(_schema_mismatch_worker, args=(2, _free_port(), str(output)), nprocs=2, join=False)
    _join(context)

    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == [
        {"message": PARAMETER_STRUCTURE_ERROR, "gradient_all_reduces": 0},
        {"message": PARAMETER_STRUCTURE_ERROR, "gradient_all_reduces": 0},
    ]

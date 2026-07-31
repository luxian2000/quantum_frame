import json
import os
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from aicir.distributed.autograd._collectives import (
    _exchange_pair,
    _gather_root_pair,
    _replicated_all_reduce,
    _scatter_root_pair,
)
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.communication import _Communicator


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join_spawn_context(context, *, timeout=30):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, (
                "distributed collective sequence timed out"
            )
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

    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode == 0 for process in context.processes)


def _communicator(rank, world_size):
    return _Communicator(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )


def _collectives_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        communicator = _communicator(rank, world_size)
        peer = rank ^ 1
        real = torch.tensor([float(rank + 1)], dtype=torch.float32, requires_grad=True)
        imag = torch.tensor([-float(rank + 1)], dtype=torch.float32, requires_grad=True)
        exchanged = _exchange_pair(
            _Pair(real, imag),
            communicator=communicator,
            peer=peer,
            operation_index=3,
            phase="forward",
        )
        exchange_loss = exchanged.abs_sq().sum()
        exchange_loss.backward()

        local = torch.tensor(float(rank + 1), dtype=torch.float32, requires_grad=True)
        global_value = _replicated_all_reduce(local, communicator=communicator)
        global_value.backward()

        local_shape = (2,)
        source = (
            _Pair(
                torch.arange(world_size * 2, dtype=torch.float32).reshape(world_size, 2).requires_grad_(),
                (-torch.arange(world_size * 2, dtype=torch.float32).reshape(world_size, 2)).requires_grad_(),
            )
            if rank == 0
            else None
        )
        scattered = _scatter_root_pair(
            source,
            communicator=communicator,
            root=0,
            local_shape=local_shape,
        )
        gathered = _gather_root_pair(scattered, communicator=communicator, root=0)
        scatter_loss = scattered.abs_sq().sum()
        scatter_loss.backward()

        torch.distributed.barrier()
        records = list(communicator.communication_records)
        payload = {
            "rank": rank,
            "exchange": {
                "real": exchanged.real.detach().tolist(),
                "imag": exchanged.imag.detach().tolist(),
                "real_grad": real.grad.tolist(),
                "imag_grad": imag.grad.tolist(),
            },
            "replicated_grad": local.grad.item(),
            "gathered": (
                {"real": gathered.real.detach().tolist(), "imag": gathered.imag.detach().tolist()}
                if gathered is not None
                else None
            ),
            "source_grad": (
                {"real": source.real.grad.tolist(), "imag": source.imag.grad.tolist()}
                if source is not None
                else None
            ),
            "records": records,
        }
        all_payloads = [None] * world_size
        torch.distributed.all_gather_object(all_payloads, payload)
        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(all_payloads, handle)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_paired_real_collectives_are_differentiable_and_float32_only(tmp_path, world_size):
    output_path = str(tmp_path / f"collectives-{world_size}.json")
    context = mp.spawn(
        _collectives_worker,
        args=(world_size, _free_port(), output_path),
        nprocs=world_size,
        join=False,
    )
    _join_spawn_context(context)

    payloads = json.loads((tmp_path / f"collectives-{world_size}.json").read_text())
    expected_real = [[float((rank ^ 1) + 1)] for rank in range(world_size)]
    expected_imag = [[-float((rank ^ 1) + 1)] for rank in range(world_size)]
    assert [payload["exchange"]["real"] for payload in payloads] == expected_real
    assert [payload["exchange"]["imag"] for payload in payloads] == expected_imag
    for rank, payload in enumerate(payloads):
        assert payload["exchange"]["real_grad"] == [2.0 * (rank + 1)]
        assert payload["exchange"]["imag_grad"] == [-2.0 * (rank + 1)]
        assert payload["replicated_grad"] == pytest.approx(1.0)

    root = payloads[0]
    assert root["gathered"] == {
        "real": [[0.0, 1.0], [2.0, 3.0]]
        if world_size == 2
        else [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        "imag": [[0.0, -1.0], [-2.0, -3.0]]
        if world_size == 2
        else [[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0], [-6.0, -7.0]],
    }
    assert root["source_grad"] == {
        "real": [[0.0, 2.0], [4.0, 6.0]]
        if world_size == 2
        else [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0], [12.0, 14.0]],
        "imag": [[0.0, -2.0], [-4.0, -6.0]]
        if world_size == 2
        else [[0.0, -2.0], [-4.0, -6.0], [-8.0, -10.0], [-12.0, -14.0]],
    }
    records = [record for payload in payloads for record in payload["records"]]
    assert records
    assert all(record["dtype"] == "torch.float32" for record in records)
    tagged = [record["tag"] for record in records if record["tag"] is not None]
    assert 24 in tagged and 25 in tagged
    assert 28 in tagged and 29 in tagged
    assert {24, 25}.isdisjoint({28, 29})
    assert all(record["bytes"] > 0 for record in records)


def _shape_mismatch_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        communicator = _communicator(rank, world_size)
        source = (
            _Pair(torch.ones((world_size, 2), dtype=torch.float32), torch.zeros((world_size, 2), dtype=torch.float32))
            if rank == 0
            else None
        )
        try:
            _scatter_root_pair(
                source,
                communicator=communicator,
                root=0,
                local_shape=(2,) if rank == 0 else (3,),
            )
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("shape mismatch must be rejected before scatter")
        torch.distributed.barrier()
        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({"message": message, "barrier": True}, handle)
    finally:
        torch.distributed.destroy_process_group()


def test_shape_mismatch_is_collective_safe_and_processes_are_cleaned_up(tmp_path):
    output_path = str(tmp_path / "shape-mismatch.json")
    context = mp.spawn(
        _shape_mismatch_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=False,
    )
    _join_spawn_context(context)

    payload = json.loads((tmp_path / "shape-mismatch.json").read_text())
    assert payload["barrier"] is True
    assert "local_shape" in payload["message"]

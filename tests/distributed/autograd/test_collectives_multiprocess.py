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


def _seven_dim_shape_mismatch_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    class _GuardedCommunicator(_Communicator):
        def scatter_from_root_real(self, *_args, **_kwargs):
            raise AssertionError("seven-dimensional mismatch reached scatter data plane")

        def gather_to_root_real(self, *_args, **_kwargs):
            raise AssertionError("seven-dimensional mismatch reached gather data plane")

    try:
        communicator = _GuardedCommunicator(
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        local_shape = (1, 1, 1, 1, 1, 1, 1 if rank == 0 else 2)
        root_pair = (
            _Pair(
                torch.ones((world_size, 1, 1, 1, 1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((world_size, 1, 1, 1, 1, 1, 1, 1), dtype=torch.float32),
            )
            if rank == 0
            else None
        )
        gather_pair = _Pair(
            torch.ones(local_shape, dtype=torch.float32),
            torch.zeros(local_shape, dtype=torch.float32),
        )
        messages = {}
        for name, call in {
            "scatter": lambda: _scatter_root_pair(
                root_pair,
                communicator=communicator,
                root=0,
                local_shape=local_shape,
            ),
            "gather": lambda: _gather_root_pair(
                gather_pair,
                communicator=communicator,
                root=0,
            ),
        }.items():
            try:
                call()
            except ValueError as error:
                messages[name] = str(error)
            else:
                raise AssertionError(f"{name} must reject a 7D mismatch before data transport")
            torch.distributed.barrier()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, messages)
        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({"messages": gathered, "barrier": True}, handle)
    finally:
        torch.distributed.destroy_process_group()


def test_seven_dim_shape_mismatch_is_collective_safe_before_scatter_or_gather(tmp_path):
    output_path = str(tmp_path / "seven-dim-shape-mismatch.json")
    context = mp.spawn(
        _seven_dim_shape_mismatch_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=False,
    )
    _join_spawn_context(context)

    payload = json.loads((tmp_path / "seven-dim-shape-mismatch.json").read_text())
    assert payload["barrier"] is True
    assert payload["messages"][0] == payload["messages"][1]
    assert "local_shape" in payload["messages"][0]["scatter"]
    assert "pair shape" in payload["messages"][0]["gather"]


def _mutated_pair(*, dtype=torch.float32, device="cpu", shape=(1,)):
    pair = _Pair(
        torch.ones(shape, dtype=torch.float32),
        torch.zeros(shape, dtype=torch.float32),
    )
    object.__setattr__(pair, "real", torch.ones(shape, dtype=dtype, device=device))
    object.__setattr__(pair, "imag", torch.zeros(shape, dtype=dtype, device=device))
    return pair


def _preflight_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        class _BadPhase:
            def __eq__(self, _other):
                raise RuntimeError("phase equality must not run")

        communicator = _communicator(rank, world_size)
        valid_pair = _Pair(
            torch.ones(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
        )
        source = _Pair(
            torch.ones((world_size, 1), dtype=torch.float32),
            torch.zeros((world_size, 1), dtype=torch.float32),
        )
        exchange_cases = {
            "backward_phase": {
                "pair": valid_pair,
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "backward",
            },
            "phase_disagreement": {
                "pair": valid_pair,
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "forward" if rank == 0 else "backward",
            },
            "phase_conversion": {
                "pair": valid_pair,
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": _BadPhase() if rank == 0 else "forward",
            },
            "peer_conversion": {
                "pair": valid_pair,
                "peer": rank ^ 1 if rank else object(),
                "operation_index": 0,
                "phase": "forward",
            },
            "peer_range": {
                "pair": valid_pair,
                "peer": world_size,
                "operation_index": 0,
                "phase": "forward",
            },
            "peer_self_or_topology": {
                "pair": valid_pair,
                "peer": rank if rank == 0 else 0,
                "operation_index": 0,
                "phase": "forward",
            },
            "pair_type": {
                "pair": valid_pair if rank else None,
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "forward",
            },
            "pair_dtype": {
                "pair": valid_pair if rank else _mutated_pair(dtype=torch.float64),
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "forward",
            },
            "pair_device": {
                "pair": valid_pair if rank else _mutated_pair(device="meta"),
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "forward",
            },
            "pair_shape": {
                "pair": valid_pair if rank else _mutated_pair(shape=(2,)),
                "peer": rank ^ 1,
                "operation_index": 0,
                "phase": "forward",
            },
            "operation_conversion": {
                "pair": valid_pair,
                "peer": rank ^ 1,
                "operation_index": object() if rank == 0 else 0,
                "phase": "forward",
            },
        }
        scatter_cases = {
            "root_conversion": {
                "pair": source if rank == 0 else None,
                "root": object() if rank == 0 else 0,
                "local_shape": (1,),
            },
            "root_range": {
                "pair": source if rank == 0 else None,
                "root": world_size,
                "local_shape": (1,),
            },
            "root_disagreement": {
                "pair": source,
                "root": rank,
                "local_shape": (1,),
            },
            "shape_conversion": {
                "pair": source if rank == 0 else None,
                "root": 0,
                "local_shape": (1,) if rank == 0 else (object(),),
            },
            "root_pair_type": {
                "pair": None,
                "root": 0,
                "local_shape": (1,),
            },
            "nonroot_pair": {
                "pair": source,
                "root": 0,
                "local_shape": (1,),
            },
            "root_pair_dtype": {
                "pair": _mutated_pair(dtype=torch.float64, shape=(world_size, 1)) if rank == 0 else None,
                "root": 0,
                "local_shape": (1,),
            },
            "root_pair_shape": {
                "pair": _mutated_pair(shape=(world_size, 2)) if rank == 0 else None,
                "root": 0,
                "local_shape": (1,),
            },
        }
        messages = {}
        for name, kwargs in exchange_cases.items():
            try:
                _exchange_pair(kwargs.pop("pair"), communicator=communicator, **kwargs)
            except (TypeError, ValueError) as error:
                messages[name] = str(error)
            else:
                raise AssertionError(f"{name} must fail before P2P")
            torch.distributed.barrier()
        for name, kwargs in scatter_cases.items():
            try:
                _scatter_root_pair(kwargs.pop("pair"), communicator=communicator, **kwargs)
            except (TypeError, ValueError) as error:
                messages[name] = str(error)
            else:
                raise AssertionError(f"{name} must fail before scatter")
            torch.distributed.barrier()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, messages)
        records = [None] * world_size
        torch.distributed.all_gather_object(
            records,
            list(communicator.communication_records),
        )
        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({"messages": gathered, "records": records}, handle)
    finally:
        torch.distributed.destroy_process_group()


def test_malformed_collective_inputs_fail_collectively_before_data_transport(tmp_path):
    output_path = str(tmp_path / "preflight.json")
    context = mp.spawn(
        _preflight_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=False,
    )
    _join_spawn_context(context)

    payload = json.loads((tmp_path / "preflight.json").read_text())
    messages = payload["messages"]
    assert messages[0] == messages[1]
    assert set(messages[0]) == {
        "backward_phase",
        "phase_disagreement",
        "phase_conversion",
        "peer_conversion",
        "peer_range",
        "peer_self_or_topology",
        "pair_type",
        "pair_dtype",
        "pair_device",
        "pair_shape",
        "operation_conversion",
        "root_conversion",
        "root_range",
        "root_disagreement",
        "shape_conversion",
        "root_pair_type",
        "nonroot_pair",
        "root_pair_dtype",
        "root_pair_shape",
    }
    assert "phase" in messages[0]["backward_phase"]
    assert all(
        record["kind"] == "all_gather"
        for rank_records in payload["records"]
        for record in rank_records
    )

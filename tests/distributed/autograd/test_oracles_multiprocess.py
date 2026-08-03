import json
import os
import socket
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir.distributed import parameter_shift_gradient


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join_spawn_context(context, *, timeout=20):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            assert time.monotonic() < deadline, (
                "distributed worker collective sequence timed out"
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


def _ordering_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        theta = np.array([0.2, -0.4], dtype=np.float64)
        shift = np.pi / 2.0
        records = []

        def objective(values):
            delta = np.asarray(values) - theta
            changed = np.flatnonzero(np.abs(delta) > 1e-12)
            assert changed.size == 1
            index = int(changed[0])
            sign = int(np.sign(delta[index]))
            assert abs(abs(delta[index]) - shift) < 1e-12
            records.append((index, sign))
            return np.sin(values).sum()

        gradient = parameter_shift_gradient(objective, theta, shift=shift)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, records)

        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "records": gathered,
                        "gradient": gradient.tolist(),
                    },
                    handle,
                )
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_gloo_records_identical_parameter_shift_order(tmp_path):
    output_path = str(tmp_path / "oracle-ordering.json")
    context = mp.spawn(
        _ordering_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=False,
    )
    _join_spawn_context(context)

    payload = json.loads((tmp_path / "oracle-ordering.json").read_text())
    assert payload["records"] == [[[0, 1], [0, -1], [1, 1], [1, -1]]] * 2
    np.testing.assert_allclose(
        payload["gradient"],
        np.cos(np.array([0.2, -0.4])),
        atol=1e-12,
    )

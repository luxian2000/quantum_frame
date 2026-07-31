import json
import os
import socket
import time

import torch
import torch.multiprocessing as mp

from aicir.distributed.communication import _Communicator
from scripts.npu import distributed_autograd_probe as probe


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


class _CpuCollectiveBackend:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._device = torch.device("cpu")
        self.communicator = _Communicator(
            rank=rank,
            world_size=world_size,
            device=self._device,
        )


def test_environment_section_is_invoked_with_its_backend(monkeypatch):
    backend = _CpuCollectiveBackend(rank=0, world_size=1)
    backend.local_rank = 0
    backend._dtype = torch.complex64
    monkeypatch.setattr(probe.torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(probe.torch.distributed, "barrier", lambda: None)

    result = probe._run_section_collectively(
        backend,
        "environment",
        runner=probe._environment_section,
    )

    assert result["passed"] is True
    assert result["device_mapping"] == {
        "rank": 0,
        "local_rank": 0,
        "device": "cpu",
    }
    assert set(result["paired_real_on_npu"]) == {
        "add",
        "mul",
        "div_real",
        "matmul",
        "dagger",
        "index_select",
        "abs_sq",
    }


def test_communication_section_reports_only_real_completed_p2p_payloads():
    class _RecordingCommunicator:
        rank = 0
        world_size = 2
        device = torch.device("cpu")

        def __init__(self):
            self._records = []

        @property
        def communication_records(self):
            return tuple(self._records)

        def clear_communication_records(self):
            self._records.clear()

        def all_gather_real(self, descriptor):
            remote = descriptor.clone()
            remote[2] = 0.0
            return [descriptor, remote]

        def exchange_real(self, tensor, *, peer, tag):
            self._records.append(
                {
                    "kind": "exchange",
                    "dtype": str(tensor.dtype),
                    "peer": peer,
                    "tag": tag,
                    "bytes": tensor.numel() * tensor.element_size() * 2,
                }
            )
            return tensor.clone()

    class _Backend:
        rank = 0
        world_size = 2
        _device = torch.device("cpu")

        def __init__(self):
            self.communicator = _RecordingCommunicator()

    result = probe._communication_section(_Backend())

    assert result["passed"] is True
    assert result["local_gate_p2p_delta"] == 0
    assert result["forward_p2p"] == 2
    assert result["backward_p2p"] == 2
    assert result["payload_dtypes"] == ["torch.float32"]
    assert result["all_handles_complete"] is True
    assert set(result["forward_tags"]).isdisjoint(result["backward_tags"])


def _failure_worker(rank, world_size, port, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        backend = _CpuCollectiveBackend(rank, world_size)

        def runner(_backend):
            if rank == 1:
                raise RuntimeError("one-rank section failure")
            return {"status": "PASS", "passed": True}

        result = probe._run_section_collectively(
            backend,
            "gates",
            runner=runner,
        )
        torch.distributed.barrier()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, result)
        if rank == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump({"results": gathered, "post_error_barrier": True}, handle)
    finally:
        torch.distributed.destroy_process_group()


def test_one_rank_section_error_is_bounded_synchronized_and_collective_safe(tmp_path):
    output_path = str(tmp_path / "section-failure.json")
    context = mp.spawn(
        _failure_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=False,
    )
    _join_spawn_context(context)

    payload = json.loads((tmp_path / "section-failure.json").read_text())
    assert payload["post_error_barrier"] is True
    expected = {
        "status": "FAIL",
        "passed": False,
        "error": {
            "rank": 1,
            "type": "RuntimeError",
            "message": "one-rank section failure",
        },
    }
    assert payload["results"] == [expected, expected]

import pytest
import torch

from aicir.distributed import DistNPUBackend


def _set_runtime_env(monkeypatch, *, world_size, rank=0, local_rank=0):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))


def test_backend_rejects_non_power_of_two_world(monkeypatch):
    _set_runtime_env(monkeypatch, world_size=3)

    with pytest.raises(ValueError, match="2 的幂"):
        DistNPUBackend.from_env(
            fallback_to_cpu=True,
            init_process_group=False,
        )


def test_backend_exposes_global_and_local_rank(monkeypatch):
    _set_runtime_env(monkeypatch, world_size=4, rank=2, local_rank=1)

    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )

    assert backend.world_size == 4
    assert backend.rank == 2
    assert backend.local_rank == 1
    assert backend.communicator.rank == 2
    assert backend.communicator.world_size == 4


def test_backend_only_accepts_complex64():
    with pytest.raises(ValueError, match="complex64"):
        DistNPUBackend(
            dtype=torch.complex128,
            device="cpu",
            fallback_to_cpu=True,
        )


def test_backend_forbids_batch_parallel_helpers(monkeypatch):
    _set_runtime_env(monkeypatch, world_size=1)
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )

    with pytest.raises(RuntimeError, match="状态分片"):
        backend.should_run_batch_index(0)
    with pytest.raises(RuntimeError, match="状态分片"):
        backend.gather_indexed_results([])


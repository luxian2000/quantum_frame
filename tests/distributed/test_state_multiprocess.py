import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _state_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    layout = _Layout.explicit(
        (0, 1),
        n_qubits=2,
        distributed_axes=1,
    )

    vector_spec = _ShardSpec.build(2, world_size, rank, "vector", layout)
    vector_local = torch.tensor(
        [[rank * 2], [rank * 2 + 1]],
        dtype=torch.complex64,
    )
    vector = DistState.from_local(
        vector_local,
        spec=vector_spec,
        backend=backend,
    )
    vector_array = vector.to_numpy(root=0)

    matrix_spec = _ShardSpec.build(2, world_size, rank, "matrix", layout)
    full_matrix = torch.eye(4, dtype=torch.complex64)
    matrix = DistState.from_local(
        full_matrix[rank * 2 : (rank + 1) * 2],
        spec=matrix_spec,
        backend=backend,
    )
    matrix_array = matrix.to_numpy(root=0)

    output = Path(output_dir)
    if rank == 0:
        np.save(output / "vector.npy", vector_array)
        np.save(output / "matrix.npy", matrix_array)
    else:
        assert vector_array is None
        assert matrix_array is None
        (output / f"rank-{rank}-none").touch()

    torch.distributed.destroy_process_group()


def test_two_rank_gloo_gathers_vector_and_matrix_on_root(tmp_path):
    world_size = 2
    mp.spawn(
        _state_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    np.testing.assert_array_equal(
        np.load(tmp_path / "vector.npy"),
        np.arange(4, dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "matrix.npy"),
        np.eye(4, dtype=np.complex64),
    )
    assert (tmp_path / "rank-1-none").exists()

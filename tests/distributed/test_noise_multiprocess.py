import os
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import AmplitudeDampingChannel
from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.density import _MatrixKernel
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _noise_worker(rank, world_size, port, output_path):
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
        (0,),
        n_qubits=1,
        distributed_axes=1,
    )
    spec = _ShardSpec.build(1, world_size, rank, "matrix", layout)
    full = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0]],
        dtype=torch.complex64,
    )
    state = DistState.from_local(
        full[spec.global_start : spec.global_stop],
        spec=spec,
        backend=backend,
    )

    result = _MatrixKernel(backend).apply_channel(
        state,
        AmplitudeDampingChannel(target_qubit=0, gamma=1.0),
        instruction_index=0,
    )

    array = result.to_numpy(root=0)
    if rank == 0:
        np.save(output_path, array)
    else:
        assert array is None
    torch.distributed.destroy_process_group()


def test_two_rank_amplitude_damping_preserves_trace(tmp_path):
    output_path = str(tmp_path / "damped.npy")
    mp.spawn(
        _noise_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=True,
    )

    expected = np.array(
        [[1.0, 0.0], [0.0, 0.0]],
        dtype=np.complex64,
    )
    actual = np.load(output_path)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.trace(actual), 1.0, atol=1e-6)

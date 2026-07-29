import os
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import cx, hadamard
from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.density import _MatrixKernel
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _density_worker(rank, world_size, port, output_path):
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
    spec = _ShardSpec.build(2, world_size, rank, "matrix", layout)
    full = torch.zeros(4, 4, dtype=torch.complex64)
    full[0, 0] = 1.0
    state = DistState.from_local(
        full[spec.global_start : spec.global_stop],
        spec=spec,
        backend=backend,
    )
    planner = _GatePlanner(backend, layout, 2)
    kernel = _MatrixKernel(backend)

    for index, gate in enumerate(
        (
            hadamard(0),
            cx(target_qubit=1, control_qubits=(0,)),
        )
    ):
        state = kernel.apply_unitary(
            state,
            planner.plan(gate, instruction_index=index),
        )

    array = state.to_numpy(root=0)
    if rank == 0:
        np.save(output_path, array)
    else:
        assert array is None
    torch.distributed.destroy_process_group()


def test_two_rank_density_evolution_builds_bell_matrix(tmp_path):
    output_path = str(tmp_path / "bell-rho.npy")
    mp.spawn(
        _density_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=True,
    )

    bell = np.array(
        [2**-0.5, 0.0, 0.0, 2**-0.5],
        dtype=np.complex64,
    )
    expected = np.outer(bell, bell.conj())
    np.testing.assert_allclose(
        np.load(output_path),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


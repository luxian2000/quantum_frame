import os
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import cx, hadamard
from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.gates import _GatePlanner, _VectorKernel
from aicir.distributed.layout import _Layout


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _bell_worker(rank, world_size, port, output_path):
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
    state = DistState.zero(2, backend=backend, layout=layout)
    planner = _GatePlanner(backend, layout, 2)
    kernel = _VectorKernel(backend)

    for index, gate in enumerate(
        (
            hadamard(0),
            cx(target_qubit=1, control_qubits=(0,)),
        )
    ):
        state = kernel.apply(
            state,
            planner.plan(gate, instruction_index=index),
        )

    array = state.to_numpy(root=0)
    if rank == 0:
        np.save(output_path, array)
    else:
        assert array is None
    torch.distributed.destroy_process_group()


def _ghz_worker(rank, world_size, port, output_path):
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
        (0, 1, 2),
        n_qubits=3,
        distributed_axes=2,
    )
    state = DistState.zero(3, backend=backend, layout=layout)
    planner = _GatePlanner(backend, layout, 3)
    kernel = _VectorKernel(backend)

    for index, gate in enumerate(
        (
            hadamard(0),
            cx(target_qubit=1, control_qubits=(0,)),
            cx(target_qubit=2, control_qubits=(1,)),
        )
    ):
        state = kernel.apply(
            state,
            planner.plan(gate, instruction_index=index),
        )

    array = state.to_numpy(root=0)
    if rank == 0:
        np.save(output_path, array)
    else:
        assert array is None
    torch.distributed.destroy_process_group()


def test_two_rank_gloo_builds_bell_state(tmp_path):
    output_path = str(tmp_path / "bell.npy")
    mp.spawn(
        _bell_worker,
        args=(2, _free_port(), output_path),
        nprocs=2,
        join=True,
    )

    expected = np.array(
        [2**-0.5, 0.0, 0.0, 2**-0.5],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(
        np.load(output_path),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_four_rank_gloo_builds_ghz_state(tmp_path):
    output_path = str(tmp_path / "ghz.npy")
    mp.spawn(
        _ghz_worker,
        args=(4, _free_port(), output_path),
        nprocs=4,
        join=True,
    )

    expected = np.zeros(8, dtype=np.complex64)
    expected[0] = 2**-0.5
    expected[7] = 2**-0.5
    np.testing.assert_allclose(
        np.load(output_path),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
